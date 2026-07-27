#!/usr/bin/env python3
"""Generate CoTs and compute one CIE-SCORER unfaithfulness score per JSONL row."""

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .circuit import (
    ALWAYS_INCLUDE_LAST,
    BATCH_SIZE,
    BETA,
    CANDIDATE_BUDGET,
    DESIRED_LOGIT_PROB,
    DIVERSITY_PENALTY,
    ENTROPY_TEMPERATURE,
    MAX_FEATURE_NODES,
    MAX_FEATURES_PER_POSITION,
    MAX_N_LOGITS,
    SENTENCE_LAMBDA,
    TOKEN_BUDGET,
    UPDATE_INTERVAL,
    attribute_compressed,
    decode_selected_tokens,
    select_positions_entropy_causal,
)
from .detector import (
    FrozenSentenceEncoder,
    JointFGWUnfaithfulnessDetector,
    TraceRecord,
)
from .graph_features import graph_to_data_compressed
from circuit_tracer import ReplacementModel


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TRANSCODERS = "facebook/crv-8b-instruct-transcoders"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a response for each input prompt and emit its learned "
            "CIE-SCORER unfaithfulness score."
        )
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--transcoder_name", default=DEFAULT_TRANSCODERS)
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--raw_prompt",
        action="store_true",
        help="Do not wrap prompts with the tokenizer's instruct chat template.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate responses, circuits, and the final output.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            row_id = str(row.get("idx", len(rows)))
            if row_id in seen_ids:
                raise ValueError(f"Duplicate idx {row_id!r} on line {line_no}")
            seen_ids.add(row_id)
            row["_cie_row_id"] = row_id
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def render_generation_prompt(
    tokenizer: AutoTokenizer, prompt: str, raw_prompt: bool
) -> str:
    if raw_prompt or not getattr(tokenizer, "chat_template", None):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def split_reasoning_steps(completion: str) -> Tuple[List[str], str]:
    """Extract Step N blocks, with a conservative fallback for free-form CoTs."""
    text = completion.strip()
    final_match = re.search(r"(?im)^\s*final\s+answer\s*:", text)
    reasoning = text[: final_match.start()].strip() if final_match else text
    final_answer = text[final_match.start() :].strip() if final_match else ""

    matches = list(re.finditer(r"(?im)^\s*(?:#+\s*)?step\s+(\d+)\s*:", reasoning))
    steps: List[str] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(reasoning)
        step = reasoning[match.start() : end].strip()
        if step:
            steps.append(step)

    if not steps:
        steps = [part.strip() for part in re.split(r"\n\s*\n+", reasoning) if part.strip()]
    if len(steps) == 1:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", reasoning)
            if part.strip()
        ]
        if len(sentences) > 1:
            steps = [f"Step {i}: {sentence}" for i, sentence in enumerate(sentences, 1)]
    if not steps and final_answer:
        steps = [final_answer]
    if not steps:
        raise ValueError("The generated completion contains no usable reasoning text")
    return steps, final_answer


@torch.inference_mode()
def generate_responses(
    rows: List[Dict[str, Any]],
    responses_dir: Path,
    *,
    model_name: str,
    prompt_field: str,
    max_new_tokens: int,
    device: str,
    raw_prompt: bool,
    overwrite: bool,
) -> None:
    pending = [
        row
        for row in rows
        if overwrite or not (responses_dir / f"response_{row['_cie_row_id']}.json").exists()
    ]
    if not pending:
        print("[generation] All cached responses are present.")
        return

    print(f"[generation] Loading {model_name} for {len(pending)} prompts.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=None,
    ).to(device)
    model.eval()

    for row in tqdm(pending, desc="Generating CoTs"):
        prompt = row.get(prompt_field)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Row idx={row['_cie_row_id']} has no non-empty {prompt_field!r} field"
            )
        rendered = render_generation_prompt(tokenizer, prompt, raw_prompt)
        inputs = tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Intentionally leave temperature/top-p/top-k/do_sample at model defaults.
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        steps, final_answer = split_reasoning_steps(completion)

        saved = {key: value for key, value in row.items() if key != "_cie_row_id"}
        saved["generation_prompt"] = rendered
        saved["sample_0"] = {
            "full_response": completion,
            **{f"step_{i}": step for i, step in enumerate(steps, 1)},
            "final_answer": final_answer,
        }
        response_path = responses_dir / f"response_{row['_cie_row_id']}.json"
        response_path.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_response(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    response = json.loads(path.read_text(encoding="utf-8"))
    sample = response["sample_0"]
    step_keys = sorted(
        (key for key in sample if re.fullmatch(r"step_\d+", key)),
        key=lambda key: int(key.split("_")[1]),
    )
    return response, [sample[key].strip() for key in step_keys]


def build_circuits(
    rows: List[Dict[str, Any]],
    responses_dir: Path,
    circuits_dir: Path,
    *,
    model_name: str,
    transcoder_name: str,
    prompt_field: str,
    overwrite: bool,
) -> None:
    missing: List[Tuple[Dict[str, Any], Path, List[str]]] = []
    for row in rows:
        row_id = row["_cie_row_id"]
        response_path = responses_dir / f"response_{row_id}.json"
        _, steps = load_response(response_path)
        trace_dir = circuits_dir / f"response_{row_id}"
        expected = []
        for i in range(len(steps) + 1):
            expected.extend(
                [
                    trace_dir / f"step_{i}.pt",
                    trace_dir / "selection_meta" / f"step_{i}_selection.json",
                ]
            )
        if overwrite or not all(path.exists() for path in expected):
            missing.append((row, response_path, steps))
    if not missing:
        print("[circuits] All cached sentence circuits are present.")
        return

    print(f"[circuits] Loading replacement model for {len(missing)} traces.")
    model = ReplacementModel.from_pretrained(
        model_name, transcoder_name, dtype=torch.bfloat16
    )

    for row, response_path, steps in tqdm(missing, desc="Tracing responses"):
        row_id = row["_cie_row_id"]
        trace_dir = circuits_dir / f"response_{row_id}"
        meta_dir = trace_dir / "selection_meta"
        trace_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        items = [row[prompt_field]] + steps

        for step_num, sentence in enumerate(items):
            graph_path = trace_dir / f"step_{step_num}.pt"
            meta_path = meta_dir / f"step_{step_num}_selection.json"
            if graph_path.exists() and meta_path.exists() and not overwrite:
                continue

            selected, input_ids, entropies, causal_scores, final_scores = (
                select_positions_entropy_causal(
                    prompt=sentence,
                    model=model,
                    top_k=TOKEN_BUDGET,
                    temperature=ENTROPY_TEMPERATURE,
                    candidate_budget=CANDIDATE_BUDGET,
                    sentence_lambda=SENTENCE_LAMBDA,
                    beta=BETA,
                    diversity_penalty=DIVERSITY_PENALTY,
                    always_include_last=ALWAYS_INCLUDE_LAST,
                    skip_first_token=True,
                )
            )
            target_pos = max(selected)
            graph = attribute_compressed(
                prompt=sentence,
                model=model,
                max_n_logits=MAX_N_LOGITS,
                desired_logit_prob=DESIRED_LOGIT_PROB,
                batch_size=BATCH_SIZE,
                max_feature_nodes=MAX_FEATURE_NODES,
                max_features_per_position=MAX_FEATURES_PER_POSITION,
                offload="cpu",
                verbose=False,
                update_interval=UPDATE_INTERVAL,
                selected_positions=selected,
                target_pos=target_pos,
            )
            graph.to_pt(graph_path)
            metadata = {
                "source_trace_file": response_path.name,
                "step_num": step_num,
                "step_text": sentence if step_num else "__base_prompt__",
                "prompt": sentence,
                "selected_positions": selected,
                "selected_tokens": decode_selected_tokens(input_ids, selected, model),
                "target_pos": target_pos,
                "entropies": [float(value) for value in entropies.tolist()],
                "causal_scores": {str(k): float(v) for k, v in causal_scores.items()},
                "final_scores": {str(k): float(v) for k, v in final_scores.items()},
            }
            meta_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            del graph, input_ids, entropies
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_internal_graphs(
    trace_dir: Path, number_of_steps: int
) -> List[Any]:
    graphs = []
    for step_num in range(number_of_steps + 1):
        graphs.append(
            graph_to_data_compressed(
                str(trace_dir / f"step_{step_num}.pt"),
                str(trace_dir / "selection_meta" / f"step_{step_num}_selection.json"),
                make_undirected=True,
            )
        )
    return graphs


def score_all(
    rows: List[Dict[str, Any]],
    responses_dir: Path,
    circuits_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    device: str,
    prompt_field: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
    config = checkpoint["model_config"]

    print("[scoring] Loading the frozen external sentence encoder.")
    encoder = FrozenSentenceEncoder(
        model_name=checkpoint["hf_model_name"],
        layer_idx=int(checkpoint["layer_idx"]),
        device=device,
    )
    detector = JointFGWUnfaithfulnessDetector(**config).to(device)
    detector.load_state_dict(checkpoint["state_dict"])
    detector.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for row in tqdm(rows, desc="Scoring traces"):
            row_id = row["_cie_row_id"]
            response_path = responses_dir / f"response_{row_id}.json"
            response, steps = load_response(response_path)
            sentences = [row[prompt_field]] + steps
            internal_graphs = load_internal_graphs(
                circuits_dir / f"response_{row_id}", len(steps)
            )
            record = TraceRecord(
                trace_id=row_id,
                sentences=sentences,
                label=0,
                internal_graphs=internal_graphs,
            )
            record.ext_embs = encoder.encode_sentences(sentences, batch_size=8)
            with torch.no_grad():
                score, raw_distance = detector(record, device=device)

            result = {key: value for key, value in row.items() if key != "_cie_row_id"}
            result["response"] = response["sample_0"]["full_response"]
            result["unfaithful_score"] = float(score.item())
            result["raw_fgw_distance"] = float(raw_distance.item())
            result["unfaithful_threshold"] = float(checkpoint["threshold"])
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()

    temporary_path.replace(output_path)
    print(f"[done] Wrote {len(rows)} scored records to {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    work_dir = Path(args.work_dir).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    responses_dir = work_dir / "responses"
    circuits_dir = work_dir / "circuits"
    responses_dir.mkdir(parents=True, exist_ok=True)
    circuits_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_path)
    print(f"Loaded {len(rows)} input records from {input_path}")

    generate_responses(
        rows,
        responses_dir,
        model_name=args.model_name,
        prompt_field=args.prompt_field,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        raw_prompt=args.raw_prompt,
        overwrite=args.overwrite,
    )
    build_circuits(
        rows,
        responses_dir,
        circuits_dir,
        model_name=args.model_name,
        transcoder_name=args.transcoder_name,
        prompt_field=args.prompt_field,
        overwrite=args.overwrite,
    )
    score_all(
        rows,
        responses_dir,
        circuits_dir,
        checkpoint_path,
        output_path,
        device=args.device,
        prompt_field=args.prompt_field,
    )


if __name__ == "__main__":
    main()
