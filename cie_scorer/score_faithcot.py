#!/usr/bin/env python3
"""Score FaithCoT-BENCH response directories with CIE-Scorer."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .score_jsonl import (
    DEFAULT_MODEL,
    DEFAULT_TRANSCODERS,
    build_circuits,
    score_all,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CIE unfaithfulness detection on a FaithCoT-BENCH response directory."
    )
    parser.add_argument(
        "--faithcot_dir",
        required=True,
        help="Directory containing FaithCoT-BENCH response_*.json files.",
    )
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--transcoder_name", default=DEFAULT_TRANSCODERS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--include_cot_prompt",
        action="store_true",
        help="Include cot_prompt before question/options when building the base prompt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate cached circuits and overwrite the output JSONL.",
    )
    return parser.parse_args()


def numeric_response_id(path: Path) -> str:
    match = re.search(r"response_(\d+)$", path.stem)
    return match.group(1) if match else path.stem


def numeric_sort_key(path: Path) -> tuple[int, str]:
    row_id = numeric_response_id(path)
    return (int(row_id), row_id) if row_id.isdigit() else (10**18, row_id)


def build_query_sentence(record: dict[str, Any], *, include_cot_prompt: bool = False) -> str:
    parts: list[str] = []
    cot_prompt = str(record.get("cot_prompt", "")).strip()
    question = str(record.get("question", "")).strip()
    options = record.get("options", [])

    if include_cot_prompt and cot_prompt:
        parts.append(cot_prompt)
    if question:
        parts.append(f"Question: {question}")
    if isinstance(options, list) and options:
        rendered_options = []
        for i, option in enumerate(options):
            text = str(option).strip()
            if not text:
                continue
            prefix = chr(65 + i)
            rendered_options.append(text if text.startswith(f"{prefix}.") else f"{prefix}. {text}")
        if rendered_options:
            parts.append("Options:\n" + "\n".join(rendered_options))

    prompt = "\n\n".join(parts).strip()
    if not prompt:
        raise ValueError("FaithCoT-BENCH record has no usable prompt fields.")
    return prompt


def load_faithcot_rows(
    faithcot_dir: Path,
    *,
    include_cot_prompt: bool = False,
) -> list[dict[str, Any]]:
    response_files = sorted(faithcot_dir.glob("response_*.json"), key=numeric_sort_key)
    if not response_files:
        raise FileNotFoundError(f"No response_*.json files found in {faithcot_dir}")

    rows: list[dict[str, Any]] = []
    for response_path in response_files:
        record = json.loads(response_path.read_text(encoding="utf-8"))
        row_id = numeric_response_id(response_path)
        row = {
            "_cie_row_id": row_id,
            "trace_id": response_path.stem,
            "prompt": build_query_sentence(record, include_cot_prompt=include_cot_prompt),
        }
        for key in (
            "cot_prompt",
            "question",
            "options",
            "label",
            "unfaithfulness",
            "faithfulness_type",
            "dataset",
            "model",
        ):
            if key in record:
                row[key] = record[key]
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    faithcot_dir = Path(args.faithcot_dir).resolve()
    output_path = Path(args.output_jsonl).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    work_dir = Path(args.work_dir).resolve()

    if not faithcot_dir.is_dir():
        raise FileNotFoundError(faithcot_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    rows = load_faithcot_rows(
        faithcot_dir,
        include_cot_prompt=args.include_cot_prompt,
    )
    circuits_dir = work_dir / "circuits"
    circuits_dir.mkdir(parents=True, exist_ok=True)

    build_circuits(
        rows,
        faithcot_dir,
        circuits_dir,
        model_name=args.model_name,
        transcoder_name=args.transcoder_name,
        prompt_field="prompt",
        overwrite=args.overwrite,
    )
    score_all(
        rows,
        faithcot_dir,
        circuits_dir,
        checkpoint_path,
        output_path,
        device=args.device,
        prompt_field="prompt",
    )


if __name__ == "__main__":
    main()
