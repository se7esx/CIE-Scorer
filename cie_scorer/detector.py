import json
import random
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from transformers import AutoTokenizer, AutoModelForCausalLM
import ot


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TraceRecord:
    trace_id: str
    sentences: List[str]
    label: int                    # 1 = unfaithful, 0 = faithful
    internal_graphs: List[Data]
    ext_embs: Optional[np.ndarray] = None


def build_query_sentence(js: dict) -> str:
    parts = []
    if js.get("cot_prompt"):
        parts.append(js["cot_prompt"].strip())
    if js.get("question"):
        parts.append(js["question"].strip())
    if js.get("options"):
        opts = js["options"]
        if isinstance(opts, list) and len(opts) > 0:
            option_text = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(opts)])
            parts.append(option_text)
    return "\n".join(parts).strip()


def extract_reasoning_steps(js: dict, sample_key: str = "sample_0") -> List[str]:
    if sample_key not in js:
        return []
    sample = js[sample_key]
    step_keys = sorted(
        [k for k in sample.keys() if re.fullmatch(r"step_\d+", k)],
        key=lambda x: int(x.split("_")[1])
    )
    return [sample[k].strip() for k in step_keys]


def load_json_traces(json_source: str) -> Dict[str, Dict[str, Any]]:
    json_source = Path(json_source)

    if json_source.is_file():
        json_files = [json_source]
    elif json_source.is_dir():
        json_files = sorted(json_source.glob("*.json"))
    else:
        raise FileNotFoundError(f"Path not found: {json_source}")

    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files found under {json_source}")

    traces: Dict[str, Dict[str, Any]] = {}

    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            js = json.load(f)

        trace_id = fp.stem
        query_sentence = build_query_sentence(js)

        sample_keys = sorted(
            [k for k in js.keys() if re.fullmatch(r"sample_\d+", k)],
            key=lambda x: int(x.split("_")[1])
        )
        if len(sample_keys) == 0:
            print(f"[WARN] No sample_* key found in {fp}, skipped.")
            continue

        if len(sample_keys) > 1:
            print(f"[WARN] Multiple sample_* keys found in {fp}; using {sample_keys[0]} only.")

        sample_key = sample_keys[0]
        reasoning_steps = extract_reasoning_steps(js, sample_key=sample_key)
        label = int(js.get("unfaithfulness", 0))

        if trace_id in traces:
            raise ValueError(f"Duplicate trace_id detected: {trace_id}")

        traces[trace_id] = {
            "trace_id": trace_id,
            "query_sentence": query_sentence,
            "reasoning_steps": reasoning_steps,
            "label": label,
        }

    print(f"[INFO] loaded {len(traces)} external traces from {json_source}")
    return traces


def load_internal_pyg_groups(internal_pyg_dir: str) -> Dict[str, List[Tuple[int, Data]]]:
    internal_pyg_dir = Path(internal_pyg_dir)
    processed_dir = internal_pyg_dir / "processed"
    index_path = internal_pyg_dir / "index.json"

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    if not isinstance(index_data, list):
        raise ValueError("Expected index.json to be a list of records.")

    grouped: Dict[str, List[Tuple[int, Data]]] = {}
    missing_files = 0

    for item in index_data:
        file_name = item["file_name"]
        trace_id = str(item["response_name"])
        step_num = int(item["step_num"])

        fp = processed_dir / file_name
        if not fp.exists():
            print(f"[WARN] graph file not found: {fp}, skipped.")
            missing_files += 1
            continue

        obj = torch.load(fp, weights_only=False)

        if isinstance(obj, Data):
            d = obj
        elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], Data):
            d = obj["data"]
        else:
            raise ValueError(f"Unsupported graph object type in {fp}: {type(obj)}")

        grouped.setdefault(trace_id, []).append((step_num, d))

    for trace_id in grouped:
        grouped[trace_id] = sorted(grouped[trace_id], key=lambda x: x[0])

    print(f"[INFO] grouped {sum(len(v) for v in grouped.values())} graphs into {len(grouped)} traces")
    if missing_files > 0:
        print(f"[INFO] missing graph files skipped: {missing_files}")
    return grouped


def build_aligned_records(json_source: str, internal_pyg_dir: str) -> List[TraceRecord]:
    external_traces = load_json_traces(json_source=json_source)
    internal_groups = load_internal_pyg_groups(internal_pyg_dir=internal_pyg_dir)

    records: List[TraceRecord] = []
    skipped = 0

    for trace_id, ext_meta in external_traces.items():
        if trace_id not in internal_groups:
            print(f"[WARN] trace {trace_id} not found in internal graphs, skipped.")
            skipped += 1
            continue

        internal_pairs = sorted(internal_groups[trace_id], key=lambda x: x[0])
        step_nums = [k for k, _ in internal_pairs]
        internal_graphs = [g for _, g in internal_pairs]

        if len(step_nums) == 0:
            print(f"[WARN] trace {trace_id} has no internal graphs, skipped.")
            skipped += 1
            continue

        query_sentence = ext_meta["query_sentence"]
        reasoning_steps = ext_meta["reasoning_steps"]

        if min(step_nums) == 0:
            candidate_sentences = [query_sentence] + reasoning_steps
        else:
            candidate_sentences = reasoning_steps

        if len(candidate_sentences) < len(internal_graphs):
            print(
                f"[WARN] trace {trace_id} length mismatch: "
                f"{len(candidate_sentences)} external sentences vs {len(internal_graphs)} internal graphs. Skipped."
            )
            skipped += 1
            continue

        sentences = candidate_sentences[:len(internal_graphs)]

        records.append(
            TraceRecord(
                trace_id=trace_id,
                sentences=sentences,
                label=int(ext_meta["label"]),
                internal_graphs=internal_graphs,
            )
        )

    print(f"[INFO] aligned traces: {len(records)}, skipped: {skipped}")
    return records


def stratified_trace_split(
    records: List[TraceRecord],
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
):
    labels = [r.label for r in records]
    idx = np.arange(len(records))

    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]

    if val_size > 0 and len(train_records) > 1:
        train_labels = [r.label for r in train_records]
        idx2 = np.arange(len(train_records))
        train2_idx, val_idx = train_test_split(
            idx2, test_size=val_size, random_state=seed, stratify=train_labels
        )
        val_records = [train_records[i] for i in val_idx]
        train_records = [train_records[i] for i in train2_idx]
    else:
        val_records = []

    return train_records, val_records, test_records


class FrozenSentenceEncoder:
    def __init__(self, model_name: str, layer_idx: int = 15, device: str = "cuda"):
        self.device = device
        self.layer_idx = layer_idx
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
            device_map=None
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_sentences(self, sentences: List[str], batch_size: int = 8) -> np.ndarray:
        all_embs = []
        for i in range(0, len(sentences), batch_size):
            batch_sent = sentences[i:i+batch_size]
            toks = self.tokenizer(
                batch_sent,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(
                **toks,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hs = out.hidden_states[self.layer_idx]
            mask = toks["attention_mask"].unsqueeze(-1)
            sent_emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_embs.append(sent_emb.float().cpu())
        return torch.cat(all_embs, dim=0).numpy()


def precompute_external_embeddings(records: List[TraceRecord], encoder: FrozenSentenceEncoder, batch_size: int = 8):
    for r in records:
        r.ext_embs = encoder.encode_sentences(r.sentences, batch_size=batch_size)


class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.out_dim = out_dim
        self.convs = nn.ModuleList()

        def make_mlp(inp, outp):
            return nn.Sequential(
                nn.Linear(inp, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, outp),
            )

        if num_layers <= 1:
            self.convs.append(GINConv(make_mlp(in_dim, out_dim)))
        else:
            self.convs.append(GINConv(make_mlp(in_dim, hidden_dim)))
            for _ in range(num_layers - 2):
                self.convs.append(GINConv(make_mlp(hidden_dim, hidden_dim)))
            self.convs.append(GINConv(make_mlp(hidden_dim, out_dim)))

    def forward(self, batch_data: Data) -> torch.Tensor:
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return global_mean_pool(x, batch)


def build_trace_graph_from_embeddings(
    x: torch.Tensor,
    seq_weight: float = 1.0,
    sim_weight: float = 0.5,
    forward_only: bool = True,
):
    T = x.size(0)
    device = x.device

    x_norm = F.normalize(x, dim=-1)
    sim = torch.matmul(x_norm, x_norm.T)

    if forward_only:
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        sim = F.relu(sim) * mask
    else:
        sim = F.relu(sim)
        sim.fill_diagonal_(0.0)

    A = sim_weight * sim

    if T > 1:
        seq_edges = torch.zeros(T, T, device=device)
        idx = torch.arange(T - 1, device=device)
        seq_edges[idx, idx + 1] = 1.0
        A = A + seq_weight * seq_edges

    row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
    A = A / row_sum
    mu = torch.full((T,), 1.0 / T, device=device)
    return {"A": A, "X": x, "mu": mu}


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class JointFGWUnfaithfulnessDetector(nn.Module):
    def __init__(
        self,
        ext_in_dim: int,
        int_in_dim: int,
        hidden_dim: int = 128,
        proj_dim: int = 128,
        gin_layers: int = 2,
        alpha: float = 0.5,
        seq_weight: float = 1.0,
        sim_weight: float = 0.5,
    ):
        super().__init__()
        self.gin = GINEncoder(
            in_dim=int_in_dim,
            hidden_dim=hidden_dim,
            out_dim=proj_dim,
            num_layers=gin_layers,
        )
        self.ext_proj = MLPProjector(ext_in_dim, proj_dim, hidden_dim=hidden_dim, dropout=0.2)
        self.int_proj = MLPProjector(proj_dim, proj_dim, hidden_dim=hidden_dim, dropout=0.2)

        self.alpha = alpha
        self.seq_weight = seq_weight
        self.sim_weight = sim_weight

    def encode_internal_trace(self, graphs: List[Data], device: str):
        loader = PyGDataLoader(graphs, batch_size=min(len(graphs), 64), shuffle=False)
        outs = []
        for batch in loader:
            batch = batch.to(device)
            emb = self.gin(batch)
            outs.append(emb)
        return torch.cat(outs, dim=0)

    def compute_fgw_distance(self, ext_graph: Dict[str, torch.Tensor], int_graph: Dict[str, torch.Tensor]):
        A_ext, X_ext, mu_ext = ext_graph["A"], ext_graph["X"], ext_graph["mu"]
        A_int, X_int, mu_int = int_graph["A"], int_graph["X"], int_graph["mu"]

        M = torch.cdist(X_ext, X_int, p=2) ** 2

        dist = ot.gromov.fused_gromov_wasserstein2(
            M=M,
            C1=A_ext,
            C2=A_int,
            p=mu_ext,
            q=mu_int,
            loss_fun="square_loss",
            alpha=self.alpha,
        )

        if not isinstance(dist, torch.Tensor):
            dist = torch.tensor(dist, dtype=X_ext.dtype, device=X_ext.device)

        return dist

    def forward(self, record: TraceRecord, device: str):
        ext_embs = torch.tensor(record.ext_embs, dtype=torch.float32, device=device)
        ext_x = self.ext_proj(ext_embs)
        ext_x = F.normalize(ext_x, dim=-1)
        ext_graph = build_trace_graph_from_embeddings(
            ext_x,
            seq_weight=self.seq_weight,
            sim_weight=self.sim_weight,
            forward_only=True,
        )

        int_embs = self.encode_internal_trace(record.internal_graphs, device=device)
        int_x = self.int_proj(int_embs)
        int_x = F.normalize(int_x, dim=-1)
        int_graph = build_trace_graph_from_embeddings(
            int_x,
            seq_weight=self.seq_weight,
            sim_weight=self.sim_weight,
            forward_only=True,
        )
      
        raw_dist = self.compute_fgw_distance(ext_graph, int_graph)
        score = torch.log1p(raw_dist.clamp(min=0.0))
        #score = torch.sigmoid(torch.log1p(raw_dist))
        return score, raw_dist


def compute_class_weights(records: List[TraceRecord]) -> Tuple[float, float]:
    labels = np.array([r.label for r in records], dtype=np.float32)
    pos = float(labels.sum())
    neg = float(len(labels) - pos)
    if pos == 0 or neg == 0:
        return 1.0, 1.0
    w_pos = neg / (pos + 1e-8)
    w_neg = 1.0
    return w_neg, w_pos


def margin_distance_loss(score: torch.Tensor, y: torch.Tensor, margin: float, w_neg: float = 1.0, w_pos: float = 1.0):
    margin_t = torch.tensor(margin, device=score.device, dtype=score.dtype)
    loss_faithful = (1.0 - y) * score
    loss_unfaithful = y * torch.relu(margin_t - score)
    return w_neg * loss_faithful + w_pos * loss_unfaithful


def select_best_threshold(scores: List[float], labels: List[int]) -> float:
    if len(scores) == 0:
        return 0.5
    uniq = sorted(set(scores))
    if len(uniq) == 1:
        return float(uniq[0])

    candidates = [uniq[0] - 1e-6]
    for i in range(len(uniq) - 1):
        candidates.append((uniq[i] + uniq[i + 1]) / 2.0)
    candidates.append(uniq[-1] + 1e-6)

    best_thr = candidates[0]
    best_f1 = -1.0
    best_acc = -1.0

    for thr in candidates:
        preds = [1 if s > thr else 0 for s in scores]
        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        if (f1 > best_f1) or (f1 == best_f1 and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_thr = thr

    return float(best_thr)


def evaluate(model, records: List[TraceRecord], device: str, threshold: Optional[float] = None):
    model.eval()
    ys, scores, raw_dists = [], [], []

    with torch.no_grad():
        for r in records:
            score, raw_dist = model(r, device=device)
            ys.append(r.label)
            scores.append(float(score.item()))
            raw_dists.append(float(raw_dist.item()))

    if threshold is None:
        threshold = select_best_threshold(scores, ys)

    preds = [1 if s > threshold else 0 for s in scores]
    acc = accuracy_score(ys, preds) if len(ys) > 0 else 0.0
    f1 = f1_score(ys, preds, zero_division=0) if len(ys) > 0 else 0.0

    return {
        "acc": acc,
        "f1": f1,
        "threshold": threshold,
        "mean_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
        "mean_raw_dist": float(np.mean(raw_dists)) if len(raw_dists) > 0 else 0.0,
    }


def train(
    model: JointFGWUnfaithfulnessDetector,
    train_records: List[TraceRecord],
    val_records: List[TraceRecord],
    test_records: List[TraceRecord],
    device: str,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    margin: float = 2.0,
):
    w_neg, w_pos = compute_class_weights(train_records)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_f1 = -1.0
    best_threshold = 0.5

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_records)
        total_loss = 0.0

        for r in train_records:
            y = torch.tensor(float(r.label), dtype=torch.float32, device=device)
            score, _ = model(r, device=device)
            loss = margin_distance_loss(score, y, margin=margin, w_neg=w_neg, w_pos=w_pos)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(train_records), 1)

        if len(val_records) > 0:
            val_probe = evaluate(model, val_records, device=device, threshold=None)
            current_threshold = val_probe["threshold"]
        else:
            train_probe = evaluate(model, train_records, device=device, threshold=None)
            current_threshold = train_probe["threshold"]

        train_metrics = evaluate(model, train_records, device=device, threshold=current_threshold)
        val_metrics = evaluate(model, val_records, device=device, threshold=current_threshold) if len(val_records) > 0 else {"acc": 0.0, "f1": 0.0}
        test_metrics = evaluate(model, test_records, device=device, threshold=current_threshold)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={avg_loss:.4f} | "
            f"thr={current_threshold:.4f} | "
            f"train_acc={train_metrics['acc']:.4f}, train_f1={train_metrics['f1']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f}, val_f1={val_metrics['f1']:.4f} | "
            f"test_acc={test_metrics['acc']:.4f}, test_f1={test_metrics['f1']:.4f}"
        )

        monitor_f1 = val_metrics["f1"] if len(val_records) > 0 else test_metrics["f1"]
        if monitor_f1 > best_val_f1:
            best_val_f1 = monitor_f1
            best_threshold = current_threshold
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_threshold


def save_detector_checkpoint(
    path: str,
    model: JointFGWUnfaithfulnessDetector,
    threshold: float,
    *,
    hf_model_name: str,
    layer_idx: int,
    ext_in_dim: int,
    int_in_dim: int,
    hidden_dim: int,
    proj_dim: int,
    gin_layers: int,
    alpha: float,
    seq_weight: float,
    sim_weight: float,
):
    """Save everything required to reproduce an inference-time score."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "threshold": float(threshold),
        "hf_model_name": hf_model_name,
        "layer_idx": int(layer_idx),
        "model_config": {
            "ext_in_dim": int(ext_in_dim),
            "int_in_dim": int(int_in_dim),
            "hidden_dim": int(hidden_dim),
            "proj_dim": int(proj_dim),
            "gin_layers": int(gin_layers),
            "alpha": float(alpha),
            "seq_weight": float(seq_weight),
            "sim_weight": float(sim_weight),
        },
    }
    torch.save(payload, output_path)
    print(f"Saved detector checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_source",
        type=str,
        required=True,
        help="Either a directory of response_i.json files or a single JSON file"
    )
    parser.add_argument(
        "--internal_pyg_dir",
        type=str,
        required=True,
        help="Directory containing index.json and processed/*.pt"
    )
    parser.add_argument("--hf_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer_idx", type=int, default=15)
    parser.add_argument("--test_size", type=float, default=0.4)
    parser.add_argument("--val_size", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--gin_layers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seq_weight", type=float, default=1.0)
    parser.add_argument("--sim_weight", type=float, default=0.4)
    parser.add_argument("--ext_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--checkpoint_out",
        type=str,
        default=None,
        help="Optional path for a reusable inference checkpoint",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    print("[1/6] Building aligned trace records...")
    records = build_aligned_records(
        json_source=args.json_source,
        internal_pyg_dir=args.internal_pyg_dir,
    )
    if len(records) == 0:
        raise RuntimeError("No aligned records found. Please check json/index alignment.")

    print("[2/6] Trace-level stratified split...")
    train_records, val_records, test_records = stratified_trace_split(
        records,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(f"Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")

    print("[3/6] Building frozen external sentence encoder and precomputing sentence embeddings...")
    sentence_encoder = FrozenSentenceEncoder(
        model_name=args.hf_model_name,
        layer_idx=args.layer_idx,
        device=args.device,
    )
    precompute_external_embeddings(train_records, sentence_encoder, batch_size=args.ext_batch_size)
    precompute_external_embeddings(val_records, sentence_encoder, batch_size=args.ext_batch_size)
    precompute_external_embeddings(test_records, sentence_encoder, batch_size=args.ext_batch_size)

    print("[4/6] Inferring feature dimensions...")
    one_graph = train_records[0].internal_graphs[0]
    int_in_dim = one_graph.x.shape[1]
    ext_in_dim = train_records[0].ext_embs.shape[1]
    print(f"External input dim = {ext_in_dim}, Internal node dim = {int_in_dim}")

    print("[5/6] Building joint FGW detector...")
    model = JointFGWUnfaithfulnessDetector(
        ext_in_dim=ext_in_dim,
        int_in_dim=int_in_dim,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        gin_layers=args.gin_layers,
        alpha=args.alpha,
        seq_weight=args.seq_weight,
        sim_weight=args.sim_weight,
    ).to(args.device)

    print("[6/6] Training...")
    model, best_threshold = train(
        model=model,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        margin=args.margin,
    )

    print("\n===== Final Evaluation =====")
    train_metrics = evaluate(model, train_records, device=args.device, threshold=best_threshold)
    val_metrics = evaluate(model, val_records, device=args.device, threshold=best_threshold) if len(val_records) > 0 else {"acc": 0.0, "f1": 0.0}
    test_metrics = evaluate(model, train_records+val_records+test_records, device=args.device, threshold=best_threshold)

    print(f"Best threshold = {best_threshold:.4f}")
    print(f"Train ACC={train_metrics['acc']:.4f}, F1={train_metrics['f1']:.4f}")
    print(f"Val   ACC={val_metrics['acc']:.4f}, F1={val_metrics['f1']:.4f}")
    print(f"Test  ACC={test_metrics['acc']:.4f}, F1={test_metrics['f1']:.4f}")
    print(f"Test mean score = {test_metrics['mean_score']:.4f}")
    print(f"Test mean raw FGW distance = {test_metrics['mean_raw_dist']:.4f}")

    if args.checkpoint_out:
        save_detector_checkpoint(
            args.checkpoint_out,
            model,
            best_threshold,
            hf_model_name=args.hf_model_name,
            layer_idx=args.layer_idx,
            ext_in_dim=ext_in_dim,
            int_in_dim=int_in_dim,
            hidden_dim=args.hidden_dim,
            proj_dim=args.proj_dim,
            gin_layers=args.gin_layers,
            alpha=args.alpha,
            seq_weight=args.seq_weight,
            sim_weight=args.sim_weight,
        )


if __name__ == "__main__":
    main()
