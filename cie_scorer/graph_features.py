import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from circuit_tracer.graph import Graph


# =========================================================
# 1. 扫描当前目录结构下的 step circuits
#    root/
#      response_0/
#         step_0.pt
#         step_1.pt
#         ...
#         selection_meta/
#            step_0_selection.json
# =========================================================
def build_graph_items_from_response_dirs(
    circuit_root: str,
) -> List[Dict]:
    circuit_root = Path(circuit_root)
    response_dirs = sorted(
        [p for p in circuit_root.iterdir() if p.is_dir() and p.name.startswith("response_")],
        key=lambda p: int(re.search(r"response_(\d+)", p.name).group(1))
    )

    items = []
    for response_dir in response_dirs:
        selection_meta_dir = response_dir / "selection_meta"

        step_pts = sorted(
            response_dir.glob("step_*.pt"),
            key=lambda p: int(re.search(r"step_(\d+)\.pt", p.name).group(1))
        )

        for step_pt in step_pts:
            m = re.search(r"step_(\d+)\.pt", step_pt.name)
            if m is None:
                continue
            step_num = int(m.group(1))

            meta_path = selection_meta_dir / f"step_{step_num}_selection.json"
            if not meta_path.exists():
                meta_path = None

            items.append({
                "response_dir": response_dir.name,
                "response_name": response_dir.name,
                "graph_path": str(step_pt),
                "step_num": step_num,
                "selection_meta_path": str(meta_path) if meta_path is not None else None,
            })

    return items


# =========================================================
# 2. 读取 selection meta
# =========================================================
def load_selection_meta(meta_path: Optional[str]) -> Dict:
    if meta_path is None:
        return {}
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 3. 当前 compressed graph 的节点类型推断
#    当前布局:
#      [selected feature nodes] + [position nodes] + [logit nodes]
# =========================================================
def infer_node_types_compressed(graph: Graph, meta: Dict, num_nodes: int) -> torch.Tensor:
    n_feature_nodes = len(graph.selected_features)
    n_logit_nodes = len(graph.logit_tokens)
    n_position_nodes = num_nodes - n_feature_nodes - n_logit_nodes

    node_type = torch.zeros(num_nodes, dtype=torch.long)
    # 0 = feature, 1 = position, 2 = logit
    if n_feature_nodes > 0:
        node_type[:n_feature_nodes] = 0
    if n_position_nodes > 0:
        node_type[n_feature_nodes:n_feature_nodes + n_position_nodes] = 1
    if n_logit_nodes > 0:
        node_type[n_feature_nodes + n_position_nodes:] = 2

    return node_type


# =========================================================
# 4. 恢复 kept_positions
#    优先用 selection meta 中的 selected_positions + target_pos
# =========================================================
def get_kept_positions(meta: Dict) -> List[int]:
    selected_positions = meta.get("selected_positions", [])
    target_pos = meta.get("target_pos", None)

    kept = list(selected_positions)
    if target_pos is not None:
        kept.append(int(target_pos))

    kept = sorted(set(int(x) for x in kept))
    return kept


# =========================================================
# 5. 度统计
# =========================================================
def compute_degree_features(adj: torch.Tensor):
    # adj: [N, N]
    binary_adj = (adj != 0).float()
    abs_adj = adj.abs()

    in_deg = binary_adj.sum(dim=0, keepdim=True).T
    out_deg = binary_adj.sum(dim=1, keepdim=True)
    w_in_deg = abs_adj.sum(dim=0, keepdim=True).T
    w_out_deg = abs_adj.sum(dim=1, keepdim=True)

    return in_deg, out_deg, w_in_deg, w_out_deg


# =========================================================
# 6. 构建 node feature matrix
#    最终特征:
#      [type_onehot(3), layer_norm, pos_norm, activation, logit_prob,
#       is_target, is_selected_pos, in_deg, out_deg, w_in_deg, w_out_deg]
# =========================================================
def build_node_feature_matrix_compressed(graph: Graph, meta: Dict) -> torch.Tensor:
    adj = graph.adjacency_matrix.float().cpu()
    num_nodes = adj.size(0)

    node_type = infer_node_types_compressed(graph, meta, num_nodes)
    node_type_onehot = F.one_hot(node_type, num_classes=3).float()  # [N, 3]

    layer_feat = torch.zeros(num_nodes, 1, dtype=torch.float)
    pos_feat = torch.zeros(num_nodes, 1, dtype=torch.float)
    activation_feat = torch.zeros(num_nodes, 1, dtype=torch.float)
    logit_prob_feat = torch.zeros(num_nodes, 1, dtype=torch.float)
    is_target_feat = torch.zeros(num_nodes, 1, dtype=torch.float)
    is_selected_pos_feat = torch.zeros(num_nodes, 1, dtype=torch.float)

    in_deg, out_deg, w_in_deg, w_out_deg = compute_degree_features(adj)

    n_feature_nodes = len(graph.selected_features)
    n_logit_nodes = len(graph.logit_tokens)
    n_position_nodes = num_nodes - n_feature_nodes - n_logit_nodes

    max_layer = max(int(graph.cfg.n_layers), 1)
    max_pos = max(int(graph.n_pos) - 1, 1)

    kept_positions = get_kept_positions(meta)
    target_pos = meta.get("target_pos", None)
    selected_positions = set(meta.get("selected_positions", []))

    # -------------------------
    # feature nodes
    # -------------------------
    for node_idx in range(n_feature_nodes):
        try:
            feat_idx = int(graph.selected_features[node_idx])

            if feat_idx < len(graph.active_features):
                layer, pos, _ = graph.active_features[feat_idx].tolist()
                layer_feat[node_idx, 0] = float(layer) / max_layer
                pos_feat[node_idx, 0] = float(pos) / max_pos

                if int(pos) in selected_positions:
                    is_selected_pos_feat[node_idx, 0] = 1.0
                if target_pos is not None and int(pos) == int(target_pos):
                    is_target_feat[node_idx, 0] = 1.0

            if feat_idx < len(graph.activation_values):
                activation_feat[node_idx, 0] = float(graph.activation_values[feat_idx].item())
        except Exception:
            continue

    # -------------------------
    # position nodes
    # 当前压缩版构图里 position nodes 的顺序是：
    # for layer in range(n_layers+1):
    #   for p in kept_positions:
    #       append(base + p)
    # -------------------------
    if len(kept_positions) > 0 and n_position_nodes > 0:
        width = len(kept_positions)
        for local_idx in range(n_position_nodes):
            node_idx = n_feature_nodes + local_idx
            layer = local_idx // width
            pos = kept_positions[local_idx % width]

            layer_feat[node_idx, 0] = float(layer) / max_layer
            pos_feat[node_idx, 0] = float(pos) / max_pos

            if pos in selected_positions:
                is_selected_pos_feat[node_idx, 0] = 1.0
            if target_pos is not None and int(pos) == int(target_pos):
                is_target_feat[node_idx, 0] = 1.0

    # -------------------------
    # logit nodes
    # -------------------------
    if n_logit_nodes > 0:
        start = n_feature_nodes + n_position_nodes
        probs = graph.logit_probabilities.float().cpu()

        # rank: 概率越高 rank 越小
        order = torch.argsort(probs, descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(len(probs))

        for i in range(n_logit_nodes):
            node_idx = start + i
            logit_prob_feat[node_idx, 0] = float(probs[i].item())
            # 把 rank 编到 layer_feat 里不合适，所以这里直接写到 pos_feat 也不理想
            # 第一版先只保留 prob，rank 不单独加；如果你想加，可再多拼一列
            if int(ranks[i].item()) == 0:
                is_target_feat[node_idx, 0] = 1.0  # top-1 logit 做一个简单标记

    x = torch.cat(
        [
            node_type_onehot,      # [N, 3]
            layer_feat,            # [N, 1]
            pos_feat,              # [N, 1]
            activation_feat,       # [N, 1]
            logit_prob_feat,       # [N, 1]
            is_target_feat,        # [N, 1]
            is_selected_pos_feat,  # [N, 1]
            in_deg,                # [N, 1]
            out_deg,               # [N, 1]
            w_in_deg,              # [N, 1]
            w_out_deg,             # [N, 1]
        ],
        dim=1,
    )

    return x


# =========================================================
# 7. 单个 graph -> PyG Data
#    默认把图对称化，避免 position nodes 只有入边没有出边的问题
# =========================================================
def graph_to_data_compressed(
    graph_path: str,
    selection_meta_path: Optional[str] = None,
    make_undirected: bool = True,
) -> Data:
    graph = Graph.from_pt(graph_path)
    graph.to("cpu")

    meta = load_selection_meta(selection_meta_path)

    adj = graph.adjacency_matrix.float().cpu()

    if make_undirected:
        adj = adj + adj.T

    edge_index, edge_weight = dense_to_sparse(adj)

    # edge_attr: [w, |w|, sign(w)]
    edge_attr = torch.stack(
        [
            edge_weight,
            edge_weight.abs(),
            torch.sign(edge_weight),
        ],
        dim=1
    ).float()

    x = build_node_feature_matrix_compressed(graph, meta)

    data = Data(
        x=x,
        edge_index=edge_index.long(),
        edge_attr=edge_attr,
    )

    data.graph_path = Path(graph_path).name
    data.selection_meta_path = Path(selection_meta_path).name if selection_meta_path is not None else ""
    data.response_name = meta.get("source_trace_file", "")
    data.step_num = int(meta.get("step_num", -1))
    data.target_pos = int(meta.get("target_pos", -1)) if "target_pos" in meta else -1
    data.num_nodes_raw = int(adj.size(0))
    data.num_edges_raw = int(edge_index.size(1))

    # 如果有 trace-level 标签，可以顺手带上
    if "unfaithfulness" in meta:
        data.y_unfaithful = torch.tensor([int(meta["unfaithfulness"])], dtype=torch.long)

    del graph, adj, edge_index, edge_weight, edge_attr, x
    return data


# =========================================================
# 8. on-disk dataset
# =========================================================
class StepCircuitOnDiskDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.processed_dir = self.root / "processed"
        self.index_file = self.root / "index.json"

        if not self.index_file.exists():
            raise FileNotFoundError(
                f"Index file not found: {self.index_file}. "
                f"Please run build_ondisk_dataset_from_step_circuits(...) first."
            )

        with open(self.index_file, "r", encoding="utf-8") as f:
            self.index = json.load(f)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]
        pt_path = self.processed_dir / item["file_name"]
        data = torch.load(pt_path, weights_only=False)
        return data


# =========================================================
# 9. 构建 on-disk dataset
# =========================================================
def build_ondisk_dataset_from_step_circuits(
    root: str,
    circuit_root: str,
    make_undirected: bool = True,
    overwrite: bool = False,
):
    root = Path(root)
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    index_file = root / "index.json"
    if index_file.exists() and not overwrite:
        raise FileExistsError(
            f"{index_file} already exists. Use overwrite=True to rebuild."
        )

    graph_items = build_graph_items_from_response_dirs(circuit_root=circuit_root)
    print(f"Found {len(graph_items)} step circuit files.", flush=True)

    index = []
    skipped = 0

    for i, item in enumerate(tqdm(graph_items, desc="Processing step circuits")):
        try:
            data = graph_to_data_compressed(
                graph_path=item["graph_path"],
                selection_meta_path=item["selection_meta_path"],
                make_undirected=make_undirected,
            )

            file_name = f"graph_{i:06d}.pt"
            save_path = processed_dir / file_name
            torch.save(data, save_path)

            index.append({
                "file_name": file_name,
                "response_name": item["response_name"],
                "step_num": int(item["step_num"]),
                "graph_path": str(item["graph_path"]),
                "selection_meta_path": item["selection_meta_path"],
                "num_nodes_raw": int(data.num_nodes_raw),
                "num_edges_raw": int(data.num_edges_raw),
            })

            del data

        except Exception as e:
            skipped += 1
            print(f"[Skip] {item['graph_path']} -> {e}", flush=True)

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Built {len(index)} graphs, skipped {skipped}.", flush=True)
    print(f"Saved dataset to: {root}", flush=True)
    print(f"Index file: {index_file}", flush=True)
# =========================================================
# 10. CLI / main
# =========================================================
import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an on-disk PyG dataset from saved step circuit graphs."
    )
    parser.add_argument(
        "--circuit_root",
        type=str,
        required=True,
        help="Absolute path to the root directory containing response_x/step_y.pt",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Absolute path to save the processed PyG dataset",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="If set, symmetrize adjacency by A <- A + A^T before building edge_index",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing index.json / processed dataset if present",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="After building, load dataset and print the first sample",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    circuit_root = Path(args.circuit_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not circuit_root.exists():
        raise FileNotFoundError(f"circuit_root does not exist: {circuit_root}")

    print(f"Building dataset from: {circuit_root}")
    print(f"Saving dataset to:    {output_root}")
    print(f"Undirected:           {args.undirected}")
    print(f"Overwrite:            {args.overwrite}")

    build_ondisk_dataset_from_step_circuits(
        root=str(output_root),
        circuit_root=str(circuit_root),
        make_undirected=args.undirected,
        overwrite=args.overwrite,
    )

    if args.inspect:
        print("\nLoading dataset for inspection...")
        dataset = StepCircuitOnDiskDataset(root=str(output_root))
        print("Dataset loaded.")
        print("Num graphs:", len(dataset))

        if len(dataset) > 0:
            sample = dataset[0]
            print(sample)
            print("x shape:", sample.x.shape)
            print("edge_index shape:", sample.edge_index.shape)
            print("edge_attr shape:", sample.edge_attr.shape)
            print("num_nodes_raw:", sample.num_nodes_raw)
            print("num_edges_raw:", sample.num_edges_raw)
            print("graph_path:", sample.graph_path)
            print("selection_meta_path:", sample.selection_meta_path)


if __name__ == "__main__":
    main()
