import json
import logging
import math
import time
from typing import Literal

import torch
import torch.nn.functional as F
from tqdm import tqdm

from circuit_tracer.graph import Graph
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils import get_default_device
from circuit_tracer.utils.disk_offload import offload_modules


MAX_N_LOGITS = 5
DESIRED_LOGIT_PROB = 0.95
BATCH_SIZE = 16
MAX_FEATURE_NODES = 128
MAX_FEATURES_PER_POSITION = 16
OFFLOAD: Literal["cpu", "disk", None] = "cpu"
VERBOSE = True
UPDATE_INTERVAL = 4

TOKEN_BUDGET = 6
CANDIDATE_BUDGET = 12
ENTROPY_TEMPERATURE = 1.0
SENTENCE_LAMBDA = 0.5
BETA = 0.5
DIVERSITY_PENALTY = 0.1
ALWAYS_INCLUDE_LAST = True
@torch.no_grad()
def compute_salient_logits(
    logits: torch.Tensor,
    unembed_proj: torch.Tensor,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    cols = unembed_proj[:, top_idx]
    demeaned = cols - unembed_proj.mean(dim=-1, keepdim=True)
    return top_idx, top_p, demeaned.T


def compute_partial_influences(edge_matrix, logit_p, row_to_node_index, max_iter=128, device=None):
    device = device or get_default_device()

    normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
    normalized_matrix = normalized_matrix.abs_()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod[-len(logit_p):] = logit_p

    for _ in range(max_iter):
        prod = prod[row_to_node_index] @ normalized_matrix
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences


def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    x_min = x.min()
    x_max = x.max()
    if float(x_max - x_min) < 1e-12:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min + 1e-12)


# =========================================================
# Token selection: entropy + causal contribution
# =========================================================
@torch.no_grad()
def forward_stats(prompt_or_ids, model: ReplacementModel, temperature: float = 1.0):
    input_ids = model.ensure_tokenized(prompt_or_ids)
    if input_ids.ndim != 1:
        input_ids = input_ids.squeeze(0)

    ctx = model.setup_attribution(input_ids)
    with ctx.install_hooks(model):
        residual = model.forward(input_ids.expand(1, -1), stop_at_layer=model.cfg.n_layers)
        final_hidden = model.ln_final(residual)[0]  # (seq_len, d_model)
        ctx._resid_activations[-1] = final_hidden.unsqueeze(0)

    logits = ctx.logits[0]
    probs = torch.softmax(logits / temperature, dim=-1)
    entropies = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    sent_repr = final_hidden.mean(dim=0)

    return (
        input_ids.detach().cpu(),
        logits.detach().cpu(),
        probs.detach().cpu(),
        entropies.detach().cpu(),
        final_hidden.detach().cpu(),
        sent_repr.detach().cpu(),
    )


@torch.no_grad()
def select_positions_entropy_causal(
    prompt: str,
    model: ReplacementModel,
    top_k: int = 8,
    temperature: float = 1.0,
    candidate_budget: int | None = None,
    sentence_lambda: float = 0.5,
    beta: float = 0.5,
    diversity_penalty: float = 0.1,
    always_include_last: bool = True,
    skip_first_token: bool = True,
):
    (
        input_ids,
        logits,
        probs,
        entropies,
        final_hidden,
        sent_repr,
    ) = forward_stats(prompt_or_ids=prompt, model=model, temperature=temperature)

    seq_len = input_ids.shape[0]
    all_positions = torch.arange(seq_len)

    if skip_first_token and seq_len > 1:
        eligible_positions = all_positions[1:]
    else:
        eligible_positions = all_positions

    if eligible_positions.numel() == 0:
        return [seq_len - 1], input_ids, entropies, {}, {}

    base_target_pos = seq_len - 1

    if candidate_budget is None:
        candidate_budget = max(top_k * 2, math.ceil(0.2 * eligible_positions.numel()))
    candidate_budget = min(candidate_budget, eligible_positions.numel())
    candidate_budget = max(candidate_budget, top_k)

    candidate_entropies = entropies[eligible_positions]
    top_local_idx = torch.topk(candidate_entropies, k=candidate_budget, largest=True).indices
    candidate_positions = eligible_positions[top_local_idx].sort().values

    causal_scores = {}
    base_target_prob = probs[base_target_pos].clamp_min(1e-12)

    for p in candidate_positions.tolist():
        if seq_len <= 1:
            causal_scores[p] = 0.0
            continue

        cf_ids = torch.cat([input_ids[:p], input_ids[p + 1:]], dim=0)

        (
            cf_input_ids,
            _cf_logits,
            cf_probs,
            _cf_entropies,
            cf_final_hidden,
            cf_sent_repr,
        ) = forward_stats(prompt_or_ids=cf_ids, model=model, temperature=temperature)

        d_sent = 1.0 - F.cosine_similarity(
            sent_repr.unsqueeze(0),
            cf_sent_repr.unsqueeze(0),
            dim=-1,
        ).item()

        if p < base_target_pos:
            cf_target_pos = max(0, base_target_pos - 1)
        else:
            cf_target_pos = min(base_target_pos, cf_input_ids.shape[0] - 1)

        cf_target_prob = cf_probs[cf_target_pos].clamp_min(1e-12)
        d_traj = torch.sum(base_target_prob * (base_target_prob.log() - cf_target_prob.log())).item()

        causal_scores[p] = sentence_lambda * d_sent + (1.0 - sentence_lambda) * d_traj

    cand_H = torch.tensor([float(entropies[p]) for p in candidate_positions.tolist()])
    cand_N = torch.tensor([float(causal_scores[p]) for p in candidate_positions.tolist()])

    H_norm = _minmax_norm(cand_H)
    N_norm = _minmax_norm(cand_N)
    cand_score = N_norm * (1.0 + beta * H_norm)

    final_scores = {int(p): float(s) for p, s in zip(candidate_positions.tolist(), cand_score.tolist())}

    universe = set(candidate_positions.tolist())
    if always_include_last:
        universe.add(seq_len - 1)
    universe = sorted(list(universe))

    selected = []
    if always_include_last:
        selected.append(seq_len - 1)

    target_budget = min(top_k, len(universe))

    while len(selected) < target_budget:
        best_p = None
        best_util = -1e18

        for p in universe:
            if p in selected:
                continue

            util = final_scores.get(p, 0.0)
            if len(selected) > 0:
                sims = []
                for q in selected:
                    sim = F.cosine_similarity(
                        final_hidden[p].unsqueeze(0),
                        final_hidden[q].unsqueeze(0),
                        dim=-1,
                    ).item()
                    sims.append(sim)
                util = util - diversity_penalty * max(sims)

            if util > best_util:
                best_util = util
                best_p = p

        if best_p is None:
            break
        selected.append(best_p)

    selected = sorted(set(selected))
    return selected, input_ids, entropies, causal_scores, final_scores


def decode_selected_tokens(input_ids: torch.Tensor, selected_positions: list[int], model: ReplacementModel) -> list[str]:
    tokens = []
    for p in selected_positions:
        tok_id = int(input_ids[p].item())
        tokens.append(model.tokenizer.decode([tok_id]))
    return tokens


# =========================================================
# Compressed attribution
# =========================================================
def attribute_compressed(
    prompt: str | torch.Tensor | list[int],
    model: ReplacementModel,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    max_features_per_position: int | None = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    selected_positions: list[int] | torch.Tensor | None = None,
    target_pos: int | None = None,
) -> Graph:
    logger = logging.getLogger("attribution_compressed")
    logger.propagate = False
    handler = None
    if verbose and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    offload_handles = []
    try:
        return _run_attribution_compressed(
            model=model,
            prompt=prompt,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            max_features_per_position=max_features_per_position,
            offload=offload,
            verbose=verbose,
            offload_handles=offload_handles,
            update_interval=update_interval,
            logger=logger,
            selected_positions=selected_positions,
            target_pos=target_pos,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()
        if handler:
            logger.removeHandler(handler)


def _run_attribution_compressed(
    model,
    prompt,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    max_features_per_position,
    offload,
    verbose,
    offload_handles,
    logger,
    update_interval=4,
    selected_positions=None,
    target_pos=None,
):
    start_time = time.time()

    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
    input_ids = model.ensure_tokenized(prompt)

    ctx = model.setup_attribution(input_ids)
    activation_matrix = ctx.activation_matrix

    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Found {activation_matrix._nnz()} active features")

    full_feat_layers, full_feat_pos, _ = activation_matrix.indices()
    full_feat_values = activation_matrix.values()
    n_layers, n_pos, _ = activation_matrix.shape
    full_total_active_feats = activation_matrix._nnz()

    if selected_positions is not None:
        if isinstance(selected_positions, torch.Tensor):
            selected_positions = selected_positions.detach().cpu().flatten().long()
        else:
            selected_positions = torch.tensor(list(selected_positions), dtype=torch.long)

        if selected_positions.numel() == 0:
            raise ValueError("selected_positions is empty.")

        selected_positions = torch.unique(selected_positions)
        selected_positions = selected_positions[(selected_positions >= 0) & (selected_positions < n_pos)]
        if selected_positions.numel() == 0:
            raise ValueError(f"No valid selected_positions remain after clipping to [0, {n_pos - 1}].")

        if target_pos is None:
            target_pos = int(selected_positions.max().item())
    else:
        if target_pos is None:
            target_pos = n_pos - 1

    if not (0 <= target_pos < n_pos):
        raise ValueError(f"target_pos={target_pos} is out of range for n_pos={n_pos}.")

    if selected_positions is None:
        kept_positions = torch.tensor([target_pos], dtype=torch.long)
    else:
        kept_positions = torch.unique(torch.cat([selected_positions, torch.tensor([target_pos])]))

    logger.info(f"Using target_pos={target_pos}")
    logger.info(f"Keeping positions={kept_positions.tolist()}")

    pos_mask = torch.isin(full_feat_pos.cpu(), kept_positions.cpu())
    keep_feat_idx = torch.where(pos_mask)[0]

    if keep_feat_idx.numel() == 0:
        raise ValueError(f"No active features found on kept positions={kept_positions.tolist()}.")

    if max_features_per_position is not None and max_features_per_position > 0:
        kept_chunks = []
        for p in kept_positions.tolist():
            local_mask = full_feat_pos[keep_feat_idx].cpu() == p
            local_ids = keep_feat_idx[local_mask]
            if local_ids.numel() == 0:
                continue
            local_vals = full_feat_values[local_ids].abs().cpu()
            k = min(max_features_per_position, local_ids.numel())
            top_local = torch.topk(local_vals, k=k, largest=True).indices
            kept_chunks.append(local_ids[top_local])

        if len(kept_chunks) == 0:
            raise ValueError(
                f"After applying max_features_per_position={max_features_per_position}, no active features remain."
            )

        keep_feat_idx = torch.unique(torch.cat(kept_chunks)).sort().values

    cand_feat_layers = full_feat_layers[keep_feat_idx]
    cand_feat_pos = full_feat_pos[keep_feat_idx]
    cand_encoder_vecs = ctx.encoder_vecs[keep_feat_idx]
    total_candidate_feats = keep_feat_idx.numel()

    logger.info(f"Retained {total_candidate_feats} candidate features after token-based compression")

    if offload:
        offload_handles += offload_modules(model.transcoders, offload)

    logger.info("Phase 1: Running forward pass")
    phase_start = time.time()
    with ctx.install_hooks(model):
        residual = model.forward(input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
        ctx._resid_activations[-1] = model.ln_final(residual)
    logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")

    if offload:
        offload_handles += offload_modules([block.mlp for block in model.blocks], offload)

    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()

    full_logit_offset = full_total_active_feats + (n_layers + 1) * n_pos

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        ctx.logits[0, target_pos],
        model.unembed.W_U,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )
    logger.info(
        f"Selected {len(logit_idx)} logits at target_pos={target_pos} with cumulative probability {logit_p.sum().item():.4f}"
    )

    if offload:
        offload_handles += offload_modules([model.unembed, model.embed], offload)

    max_feature_nodes = min(max_feature_nodes or total_candidate_feats, total_candidate_feats)
    logger.info(f"Will include {max_feature_nodes} of {total_candidate_feats} candidate features")

    n_logits = len(logit_idx)
    full_total_nodes = full_logit_offset + n_logits

    edge_matrix = torch.zeros(max_feature_nodes + n_logits, full_total_nodes)
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int64)

    logger.info(f"Input vectors built in {time.time() - phase_start:.2f}s")

    logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()
    for i in range(0, len(logit_idx), batch_size):
        batch = logit_vecs[i : i + batch_size]
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full((batch.shape[0],), target_pos),
            inject_values=batch,
        )
        edge_matrix[i : i + batch.shape[0], :full_logit_offset] = rows.cpu()
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0], dtype=torch.int64) + full_logit_offset
        )
    logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    logger.info("Phase 4: Computing feature attributions")
    phase_start = time.time()
    st = n_logits
    visited_local = torch.zeros(total_candidate_feats, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation", disable=not verbose)

    while n_visited < max_feature_nodes:
        if max_feature_nodes == total_candidate_feats:
            pending_local = torch.arange(total_candidate_feats)
        else:
            influences = compute_partial_influences(edge_matrix[:st], logit_p, row_to_node_index[:st])
            candidate_influences = influences[keep_feat_idx]
            feature_rank_local = torch.argsort(candidate_influences, descending=True).cpu()
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            pending_local = feature_rank_local[~visited_local[feature_rank_local]][:queue_size]

        queue = [pending_local[i : i + batch_size] for i in range(0, len(pending_local), batch_size)]

        for idx_batch_local in queue:
            n_visited += len(idx_batch_local)

            rows = ctx.compute_batch(
                layers=cand_feat_layers[idx_batch_local],
                positions=cand_feat_pos[idx_batch_local],
                inject_values=cand_encoder_vecs[idx_batch_local],
                retain_graph=n_visited < max_feature_nodes,
            )

            end = min(st + batch_size, st + rows.shape[0])
            edge_matrix[st:end, :full_logit_offset] = rows.cpu()
            row_to_node_index[st:end] = keep_feat_idx[idx_batch_local].to(torch.int64)

            visited_local[idx_batch_local] = True
            st = end
            pbar.update(len(idx_batch_local))

    pbar.close()
    logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    logger.info("Phase 5: Compressing graph and packaging")
    phase_start = time.time()

    selected_features = keep_feat_idx[torch.where(visited_local)[0]].sort().values

    position_cols = []
    for layer in range(n_layers + 1):
        base = full_total_active_feats + layer * n_pos
        for p in kept_positions.tolist():
            position_cols.append(base + p)
    position_cols = torch.tensor(position_cols, dtype=torch.int64)
    logit_cols = torch.arange(full_logit_offset, full_total_nodes, dtype=torch.int64)

    col_read = torch.cat([selected_features.to(torch.int64), position_cols, logit_cols])
    edge_matrix = edge_matrix[:, col_read]
    edge_matrix = edge_matrix[row_to_node_index.argsort()]

    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]

    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_tokens=logit_idx,
        logit_probabilities=logit_p,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=model.cfg,
        scan=model.scan,
    )

    try:
        graph.selected_positions = kept_positions.tolist()
        graph.target_pos = int(target_pos)
        graph.max_features_per_position = max_features_per_position
    except Exception:
        pass

    logger.info(
        f"Compressed graph packaged in {time.time() - phase_start:.2f}s; final adjacency shape = {tuple(full_edge_matrix.shape)}"
    )
    logger.info(f"Attribution completed in {time.time() - start_time:.2f}s")
    return graph

