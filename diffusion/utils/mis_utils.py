import torch
import numpy as np
from typing import Optional, Tuple


def mis_decode_np(predictions, adj_matrix):
    """Decode the labels to the MIS (NumPy / SciPy CSR adj)."""
    solution = np.zeros_like(predictions.astype(int))
    sorted_predict_labels = np.argsort(-predictions)
    csr_adj_matrix = adj_matrix.tocsr()

    for i in sorted_predict_labels:
        next_node = i
        if solution[next_node] == -1:
            continue
        solution[csr_adj_matrix[next_node].nonzero()[1]] = -1
        solution[next_node] = 1

    return (solution == 1).astype(int)


def greedy_mis_from_scores(scores: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """Greedy MIS insertion from node scores.

    Args:
        scores: [B, N] float tensor of node scores (higher = prefer to select)
        adj: [B, N, N] or [N, N] binary adjacency matrix (0/1).
    Returns:
        solutions: [B, N] bool tensor representing a maximal independent set.
    """
    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(scores.shape[0], -1, -1)
    B, N = scores.shape
    device = scores.device
    solutions = torch.zeros((B, N), dtype=torch.bool, device=device)

    for b in range(B):
        order = torch.argsort(scores[b], descending=True)
        masked = torch.zeros(N, dtype=torch.bool, device=device)
        sol = torch.zeros(N, dtype=torch.bool, device=device)
        adj_b = adj[b].bool()
        for idx in order:
            if not masked[idx]:
                sol[idx] = True
                # mask node itself and its neighbors
                nbrs = adj_b[idx]
                masked = masked | nbrs
                masked[idx] = True
        solutions[b] = sol
    return solutions


def greedy_with_initial(
    scores: torch.Tensor,
    adj: torch.Tensor,
    initial_sol: Optional[torch.Tensor] = None,
    target_num: Optional[int] = None,
) -> torch.Tensor:
    """Greedy insertion that respects an initial solution.

    Mask neighbors of initial selected nodes first; then insert according to scores.

    Args:
        scores: [B, N]
        adj: [B, N, N] or [N, N]
        initial_sol: [B, N] bool or None
        target_num: optional int, stop when reached
    Returns:
        solutions: [B, N] bool
    """
    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(scores.shape[0], -1, -1)
    B, N = scores.shape
    device = scores.device
    solutions = torch.zeros((B, N), dtype=torch.bool, device=device)

    for b in range(B):
        order = torch.argsort(scores[b], descending=True)
        masked = torch.zeros(N, dtype=torch.bool, device=device)
        sol = torch.zeros(N, dtype=torch.bool, device=device)
        adj_b = adj[b].bool()

        if initial_sol is not None:
            init = initial_sol[b].bool()
            sol = init.clone()
            # mask neighbors + self of initial
            init_idx = torch.nonzero(init, as_tuple=False).squeeze(-1)
            if init_idx.numel() > 0:
                masked = masked | adj_b[init_idx].any(dim=0)
                masked[init_idx] = True

        inserted = int(sol.sum().item())
        for idx in order:
            if target_num is not None and inserted >= target_num:
                break
            if not masked[idx]:
                sol[idx] = True
                nbrs = adj_b[idx]
                masked = masked | nbrs
                masked[idx] = True
                inserted += 1

        solutions[b] = sol
    return solutions


def sample_candidates_from_anchor(
    anchor_probs: torch.Tensor,
    S: int = 3,
    use_gumbel: bool = True,
    sigma: float = 1.8,
    gumbel_scale: float = 1.8,
) -> torch.Tensor:
    """Sample S candidate probability vectors in logit space from an anchor.

    Args:
        anchor_probs: [B, N] in (0,1)
    Returns:
        noisy_prob: [B*S, N]
    """
    eps = 1e-6
    B, N = anchor_probs.shape
    device = anchor_probs.device

    anchor_rep = anchor_probs.unsqueeze(1).repeat(1, S, 1).reshape(B * S, N)
    logit_anchor = torch.logit(anchor_rep.clamp(eps, 1 - eps))

    if use_gumbel:
        U = torch.rand_like(logit_anchor, device=device)
        gumbel = -torch.log(-torch.log(U + eps) + eps)
        noisy_logit = logit_anchor + gumbel * gumbel_scale
    else:
        noisy_logit = logit_anchor + torch.randn_like(logit_anchor, device=device) * sigma

    noisy_prob = torch.sigmoid(noisy_logit)
    return noisy_prob


def mis_entropic_projection_from_logits(
    logits: torch.Tensor,
    adj: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    tau: float = 1.0,
    num_iters: int = 8,
    step_size: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project logits onto the local MIS relaxation with Bernoulli entropy.

    We solve an entropic projection onto:
        Q_MIS = {x in [0,1]^N : x_i + x_j <= 1 for all (i,j) in E}

    using a simple projected dual ascent scheme on the local edge constraints.

    Args:
        logits: [B, N] node logits.
        adj: [B, N, N] or [N, N] adjacency matrix with non-zero entries for edges.
        node_mask: optional [B, N] bool mask for valid nodes.
        tau: entropy temperature; smaller values enforce sharper probabilities.
        num_iters: number of dual ascent steps.
        step_size: dual ascent step size.
        eps: numerical stability.

    Returns:
        probs: [B, N] projected probabilities.
        violation: [B] maximum edge-constraint violation after projection.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must have shape [B, N], got {tuple(logits.shape)}")

    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(logits.shape[0], -1, -1)
    elif adj.dim() != 3:
        raise ValueError(f"adj must have shape [B, N, N] or [N, N], got {tuple(adj.shape)}")

    B, N = logits.shape
    if adj.shape[0] != B or adj.shape[1] != N or adj.shape[2] != N:
        raise ValueError(
            f"adj shape {tuple(adj.shape)} is incompatible with logits shape {tuple(logits.shape)}"
        )

    dtype = logits.dtype
    device = logits.device

    tau = float(max(tau, eps))
    scaled_logits = logits / tau

    edge_mask = adj.to(dtype=torch.bool)
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    edge_mask = edge_mask & (~eye)

    if node_mask is not None:
        valid = node_mask.to(dtype=torch.bool)
        edge_mask = edge_mask & valid.unsqueeze(-1) & valid.unsqueeze(-2)
    else:
        valid = None

    lam = torch.zeros((B, N, N), dtype=dtype, device=device)
    edge_mask_f = edge_mask.to(dtype)

    for _ in range(max(0, int(num_iters))):
        node_penalty = lam.sum(dim=-1)
        probs = torch.sigmoid(scaled_logits - node_penalty)
        if valid is not None:
            probs = probs * valid.to(dtype)

        grad = (probs.unsqueeze(-1) + probs.unsqueeze(-2) - 1.0) * edge_mask_f
        lam = (lam + float(step_size) * grad).clamp_min_(0.0)
        lam = lam * edge_mask_f
        lam = 0.5 * (lam + lam.transpose(-1, -2))

    node_penalty = lam.sum(dim=-1)
    probs = torch.sigmoid(scaled_logits - node_penalty)
    if valid is not None:
        probs = probs * valid.to(dtype)

    violation = (
        (probs.unsqueeze(-1) + probs.unsqueeze(-2) - 1.0).clamp_min(0.0) * edge_mask_f
    ).amax(dim=(-1, -2))
    return probs, violation


def mis_projection_residual(
    probs: torch.Tensor,
    adj: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return max edge-constraint violation for MIS local-relaxation states."""
    if probs.dim() != 2:
        raise ValueError(f"probs must have shape [B, N], got {tuple(probs.shape)}")

    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(probs.shape[0], -1, -1)
    elif adj.dim() != 3:
        raise ValueError(f"adj must have shape [B, N, N] or [N, N], got {tuple(adj.shape)}")

    B, N = probs.shape
    if adj.shape[0] != B or adj.shape[1] != N or adj.shape[2] != N:
        raise ValueError(
            f"adj shape {tuple(adj.shape)} is incompatible with probs shape {tuple(probs.shape)}"
        )

    edge_mask = adj.to(dtype=torch.bool)
    eye = torch.eye(N, device=probs.device, dtype=torch.bool).unsqueeze(0)
    edge_mask = edge_mask & (~eye)
    if node_mask is not None:
        valid = node_mask.to(dtype=torch.bool)
        edge_mask = edge_mask & valid.unsqueeze(-1) & valid.unsqueeze(-2)

    edge_mask_f = edge_mask.to(probs.dtype)
    return (
        (probs.unsqueeze(-1) + probs.unsqueeze(-2) - 1.0).clamp_min(0.0) * edge_mask_f
    ).amax(dim=(-1, -2))


def _repair_independent_set(sol: torch.Tensor, adj_bool: torch.Tensor, scores: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Ensure sol is an independent set by removing nodes from conflicts.

    If scores is provided, remove the lower-score endpoint in each conflict.
    Args:
        sol: [N] bool
        adj_bool: [N, N] bool, symmetric, diag=0
        scores: [N] float or None
    """
    sol = sol.clone()
    N = sol.numel()
    if scores is None:
        scores = torch.arange(N, device=sol.device, dtype=torch.float32)  # deterministic tie-break

    # Repeatedly remove conflicts until independent
    while True:
        idx = torch.nonzero(sol, as_tuple=False).squeeze(-1)
        if idx.numel() <= 1:
            break
        sub = adj_bool[idx][:, idx]
        # any conflict?
        if not sub.any():
            break
        # find one conflict edge (i,j)
        i_pos, j_pos = torch.nonzero(sub, as_tuple=True)
        # take the first (upper triangle or any)
        i = idx[i_pos[0]]
        j = idx[j_pos[0]]
        # remove lower score
        if scores[i] >= scores[j]:
            sol[j] = False
        else:
            sol[i] = False
    return sol


def local_improvement_swap_12(
    sol: torch.Tensor,
    adj: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    node_mask: Optional[torch.Tensor] = None,
    max_iters: int = 50,
    top_k: int = 60,
) -> torch.Tensor:
    """Simple (1 -> 2) swap local improvement for MIS.

    Tries to find an improving move that removes 1 selected node v and adds 2 non-selected nodes
    (u,w) such that:
      - u and w are not adjacent
      - u and w do not conflict with S \ {v}
    This guarantees a +1 improvement in size.

    Args:
        sol: [N] bool (initial independent set; if not, it will be repaired)
        adj: [N, N] (0/1) or bool
        scores: [N] float scores used to guide swaps; defaults to zeros
        node_mask: [N] bool; padded nodes will be ignored
    """
    device = sol.device
    N = sol.numel()

    if scores is None:
        scores = torch.zeros(N, device=device)

    adj_bool = adj.bool()
    if node_mask is not None:
        m = node_mask.bool()
        # ignore padded nodes
        adj_bool = adj_bool & (m[:, None] & m[None, :])
        sol = sol & m
        # push padded nodes scores very low
        scores = scores.masked_fill(~m, -1e9)

    # repair + make maximal by greedy fill
    sol = _repair_independent_set(sol, adj_bool, scores=scores)

    # Make maximal by greedy add (only for uncovered nodes)
    # This is cheaper than re-running greedy from scratch.
    def _fill_maximal(sol_in: torch.Tensor) -> torch.Tensor:
        sol_out = sol_in.clone()
        # conflicts count for each node with current sol
        conflicts = (adj_bool.float() @ sol_out.float())
        addable = (~sol_out) & (conflicts == 0)
        if addable.any():
            order = torch.argsort(scores, descending=True)
            for u in order.tolist():
                if addable[u]:
                    sol_out[u] = True
                    # update conflicts/addable
                    conflicts = conflicts + adj_bool[u].float()
                    addable = (~sol_out) & (conflicts == 0)
        return sol_out

    sol = _fill_maximal(sol)

    for _ in range(max_iters):
        idx_sel = torch.nonzero(sol, as_tuple=False).squeeze(-1)
        if idx_sel.numel() == 0:
            break

        conflicts = (adj_bool.float() @ sol.float())  # [N]
        outside = ~sol
        # nodes that conflict with exactly one selected node
        cands_global = outside & (conflicts == 1)

        best_move = None  # (gain_score, v, u, w)
        # iterate v in selected nodes (could be randomized; keep deterministic)
        for v in idx_sel.tolist():
            # candidates that are adjacent to v and have v as their only selected neighbor
            # since conflicts==1, adjacency to v implies the unique selected neighbor is v
            L_mask = cands_global & adj_bool[:, v]
            L = torch.nonzero(L_mask, as_tuple=False).squeeze(-1)
            if L.numel() < 2:
                continue

            # sort by score, take top_k
            L_scores = scores[L]
            k = min(top_k, L.numel())
            top_idx = torch.argsort(L_scores, descending=True)[:k]
            Lk = L[top_idx]

            # find best pair in Lk that are non-adjacent
            # try u in order, pick best w among nonadj
            for u in Lk.tolist():
                partners = Lk
                nonadj = ~adj_bool[u, partners]
                # remove self if present
                nonadj = nonadj & (partners != u)
                if not nonadj.any():
                    continue
                w_candidates = partners[nonadj]
                # choose w with max score
                w = w_candidates[torch.argmax(scores[w_candidates]).item()].item()
                gain_score = scores[u].item() + scores[w].item() - scores[v].item()
                if (best_move is None) or (gain_score > best_move[0]):
                    best_move = (gain_score, v, u, w)

        if best_move is None:
            break

        _, v, u, w = best_move
        # apply swap v -> {u,w}
        sol[v] = False
        sol[u] = True
        sol[w] = True
        # safety repair & fill (should still be independent, but keep robust)
        sol = _repair_independent_set(sol, adj_bool, scores=scores)
        sol = _fill_maximal(sol)

    return sol


def local_improvement_swap_12_batched(
    solutions: torch.Tensor,
    adj: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    node_mask: Optional[torch.Tensor] = None,
    max_iters: int = 50,
    top_k: int = 60,
) -> torch.Tensor:
    """Batched wrapper for local_improvement_swap_12.

    Args:
        solutions: [B, N] bool
        adj: [B, N, N] or [N, N]
        scores: [B, N] or None
        node_mask: [B, N] or None
    """
    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(solutions.shape[0], -1, -1)
    B, N = solutions.shape
    out = solutions.clone()
    if scores is None:
        scores = torch.zeros((B, N), device=solutions.device)

    for b in range(B):
        out[b] = local_improvement_swap_12(
            out[b],
            adj[b],
            scores=scores[b],
            node_mask=None if node_mask is None else node_mask[b],
            max_iters=max_iters,
            top_k=top_k,
        )
    return out


def evaluate_solutions(
    solutions: torch.Tensor,
    labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Evaluate solutions.

    Returns:
        sizes: [B] tensor (number of selected nodes)
        stats: if labels is provided, recall vs labels.
    """
    sizes = solutions.sum(dim=1).float()
    stats = None
    if labels is not None:
        gt = labels.bool()
        overlap = (solutions & gt).sum(dim=1).float()
        gt_size = gt.sum(dim=1).float().clamp(min=1.0)
        recall = (overlap / gt_size).cpu()
        stats = recall
    return sizes, stats
