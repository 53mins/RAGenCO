from __future__ import annotations

from typing import Optional, Tuple, Union
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch

def batched_atsp_two_opt_torch(
    dists: torch.Tensor,
    tours: torch.Tensor,
    max_iterations: int = 1000,
    improvement_threshold: float = -1e-6,
) -> Tuple[torch.Tensor, int]:
    """
    Batched 2-opt local search for ATSP (directed / asymmetric costs) using an *exact* delta.

    Args:
      dists: [B, N, N] or [N, N]. Directed cost matrix.
      tours: [B, N] (open) or [B, N+1] (closed; tours[:, -1] == tours[:, 0]).
      max_iterations: max improving moves per instance (global loop count; early-stops if no improvement).
      improvement_threshold: consider a move improving if delta < threshold.

    Returns:
      refined_tours: [B, N+1] closed tours
      iters_done: number of improvement rounds applied (global rounds; instances stop independently)
    """
    if dists.ndim == 2:
        dists = dists.unsqueeze(0)
    if tours.ndim == 1:
        tours = tours.unsqueeze(0)

    B, N, _ = dists.shape
    device = dists.device

    tours = tours.to(device=device, dtype=torch.long)
    if tours.shape[1] == N:
        tours = torch.cat([tours, tours[:, :1]], dim=1)
    elif tours.shape[1] == N + 1:
        # ensure closed
        if (tours[:, -1] != tours[:, 0]).any():
            tours = tours.clone()
            tours[:, -1] = tours[:, 0]
    else:
        raise ValueError(f"tours length must be N or N+1 (N={N}), got {tours.shape[1]}")
    
    d = dists.float()

    iters = 0
    batch_idx = torch.arange(B, device=device)[:, None]  # [B,1]
    b3 = torch.arange(B, device=device)[:, None, None]   # [B,1,1]

    while iters < int(max_iterations):
        nodes = tours[:, :-1]  # [B, N]
        nxt = tours[:, 1:]     # [B, N]
        
        c = d[batch_idx, nodes, nxt]  # [B, N]
        r = d[batch_idx, nxt, nodes]  # [B, N]

        pc = torch.cat([torch.zeros(B, 1, device=device), c.cumsum(dim=1)], dim=1)  # [B, N+1]
        pr = torch.cat([torch.zeros(B, 1, device=device), r.cumsum(dim=1)], dim=1)  # [B, N+1]

        vi = nodes.unsqueeze(2)  # [B, N, 1]  v_i
        vj = nodes.unsqueeze(1)  # [B, 1, N]  v_j
        vi1 = nxt.unsqueeze(2)   # [B, N, 1]  v_{i+1}
        vj1 = nxt.unsqueeze(1)   # [B, 1, N]  v_{j+1}

        new1 = d[b3, vi, vj]     # d(v_i, v_j)           [B, N, N]
        new2 = d[b3, vi1, vj1]   # d(v_{i+1}, v_{j+1})   [B, N, N]
        pc_j = pc[:, :N]         # pc[j] for j=0..N-1
        pc_i1 = pc[:, 1:N+1]     # pc[i+1] for i=0..N-1
        pr_j = pr[:, :N]
        pr_i1 = pr[:, 1:N+1]

        sum_c = pc_j.unsqueeze(1) - pc_i1.unsqueeze(2)  # [B, N, N]
        sum_r = pr_j.unsqueeze(1) - pr_i1.unsqueeze(2)  # [B, N, N]
        
        c_i = c.unsqueeze(2)  # d(v_i, v_{i+1}) [B, N, 1]
        c_j = c.unsqueeze(1)  # d(v_j, v_{j+1}) [B, 1, N]

        delta = new1 + new2 + sum_r - (c_i + c_j + sum_c)  # [B, N, N]

        valid = torch.triu(delta, diagonal=2)

        min_delta, flat_idx = valid.view(B, -1).min(dim=-1)  # per instance
        improved_mask = min_delta < float(improvement_threshold)
        if not bool(improved_mask.any()):
            break

        min_i = torch.div(flat_idx, N, rounding_mode="floor")
        min_j = torch.remainder(flat_idx, N)

        tours_new = tours.clone()
        improved_rows = torch.nonzero(improved_mask, as_tuple=False).view(-1)
        for b in improved_rows.tolist():
            i = int(min_i[b].item())
            j = int(min_j[b].item())
            tours_new[b, i + 1 : j + 1] = torch.flip(tours_new[b, i + 1 : j + 1], dims=(0,))
        tours = tours_new
        iters += 1

    return tours, iters


def atsp_2opt_local_search(
    init_tours: np.ndarray,
    dists: np.ndarray,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """
    Numpy-friendly wrapper around batched_atsp_two_opt_torch.

    Notes:
      - Accepts tours as [N] (open) or [N+1] (closed), and will always return closed tours.
      - Runs on CUDA if available by default.
    """
    if init_tours.ndim == 1:
        init_tours = np.expand_dims(init_tours, axis=0)
        single = True
    else:
        single = False
    if init_tours.ndim != 2:
        raise ValueError("init_tours must be a 1D or 2D array")

    if dists.ndim == 2:
        dists = np.expand_dims(dists, axis=0)
    if dists.ndim != 3:
        raise ValueError("dists must be a 2D or 3D array")

    B = init_tours.shape[0]
    if dists.shape[0] == 1 and B > 1:
        dists = np.repeat(dists, B, axis=0)
    if dists.shape[0] < B:
        raise ValueError(f"dists batch ({dists.shape[0]}) smaller than number of tours ({B})")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tours_t = torch.from_numpy(init_tours.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)
    dists_t = torch.from_numpy(dists.astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)

    refined_t, _ = batched_atsp_two_opt_torch(
        dists_t,
        tours_t,
    )
    refined = refined_t.detach().cpu().numpy()
    if single:
        refined = refined[0]
    return refined

def _would_form_cycle(u: int, v: int, succ: np.ndarray) -> bool:
    """Return True if adding edge u->v would create a directed cycle in current successor map."""
    cur = v
    while cur != -1:
        if cur == u:
            return True
        cur = int(succ[cur])
    return False


def _mapping_to_single_tour(succ: np.ndarray) -> np.ndarray:
    """Convert a complete successor map to a single closed tour starting at node 0."""
    N = int(succ.shape[0])
    tour = [0]
    cur = 0
    seen = {0}
    for _ in range(N):
        nxt = int(succ[cur])
        if nxt < 0:
            break
        tour.append(nxt)
        cur = nxt
        if cur in seen:
            break
        seen.add(cur)
    if len(tour) == N + 1 and tour[-1] == 0 and len(seen) == N:
        return np.asarray(tour, dtype=np.int64)
    return None


def atsp_greedy_merge_decode(
    prob_mats: np.ndarray,
    cost_mat: np.ndarray,
    greedy_cost_alpha: float = 1.0,
    score_noise_scale: float = 0.0,
    score_noise_type: str = "gaussian",
    deterministic: bool = True,
    rng=None,
    eps: float = 1e-9,
    forbid_self: bool = True,
) -> np.ndarray:
    """
    Greedy edge-insertion decoder for ATSP analogous in spirit to TSP's merge_tours.

    Enforces:
      - exactly one outgoing and one incoming edge per node in final tour,
      - subtour elimination during edge insertion (no directed cycles before completion),
      - returns a closed Hamiltonian cycle as [S, N+1] with tour[-1] == tour[0].

    Args:
        prob_mats: [S, N, N] non-negative edge probabilities/scores.
        cost_mat:  [N, N] directed/asymmetric costs (used as a weak bias).
        greedy_cost_alpha: weight on normalized costs in the edge score.
        score_noise_scale: optional per-edge noise added to the score to break ties / add diversity.
        score_noise_type: "gaussian" or "gumbel".
        deterministic: if True, disable score noise.
        rng: numpy Generator for reproducibility.
        eps: numerical stability.
        forbid_self: disallow i->i edges.
    """
    prob_mats = np.asarray(prob_mats, dtype=np.float64)
    if prob_mats.ndim == 2:
        prob_mats = prob_mats[None, ...]
    S, N, _ = prob_mats.shape

    cost = np.asarray(cost_mat, dtype=np.float64)
    if cost.shape != (N, N):
        raise ValueError(f"cost_mat must be [N,N] with N={N}, got {cost.shape}")

    cmin = np.min(cost[np.isfinite(cost)])
    cmax = np.max(cost[np.isfinite(cost)])
    denom = (cmax - cmin) + eps
    cost_norm = (cost - cmin) / denom
    cost_norm = np.clip(cost_norm, 0.0, 1.0)

    tours = np.zeros((S, N + 1), dtype=np.int64)

    # Precompute edge order per candidate (descending score)
    for s in range(S):
        prob = prob_mats[s].copy()
        if forbid_self:
            np.fill_diagonal(prob, 0.0)

        score = np.log(prob + eps) - float(greedy_cost_alpha) * cost_norm
        if (not bool(deterministic)) and float(score_noise_scale) > 0.0:
            if rng is None:
                rng = np.random.default_rng()
            if str(score_noise_type).lower() == "gaussian":
                jitter = rng.standard_normal(size=score.shape)
            else:
                U = rng.random(size=score.shape)
                jitter = -np.log(-np.log(U + eps) + eps)
            score = score + float(score_noise_scale) * jitter
        # Force disallowed/self edges to -inf so they never get selected.
        if forbid_self:
            np.fill_diagonal(score, -np.inf)

        flat_order = np.argsort(score.reshape(-1))[::-1]

        succ = -np.ones(N, dtype=np.int64)
        pred = -np.ones(N, dtype=np.int64)
        selected = 0

        for e in flat_order:
            u = int(e // N)
            v = int(e % N)
            if forbid_self and u == v:
                continue
            if succ[u] != -1 or pred[v] != -1:
                continue
            if selected < N - 1 and _would_form_cycle(u, v, succ):
                continue
            succ[u] = v
            pred[v] = u
            selected += 1
            if selected == N:
                break

        # Fallback: if greedy couldn't complete a valid mapping, use Hungarian projection and merge cycles.
        if (succ == -1).any() or (pred == -1).any():
            p = decode_permutation_from_heat(prob, eps=eps, forbid_self=forbid_self, deterministic=True)
            tour_list = permutation_to_tour_from_mapping(p, dists=cost)
            tours[s] = np.asarray(tour_list, dtype=np.int64)
            continue

        tour = _mapping_to_single_tour(succ)
        if tour is None:
            tour_list = permutation_to_tour_from_mapping(succ, dists=cost)
            tours[s] = np.asarray(tour_list, dtype=np.int64)
        else:
            tours[s] = tour

    return tours


def tour_cost_from_matrix(tour: np.ndarray, dists: np.ndarray) -> float:
    """Compute total cost of a closed tour given a cost matrix."""
    N = len(tour) - 1
    cost = 0.0
    for i in range(N):
        a = int(tour[i])
        b = int(tour[i + 1])
        cost += float(dists[a, b])
    return cost


def decode_permutation_from_heat(
    prob_matrix,
    eps: float = 1e-9,
    forbid_self: bool = True,
    tau: float = 1.0,
    noise_scale: float = 0.0,
    noise_type: str = "gumbel",
    deterministic: bool = False,
    rng=None,
):
    """Map a soft assignment matrix to a permutation via Hungarian algorithm with optional diversification.

    Args:
        prob_matrix: [N, N] non-negative scores (higher is better).
        forbid_self: if True, disallow i->i assignments.
        noise_scale: per-entry noise added to the cost matrix (0 disables). Applied only if not deterministic.
        noise_type: "gumbel" or "gaussian".
        deterministic: if True, disable noise.
        rng: numpy Generator to make noise reproducible.
    Returns:
        p: [N] int64 array where p[i] = next node.
    """
    prob = np.asarray(prob_matrix, dtype=np.float64).copy()
    if prob.ndim != 2 or prob.shape[0] != prob.shape[1]:
        raise ValueError("prob_matrix must be a square 2D array")
    if not np.all(np.isfinite(prob)):
        raise ValueError("prob_matrix contains non-finite values")

    if forbid_self:
        np.fill_diagonal(prob, 0.0)

    tau = float(tau)
    if tau <= 0.0:
        tau = 1.0
    cost = -np.log(prob + eps) / tau

    if (not deterministic) and float(noise_scale) > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        if str(noise_type).lower() == "gaussian":
            noise = rng.standard_normal(size=cost.shape)
        else:
            # i.i.d. Gumbel(0,1)
            U = rng.random(size=cost.shape)
            noise = -np.log(-np.log(U + eps) + eps)
        cost = cost + float(noise_scale) * noise

    if forbid_self:
        np.fill_diagonal(cost, 1e9)

    row_ind, col_ind = linear_sum_assignment(cost)
    p = np.empty(prob.shape[0], dtype=np.int64)
    p[row_ind] = col_ind
    return p


def permutation_to_tour_from_mapping(p, dists=None):
    """Convert mapping p (p[i] = next_node) to a single closed tour.
    If multiple cycles exist, merge them greedily using cost matrix and refine with 2-opt.
    """
    N = len(p)
    visited = [False] * N
    cycles = []
    for start in range(N):
        if not visited[start]:
            cur = start
            cycle = []
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = int(p[cur])
            cycles.append(cycle)

    if len(cycles) == 1:
        c = cycles[0]
        return c + [c[0]]

    if dists is None:
        # fallback: concatenate cycles and ensure all nodes are included
        tour = []
        for c in cycles:
            tour.extend(c)
        seen = set()
        final = []
        for v in tour:
            if v not in seen:
                final.append(v)
                seen.add(v)
        for i in range(N):
            if i not in seen:
                final.append(i)
        final.append(final[0])
        return final

    cycles_copy = [list(c) for c in cycles]

    def merge_cost_and_merged_cycle(ca, cb, dmat):
        best_delta = None
        best_merged = None
        for ia in range(len(ca)):
            a_i = ca[ia]
            a_next = ca[(ia + 1) % len(ca)]
            for jb in range(len(cb)):
                b_j = cb[jb]
                b_next = cb[(jb + 1) % len(cb)]
                cost_before = dmat[a_i, a_next] + dmat[b_j, b_next]
                cost_after = dmat[a_i, b_next] + dmat[b_j, a_next]
                delta = cost_after - cost_before
                if (best_delta is None) or (delta < best_delta):
                    merged = ca[0:ia+1] + cb[jb+1:] + cb[:jb+1] + ca[ia+1:]
                    best_delta = delta
                    best_merged = merged
        return best_delta, best_merged

    if isinstance(dists, torch.Tensor):
        dmat = dists.detach().cpu().numpy()
    else:
        dmat = np.array(dists, dtype=float)

    while len(cycles_copy) > 1:
        best_pair = None
        best_pair_delta = None
        best_merged_cycle = None
        for i in range(len(cycles_copy)):
            for j in range(i+1, len(cycles_copy)):
                ca = cycles_copy[i]
                cb = cycles_copy[j]
                delta, merged = merge_cost_and_merged_cycle(ca, cb, dmat)
                if merged is None:
                    continue
                if (best_pair_delta is None) or (delta < best_pair_delta):
                    best_pair_delta = delta
                    best_pair = (i, j)
                    best_merged_cycle = merged
        if best_pair is None:
            break
        i, j = best_pair
        if i < j:
            cycles_copy.pop(j)
            cycles_copy.pop(i)
        else:
            cycles_copy.pop(i)
            cycles_copy.pop(j)
        cycles_copy.append(best_merged_cycle)

    merged_cycle = cycles_copy[0]
    seen = set(merged_cycle)
    for v in range(N):
        if v not in seen:
            merged_cycle.append(v)
            seen.add(v)

    refined = atsp_2opt_local_search(np.array(merged_cycle), dmat)
    if isinstance(refined, np.ndarray) and refined.ndim == 2:
        refined_tour = refined[0]
    else:
        refined_tour = refined
    return refined_tour.tolist()


class ATSPEvaluator:
    """Evaluate tour cost on a cost matrix."""
    def __init__(self, cost_matrix):
        self.cost_matrix = cost_matrix

    def evaluate(self, tour):
        N = len(tour) - 1
        c = 0.0
        for i in range(N):
            a = int(tour[i])
            b = int(tour[i + 1])
            c += float(self.cost_matrix[a, b])
        return c