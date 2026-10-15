import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import math

from diffusion.pl_tsp_model import TSPModel
from diffusion.utils.tsp_utils import batched_two_opt_torch, merge_tours
from diffusion.utils.diffusion_schedulers import InferenceSchedule
from diffusion.models.ragenco_transformer import get_ragenco_model

# -------- Sinkhorn --------
def sinkhorn_normalize(prob, iters=10, tau=0.1, eps=1e-9, mask=None):
    """
    Approximate row/column normalization to produce a near doubly-stochastic matrix.

    Args:
        prob: [B, N, N] non-negative scores.
        iters: number of Sinkhorn iterations.
        tau: temperature applied before exponentiation.
        eps: small constant for numerical stability.
        mask: optional boolean mask (broadcastable to prob) where True indicates
              allowed entries. Disallowed entries are forced to 0 and excluded
              from normalization.
    Returns:
        p: normalized matrix with same shape as prob.
    """
    logp = prob / tau
    # numerical stabilization
    logp = logp - logp.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        # Disallow masked entries BEFORE exp so they never get mass.
        logp = logp.masked_fill(~mask, -1e9)

    p = torch.exp(logp).clamp(min=eps)

    if mask is not None:
        # clamp(min=eps) would otherwise revive disallowed entries.
        p = p.masked_fill(~mask, 0.0)

    for _ in range(iters):
        p = p / (p.sum(dim=-1, keepdim=True) + eps)  # row norm
        p = p / (p.sum(dim=-2, keepdim=True) + eps)  # col norm
        if mask is not None:
            p = p.masked_fill(~mask, 0.0)
    return p


# -------- Hungarian decoding (optional) --------
def decode_permutation_from_heat(prob_matrix, eps=1e-9, forbid_self=True):
    """Map a soft assignment matrix to a permutation (next-node mapping) via Hungarian algorithm.

    Args:
        prob_matrix: [N, N] numpy array of non-negative scores (higher is better).
        eps: numerical stability for log.
        forbid_self: if True, disallow i->i assignments.
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
    cost = -np.log(prob + eps)
    row_ind, col_ind = linear_sum_assignment(cost)
    p = np.empty(prob.shape[0], dtype=np.int64)
    p[row_ind] = col_ind
    return p


def euclidean_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix from points [N,2]."""
    pts = np.asarray(points, dtype=np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def tour_length_np(points_xy: np.ndarray, tours: np.ndarray) -> np.ndarray:
    """Vectorized TSP tour length.

    Args:
        points_xy: [N,2] float array
        tours: [K, N+1] int array, where tours[:, 0] == tours[:, -1] typically.
    Returns:
        lengths: [K] float array
    """
    pts = np.asarray(points_xy)
    tr = np.asarray(tours, dtype=np.int64)
    if tr.ndim == 1:
        tr = tr[None, :]
    # [K, N+1, 2]
    seq = pts[tr]
    # [K, N, 2]
    seg = seq[:, 1:] - seq[:, :-1]
    return np.sqrt((seg * seg).sum(axis=-1)).sum(axis=-1)


def permutation_to_tour_from_mapping(p, dists=None):
    """Convert mapping p (p[i] = next node) to a single tour (node order of length N).
    If multiple cycles exist, merge them greedily using the distance matrix.
    """
    p = np.asarray(p, dtype=np.int64)
    if p.ndim != 1:
        raise ValueError("p must be a 1D array")
    N = int(p.shape[0])

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
        return cycles[0]

    if dists is None:
        tour = []
        for c in cycles:
            tour.extend(c)
        seen = set()
        out = []
        for v in tour:
            if v not in seen:
                out.append(v)
                seen.add(v)
        for v in range(N):
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out

    dmat = np.asarray(dists, dtype=np.float64)
    cycles_copy = [list(c) for c in cycles]

    def merge_cost_and_merged_cycle(ca, cb, dmat_):
        best_delta = None
        best_merged = None
        la = len(ca)
        lb = len(cb)
        for ia in range(la):
            a_i = ca[ia]
            a_next = ca[(ia + 1) % la]
            for jb in range(lb):
                b_j = cb[jb]
                b_next = cb[(jb + 1) % lb]
                cost_before = dmat_[a_i, a_next] + dmat_[b_j, b_next]
                cost_after = dmat_[a_i, b_next] + dmat_[b_j, a_next]
                delta = cost_after - cost_before
                if (best_delta is None) or (delta < best_delta):
                    merged = ca[0:ia + 1] + cb[jb + 1:] + cb[:jb + 1] + ca[ia + 1:]
                    best_delta = delta
                    best_merged = merged
        return best_delta, best_merged

    while len(cycles_copy) > 1:
        best_pair = None
        best_pair_delta = None
        best_merged_cycle = None
        for i in range(len(cycles_copy)):
            for j in range(i + 1, len(cycles_copy)):
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
    return merged_cycle

class RAGenCOTSPModel(TSPModel):
    """
    RAGenCO TSP model for relaxation-aligned test-time scaling.

    The denoiser predicts edge logits, Sinkhorn projection forms the projected
    control state, and anchor best-of-B branching expands candidates around that
    aligned state before greedy tour decoding and optional 2Opt.
    """
    def __init__(self, param_args=None, load_dataset=True):
        super().__init__(param_args=param_args, load_dataset=load_dataset)
        self.model = get_ragenco_model(self.args)

        # compile controls
        self.use_torch_compile = getattr(self.args, "use_torch_compile", False)
        self.use_compile_warmup = getattr(self.args, "use_compile_warmup", False)
        self.compile_warmup_nodes = getattr(self.args, "compile_warmup_nodes", 50)
        self._compiled = False
        self._compile_warmed = False
        self._compiled_infer_fn = None

        # sampling params (logit-normal)
        self.mu = -0.8
        self.sigma = 0.8

        # Sinkhorn options
        self.use_sinkhorn = getattr(self.args, "use_sinkhorn", True)
        self.sinkhorn_iters = getattr(self.args, "sinkhorn_iters", 1)
        self.sinkhorn_tau = getattr(self.args, "sinkhorn_tau", 0.05)

        # Fast candidate sampling (parallel candidates)
        self.fast_num_candidates = getattr(self.args, "fast_num_candidates", 3)
        self.fast_sigma = getattr(self.args, "fast_sigma", 1.8)
        self.fast_t_anchor = float(torch.sigmoid(torch.tensor(self.mu)))
        self.final_top_m = getattr(self.args, "final_top_m", 3)
        self.use_final_two_opt = getattr(self.args, "use_final_two_opt", True)

        # Gumbel sampling options for parallel candidates
        self.use_gumbel = getattr(self.args, "use_gumbel", True)
        self.fast_gumbel_scale = getattr(self.args, "fast_gumbel_scale", self.fast_sigma)

        self.use_hungarian = getattr(self.args, "use_hungarian", False)

        # cached inference schedule
        self._cached_schedule = None

    def _infer_anchor_and_candidates_eager(self, points, neutral, t_tensor, no_self_mask):
        """Eager version of the compiled inference path.

        This matches the compiled infer function: candidate generation + model forward +
        sigmoid + optional Sinkhorn.

        Returns:
            x0_prob_rep: [B*S, N, N]
            anchor:      [B,   N, N]
        """
        B, N, _ = points.shape
        S = int(self.fast_num_candidates)
        eps = 1e-6

        # 1) anchor
        init_logits = self.model(points, t_tensor, x_t_adj=neutral)
        anchor = torch.sigmoid(init_logits)
        if self.use_sinkhorn:
            anchor = sinkhorn_normalize(
                anchor,
                iters=max(1, int(self.sinkhorn_iters)),
                tau=float(self.sinkhorn_tau),
                mask=no_self_mask,
            )

        # 2) candidates (batched over B*S)
        anchor_rep = anchor.unsqueeze(1).expand(B, S, N, N).reshape(B * S, N, N)
        logit_anchor = torch.logit(anchor_rep.clamp(eps, 1 - eps))

        if self.use_gumbel:
            U = torch.rand_like(logit_anchor)
            g = -torch.log(-torch.log(U + eps) + eps)
            noisy_logit = logit_anchor + g * float(self.fast_gumbel_scale)
        else:
            noisy_logit = logit_anchor + torch.randn_like(logit_anchor) * float(self.fast_sigma)

        noisy_heat = torch.sigmoid(noisy_logit)
        if self.use_sinkhorn:
            noisy_heat = sinkhorn_normalize(
                noisy_heat,
                iters=max(1, int(self.sinkhorn_iters)),
                tau=float(self.sinkhorn_tau),
                mask=no_self_mask,
            )

        points_rep = points.unsqueeze(1).expand(B, S, N, 2).reshape(B * S, N, 2)
        t_rep = t_tensor.repeat_interleave(S)
        pred_logits_rep = self.model(points_rep, t_rep, x_t_adj=noisy_heat)
        x0_prob_rep = torch.sigmoid(pred_logits_rep)
        if self.use_sinkhorn:
            x0_prob_rep = sinkhorn_normalize(
                x0_prob_rep,
                iters=max(1, int(self.sinkhorn_iters)),
                tau=float(self.sinkhorn_tau),
                mask=no_self_mask,
            )

        return x0_prob_rep, anchor

    def _maybe_compile_model(self):
        """Compile a single, larger inference function (anchor + candidates + sigmoid + sinkhorn).

        We intentionally compile a *function* instead of compiling self.model in-place.
        This keeps the compiled graph larger/cleaner (fewer graph breaks) and avoids
        nested compilation (compiled module called from a compiled function).
        """
        if (not self.use_torch_compile) or self._compiled:
            return

        S = int(self.fast_num_candidates)
        use_gumbel = bool(self.use_gumbel)
        use_sinkhorn = bool(self.use_sinkhorn)
        sinkhorn_iters = max(1, int(self.sinkhorn_iters))
        sinkhorn_tau = float(self.sinkhorn_tau)
        sigma = float(self.fast_sigma)
        gumbel_scale = float(self.fast_gumbel_scale)
        eps = 1e-6

        # Cache module reference to avoid attribute lookups inside compiled region.
        model = self.model

        def infer_anchor_and_candidates(points, neutral, t_tensor, no_self_mask):
            """Return:
                x0_prob_rep: [B*S, N, N]
                anchor:      [B,   N, N]
            """
            B, N, _ = points.shape

            # 1) anchor
            init_logits = model(points, t_tensor, x_t_adj=neutral)
            anchor = torch.sigmoid(init_logits)
            if use_sinkhorn:
                anchor = sinkhorn_normalize(anchor, iters=sinkhorn_iters, tau=sinkhorn_tau, mask=no_self_mask)

            # 2) candidates (batched over B*S)
            anchor_rep = anchor.unsqueeze(1).expand(B, S, N, N).reshape(B * S, N, N)
            logit_anchor = torch.logit(anchor_rep.clamp(eps, 1 - eps))

            if use_gumbel:
                U = torch.rand_like(logit_anchor)
                g = -torch.log(-torch.log(U + eps) + eps)
                noisy_logit = logit_anchor + g * gumbel_scale
            else:
                noisy_logit = logit_anchor + torch.randn_like(logit_anchor) * sigma

            noisy_heat = torch.sigmoid(noisy_logit)
            if use_sinkhorn:
                noisy_heat = sinkhorn_normalize(noisy_heat, iters=sinkhorn_iters, tau=sinkhorn_tau, mask=no_self_mask)

            points_rep = points.unsqueeze(1).expand(B, S, N, 2).reshape(B * S, N, 2)
            t_rep = t_tensor.repeat_interleave(S)

            pred_logits_rep = model(points_rep, t_rep, x_t_adj=noisy_heat)
            x0_prob_rep = torch.sigmoid(pred_logits_rep)
            if use_sinkhorn:
                x0_prob_rep = sinkhorn_normalize(x0_prob_rep, iters=sinkhorn_iters, tau=sinkhorn_tau, mask=no_self_mask)

            return x0_prob_rep, anchor

        # Compile with a speed-oriented mode. "max-autotune" enables more aggressive tuning
        self._compiled_infer_fn = torch.compile(
            infer_anchor_and_candidates,
            mode="max-autotune",
        )
        self._compiled = True
        print("[info] torch.compile enabled for inference function (max-autotune).")

    def _maybe_warmup_compile(self):
        if self._compiled and self.use_compile_warmup and (not self._compile_warmed):
            N = int(self.compile_warmup_nodes)
            B = 1
            device = self.device
            dummy_points = torch.zeros(B, N, 2, device=device)
            dummy_adj = torch.zeros(B, N, N, device=device)
            neutral = torch.full((B, N, N), 0.5, device=device)
            no_self_mask = ~torch.eye(N, device=device, dtype=torch.bool)

            amp_dtype = torch.bfloat16 if getattr(self.args, "bf16", False) else (
                torch.float16 if getattr(self.args, "fp16", False) else torch.float32
            )

            # Match real inference dtypes as much as possible to avoid recompiles.
            t_anchor = float(self.fast_t_anchor)
            t_tensor = torch.full((B,), t_anchor, device=device)
            
            if self._compiled_infer_fn is not None:
                with torch.no_grad(), torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), dtype=amp_dtype):
                    _ = self._compiled_infer_fn(dummy_points, neutral, t_tensor, no_self_mask)
            self._compile_warmed = True
            print(f"[info] compile warmup done with N={N}.")

    def on_fit_start(self):
        self._maybe_compile_model()
        self._maybe_warmup_compile()

    def on_validation_start(self):
        self._maybe_compile_model()
        self._maybe_warmup_compile()

    def on_test_start(self):
        self._maybe_compile_model()
        self._maybe_warmup_compile()

    def forward(self, x, t, x_t_adj=None):
        return self.model(x, t, x_t_adj)

    def sample_time_logit_normal(self, num_samples, device):
        s = torch.randn(num_samples, device=device) * self.sigma + self.mu
        t = torch.sigmoid(s)
        t = t.clamp(1e-5, 1.0 - 1e-5)
        t_idx = (t * self.diffusion.T).long().clamp(0, self.diffusion.T - 1)
        return t_idx

    def training_step(self, batch, batch_idx):
        # Unpack batch
        if self.sparse:
            _, graph_data, point_indicator, edge_indicator, _ = batch
            points = graph_data.x
            edge_index = graph_data.edge_index
            route_edge_flags = graph_data.edge_attr
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
        else:
            _, points, adj_matrix, _ = batch
            batch_size = points.shape[0]

        device = points.device

        # Time sampling (logit-normal)
        t_idx = self.sample_time_logit_normal(batch_size, device)
        t_idx_np = t_idx.cpu().numpy()

        # Noise injection (diffusion API expects one-hot)
        adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
        if self.sparse:
            adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)
        xt = self.diffusion.sample(adj_matrix_onehot, t_idx_np)

        xt_input = xt * 2 - 1
        xt_input = xt_input * (1.0 + 0.05 * torch.rand_like(xt_input.float()))
        x_t_adj_hint = xt.float()

        # Model forward (predict adjacency logits)
        if self.sparse:
            raise NotImplementedError("Sparse training for RAGenCO Transformer not implemented yet.")
        pred_logits = self.model(points, t_idx.float(), x_t_adj_hint)

        # BCE loss on adjacency logits
        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(pred_logits, adj_matrix.float())

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx, split="test"):
        real_batch_idx, points, gt_adj, gt_tour = batch
        device = points.device
        B, N, _ = points.shape

        # Disable self-loops (diagonal) before Sinkhorn via masking.
        no_self_mask = ~torch.eye(N, device=device, dtype=torch.bool)

        S = int(self.fast_num_candidates)
        sigma = float(self.fast_sigma)
        t_anchor = float(self.fast_t_anchor)
        eps = 1e-6

        amp_dtype = torch.bfloat16 if getattr(self.args, "bf16", False) else (torch.float16 if getattr(self.args, "fp16", False) else torch.float32)

        # (Optional) reuse or build an InferenceSchedule
        schedule = None
        inf_sched_name = getattr(self.args, "inference_schedule", None)
        inf_T = int(getattr(self.args, "inference_diffusion_steps", 1))
        if inf_sched_name is not None and inf_T > 1:
            if (self._cached_schedule is None) or (getattr(self._cached_schedule, "inference_schedule", None) != inf_sched_name) or (getattr(self._cached_schedule, "inference_T", None) != inf_T):
                schedule = InferenceSchedule(
                    inference_schedule=inf_sched_name,
                    T=self.diffusion.T,
                    inference_T=inf_T,
                )
                self._cached_schedule = schedule
            else:
                schedule = self._cached_schedule

        # 1) obtain an anchor via single-step or multi-step denoising
        neutral = torch.full((B, N, N), 0.5, device=device)
        if schedule is None:
            t_tensor = torch.full((B,), t_anchor, device=device)
            if self._compiled_infer_fn is not None:
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype):
                    x0_prob_rep, anchor = self._compiled_infer_fn(points, neutral, t_tensor, no_self_mask)
            else:
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype):
                    x0_prob_rep, anchor = self._infer_anchor_and_candidates_eager(points, neutral, t_tensor, no_self_mask)
        else:
            current_adj = neutral
            for i in range(schedule.inference_T):
                t1, t2 = schedule(i)
                t_float = float(t1)
                t_tensor = torch.full((B,), t_float, device=device)
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype):
                    logits = self.model(points, t_tensor, x_t_adj=current_adj)
                    prob = torch.sigmoid(logits).detach()

                if self.use_sinkhorn:
                    prob = sinkhorn_normalize(prob, iters=max(1, int(self.sinkhorn_iters)), tau=self.sinkhorn_tau, mask=no_self_mask)

                current_adj = prob

            anchor = current_adj

            # candidates for the multi-step case (keep eager; schedule(i) is Python-side)
            anchor_rep = anchor.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, N, N)
            logit_anchor = torch.logit(anchor_rep.clamp(eps, 1 - eps))

            if self.use_gumbel:
                U = torch.rand_like(logit_anchor, device=device)
                gumbel = -torch.log(-torch.log(U + eps) + eps)
                noisy_logit = logit_anchor + gumbel * float(self.fast_gumbel_scale)
            else:
                noisy_logit = logit_anchor + torch.randn_like(logit_anchor, device=device) * sigma

            noisy_heat = torch.sigmoid(noisy_logit)
            if self.use_sinkhorn:
                noisy_heat = sinkhorn_normalize(noisy_heat, iters=max(1, int(self.sinkhorn_iters)), tau=self.sinkhorn_tau, mask=no_self_mask)

            points_rep = points.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, N, 2)
            t_rep = torch.full((B * S,), t_anchor, device=device)
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype):
                pred_logits_rep = self.model(points_rep, t_rep, x_t_adj=noisy_heat)
                x0_prob_rep = torch.sigmoid(pred_logits_rep)
            if self.use_sinkhorn:
                x0_prob_rep = sinkhorn_normalize(x0_prob_rep, iters=max(1, int(self.sinkhorn_iters)), tau=self.sinkhorn_tau, mask=no_self_mask)

        # 4) decode candidates, optionally refine with 2-opt, evaluate and pick best
        adj_np = x0_prob_rep.view(B, S, N, N).cpu().float().numpy()
        np_points_batch = points.detach().cpu().numpy()

        total_gaps = []
        total_gt_costs = []
        total_solved_costs = []

        for i in range(B):
            np_points = np_points_batch[i]

            if not self.use_hungarian:
                new_tours, _ = merge_tours(adj_np[i], np_points, None, sparse_graph=False, parallel_sampling=S)
            else:
                prob_mats = adj_np[i]  # [S, N, N]
                dmat = euclidean_distance_matrix(np_points)
                new_tours = []
                for s_idx in range(S):
                    prob_mat = prob_mats[s_idx]
                    p = decode_permutation_from_heat(prob_mat, forbid_self=True)
                    tour = permutation_to_tour_from_mapping(p, dists=dmat)
                    tour = np.asarray(tour, dtype=np.int64)
                    # close the tour for cost/two-opt compatibility
                    tour = np.concatenate([tour, tour[:1]], axis=0)
                    new_tours.append(tour)

            tours_arr = np.asarray(new_tours, dtype=np.int64)
            costs = tour_length_np(np_points, tours_arr)
            sort_idx = np.argsort(costs)
            M = min(self.final_top_m, S)
            top_idx = sort_idx[:M]
            top_tours = [new_tours[j] for j in top_idx]

            if self.use_final_two_opt and (self.args.two_opt_iterations > 0) and (M > 0):
                refined, _ = batched_two_opt_torch(
                    np_points.astype("float32"),
                    np.asarray(top_tours, dtype=np.int64),
                    max_iterations=int(self.args.two_opt_iterations),
                    device=device
                )
                refined_costs = tour_length_np(np_points, np.asarray(refined, dtype=np.int64))
                for ii, idx in enumerate(top_idx):
                    costs[idx] = refined_costs[ii]
                    new_tours[idx] = refined[ii]

            best_idx = int(np.argmin(costs))
            best_tour = new_tours[best_idx]
            best_cost = float(costs[best_idx])

            gt_tour_i = gt_tour[i].detach().cpu().numpy().astype(np.int64)
            gt_cost = float(tour_length_np(np_points, gt_tour_i[None, :])[0])
            gap = (best_cost - gt_cost) / gt_cost * 100

            total_gaps.append(gap)
            total_gt_costs.append(gt_cost)
            total_solved_costs.append(best_cost)

        # aggregate metrics
        mean_gap = float(np.mean(total_gaps))
        mean_gt_cost = float(np.mean(total_gt_costs))
        mean_solved_cost = float(np.mean(total_solved_costs))

        metrics = {
            f"{split}/gap": mean_gap,
            f"{split}/gt_cost": mean_gt_cost,
            f"{split}/solved_cost": mean_solved_cost,
        }
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        return metrics
