import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from diffusion.pl_meta_model import COMetaModel
from diffusion.utils.diffusion_schedulers import InferenceSchedule
from diffusion.utils.atsp_utils import (
    atsp_2opt_local_search,
    batched_atsp_two_opt_torch,
    atsp_greedy_merge_decode,
    decode_permutation_from_heat,
    permutation_to_tour_from_mapping,
)
from diffusion.models.ragenco_transformer import get_ragenco_model
from diffusion.co_datasets.atsp_graph_dataset import ATSPGraphDataset


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
    logp = logp - logp.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        logp = logp.masked_fill(~mask, -1e9)

    p = torch.exp(logp).clamp(min=eps)

    if mask is not None:
        p = p.masked_fill(~mask, 0.0)

    for _ in range(iters):
        p = p / (p.sum(dim=-1, keepdim=True) + eps)  # row norm
        p = p / (p.sum(dim=-2, keepdim=True) + eps)  # col norm
        if mask is not None:
            p = p.masked_fill(~mask, 0.0)
    return p

class RAGenCOATSPModel(COMetaModel):
    """RAGenCO ATSP model with Sinkhorn-aligned control states.

    Inputs are directed cost matrices. Inference projects matrix states toward
    the cycle-cover relaxation, branches around the aligned anchor, and decodes
    with assignment-based repair.
    """
    def __init__(self, param_args=None, load_dataset=True):
        super().__init__(param_args=param_args, node_feature_only=False)

        if load_dataset:
            if self.args.training_split:
                self.train_dataset = ATSPGraphDataset(
                    data_file=os.path.join(self.args.training_split),
                    graph_type="directed",
                )
            if self.args.test_split:
                self.test_dataset = ATSPGraphDataset(
                    data_file=os.path.join(self.args.test_split),
                    graph_type="directed",
                )
            if self.args.validation_split:
                self.validation_dataset = ATSPGraphDataset(
                    data_file=os.path.join(self.args.validation_split),
                    graph_type="directed",
                )

        if load_dataset and getattr(self, "train_dataset", None) is not None and len(self.train_dataset) > 0:
            _, cost_mat, _, _ = self.train_dataset[0]
            self.N = cost_mat.shape[0]

        self.model = get_ragenco_model(self.args, input_dim=self.N, edge_asymmetric=True)

        # -------- torch.compile controls (mirrors TSP fast path) --------
        self.use_torch_compile = getattr(self.args, "use_torch_compile", False)
        self.use_compile_warmup = getattr(self.args, "use_compile_warmup", False)
        self.compile_warmup_nodes = getattr(self.args, "compile_warmup_nodes", getattr(self, "N", 50))
        # ATSP backbone uses input_dim=self.N, so warmup must match self.N (if known)
        if hasattr(self, "N"):
            self.compile_warmup_nodes = int(self.N)
        self._compiled = False
        self._compile_warmed = False
        self._compiled_infer_fn = None

        # sampling params
        self.mu = -0.8
        self.sigma = 0.8

        # sinkhorn & sampling options
        self.use_sinkhorn = getattr(self.args, "use_sinkhorn", True)
        self.sinkhorn_iters = getattr(self.args, "sinkhorn_iters", 5)
        self.sinkhorn_tau = getattr(self.args, "sinkhorn_tau", 0.15)

        self.fast_num_candidates = getattr(self.args, "fast_num_candidates", 3)
        self.fast_sigma = getattr(self.args, "fast_sigma", 1.8)
        self.fast_t_anchor = float(torch.sigmoid(torch.tensor(self.mu)))
        self.final_top_m = getattr(self.args, "final_top_m", 3)
        self.use_final_two_opt = getattr(self.args, "use_final_two_opt", True)

        self.use_gumbel = getattr(self.args, "use_gumbel", False)
        self.fast_gumbel_scale = getattr(self.args, "fast_gumbel_scale", self.fast_sigma)

        self.use_hungarian = getattr(self.args, "use_hungarian", False)

        self._cached_schedule = None

    # -------------------------
    # torch.compile helpers
    # -------------------------
    def _infer_anchor_and_candidates_eager(self, model_points, neutral, t_tensor, no_self_mask):
        """Eager version of the compiled inference path (single-step anchor)."""
        B, N, _ = model_points.shape
        S = int(self.fast_num_candidates)
        eps = 1e-6

        # 1) anchor (NOTE: keep parity with original ATSP code: do NOT Sinkhorn anchor here)
        init_logits = self.model(model_points, t_tensor, x_t_adj=neutral)
        anchor = torch.sigmoid(init_logits)
        anchor = anchor.masked_fill(~no_self_mask, 0.0)

        # 2) noisy candidates in logit space
        anchor_rep = anchor.unsqueeze(1).expand(B, S, N, N).reshape(B * S, N, N)
        logit_anchor = torch.logit(anchor_rep.clamp(eps, 1.0 - eps))

        if self.use_gumbel:
            U = torch.rand_like(logit_anchor)
            g = -torch.log(-torch.log(U + eps) + eps)
            noisy_logit = logit_anchor + g * float(self.fast_gumbel_scale)
        else:
            noisy_logit = logit_anchor + torch.randn_like(logit_anchor) * float(self.fast_sigma)

        noisy_heat = torch.sigmoid(noisy_logit).masked_fill(~no_self_mask, 0.0)
        if self.use_sinkhorn:
            noisy_heat = sinkhorn_normalize(
                noisy_heat,
                iters=max(1, int(self.sinkhorn_iters)),
                tau=float(self.sinkhorn_tau),
                mask=no_self_mask,
            )

        # 3) one-step denoise
        points_rep = model_points.unsqueeze(1).expand(B, S, N, N).reshape(B * S, N, N)
        t_rep = t_tensor.repeat_interleave(S)
        pred_logits_rep = self.model(points_rep, t_rep, x_t_adj=noisy_heat)
        x0_prob_rep = torch.sigmoid(pred_logits_rep).masked_fill(~no_self_mask, 0.0)

        if self.use_sinkhorn:
            x0_prob_rep = sinkhorn_normalize(
                x0_prob_rep,
                iters=max(1, int(self.sinkhorn_iters)),
                tau=float(self.sinkhorn_tau),
                mask=no_self_mask,
            )

        candidate_mix_lambda = float(getattr(self.args, "candidate_mix_lambda", 0.05))
        if candidate_mix_lambda > 0.0:
            x0_prob_rep = (1.0 - candidate_mix_lambda) * x0_prob_rep + candidate_mix_lambda * noisy_heat
            x0_prob_rep = x0_prob_rep.masked_fill(~no_self_mask, 0.0)

        return x0_prob_rep, anchor

    def _maybe_compile_model(self):
        """Compile a single inference function (anchor + candidates + denoise + postproc).

        We compile a *function* (not the module) to reduce graph breaks and keep
        one larger graph, mirroring the TSP implementation.
        """
        if (not self.use_torch_compile) or self._compiled:
            return
        if not hasattr(torch, "compile"):
            return

        S = int(self.fast_num_candidates)
        use_gumbel = bool(self.use_gumbel)
        use_sinkhorn = bool(self.use_sinkhorn)
        sinkhorn_iters = max(1, int(self.sinkhorn_iters))
        sinkhorn_tau = float(self.sinkhorn_tau)
        sigma = float(self.fast_sigma)
        gumbel_scale = float(self.fast_gumbel_scale)
        candidate_mix_lambda = float(getattr(self.args, "candidate_mix_lambda", 0.05))
        eps = 1e-6

        model = self.model

        def infer_anchor_and_candidates(model_points, neutral, t_tensor, no_self_mask):
            B, N, _ = model_points.shape

            # 1) anchor (keep parity: no Sinkhorn here)
            init_logits = model(model_points, t_tensor, x_t_adj=neutral)
            anchor = torch.sigmoid(init_logits)
            anchor = anchor.masked_fill(~no_self_mask, 0.0)

            # 2) noisy candidates
            anchor_rep = anchor.unsqueeze(1).expand(B, S, N, N).reshape(B * S, N, N)
            logit_anchor = torch.logit(anchor_rep.clamp(eps, 1.0 - eps))

            if use_gumbel:
                U = torch.rand_like(logit_anchor)
                g = -torch.log(-torch.log(U + eps) + eps)
                noisy_logit = logit_anchor + g * gumbel_scale
            else:
                noisy_logit = logit_anchor + torch.randn_like(logit_anchor) * sigma

            noisy_heat = torch.sigmoid(noisy_logit)
            noisy_heat = noisy_heat.masked_fill(~no_self_mask, 0.0)

            if use_sinkhorn:
                noisy_heat = sinkhorn_normalize(noisy_heat, iters=sinkhorn_iters, tau=sinkhorn_tau, mask=no_self_mask)

            # 3) denoise
            points_rep = model_points.unsqueeze(1).expand(B, S, N, N).reshape(B * S, N, N)
            t_rep = t_tensor.repeat_interleave(S)
            pred_logits_rep = model(points_rep, t_rep, x_t_adj=noisy_heat)
            x0_prob_rep = torch.sigmoid(pred_logits_rep)
            x0_prob_rep = x0_prob_rep.masked_fill(~no_self_mask, 0.0)

            if use_sinkhorn:
                x0_prob_rep = sinkhorn_normalize(x0_prob_rep, iters=sinkhorn_iters, tau=sinkhorn_tau, mask=no_self_mask)

            if candidate_mix_lambda > 0.0:
                x0_prob_rep = (1.0 - candidate_mix_lambda) * x0_prob_rep + candidate_mix_lambda * noisy_heat
                x0_prob_rep = x0_prob_rep.masked_fill(~no_self_mask, 0.0)

            return x0_prob_rep, anchor

        try:
            self._compiled_infer_fn = torch.compile(infer_anchor_and_candidates, mode="max-autotune")
            self._compiled = True
            print(f"[info] ATSP torch.compile enabled for inference function (mode=max-autotune).")
        except Exception as e:
            # Safe fallback: keep eager
            self._compiled_infer_fn = None
            self._compiled = False
            print(f"[warn] ATSP torch.compile failed, falling back to eager. Error: {e}")

    def _maybe_warmup_compile(self):
        if (not self._compiled) or (not self.use_compile_warmup) or self._compile_warmed:
            return
        if self._compiled_infer_fn is None:
            return

        # Warmup shapes (avoid recompiles on first real batch)
        N = int(self.compile_warmup_nodes)
        B = 1
        device = self.device

        dummy_cost = torch.zeros(B, N, N, device=device, dtype=torch.float32)
        neutral = torch.full((B, N, N), 0.5, device=device, dtype=torch.float32)
        no_self_mask = ~torch.eye(N, device=device, dtype=torch.bool)

        amp_dtype = torch.bfloat16 if bool(getattr(self.args, "bf16", False)) else (
            torch.float16 if bool(getattr(self.args, "fp16", False)) else torch.float32
        )
        t_anchor = float(self.fast_t_anchor)
        t_tensor = torch.full((B,), t_anchor, device=device, dtype=torch.float32)

        with torch.no_grad(), torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), dtype=amp_dtype):
            _ = self._compiled_infer_fn(dummy_cost, neutral, t_tensor, no_self_mask)

        self._compile_warmed = True
        print(f"[info] ATSP compile warmup done with N={N}.")

    def on_fit_start(self):
        self._maybe_compile_model()
        self._maybe_warmup_compile()

    def on_validation_start(self):
        self._maybe_compile_model()
        self._maybe_warmup_compile()

    def on_test_start(self):
        self._maybe_compile_model()
        self._maybe_warmup_compile()

    # -------------------------
    # Data loader helpers
    # -------------------------
    def _collate(self, batch):
        idx = torch.stack([b[0].squeeze(0) if b[0].ndim == 1 else b[0].squeeze() for b in batch], dim=0)
        cost = torch.stack([b[1] for b in batch], dim=0)
        adj = torch.stack([b[2] for b in batch], dim=0)
        tour = torch.stack([b[3] for b in batch], dim=0)
        return idx, cost, adj, tour

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        val_count = min(getattr(self.args, "validation_examples", 1), len(self.validation_dataset))
        val_indices = list(range(val_count))
        val_subset = torch.utils.data.Subset(self.validation_dataset, val_indices)
        return DataLoader(
            val_subset,
            batch_size=getattr(self.args, "val_batch_size", 1),
            shuffle=False,
            num_workers=max(0, getattr(self.args, "num_workers", 0)),
            collate_fn=self._collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=getattr(self.args, "test_batch_size", 1),
            shuffle=False,
            num_workers=max(0, getattr(self.args, "num_workers", 0)),
            collate_fn=self._collate,
        )

    # -------------------------
    # Forward / helpers
    # -------------------------
    def forward(self, x, t, x_t_adj=None):
        return self.model(x, t, x_t_adj)

    def sample_time_logit_normal(self, num_samples, device):
        s = torch.randn(num_samples, device=device) * self.sigma + self.mu
        t = torch.sigmoid(s)
        t = t.clamp(1e-5, 1.0 - 1e-5)
        t_idx = (t * self.diffusion.T).long().clamp(0, self.diffusion.T - 1)
        return t_idx

    def training_step(self, batch, batch_idx):
        _, cost_matrix, adj_matrix, _ = batch
        B = cost_matrix.shape[0]
        device = cost_matrix.device

        t_idx = self.sample_time_logit_normal(B, device)
        t_idx_np = t_idx.cpu().numpy()

        adj_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
        xt = self.diffusion.sample(adj_onehot, t_idx_np)
        x_t_adj_hint = xt.float()

        denom = cost_matrix.view(B, -1).abs().max(dim=1)[0].view(B, 1, 1).clamp(min=1e-6)
        x_input = cost_matrix / denom

        pred_logits = self.model(x_input, t_idx.float(), x_t_adj_hint)

        total = float(pred_logits.numel())
        pos = float(adj_matrix.sum().item())
        if pos <= 0.0:
            pos_weight = torch.tensor(1.0, device=device)
        else:
            neg = total - pos
            pos_weight = torch.tensor(max(1.0, neg / pos), device=device)
        loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_func(pred_logits, adj_matrix.float())

        with torch.no_grad():
            probs = torch.sigmoid(pred_logits)
            pred_pos = (probs > 0.5).float()
            pos_gt = adj_matrix.float()
            if pos > 0:
                recall_pos = ((pred_pos * pos_gt).sum() / (pos_gt.sum())).clamp(0.0, 1.0)
            else:
                recall_pos = torch.tensor(0.0, device=device)
            pos_rate = probs.mean()
            self.log("train/pos_rate", pos_rate, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/pos_recall", recall_pos, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/pos_weight", pos_weight.item(), on_step=False, on_epoch=True, sync_dist=False)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx, split="test"):
        real_batch_idx, points_or_cost, gt_adj, gt_tour = batch
        device = points_or_cost.device
        B = points_or_cost.shape[0]
        N = points_or_cost.shape[1]

        # Disallow self-loops (diagonal) throughout decoding.
        no_self_mask = ~torch.eye(N, device=device, dtype=torch.bool)

        S = int(self.fast_num_candidates)
        t_anchor = float(self.fast_t_anchor)
        eps = 1e-6

        amp_dtype = torch.bfloat16 if bool(getattr(self.args, "bf16", False)) else (
            torch.float16 if bool(getattr(self.args, "fp16", False)) else torch.float32
        )

        schedule = None
        inf_sched_name = getattr(self.args, "inference_schedule", None)
        inf_T = int(getattr(self.args, "inference_diffusion_steps", 1))
        if inf_sched_name is not None and inf_T > 1:
            if (
                (self._cached_schedule is None)
                or (getattr(self._cached_schedule, "inference_schedule", None) != inf_sched_name)
                or (getattr(self._cached_schedule, "inference_T", None) != inf_T)
            ):
                schedule = InferenceSchedule(
                    inference_schedule=inf_sched_name,
                    T=self.diffusion.T,
                    inference_T=inf_T,
                )
                self._cached_schedule = schedule
            else:
                schedule = self._cached_schedule

        denom = points_or_cost.view(B, -1).abs().max(dim=1)[0].view(B, 1, 1).clamp(min=1e-6)
        model_points = points_or_cost / denom  # [B, N, N]

        neutral = torch.full((B, N, N), 0.5, device=device, dtype=torch.float32)
        if schedule is None:
            t_tensor = torch.full((B,), t_anchor, device=device, dtype=torch.float32)
            with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), dtype=amp_dtype):
                if self._compiled_infer_fn is not None:
                    x0_prob_rep, anchor = self._compiled_infer_fn(model_points, neutral, t_tensor, no_self_mask)
                else:
                    x0_prob_rep, anchor = self._infer_anchor_and_candidates_eager(model_points, neutral, t_tensor, no_self_mask)
        else:
            # Multi-step schedule is kept eager (schedule() is Python-side)
            current_adj = neutral
            for i in range(schedule.inference_T):
                t1, t2 = schedule(i)
                t_tensor = torch.full((B,), float(t1), device=device, dtype=torch.float32)
                with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), dtype=amp_dtype):
                    logits = self.model(model_points, t_tensor, x_t_adj=current_adj)
                    prob = torch.sigmoid(logits)

                prob = prob.masked_fill(~no_self_mask, 0.0)
                if self.use_sinkhorn:
                    prob = sinkhorn_normalize(
                        prob,
                        iters=max(1, int(self.sinkhorn_iters)),
                        tau=float(self.sinkhorn_tau),
                        mask=no_self_mask,
                    )
                current_adj = prob

            anchor = current_adj
            anchor = anchor.masked_fill(~no_self_mask, 0.0)

            # candidates for multi-step case (eager)
            anchor_rep = anchor.unsqueeze(1).expand(B, S, N, N).reshape(B * S, N, N)
            logit_anchor = torch.logit(anchor_rep.clamp(eps, 1.0 - eps))

            if self.use_gumbel:
                U = torch.rand_like(logit_anchor, device=device)
                gumbel = -torch.log(-torch.log(U + eps) + eps)
                noisy_logit = logit_anchor + gumbel * float(self.fast_gumbel_scale)
            else:
                noisy_logit = logit_anchor + torch.randn_like(logit_anchor, device=device) * float(self.fast_sigma)

            noisy_heat = torch.sigmoid(noisy_logit).masked_fill(~no_self_mask, 0.0)
            if self.use_sinkhorn:
                noisy_heat = sinkhorn_normalize(
                    noisy_heat,
                    iters=max(1, int(self.sinkhorn_iters)),
                    tau=float(self.sinkhorn_tau),
                    mask=no_self_mask,
                )

            points_rep = model_points.unsqueeze(1).expand(B, S, N, N).reshape(B * S, N, N)
            t_rep = torch.full((B * S,), t_anchor, device=device, dtype=torch.float32)
            with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), dtype=amp_dtype):
                pred_logits_rep = self.model(points_rep, t_rep, x_t_adj=noisy_heat)
                x0_prob_rep = torch.sigmoid(pred_logits_rep)
            x0_prob_rep = x0_prob_rep.masked_fill(~no_self_mask, 0.0)

            if self.use_sinkhorn:
                x0_prob_rep = sinkhorn_normalize(
                    x0_prob_rep,
                    iters=max(1, int(self.sinkhorn_iters)),
                    tau=float(self.sinkhorn_tau),
                    mask=no_self_mask,
                )

            candidate_mix_lambda = float(getattr(self.args, "candidate_mix_lambda", 0.05))
            if candidate_mix_lambda > 0.0:
                x0_prob_rep = (1.0 - candidate_mix_lambda) * x0_prob_rep + candidate_mix_lambda * noisy_heat
                x0_prob_rep = x0_prob_rep.masked_fill(~no_self_mask, 0.0)

        # Decode on CPU (numpy)
        heat_np = x0_prob_rep.view(B, S, N, N).cpu().float().numpy()

        total_gaps = []
        total_gt_costs = []
        total_solved_costs = []

        two_opt_iters = int(getattr(self.args, "two_opt_iterations", 0))

        # Cache cost matrices on CPU for fast evaluation/decoding (numpy)
        cost_matrices_np = points_or_cost.detach().cpu().float().numpy()
        greedy_cost_alpha = float(getattr(self.args, "greedy_cost_alpha", 1.0))

        # Greedy decoding diversification controls (default OFF)
        greedy_score_noise_scale = float(getattr(self.args, "greedy_score_noise_scale", 0.0))
        greedy_score_noise_type = getattr(self.args, "greedy_score_noise_type", "gaussian")
        greedy_deterministic = bool(getattr(self.args, "greedy_deterministic", greedy_score_noise_scale <= 0.0))

        # Hungarian diversification controls (active only when use_hungarian=True)
        hungarian_tau = float(getattr(self.args, "hungarian_tau", 0.75))
        hungarian_noise_scale = float(getattr(self.args, "hungarian_noise_scale", 0.015))
        hungarian_noise_type = getattr(self.args, "hungarian_noise_type", "gumbel")
        hungarian_deterministic = bool(getattr(self.args, "hungarian_deterministic", False))
        if hungarian_deterministic:
            hungarian_noise_scale = 0.0

        tours_all = np.zeros((B, S, N + 1), dtype=np.int64)
        unique_counts = []
        cand_l1_list = []
        cand_std_list = []
        base_seed = int(torch.initial_seed() % (2**32))

        # 4) decode all candidates on CPU (numpy)
        for bi in range(B):
            mats = heat_np[bi]              # [S, N, N]
            cost_mat = cost_matrices_np[bi] # [N, N]

            # Candidate heat diversity diagnostics (before decoding)
            if S > 1:
                cand_mean = mats.mean(axis=0, keepdims=True)
                cand_l1_list.append(float(np.mean(np.abs(mats - cand_mean))))
                cand_std_list.append(float(np.mean(np.std(mats, axis=0))))
            else:
                cand_l1_list.append(0.0)
                cand_std_list.append(0.0)

            if not self.use_hungarian:
                tours = atsp_greedy_merge_decode(
                    mats,
                    cost_mat,
                    greedy_cost_alpha=greedy_cost_alpha,
                    score_noise_scale=greedy_score_noise_scale,
                    score_noise_type=greedy_score_noise_type,
                    deterministic=greedy_deterministic,
                    eps=eps,
                    forbid_self=True,
                )  # [S, N+1]
            else:
                tours = np.zeros((S, N + 1), dtype=np.int64)
                for s_idx in range(S):
                    rng = np.random.default_rng(base_seed + bi * 1000003 + s_idx)
                    p = decode_permutation_from_heat(
                        mats[s_idx],
                        eps=1e-9,
                        forbid_self=True,
                        tau=hungarian_tau,
                        noise_scale=hungarian_noise_scale,
                        noise_type=hungarian_noise_type,
                        deterministic=hungarian_deterministic,
                        rng=rng,
                    )
                    tour = permutation_to_tour_from_mapping(p, dists=cost_mat)
                    tours[s_idx] = np.asarray(tour, dtype=np.int64)

            tours_all[bi] = tours
            unique_counts.append(len({tuple(t.tolist()) for t in tours}))

        self.log(f"{split}/unique_tours", float(np.mean(unique_counts)), on_epoch=True, sync_dist=True)
        self.log(f"{split}/cand_heat_l1", float(np.mean(cand_l1_list)), on_epoch=True, sync_dist=True)
        self.log(f"{split}/cand_heat_std", float(np.mean(cand_std_list)), on_epoch=True, sync_dist=True)

        # 5) evaluate all candidates (vectorized)
        costs = np.zeros((B, S), dtype=np.float64)
        for bi in range(B):
            cost_mat = cost_matrices_np[bi]
            frm = tours_all[bi, :, :-1].astype(np.int64)
            to = tours_all[bi, :, 1:].astype(np.int64)
            costs[bi] = cost_mat[frm, to].sum(axis=1)

        # 6) optionally refine top-M per instance with ATSP 2-opt (batched), then select best
        M = min(int(self.final_top_m), S)
        if self.use_final_two_opt and (two_opt_iters > 0) and (M > 0):
            top_idx = np.argsort(costs, axis=1)[:, :M]  # [B, M]
            top_tours_np = np.concatenate([tours_all[bi, top_idx[bi]] for bi in range(B)], axis=0).astype(np.int64)

            top_dists_t = points_or_cost.detach().float().repeat_interleave(M, dim=0).to(device=device, dtype=torch.float32)
            top_tours_t = torch.from_numpy(top_tours_np).to(device=device, dtype=torch.long)

            top_refined_t, _ = batched_atsp_two_opt_torch(top_dists_t, top_tours_t, max_iterations=two_opt_iters)
            top_refined_np = top_refined_t.detach().cpu().numpy()

            k = 0
            for bi in range(B):
                cost_mat = cost_matrices_np[bi]
                for mi in range(M):
                    s_idx = int(top_idx[bi, mi])
                    tours_all[bi, s_idx] = top_refined_np[k]
                    frm = top_refined_np[k, :-1].astype(np.int64)
                    to = top_refined_np[k, 1:].astype(np.int64)
                    costs[bi, s_idx] = cost_mat[frm, to].sum()
                    k += 1

        # 7) select best tour per instance and compute gaps
        for bi in range(B):
            best_idx = int(np.argmin(costs[bi]))
            best_cost = float(costs[bi, best_idx])

            gt_tour_i = gt_tour[bi].detach().cpu().numpy()
            cost_mat = cost_matrices_np[bi]
            gt_cost = float(cost_mat[gt_tour_i[:-1].astype(np.int64), gt_tour_i[1:].astype(np.int64)].sum())
            gap = (best_cost - gt_cost) / gt_cost * 100.0

            total_gaps.append(gap)
            total_gt_costs.append(gt_cost)
            total_solved_costs.append(best_cost)

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

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split="val")
