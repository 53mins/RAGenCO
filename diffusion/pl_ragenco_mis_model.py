import os
import math
from contextlib import nullcontext
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusion.co_datasets.mis_dataset import MISDataset

from diffusion.pl_meta_model import COMetaModel
from diffusion.utils.diffusion_schedulers import InferenceSchedule
from diffusion.models.mis_node_transformer import get_mis_node_transformer
from diffusion.utils.mis_utils import (
    greedy_mis_from_scores,
    sample_candidates_from_anchor,
    local_improvement_swap_12_batched,
    mis_entropic_projection_from_logits,
    mis_projection_residual,
)


class RAGenCOMISModel(COMetaModel):
    """
    RAGenCO MIS model with decode-aware Bernoulli projection.

    The projected independent-set control state suppresses local conflicts
    during refinement, while terminal greedy decoding uses raw logits so the
    rank signal is preserved. Node features include:
      - degree (normalized)
      - average neighbor degree (normalized)
      - 2‑hop neighbour counts (A^2 row sum, normalized)
      - PageRank (power iteration, few iters)
      - local clustering coefficient (approx)
      - sincos index positional encoding

    Notes:
      - This class avoids online eigendecomposition; features are computed efficiently
        (matvecs, small iterates). For very large graphs consider offline precompute or
        additional approximations.
      - Assumes DataLoader provides batches as (idx_tensor, graph_data_list, point_indicator_tensor)
    """

    def __init__(self, param_args=None, load_dataset=True):
        super().__init__(param_args=param_args, node_feature_only=False)

        if load_dataset:
            if self.args.training_split:
                self.train_dataset = MISDataset(
                    data_file=os.path.join(self.args.training_split),
                    data_label_dir=getattr(self.args, "training_split_label_dir", None),
                )
            if self.args.test_split:
                self.test_dataset = MISDataset(
                    data_file=os.path.join(self.args.test_split),
                    data_label_dir=getattr(self.args, "test_split_label_dir", None),
                )
            if self.args.validation_split:
                self.validation_dataset = MISDataset(
                    data_file=os.path.join(self.args.validation_split),
                    data_label_dir=getattr(self.args, "validation_split_label_dir", None),
                )

        # structural PE params (lightweight)
        # self.sincos_dim = getattr(self.args, "mis_sincos_dim", 32)  # sincos PE dimension
        self.use_degree = True
        self.use_pagerank = True
        self.pagerank_iters = int(getattr(self.args, "mis_pagerank_iters", 10))

        # feature composition:
        # degree(1) + avg_neighbor_deg(1) + two_hop(1) + pagerank(1) + clustering(1) + sincos_dim
        self.feature_dim = 1 + 1 + 1 + 1 + 1

        # instantiate RAGenCO transformer for node outputs with input_dim = feature_dim
        self.model = get_mis_node_transformer(self.args, input_dim=self.feature_dim + 1)

        # compile controls (mirrors TSP/CVRP branches)
        self.use_torch_compile = getattr(self.args, "use_torch_compile", False)
        self.use_compile_warmup = getattr(self.args, "use_compile_warmup", False)
        self.compile_warmup_nodes = getattr(self.args, "compile_warmup_nodes", 50)
        self._compiled = False
        self._compile_warmed = False
        self._compiled_infer_fn = None


        # sampling params (logit-normal anchor)
        self.mu = -0.8
        self.sigma = 0.8

        # parallel candidate sampling (same config keys as TSP/ATSP branches)
        self.fast_num_candidates = getattr(self.args, "fast_num_candidates", 3)
        self.fast_sigma = getattr(self.args, "fast_sigma", 1.8)
        self.fast_gumbel_scale = getattr(self.args, "fast_gumbel_scale", 1.8)
        self.use_gumbel = getattr(self.args, "use_gumbel", True)
        self.node_decode_source = str(getattr(self.args, "node_decode_source", "raw_logits"))
        if self.node_decode_source not in {"raw_logits", "projected_state"}:
            raise ValueError(f"Unsupported node_decode_source: {self.node_decode_source}")

        # MIS-specific in-loop entropic projection onto the local stable-set relaxation.
        self.mis_use_entropic_projection = bool(getattr(self.args, "mis_use_entropic_projection", True))
        self.mis_proj_tau = float(getattr(self.args, "mis_proj_tau", 1.0))
        self.mis_proj_steps = int(getattr(self.args, "mis_proj_steps", 8))
        self.mis_proj_step_size = float(getattr(self.args, "mis_proj_step_size", 0.5))
        self.mis_proj_tol = float(getattr(self.args, "mis_proj_tol", 1e-4))

        # caching / schedule
        self._cached_schedule = None

    def _select_node_decoder_scores(
        self,
        state_probs: torch.Tensor,
        decode_scores: torch.Tensor,
    ) -> torch.Tensor:
        if self.node_decode_source == "projected_state":
            return state_probs
        return decode_scores

    def sample_time_logit_normal(self, num_samples, device):
        s = torch.randn(num_samples, device=device) * self.sigma + self.mu
        t = torch.sigmoid(s)
        t = t.clamp(1e-5, 1.0 - 1e-5)
        t_idx = (t * self.diffusion.T).long().clamp(0, self.diffusion.T - 1)
        return t_idx

    def _autocast_context(self):
        if torch.cuda.is_available():
            if bool(getattr(self.args, "bf16", False)):
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if bool(getattr(self.args, "fp16", False)):
                return torch.autocast(device_type="cuda", dtype=torch.float16)
            return nullcontext()

        if bool(getattr(self.args, "bf16", False)):
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)

        return nullcontext()

    def _resolve_timestep(self, t_value: float) -> float:
        """Resolve user/schedule timestep into the same scale as training.

        Training conditions the model on discrete indices in [0, T-1] (passed as float).
        For convenience, treat values <= 1.0 as a fraction of the diffusion horizon.
        """
        T = float(getattr(self.diffusion, "T", 1000))
        t = float(t_value)
        if t <= 1.0 + 1e-9:
            t = t * (T - 1.0)
        # clamp to valid range
        t = max(0.0, min(t, T - 1.0))
        return t

    def _apply_mis_projection_from_logits(
        self,
        logits: torch.Tensor,
        adj: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ):
        if not self.mis_use_entropic_projection:
            probs = torch.sigmoid(logits)
            if node_mask is not None:
                probs = probs * node_mask.to(probs.dtype)
            violation = mis_projection_residual(probs, adj, node_mask=node_mask)
            return probs, violation

        return mis_entropic_projection_from_logits(
            logits,
            adj,
            node_mask=node_mask,
            tau=self.mis_proj_tau,
            num_iters=self.mis_proj_steps,
            step_size=self.mis_proj_step_size,
        )


    # -------------------------
    # torch.compile helpers (inference only)
    # -------------------------
    def _infer_anchor_candidates_refine_eager(
        self,
        node_features: torch.Tensor,
        neutral_node: torch.Tensor,
        t_tensor: torch.Tensor,
        adj_batch: torch.Tensor,
        mask: torch.Tensor,
        refine_steps: int,
    ):
        """Eager fast path: anchor -> candidates -> optional refinement.

        Returns:
            state_probs: [B, S, N]
            decode_scores: [B, S, N]
            anchor: [B, N]
        """
        B, N, _ = node_features.shape
        S = max(1, int(self.fast_num_candidates))
        sigma = float(self.fast_sigma)
        gumbel_scale = float(self.fast_gumbel_scale)
        use_gumbel = bool(self.use_gumbel)
        eps = 1e-6

        # 1) anchor
        anchor_logits = self.model(
            node_features,
            t_tensor,
            x_t_node=neutral_node,
            adj_bias=adj_batch,
            node_mask=mask,
        )
        anchor, _ = self._apply_mis_projection_from_logits(anchor_logits, adj_batch, node_mask=mask)
        anchor = anchor.detach()  # [B, N]

        # 2) candidates sampled in logit space: [B*S, N]
        anchor_rep = anchor.unsqueeze(1).expand(B, S, N).reshape(B * S, N)
        logit_anchor = torch.logit(anchor_rep.clamp(eps, 1.0 - eps))
        adj_rep = adj_batch.unsqueeze(1).expand(B, S, N, N).reshape(B * S, N, N)
        mask_rep = mask.unsqueeze(1).expand(B, S, N).reshape(B * S, N)

        if use_gumbel:
            U = torch.rand_like(logit_anchor)
            g = -torch.log(-torch.log(U + eps) + eps)
            noisy_logit = logit_anchor + g * gumbel_scale
        else:
            noisy_logit = logit_anchor + torch.randn_like(logit_anchor) * sigma

        decode_scores = noisy_logit
        current, _ = self._apply_mis_projection_from_logits(noisy_logit, adj_rep, node_mask=mask_rep)
        current = current.detach()  # [B*S, N]

        # 3) optional refinement steps (linear anneal t_anchor -> 0)
        if refine_steps > 0:
            node_feat_rep = node_features.unsqueeze(1).expand(B, S, N, -1).reshape(B * S, N, -1)
            t_rep = t_tensor.repeat_interleave(S)

            if refine_steps == 1:
                ratios = (1.0,)
            else:
                ratios = tuple(1.0 - (k / float(refine_steps - 1)) for k in range(refine_steps))

            for r in ratios:
                t_step = t_rep * float(r)
                logits = self.model(
                    node_feat_rep,
                    t_step,
                    x_t_node=current,
                    adj_bias=adj_rep,
                    node_mask=mask_rep,
                )
                decode_scores = logits
                current, _ = self._apply_mis_projection_from_logits(logits, adj_rep, node_mask=mask_rep)
                current = current.detach()

        state_probs = current.view(B, S, N)
        decode_scores = decode_scores.view(B, S, N)
        return state_probs, decode_scores, anchor

    def _maybe_compile_model(self):
        """Compile a single larger inference function (anchor + candidates + refinement).

        We compile a *function* instead of compiling self.model in-place, mirroring TSP/CVRP.
        """
        if (not self.use_torch_compile) or self._compiled:
            return
        if not hasattr(torch, "compile"):
            return

        S = max(1, int(self.fast_num_candidates))
        use_gumbel = bool(self.use_gumbel)
        sigma = float(self.fast_sigma)
        gumbel_scale = float(self.fast_gumbel_scale)
        refine_steps = int(getattr(self.args, "mis_refine_steps", 1))
        use_projection = bool(self.mis_use_entropic_projection)
        proj_tau = float(self.mis_proj_tau)
        proj_steps = int(self.mis_proj_steps)
        proj_step_size = float(self.mis_proj_step_size)
        eps = 1e-6

        model = self.model

        def infer_anchor_candidates_refine(node_features, neutral_node, t_tensor, adj_batch, mask):
            B, N, _ = node_features.shape

            # 1) anchor
            anchor_logits = model(
                node_features,
                t_tensor,
                x_t_node=neutral_node,
                adj_bias=adj_batch,
                node_mask=mask,
            )
            if use_projection:
                anchor, _ = mis_entropic_projection_from_logits(
                    anchor_logits,
                    adj_batch,
                    node_mask=mask,
                    tau=proj_tau,
                    num_iters=proj_steps,
                    step_size=proj_step_size,
                )
            else:
                anchor = torch.sigmoid(anchor_logits)
                anchor = anchor * mask.to(anchor.dtype)

            # 2) candidates in logit space
            anchor_rep = anchor.unsqueeze(1).expand(B, S, N).reshape(B * S, N)
            logit_anchor = torch.logit(anchor_rep.clamp(eps, 1.0 - eps))
            adj_rep = adj_batch.unsqueeze(1).expand(B, S, N, N).reshape(B * S, N, N)
            mask_rep = mask.unsqueeze(1).expand(B, S, N).reshape(B * S, N)

            if use_gumbel:
                U = torch.rand_like(logit_anchor)
                g = -torch.log(-torch.log(U + eps) + eps)
                noisy_logit = logit_anchor + g * gumbel_scale
            else:
                noisy_logit = logit_anchor + torch.randn_like(logit_anchor) * sigma

            decode_scores = noisy_logit
            if use_projection:
                current, _ = mis_entropic_projection_from_logits(
                    noisy_logit,
                    adj_rep,
                    node_mask=mask_rep,
                    tau=proj_tau,
                    num_iters=proj_steps,
                    step_size=proj_step_size,
                )
            else:
                current = torch.sigmoid(noisy_logit)
                current = current * mask_rep.to(current.dtype)

            # 3) refinement (optional)
            if refine_steps > 0:
                node_feat_rep = node_features.unsqueeze(1).expand(B, S, N, -1).reshape(B * S, N, -1)
                t_rep = t_tensor.repeat_interleave(S)

                if refine_steps == 1:
                    ratios = (1.0,)
                else:
                    ratios = tuple(1.0 - (k / float(refine_steps - 1)) for k in range(refine_steps))

                for r in ratios:
                    t_step = t_rep * float(r)
                    logits = model(
                        node_feat_rep,
                        t_step,
                        x_t_node=current,
                        adj_bias=adj_rep,
                        node_mask=mask_rep,
                    )
                    decode_scores = logits
                    if use_projection:
                        current, _ = mis_entropic_projection_from_logits(
                            logits,
                            adj_rep,
                            node_mask=mask_rep,
                            tau=proj_tau,
                            num_iters=proj_steps,
                            step_size=proj_step_size,
                        )
                    else:
                        current = torch.sigmoid(logits)
                        current = current * mask_rep.to(current.dtype)

            state_probs = current.view(B, S, N)
            decode_scores = decode_scores.view(B, S, N)
            return state_probs, decode_scores, anchor

        self._compiled_infer_fn = torch.compile(infer_anchor_candidates_refine, mode="max-autotune")
        self._compiled = True
        print("[info] MIS torch.compile enabled for inference function (mode=max-autotune).")

    def _maybe_warmup_compile(self):
        if self._compiled and self.use_compile_warmup and (not self._compile_warmed):
            N = int(self.compile_warmup_nodes)
            B = 1
            device = torch.device(self.device if hasattr(self, "device") else ("cuda" if torch.cuda.is_available() else "cpu"))

            dummy_nodes = torch.zeros((B, N, self.feature_dim), device=device, dtype=torch.float32)
            dummy_adj = torch.zeros((B, N, N), device=device, dtype=torch.float32)
            dummy_mask = torch.ones((B, N), device=device, dtype=torch.bool)
            dummy_neutral = torch.full((B, N), 0.5, device=device, dtype=torch.float32)

            t_anchor_raw = float(getattr(self.args, "fast_t_anchor", 0.6))
            t_anchor = float(self._resolve_timestep(t_anchor_raw))
            t_tensor = torch.full((B,), t_anchor, device=device, dtype=torch.float32)

            if self._compiled_infer_fn is not None:
                with torch.no_grad(), self._autocast_context():
                    _ = self._compiled_infer_fn(dummy_nodes, dummy_neutral, t_tensor, dummy_adj, dummy_mask)

            self._compile_warmed = True
            print(f"[info] MIS compile warmup done with N={N}.")

    def on_fit_start(self):
        self._maybe_compile_model()
        self._maybe_warmup_compile()

    def on_validation_start(self):
        self._maybe_compile_model()
        self._maybe_warmup_compile()

    def on_test_start(self):
        self._maybe_compile_model()
        self._maybe_warmup_compile()


    # ----------------- low-cost structural feature builders -----------------

    def _extract_xt_nodeprob(self, xt, node_mask, device):
        """Normalize diffusion.sample output to node probabilities in [0,1] with shape [B,N].

        Supports several diffusion implementations:
          - categorical one-hot: [B, N, 2] or [B, 2, N]
          - Bernoulli / continuous: [B, N] or [B, N, 1]
          - latent continuous: [B, D, N] or [B, N, D]

        Padding nodes (node_mask==0) are set to 0.5 (neutral).
        """
        import numpy as _np
        if isinstance(xt, _np.ndarray):
            xt = torch.from_numpy(xt)
        if not torch.is_tensor(xt):
            raise TypeError(f"diffusion.sample returned unsupported type: {type(xt)}")

        xt = xt.to(device=device, dtype=torch.float32)
        node_mask_f = node_mask.to(device=device, dtype=torch.float32)
        B, N = node_mask_f.shape

        # derive x: [B,N]
        if xt.dim() == 3:
            if xt.shape[0] == B and xt.shape[1] == N and xt.shape[-1] == 2:
                x = xt[..., 1]
            elif xt.shape[0] == B and xt.shape[1] == 2 and xt.shape[2] == N:
                x = xt[:, 1, :]
            elif xt.shape[0] == B and xt.shape[1] == N and xt.shape[-1] == 1:
                x = xt.squeeze(-1)
            elif xt.shape[0] == B and xt.shape[-1] == N:
                # [B, D, N]
                x = xt.mean(dim=1)
            elif xt.shape[0] == B and xt.shape[1] == N:
                # [B, N, D]
                x = xt.mean(dim=-1)
            else:
                raise ValueError(f"Unexpected xt shape from diffusion.sample: {tuple(xt.shape)} (expected one of [B,N,2],[B,2,N],[B,N],[B,D,N],[B,N,D])")
        elif xt.dim() == 2:
            if xt.shape[0] == B and xt.shape[1] == N:
                x = xt
            else:
                raise ValueError(f"Unexpected xt shape from diffusion.sample: {tuple(xt.shape)} for mask (B,N)=({B},{N})")
        else:
            raise ValueError(f"Unexpected xt dim from diffusion.sample: {tuple(xt.shape)}")

        # map to [0,1] if needed
        if x.min() < -1e-4 or x.max() > 1.0 + 1e-4:
            x = torch.sigmoid(x)
        else:
            x = x.clamp(0.0, 1.0)

        # neutral for padding
        x = x * node_mask_f + 0.5 * (1.0 - node_mask_f)
        return x

    def sincos_positional(self, n, dim, device):
        """
        sincos positional encoding for indices 0..n-1 (returns [n, dim])
        """
        half = dim // 2
        positions = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)  # [n,1]
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=device) / max(half, 1))
        args = positions * freqs.unsqueeze(0)  # [n, half]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [n, 2*half]
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((n, 1), device=device)], dim=-1)
        return emb  # [n, dim]

    def compute_structural_features(self, adj):
        """
        adj: [N, N] binary adjacency matrix (dtype float32 or uint8) on correct device
        returns features: [N, F] float32 where F == self.feature_dim
        Features:
           - deg_norm (deg / max_deg)
           - avg_neighbor_deg ( (A @ deg) / deg_max )
           - two_hop_count_norm (row-sum of A^2 normalized)
           - pagerank (iterative, normalized)
           - clustering_coeff (local clustering)
           - sincos positional [N, sincos_dim]
        Notes: uses vectorized ops and small number of matvecs; avoids eigendecomposition.
        """
        device = adj.device
        A = adj.float()
        N = A.shape[0]
        if N == 0:
            return torch.zeros((0, self.feature_dim), dtype=torch.float32, device=device)

        # ensure no diag self-loops for structural stats (optional)
        A = A.clone()
        A.fill_diagonal_(0)

        # degree
        deg = A.sum(dim=1)  # [N]
        max_deg = deg.max().clamp(min=1.0)
        deg_norm = deg / max_deg  # [N]

        # average neighbor degree: (A @ deg) / deg (for nodes with deg>0)
        neigh_deg_sum = A @ deg  # [N]
        avg_neigh_deg = neigh_deg_sum / deg.clamp(min=1.0)
        # normalize by max_deg
        avg_neigh_deg_norm = avg_neigh_deg / max_deg

        # two-hop counts: A2 = A @ A, then row sum
        A2 = torch.matmul(A, A)  # [N,N] (dense matmul; optimized on GPU)
        two_hop = A2.sum(dim=1)
        two_hop_norm = two_hop / (two_hop.max().clamp(min=1.0))

        # approximate PageRank (power iteration on stochastic matrix)
        if self.use_pagerank:
            # build transition matrix P^T for matvec: pr_next = alpha * P^T @ pr + (1-alpha)/N
            alpha = 0.85
            deg_safe = deg.clone()
            deg_safe[deg_safe == 0] = 1.0
            P = A / deg_safe.unsqueeze(1)  # row-normalized: each row sum = 1 for nodes with deg>0
            P_T = P.transpose(0, 1)  # we will do P_T @ pr
            pr = torch.full((N,), 1.0 / N, device=device, dtype=torch.float32)
            for _ in range(self.pagerank_iters):
                pr = alpha * (P_T @ pr) + (1.0 - alpha) / float(N)
            pagerank = pr
            pagerank_norm = (pagerank - pagerank.min()).clamp(min=0.0)
            pagerank_norm = pagerank_norm / (pagerank_norm.max().clamp(min=1e-9))
        else:
            pagerank_norm = torch.zeros_like(deg_norm)

        # local clustering coefficient (approx via A2 * A diag sum)
        # triangles_i = 0.5 * sum_j ( (A2 * A)[i,j] ) = 0.5 * (A2 * A).sum(dim=1)
        tri_counts = (A2 * A).sum(dim=1) * 0.5  # [N]
        possible = deg * (deg - 1) * 0.5
        clustering = torch.zeros_like(deg, dtype=torch.float32)
        mask_poss = possible > 0
        clustering[mask_poss] = (tri_counts[mask_poss] / possible[mask_poss])
        clustering = torch.nan_to_num(clustering, nan=0.0, posinf=0.0, neginf=0.0)
        clustering = clustering.clamp(min=0.0, max=1.0)

        # sincos positional
        # sincos = self.sincos_positional(N, self.sincos_dim, device=device)  # [N, sincos_dim]

        # assemble features
        feats = torch.stack([
            deg_norm.to(torch.float32),
            avg_neigh_deg_norm.to(torch.float32),
            two_hop_norm.to(torch.float32),
            pagerank_norm.to(torch.float32),
            clustering.to(torch.float32),
        ], dim=1)  # [N,5]

        # feats = torch.cat([feats, sincos.to(torch.float32)], dim=1)  # [N, feature_dim]
        # final safety: if dims mismatch, pad/trim
        if feats.shape[1] < self.feature_dim:
            pad = torch.zeros((N, self.feature_dim - feats.shape[1]), device=device)
            feats = torch.cat([feats, pad], dim=1)
        elif feats.shape[1] > self.feature_dim:
            feats = feats[:, :self.feature_dim]

        return feats  # [N, feature_dim]

    def build_node_features(self, graph_data_list, point_indicator):
        """
        graph_data_list: list-like of GraphData objects (length B) or a torch_geometric Batch
        point_indicator: tensor shape [B] with num_nodes per sample
        returns:
           node_features: [B, N_max, feature_dim] float tensor
           adj_batch: [B, N_max, N_max] float32 tensor
           mask: [B, N_max] bool tensor (True for real nodes)
        """
        # If graph_data_list is a Batch from torch_geometric, convert to list of Data
        if hasattr(graph_data_list, "to_data_list"):
            graph_data_list = graph_data_list.to_data_list()

        # Normalize point_indicator into a list of ints robustly
        if isinstance(point_indicator, torch.Tensor):
            if point_indicator.dim() > 1:
                point_indicator = point_indicator.view(point_indicator.shape[0], -1)[:, 0]
            num_nodes_list = [int(x) for x in point_indicator.tolist()]
        else:
            num_nodes_list = [int(x) for x in list(point_indicator)]

        B = len(num_nodes_list)
        if B == 0:
            return torch.zeros((0, 0, self.feature_dim)), torch.zeros((0, 0, 0)), torch.zeros((0, 0), dtype=torch.bool)

        N_max = max(num_nodes_list)
        device = torch.device(self.device if hasattr(self, 'device') else 'cpu')

        # preallocate
        adj_batch = torch.zeros((B, N_max, N_max), dtype=torch.float32, device=device)
        feat_batch = torch.zeros((B, N_max, self.feature_dim), dtype=torch.float32, device=device)
        mask = torch.zeros((B, N_max), dtype=torch.bool, device=device)

        for b in range(B):
            num_nodes = num_nodes_list[b]
            mask[b, :num_nodes] = True
            gd = graph_data_list[b]
            edge_index = gd.edge_index if hasattr(gd, 'edge_index') else None

            # build dense adjacency as float32 on device
            adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
            if edge_index is not None:
                ei = edge_index.long()
                if ei.dim() == 2 and ei.shape[0] == 2:
                    rows = ei[0].to(device)
                    cols = ei[1].to(device)
                    # if collated edge_index has batch offsets, this code expects per-sample edge_index
                    adj[rows, cols] = 1.0
                    adj[cols, rows] = 1.0
            # ensure zeros on diagonal
            adj.fill_diagonal_(0.0)
            adj_batch[b, :num_nodes, :num_nodes] = adj

            # compute low-cost structural features
            feats = self.compute_structural_features(adj)  # [num_nodes, feature_dim]
            feat_batch[b, :num_nodes, :] = feats

        return feat_batch, adj_batch, mask

    # ----------------- training / inference -----------------
    def training_step(self, batch, batch_idx):
        """
        Expect batch = (idx_tensor, graph_data_list, point_indicator_tensor)
        """
        idxs, graph_data_list, point_indicator = batch

        # If graph_data_list is a Batch, convert to list
        if hasattr(graph_data_list, "to_data_list"):
            graph_data_list = graph_data_list.to_data_list()

        device = torch.device(self.device if hasattr(self, 'device') else 'cpu')
        B = len(graph_data_list)

        # Build node features + adj + mask
        node_features, adj_batch, mask = self.build_node_features(graph_data_list, point_indicator)
        node_features = node_features.to(device)
        adj_batch = adj_batch.to(device)
        mask = mask.to(device)

        # get labels: read original labels from graph_data.x (MISDataset stores labels in x)
        labels_list = []
        for b in range(B):
            labels = graph_data_list[b].x
            labels = labels.view(-1).float().to(device)  # shape [num_nodes]
            pad_len = node_features.shape[1] - labels.shape[0]
            if pad_len > 0:
                labels = torch.cat([labels, torch.zeros(pad_len, device=device)], dim=0)
            labels_list.append(labels)
        labels_tensor = torch.stack(labels_list, dim=0)  # [B, N_max]

        # time sampling
        t_idx = self.sample_time_logit_normal(B, device)
        t_idx_np = t_idx.cpu().numpy()

        # Construct x_t from x_0 via diffusion.sample (like TSP/ATSP branches)
        x0 = labels_tensor.long().clamp(0, 1)  # [B, N]
        x0_onehot = F.one_hot(x0, num_classes=2).float()  # [B, N, 2]
        with torch.no_grad():
            xt = self.diffusion.sample(x0_onehot, t_idx_np)
        x_t_node = self._extract_xt_nodeprob(xt, node_mask=mask, device=device)  # [B, N]

        # forward: pass node features, x_t_node and adj bias into MIS node transformer
        adj_bias = adj_batch  # [B, N, N]
        pred_logits = self.model(
            node_features,
            t_idx.float().to(device),
            x_t_node=x_t_node,
            adj_bias=adj_bias,
            node_mask=mask,
        )  # [B, N]

        # compute BCE loss masked
        mask_f = mask.float()
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        per_elem = loss_fn(pred_logits, labels_tensor)
        per_elem = per_elem * mask_f
        loss = per_elem.sum() / (mask_f.sum().clamp(min=1.0))

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx, split="test"):
        """
        Inference pipeline (MIS):
          1) build node features and adj
          2) obtain an anchor via single-step or multi-step denoising from neutral x_t
          3) perturb anchor in logit space to get S candidates
          4) continue denoising each candidate for a few refinement steps
          5) greedy MIS insertion + swap-based local improvement; pick best by size
        """
        idxs, graph_data_list, point_indicator = batch

        if hasattr(graph_data_list, "to_data_list"):
            graph_data_list = graph_data_list.to_data_list()

        device = torch.device(self.device if hasattr(self, "device") else "cpu")
        B = len(graph_data_list)

        # Build node features + adj + mask
        node_features, adj_batch, mask = self.build_node_features(graph_data_list, point_indicator)
        node_features = node_features.to(device)
        adj_batch = adj_batch.to(device)
        mask = mask.to(device)

        B2, N, _ = node_features.shape
        assert B2 == B, "Batch size mismatch"

        # inference params
        S = int(getattr(self, "fast_num_candidates", 3))
        sigma = float(getattr(self, "fast_sigma", 1.8))
        t_anchor_raw = float(getattr(self.args, "fast_t_anchor", 0.6))
        t_anchor = self._resolve_timestep(t_anchor_raw)
        refine_steps = int(getattr(self.args, "mis_refine_steps", 1))
        eps = 1e-6

        # (Optional) build an InferenceSchedule (same knobs as TSP/ATSP) with caching
        schedule = None
        inf_sched_name = getattr(self.args, "inference_schedule", None)
        inf_T = int(getattr(self.args, "inference_diffusion_steps", 1))
        if inf_sched_name is not None and inf_T > 1:
            T = int(getattr(self.diffusion, "T", 1000))
            if (
                (self._cached_schedule is None)
                or (getattr(self._cached_schedule, "inference_schedule", None) != inf_sched_name)
                or (getattr(self._cached_schedule, "inference_T", None) != inf_T)
                or (getattr(self._cached_schedule, "T", None) != T)
            ):
                schedule = InferenceSchedule(inf_sched_name, T, inf_T)
                self._cached_schedule = schedule
            else:
                schedule = self._cached_schedule

        # 1) anchor from neutral
        neutral_node = torch.full((B, N), 0.5, device=device, dtype=torch.float32)

        if schedule is None:
            t_tensor = torch.full((B,), float(t_anchor), device=device, dtype=torch.float32)

            if self._compiled_infer_fn is not None:
                with self._autocast_context():
                    state_probs, decode_scores, anchor = self._compiled_infer_fn(
                        node_features, neutral_node, t_tensor, adj_batch, mask
                    )
                anchor = anchor.detach()
                state_probs = state_probs.detach()
                decode_scores = decode_scores.detach()
            else:
                with self._autocast_context():
                    state_probs, decode_scores, anchor = self._infer_anchor_candidates_refine_eager(
                        node_features=node_features,
                        neutral_node=neutral_node,
                        t_tensor=t_tensor,
                        adj_batch=adj_batch,
                        mask=mask,
                        refine_steps=refine_steps,
                    )
                state_probs = state_probs.detach()
                decode_scores = decode_scores.detach()
        else:
            current = neutral_node
            for i in range(schedule.inference_T):
                t1, _t2 = schedule(i)
                t_tensor = torch.full((B,), float(self._resolve_timestep(float(t1))), device=device, dtype=torch.float32)
                with self._autocast_context():
                    logits = self.model(
                        node_features,
                        t_tensor,
                        x_t_node=current,
                        adj_bias=adj_batch,
                        node_mask=mask,
                    )
                    current, _ = self._apply_mis_projection_from_logits(logits, adj_batch, node_mask=mask)
                    current = current.detach()
            anchor = current

        # 2) perturb anchor -> S candidates.
        # If schedule is None, both state_probs and decode_scores are already produced by the fast path.
        if schedule is not None:
            state_probs = sample_candidates_from_anchor(
                anchor,
                S,
                use_gumbel=self.use_gumbel,
                sigma=sigma,
                gumbel_scale=float(getattr(self, "fast_gumbel_scale", 1.8)),
            )  # [B*S, N] or [B, S, N]

            # normalize shape -> [B, S, N]
            if state_probs.dim() == 2:
                state_probs = state_probs.view(B, S, -1)
            elif state_probs.dim() != 3:
                raise ValueError(f"Unexpected state_probs shape: {tuple(state_probs.shape)}")

            raw_decode_scores = torch.logit(state_probs.reshape(B * S, N).clamp(eps, 1.0 - eps))
            decode_scores = raw_decode_scores.view(B, S, N)

            if self.mis_use_entropic_projection:
                adj_rep = adj_batch.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, N, N)
                mask_rep = mask.unsqueeze(1).repeat(1, S, 1).reshape(B * S, N)
                state_probs, _ = self._apply_mis_projection_from_logits(
                    raw_decode_scores,
                    adj_rep,
                    node_mask=mask_rep,
                )
                state_probs = state_probs.view(B, S, N)

        # 3) continue denoising for refinement steps (optional)
        if refine_steps > 0 and schedule is not None:
            # pick a list of timesteps to run after perturb
            if schedule is not None:
                # use the last 'refine_steps' timesteps of the schedule (closest to clean)
                start_i = max(0, schedule.inference_T - refine_steps)
                t_list = [self._resolve_timestep(float(schedule(i)[0])) for i in range(start_i, schedule.inference_T)]
            else:
                # simple linear anneal from t_anchor -> 0
                if refine_steps == 1:
                    t_list = [t_anchor]
                else:
                    t_list = [float(t_anchor * (1.0 - k / (refine_steps - 1))) for k in range(refine_steps)]
            # batched refinement
            node_feat_rep = node_features.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, N, -1)
            adj_rep = adj_batch.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, N, N)
            mask_rep = mask.unsqueeze(1).repeat(1, S, 1).reshape(B * S, N)

            current = state_probs.reshape(B * S, N)
            for t_val in t_list:
                t_tensor = torch.full((B * S,), float(t_val), device=device)
                logits = self.model(
                    node_feat_rep,
                    t_tensor,
                    x_t_node=current,
                    adj_bias=adj_rep,
                    node_mask=mask_rep,
                )
                decode_scores = logits.view(B, S, N)
                current, _ = self._apply_mis_projection_from_logits(logits, adj_rep, node_mask=mask_rep)
                current = current.detach()
            state_probs = current.view(B, S, N)

        if self.mis_use_entropic_projection:
            adj_rep = adj_batch.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, N, N)
            mask_rep = mask.unsqueeze(1).repeat(1, S, 1).reshape(B * S, N)
            proj_residual = mis_projection_residual(
                state_probs.reshape(B * S, N),
                adj_rep,
                node_mask=mask_rep,
            )
            self.log(
                f"{split}/proj_residual",
                float(proj_residual.mean().item()),
                on_epoch=True,
                sync_dist=True,
            )

        # 4) greedy decode + local improvement; pick best candidate
        candidate_scores = self._select_node_decoder_scores(state_probs, decode_scores)

        sizes = []
        recalls = []
        best_solutions = []
        for b in range(B):
            num_nodes = int(point_indicator[b].item()) if isinstance(point_indicator[b], torch.Tensor) else int(point_indicator[b])
            adj_b = adj_batch[b:b+1]  # [1, N, N]
            label = graph_data_list[b].x.view(-1).to(device)

            cand_scores = candidate_scores[b]
            best_size = -1
            best_sol = None
            for s_idx in range(S):
                scores_s = cand_scores[s_idx:s_idx+1]  # [1, N]
                sol = greedy_mis_from_scores(scores_s, adj_b)[0]  # [N] bool

                # local improvement (swap 1->2) on padded graph, then trim
                sol_imp = local_improvement_swap_12_batched(
                    sol.unsqueeze(0),
                    adj_b,
                    scores=scores_s,
                    node_mask=mask[b:b+1],
                    max_iters=int(getattr(self.args, "mis_swap_iters", 5000)),
                    top_k=int(getattr(self.args, "mis_swap_top_k", 200)),
                )[0]  # [N] bool

                sol_trim = sol_imp[:num_nodes]
                size = int(sol_trim.sum().item())
                if size > best_size:
                    best_size = size
                    best_sol = sol_trim.clone()

            sizes.append(best_size)
            best_solutions.append(best_sol)

            # recall vs ground-truth (if provided)
            gt = label[:num_nodes].bool()
            overlap = int((best_sol & gt).sum().item())
            gt_size = max(1, int(gt.sum().item()))
            recalls.append(float(overlap / gt_size) if gt_size > 0 else 0.0)

        gap_list = []
        gt_sizes = []
        for i in range(B):
            label = graph_data_list[i].x.view(-1).to(device).float()
            num_nodes_i = int(point_indicator[i].item()) if isinstance(point_indicator[i], torch.Tensor) else int(point_indicator[i])
            gt_bin = (label[:num_nodes_i] > 0.5)
            gt_size = int(gt_bin.sum().item())
            gt_sizes.append(gt_size)

            found = int(best_solutions[i].sum().item())
            if gt_size > 0:
                gap_pct = (gt_size - found) / gt_size * 100.0
                # clamp at 0 for reporting optimality gap (in case labels are suboptimal)
                gap_pct = max(0.0, float(gap_pct))
            else:
                gap_pct = 0.0
            gap_list.append(gap_pct)

        mean_gap = float(np.mean(gap_list)) if len(gap_list) > 0 else 0.0
        mean_gt_size = float(np.mean(gt_sizes)) if len(gt_sizes) > 0 else 0.0
        mean_size = float(np.mean(sizes))
        mean_recall = float(np.mean(recalls))

        metrics = {
            f"{split}/avg_size": mean_size,
            f"{split}/mean_recall": mean_recall,
            f"{split}/gt_size": mean_gt_size,
            f"{split}/gap": mean_gap,
        }
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True, batch_size=B)
        return metrics
    
    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split="val")
