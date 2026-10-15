import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.co_datasets.mvc_dataset import MVCDataset
from diffusion.pl_ragenco_mis_model import RAGenCOMISModel
from diffusion.utils.diffusion_schedulers import InferenceSchedule
from diffusion.utils.mis_utils import (
    greedy_mis_from_scores,
    local_improvement_swap_12_batched,
    mis_projection_residual,
    sample_candidates_from_anchor,
)


class RAGenCOMVCModel(RAGenCOMISModel):
    """RAGenCO MVC model via the MVC <-> MIS complement relation.

    The raw dataset stores MVC cover labels. Internally we supervise the complement
    independent-set indicator so the decode-aware projected control state,
    anchor branching, and local improvement are shared with the MIS branch. The
    final cover is obtained by complementing the decoded independent set.
    """

    def __init__(self, param_args=None, load_dataset=True):
        super().__init__(param_args=param_args, load_dataset=False)

        if load_dataset:
            if self.args.training_split:
                self.train_dataset = MVCDataset(
                    data_file=os.path.join(self.args.training_split),
                    data_label_dir=getattr(self.args, "training_split_label_dir", None),
                )
            if self.args.test_split:
                self.test_dataset = MVCDataset(
                    data_file=os.path.join(self.args.test_split),
                    data_label_dir=getattr(self.args, "test_split_label_dir", None),
                )
            if self.args.validation_split:
                self.validation_dataset = MVCDataset(
                    data_file=os.path.join(self.args.validation_split),
                    data_label_dir=getattr(self.args, "validation_split_label_dir", None),
                )

        # Map MVC-specific CLI knobs to the MIS-style internals used by the shared
        # refinement path. The model still predicts the complement independent set.
        refine_steps = int(
            getattr(self.args, "mvc_refine_steps", getattr(self.args, "mis_refine_steps", 1))
        )
        swap_iters = int(
            getattr(self.args, "mvc_swap_iters", getattr(self.args, "mis_swap_iters", 5000))
        )
        swap_top_k = int(
            getattr(self.args, "mvc_swap_top_k", getattr(self.args, "mis_swap_top_k", 200))
        )
        use_proj = bool(
            getattr(
                self.args,
                "mvc_use_entropic_projection",
                getattr(self.args, "mis_use_entropic_projection", True),
            )
        )
        proj_tau = float(getattr(self.args, "mvc_proj_tau", getattr(self.args, "mis_proj_tau", 1.0)))
        proj_steps = int(getattr(self.args, "mvc_proj_steps", getattr(self.args, "mis_proj_steps", 8)))
        proj_step_size = float(
            getattr(self.args, "mvc_proj_step_size", getattr(self.args, "mis_proj_step_size", 0.5))
        )
        proj_tol = float(getattr(self.args, "mvc_proj_tol", getattr(self.args, "mis_proj_tol", 1e-4)))

        setattr(self.args, "mis_refine_steps", refine_steps)
        setattr(self.args, "mis_swap_iters", swap_iters)
        setattr(self.args, "mis_swap_top_k", swap_top_k)
        setattr(self.args, "mis_use_entropic_projection", use_proj)
        setattr(self.args, "mis_proj_tau", proj_tau)
        setattr(self.args, "mis_proj_steps", proj_steps)
        setattr(self.args, "mis_proj_step_size", proj_step_size)
        setattr(self.args, "mis_proj_tol", proj_tol)

        self.mis_use_entropic_projection = use_proj
        self.mis_proj_tau = proj_tau
        self.mis_proj_steps = proj_steps
        self.mis_proj_step_size = proj_step_size
        self.mis_proj_tol = proj_tol

    def _stack_cover_labels(self, graph_data_list, padded_num_nodes: int, device: torch.device):
        labels_list = []
        for graph_data in graph_data_list:
            cover_labels = graph_data.x.view(-1).float().to(device)
            pad_len = padded_num_nodes - cover_labels.shape[0]
            if pad_len > 0:
                cover_labels = torch.cat([cover_labels, torch.zeros(pad_len, device=device)], dim=0)
            labels_list.append(cover_labels)
        return torch.stack(labels_list, dim=0)

    def training_step(self, batch, batch_idx):
        idxs, graph_data_list, point_indicator = batch

        if hasattr(graph_data_list, "to_data_list"):
            graph_data_list = graph_data_list.to_data_list()

        device = torch.device(self.device if hasattr(self, "device") else "cpu")
        batch_size = len(graph_data_list)

        node_features, adj_batch, mask = self.build_node_features(graph_data_list, point_indicator)
        node_features = node_features.to(device)
        adj_batch = adj_batch.to(device)
        mask = mask.to(device)

        cover_labels = self._stack_cover_labels(graph_data_list, node_features.shape[1], device)
        independent_labels = (1.0 - cover_labels).clamp(0.0, 1.0)

        t_idx = self.sample_time_logit_normal(batch_size, device)
        t_idx_np = t_idx.cpu().numpy()

        x0 = independent_labels.long().clamp(0, 1)
        x0_onehot = F.one_hot(x0, num_classes=2).float()
        with torch.no_grad():
            xt = self.diffusion.sample(x0_onehot, t_idx_np)
        x_t_node = self._extract_xt_nodeprob(xt, node_mask=mask, device=device)

        pred_logits = self.model(
            node_features,
            t_idx.float().to(device),
            x_t_node=x_t_node,
            adj_bias=adj_batch,
            node_mask=mask,
        )

        mask_f = mask.float()
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        per_elem = loss_fn(pred_logits, independent_labels)
        per_elem = per_elem * mask_f
        loss = per_elem.sum() / (mask_f.sum().clamp(min=1.0))

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx, split="test"):
        idxs, graph_data_list, point_indicator = batch

        if hasattr(graph_data_list, "to_data_list"):
            graph_data_list = graph_data_list.to_data_list()

        device = torch.device(self.device if hasattr(self, "device") else "cpu")
        batch_size = len(graph_data_list)

        node_features, adj_batch, mask = self.build_node_features(graph_data_list, point_indicator)
        node_features = node_features.to(device)
        adj_batch = adj_batch.to(device)
        mask = mask.to(device)

        _, num_nodes, _ = node_features.shape
        num_candidates = int(getattr(self, "fast_num_candidates", 3))
        sigma = float(getattr(self, "fast_sigma", 1.8))
        t_anchor_raw = float(getattr(self.args, "fast_t_anchor", 0.6))
        t_anchor = self._resolve_timestep(t_anchor_raw)
        refine_steps = int(getattr(self.args, "mis_refine_steps", 1))
        eps = 1e-6

        schedule = None
        inf_sched_name = getattr(self.args, "inference_schedule", None)
        inf_steps = int(getattr(self.args, "inference_diffusion_steps", 1))
        if inf_sched_name is not None and inf_steps > 1:
            total_steps = int(getattr(self.diffusion, "T", 1000))
            if (
                (self._cached_schedule is None)
                or (getattr(self._cached_schedule, "inference_schedule", None) != inf_sched_name)
                or (getattr(self._cached_schedule, "inference_T", None) != inf_steps)
                or (getattr(self._cached_schedule, "T", None) != total_steps)
            ):
                schedule = InferenceSchedule(inf_sched_name, total_steps, inf_steps)
                self._cached_schedule = schedule
            else:
                schedule = self._cached_schedule

        neutral_node = torch.full((batch_size, num_nodes), 0.5, device=device, dtype=torch.float32)

        if schedule is None:
            t_tensor = torch.full((batch_size,), float(t_anchor), device=device, dtype=torch.float32)

            if self._compiled_infer_fn is not None:
                with self._autocast_context():
                    state_probs, decode_scores, anchor = self._compiled_infer_fn(
                        node_features, neutral_node, t_tensor, adj_batch, mask
                    )
                state_probs = state_probs.detach()
                decode_scores = decode_scores.detach()
                anchor = anchor.detach()
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
                t1, _ = schedule(i)
                t_tensor = torch.full(
                    (batch_size,),
                    float(self._resolve_timestep(float(t1))),
                    device=device,
                    dtype=torch.float32,
                )
                with self._autocast_context():
                    logits = self.model(
                        node_features,
                        t_tensor,
                        x_t_node=current,
                        adj_bias=adj_batch,
                        node_mask=mask,
                    )
                    current, _ = self._apply_mis_projection_from_logits(
                        logits, adj_batch, node_mask=mask
                    )
                    current = current.detach()
            anchor = current

        if schedule is not None:
            state_probs = sample_candidates_from_anchor(
                anchor,
                num_candidates,
                use_gumbel=self.use_gumbel,
                sigma=sigma,
                gumbel_scale=float(getattr(self, "fast_gumbel_scale", 1.8)),
            )

            if state_probs.dim() == 2:
                state_probs = state_probs.view(batch_size, num_candidates, -1)
            elif state_probs.dim() != 3:
                raise ValueError(f"Unexpected state_probs shape: {tuple(state_probs.shape)}")

            raw_decode_scores = torch.logit(
                state_probs.reshape(batch_size * num_candidates, num_nodes).clamp(eps, 1.0 - eps)
            )
            decode_scores = raw_decode_scores.view(batch_size, num_candidates, num_nodes)

            if self.mis_use_entropic_projection:
                adj_rep = (
                    adj_batch.unsqueeze(1)
                    .repeat(1, num_candidates, 1, 1)
                    .reshape(batch_size * num_candidates, num_nodes, num_nodes)
                )
                mask_rep = (
                    mask.unsqueeze(1)
                    .repeat(1, num_candidates, 1)
                    .reshape(batch_size * num_candidates, num_nodes)
                )
                state_probs, _ = self._apply_mis_projection_from_logits(
                    raw_decode_scores,
                    adj_rep,
                    node_mask=mask_rep,
                )
                state_probs = state_probs.view(batch_size, num_candidates, num_nodes)

        if refine_steps > 0 and schedule is not None:
            start_i = max(0, schedule.inference_T - refine_steps)
            t_list = [
                self._resolve_timestep(float(schedule(i)[0]))
                for i in range(start_i, schedule.inference_T)
            ]
            node_feat_rep = (
                node_features.unsqueeze(1)
                .repeat(1, num_candidates, 1, 1)
                .reshape(batch_size * num_candidates, num_nodes, -1)
            )
            adj_rep = (
                adj_batch.unsqueeze(1)
                .repeat(1, num_candidates, 1, 1)
                .reshape(batch_size * num_candidates, num_nodes, num_nodes)
            )
            mask_rep = (
                mask.unsqueeze(1)
                .repeat(1, num_candidates, 1)
                .reshape(batch_size * num_candidates, num_nodes)
            )

            current = state_probs.reshape(batch_size * num_candidates, num_nodes)
            for t_val in t_list:
                t_tensor = torch.full((batch_size * num_candidates,), float(t_val), device=device)
                logits = self.model(
                    node_feat_rep,
                    t_tensor,
                    x_t_node=current,
                    adj_bias=adj_rep,
                    node_mask=mask_rep,
                )
                decode_scores = logits.view(batch_size, num_candidates, num_nodes)
                current, _ = self._apply_mis_projection_from_logits(
                    logits, adj_rep, node_mask=mask_rep
                )
                current = current.detach()
            state_probs = current.view(batch_size, num_candidates, num_nodes)

        if self.mis_use_entropic_projection:
            adj_rep = (
                adj_batch.unsqueeze(1)
                .repeat(1, num_candidates, 1, 1)
                .reshape(batch_size * num_candidates, num_nodes, num_nodes)
            )
            mask_rep = (
                mask.unsqueeze(1)
                .repeat(1, num_candidates, 1)
                .reshape(batch_size * num_candidates, num_nodes)
            )
            proj_residual = mis_projection_residual(
                state_probs.reshape(batch_size * num_candidates, num_nodes),
                adj_rep,
                node_mask=mask_rep,
            )
            self.log(
                f"{split}/proj_residual",
                float(proj_residual.mean().item()),
                on_epoch=True,
                sync_dist=True,
            )

        candidate_scores = self._select_node_decoder_scores(state_probs, decode_scores)

        independent_sizes = []
        cover_sizes = []
        independent_recalls = []
        cover_recalls = []
        best_independent_solutions = []

        for batch_i in range(batch_size):
            sample_nodes = (
                int(point_indicator[batch_i].item())
                if isinstance(point_indicator[batch_i], torch.Tensor)
                else int(point_indicator[batch_i])
            )
            adj_b = adj_batch[batch_i : batch_i + 1]
            raw_cover = graph_data_list[batch_i].x.view(-1).to(device).float()
            gt_independent = (1.0 - raw_cover[:sample_nodes]).bool()
            gt_cover = raw_cover[:sample_nodes].bool()

            cand_scores = candidate_scores[batch_i]
            best_size = -1
            best_sol = None
            for cand_i in range(num_candidates):
                scores_i = cand_scores[cand_i : cand_i + 1]
                sol = greedy_mis_from_scores(scores_i, adj_b)[0]
                sol_imp = local_improvement_swap_12_batched(
                    sol.unsqueeze(0),
                    adj_b,
                    scores=scores_i,
                    node_mask=mask[batch_i : batch_i + 1],
                    max_iters=int(getattr(self.args, "mis_swap_iters", 5000)),
                    top_k=int(getattr(self.args, "mis_swap_top_k", 200)),
                )[0]

                sol_trim = sol_imp[:sample_nodes]
                size = int(sol_trim.sum().item())
                if size > best_size:
                    best_size = size
                    best_sol = sol_trim.clone()

            pred_cover = ~best_sol
            independent_sizes.append(best_size)
            cover_sizes.append(int(pred_cover.sum().item()))
            best_independent_solutions.append(best_sol)

            gt_ind_size = max(1, int(gt_independent.sum().item()))
            independent_overlap = int((best_sol & gt_independent).sum().item())
            independent_recalls.append(float(independent_overlap / gt_ind_size))

            gt_cover_size = max(1, int(gt_cover.sum().item()))
            cover_overlap = int((pred_cover & gt_cover).sum().item())
            cover_recalls.append(float(cover_overlap / gt_cover_size))

        gap_list = []
        gt_cover_sizes = []
        gt_independent_sizes = []
        for batch_i in range(batch_size):
            raw_cover = graph_data_list[batch_i].x.view(-1).to(device).float()
            sample_nodes = (
                int(point_indicator[batch_i].item())
                if isinstance(point_indicator[batch_i], torch.Tensor)
                else int(point_indicator[batch_i])
            )
            gt_cover_size = int(raw_cover[:sample_nodes].sum().item())
            gt_independent_size = int(sample_nodes - gt_cover_size)
            pred_cover_size = int(sample_nodes - best_independent_solutions[batch_i].sum().item())

            gt_cover_sizes.append(gt_cover_size)
            gt_independent_sizes.append(gt_independent_size)

            if gt_cover_size > 0:
                gap_pct = (pred_cover_size - gt_cover_size) / gt_cover_size * 100.0
                gap_pct = max(0.0, float(gap_pct))
            else:
                gap_pct = 0.0
            gap_list.append(gap_pct)

        metrics = {
            f"{split}/avg_independent_size": float(np.mean(independent_sizes)) if independent_sizes else 0.0,
            f"{split}/avg_cover_size": float(np.mean(cover_sizes)) if cover_sizes else 0.0,
            f"{split}/mean_independent_recall": float(np.mean(independent_recalls)) if independent_recalls else 0.0,
            f"{split}/mean_cover_recall": float(np.mean(cover_recalls)) if cover_recalls else 0.0,
            f"{split}/gt_independent_size": float(np.mean(gt_independent_sizes)) if gt_independent_sizes else 0.0,
            f"{split}/gt_cover_size": float(np.mean(gt_cover_sizes)) if gt_cover_sizes else 0.0,
            f"{split}/gap": float(np.mean(gap_list)) if gap_list else 0.0,
        }
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return metrics

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split="val")
