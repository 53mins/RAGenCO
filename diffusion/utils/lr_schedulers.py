"""Misc. optimizer implementations and warmup-aware schedulers."""
from functools import partial
import math
from typing import List, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR

from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule

def get_schedule_fn(scheduler: str, num_training_steps: int, warmup_steps: int = 0):
    """Returns a callable scheduler_fn(optimizer) -> lr scheduler.

    This function now supports optional linear warmup for the first `warmup_steps`.
    After warmup the chosen schedule (cosine-decay / one-cycle) continues from the
    post-warmup step (i.e. warmup + decay compose).

    Supported scheduler strings:
      - "cosine-decay": cosine annealing from base_lr -> 0 over num_training_steps
      - "one-cycle": simplified single-cycle (ramp up then ramp down) over num_training_steps

    Note: warmup_steps is in the same unit as num_training_steps (both in optimizer steps).
    """
    warmup_steps = int(warmup_steps)

    def _with_warmup(lr_lambda_after_warmup):
        def wrapped(current_step):
            # current_step is an integer step (starting from 0)
            if current_step < 0:
                return 0.0
            # Warmup portion (linear scale from 0 -> 1)
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            # After warmup: map current_step -> step' relative to post-warmup horizon
            if warmup_steps >= num_training_steps:
                # Trivial case: everything is warmup (no decay)
                return 1.0
            post_step = current_step - warmup_steps
            post_total = max(1, num_training_steps - warmup_steps)
            return lr_lambda_after_warmup(post_step, post_total)
        return wrapped

    if scheduler == "cosine-decay":
        def _cosine_lambda(post_step, post_total):
            if post_step >= post_total:
                return 0.0
            progress = float(post_step) / float(max(1, post_total))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_fn = _with_warmup(_cosine_lambda)
        scheduler_fn = lambda optimizer: LambdaLR(optimizer, lr_fn, -1)

    elif scheduler == "one-cycle":
        def _one_cycle_lambda(post_step, post_total):
            # simplified single-cycle (ramp up then ramp down) over post_total
            half = post_total / 2.0
            if post_step < 0:
                return 0.0
            if post_step < half:
                return float(post_step) / float(max(1.0, half))
            else:
                return float(2.0 - post_step / float(max(1.0, half)))
        lr_fn = _with_warmup(_one_cycle_lambda)
        scheduler_fn = lambda optimizer: LambdaLR(optimizer, lr_fn, -1)

    else:
        raise ValueError(f"Invalid schedule {scheduler} given.")
    return scheduler_fn


def get_one_cycle(optimizer, num_training_steps):
    # Backwards-compatible helper (no warmup here). Prefer get_schedule_fn("one-cycle", ...)
    def lr_lambda(current_step):
        if current_step < num_training_steps / 2:
            return float(current_step / (num_training_steps / 2))
        else:
            return float(2 - current_step / (num_training_steps / 2))
    return LambdaLR(optimizer, lr_lambda, -1)


class LRWarmupCallback(Callback):
    """
    Backward-compatible warmup callback.

    Notes:
    - Recommended approach: use get_schedule_fn(..., warmup_steps=...) and register
      the returned scheduler via LightningModule.configure_optimizers. That integrates
      warmup into the scheduler and avoids conflicts.
    - If a scheduler is already registered with the trainer (e.g., returned by
      configure_optimizers), this callback will NOT perform per-step lr writes to avoid
      clobbering the scheduler. In that case the user should integrate warmup into the
      scheduler via get_schedule_fn.
    - If no scheduler is present and warmup_steps>0, this callback performs the classic
      per-step warmup by setting optimizer.param_groups[*]['lr'] = base_lr * scale.
    """
    def __init__(self, warmup_steps: int = 0):
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        # recorded base lrs per optimizer param_group
        self._base_lrs: Optional[List[float]] = None
        self._initialized = False
        self._disabled_due_to_scheduler = False

    def _init_base_lrs(self, trainer: Trainer):
        opt_list = trainer.optimizers
        if not opt_list:
            self._base_lrs = []
            return
        opt = opt_list[0]
        # record current lr as base
        self._base_lrs = [float(group.get("lr", 0.0)) for group in opt.param_groups]
        # store backup key for cleanup
        for idx, group in enumerate(opt.param_groups):
            group.setdefault("initial_lr_saved_by_warmup", self._base_lrs[idx])
        self._initialized = True

        # If the trainer already has lr schedulers registered, prefer scheduler-based warmup
        # and disable per-step lr writes to avoid conflicts.
        has_scheduler = False
        try:
            # trainer.lr_schedulers is populated by Lightning when schedulers are set up.
            # It may be a list of dicts; check for non-empty.
            has_scheduler = bool(getattr(trainer, "lr_schedulers", None))
        except Exception:
            has_scheduler = False

        if has_scheduler:
            self._disabled_due_to_scheduler = True

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.warmup_steps <= 0:
            return
        if not self._initialized:
            self._init_base_lrs(trainer)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int, dataloader_idx: int = 0):
        # If scheduler-based warmup is available/used, do nothing here.
        if self.warmup_steps <= 0:
            return
        if not self._initialized:
            self._init_base_lrs(trainer)
            if not self._initialized:
                return
        if self._disabled_due_to_scheduler:
            # A scheduler has been detected; we avoid clobbering scheduler-managed LR.
            return

        step = getattr(trainer, "global_step", None)
        if step is None:
            return
        opt = trainer.optimizers[0]
        if step >= self.warmup_steps:
            for idx, group in enumerate(opt.param_groups):
                group["lr"] = float(self._base_lrs[idx])
            return

        scale = float(step + 1) / float(max(1, self.warmup_steps))
        for idx, group in enumerate(opt.param_groups):
            base = float(self._base_lrs[idx])
            group["lr"] = base * scale

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # restore base lrs
        if not self._initialized:
            return
        if not getattr(trainer, "optimizers", None):
            return
        opt = trainer.optimizers[0]
        for idx, group in enumerate(opt.param_groups):
            group["lr"] = float(self._base_lrs[idx])
            if "initial_lr_saved_by_warmup" in group:
                group.pop("initial_lr_saved_by_warmup", None)