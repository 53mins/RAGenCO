import os
import sys
import types
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from diffusion.pl_ragenco_mvc_model import RAGenCOMVCModel
from diffusion.utils.lr_schedulers import get_schedule_fn
from train import arg_parser, build_experiment_logger, finalize_experiment_logger

warnings.filterwarnings("ignore")


def main(args):
    print("Initializing MVC Training...")
    print(args)

    project_name = args.project_name if args.project_name else "ragenco_mvc"

    model = RAGenCOMVCModel(param_args=args)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank_zero_info(f"MVC Model Initialized. Trainable Parameters: {param_count:,}")
    rank_zero_info(f"Time Sampling: Logit-Normal (mu={model.mu}, sigma={model.sigma})")

    if args.do_train:
        train_loader = model.train_dataloader()
        steps_per_epoch = len(train_loader)
        rank_zero_info(f"Inferred steps_per_epoch from model.train_dataloader(): {steps_per_epoch}")
        if (steps_per_epoch is None) or (steps_per_epoch == 0):
            num_training_steps = max(1, args.num_epochs)
            rank_zero_info(f"Using fallback num_training_steps={num_training_steps}")
        else:
            num_training_steps = int(steps_per_epoch) * int(args.num_epochs)
            rank_zero_info(
                f"Computed num_training_steps = {steps_per_epoch} * {args.num_epochs} = {num_training_steps}"
            )

        args.num_training_steps = num_training_steps
        setattr(model.args, "num_training_steps", num_training_steps)
        setattr(model.args, "lr_warmup_steps", int(getattr(args, "lr_warmup_steps", 0)))
        rank_zero_info(
            f"lr_warmup_steps set to {model.args.lr_warmup_steps} (will be used by scheduler if model registers one)"
        )

        base_lr = float(getattr(args, "learning_rate", 1e-4))
        weight_decay = float(getattr(args, "weight_decay", 0.0))

        def _configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr, weight_decay=weight_decay)
            lr_scheduler_name = getattr(args, "lr_scheduler", None)
            if lr_scheduler_name is None:
                return optimizer
            sched_fn = get_schedule_fn(
                lr_scheduler_name,
                num_training_steps=num_training_steps,
                warmup_steps=getattr(args, "lr_warmup_steps", 0),
            )
            scheduler = sched_fn(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }

        model.configure_optimizers = types.MethodType(_configure_optimizers, model)

    os.makedirs(os.path.join(args.storage_path), exist_ok=True)

    logger, run_name, run_id = build_experiment_logger(
        args,
        project_name=project_name,
        default_run_name="ragenco_mvc_run",
    )

    rank_zero_info(f"Logging to {logger.save_dir}/{run_name}/{run_id}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val/gap",
        mode="min",
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(
            logger.save_dir,
            run_name,
            run_id,
            "checkpoints",
        ),
    )

    lr_callback = LearningRateMonitor(logging_interval="step")
    callbacks_list = [TQDMProgressBar(refresh_rate=10), checkpoint_callback, lr_callback]

    if args.bf16:
        precision_strategy = "bf16"
    elif args.fp16:
        precision_strategy = 16
    else:
        precision_strategy = 32

    trainer = Trainer(
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        max_epochs=args.num_epochs,
        callbacks=callbacks_list,
        logger=logger,
        check_val_every_n_epoch=1,
        strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto",
        precision=precision_strategy,
        inference_mode=False,
    )

    ckpt_path = args.ckpt_path

    if args.do_train:
        if args.resume_weight_only and ckpt_path:
            print(
                f"Loading weights from {ckpt_path} into a new model instance and re-binding optim/scheduler..."
            )
            loaded_model = RAGenCOMVCModel.load_from_checkpoint(ckpt_path, param_args=args, strict=False)
            setattr(loaded_model.args, "num_training_steps", num_training_steps)
            setattr(loaded_model.args, "lr_warmup_steps", args.lr_warmup_steps)
            rank_zero_info(f"Re-set loaded_model.args.num_training_steps = {num_training_steps}")
            rank_zero_info(
                f"Re-set loaded_model.args.lr_warmup_steps = {loaded_model.args.lr_warmup_steps}"
            )
            loaded_model.configure_optimizers = types.MethodType(_configure_optimizers, loaded_model)
            model = loaded_model
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)

        if args.do_test:
            trainer.test(model)
    elif args.do_test:
        trainer.test(model, ckpt_path=ckpt_path)

    finalize_experiment_logger(trainer.logger)


if __name__ == "__main__":
    args = arg_parser()
    main(args)
