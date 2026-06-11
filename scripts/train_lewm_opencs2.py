from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from les1mple.data import LocalTensorSequenceDataset
from les1mple.data.transforms import ViewDeltaNormalizedDataset, compute_view_delta_stats
from les1mple.models import build_lewm_model
from les1mple.training import lejepa_forward, split_dataset
from stable_worldmodel.wm.loss import SIGReg


class MetricsCsvCallback(Callback):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.file = None
        self.writer = None

    def on_train_start(self, trainer, pl_module) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=(
                "step",
                "train_loss",
                "train_pred_loss",
                "train_sigreg_loss",
                "val_loss",
                "val_pred_loss",
                "val_sigreg_loss",
                "cuda_peak_allocated_mb",
                "cuda_peak_reserved_mb",
            ),
        )
        self.writer.writeheader()
        self.file.flush()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        _reset_cuda_peak_memory()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self.writer or outputs is None:
            return
        self.writer.writerow(
            {
                "step": trainer.global_step,
                "train_loss": _item(outputs.get("raw_loss", outputs.get("loss"))),
                "train_pred_loss": _item(outputs.get("pred_loss")),
                "train_sigreg_loss": _item(outputs.get("sigreg_loss")),
                "val_loss": "",
                "val_pred_loss": "",
                "val_sigreg_loss": "",
                "cuda_peak_allocated_mb": _cuda_peak_allocated_mb(),
                "cuda_peak_reserved_mb": _cuda_peak_reserved_mb(),
            }
        )
        self.file.flush()

    def on_validation_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        _reset_cuda_peak_memory()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        if self.writer and outputs is not None:
            self.writer.writerow(
                {
                    "step": trainer.global_step,
                    "train_loss": "",
                    "train_pred_loss": "",
                    "train_sigreg_loss": "",
                    "val_loss": _item(outputs.get("loss")),
                    "val_pred_loss": _item(outputs.get("pred_loss")),
                    "val_sigreg_loss": _item(outputs.get("sigreg_loss")),
                    "cuda_peak_allocated_mb": _cuda_peak_allocated_mb(),
                    "cuda_peak_reserved_mb": _cuda_peak_reserved_mb(),
                }
            )
            self.file.flush()

    def on_train_end(self, trainer, pl_module) -> None:
        if self.file:
            self.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LeWM on cached OpenCS2 tensors.")
    parser.add_argument("--data-dir", default="/tmp/les1mple-opencs2-preview-batch-128")
    parser.add_argument("--run-dir", default="/tmp/les1mple-runs/opencs2-lewm")
    parser.add_argument(
        "--encoder",
        choices=("vit-hf", "timm", "tiny-cnn"),
        default="vit-hf",
    )
    parser.add_argument("--vit-size", default="tiny")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--timm-model", default="vit_tiny_patch16_224")
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--encoder-chunk-size", type=int, default=0)
    parser.add_argument(
        "--checkpoint-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--num-preds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--action-smoothed-dim", type=int, default=10)
    parser.add_argument(
        "--predictor-type",
        choices=("transformer", "mamba3"),
        default="transformer",
    )
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--predictor-heads", type=int, default=16)
    parser.add_argument("--predictor-mlp-dim", type=int, default=2048)
    parser.add_argument("--predictor-dim-head", type=int, default=64)
    parser.add_argument("--predictor-dropout", type=float, default=0.1)
    parser.add_argument("--predictor-emb-dropout", type=float, default=0.0)
    parser.add_argument("--mamba-d-state", type=int, default=64)
    parser.add_argument("--mamba-headdim", type=int, default=64)
    parser.add_argument("--projection-hidden-dim", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normalize-view-deltas",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.accumulate_grad_batches < 1:
        raise SystemExit("--accumulate-grad-batches must be positive")

    pl.seed_everything(args.seed, workers=True)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = LocalTensorSequenceDataset(args.data_dir)
    train_set, val_set = split_dataset(dataset, args.val_split, args.seed)
    view_delta_mean = None
    view_delta_std = None
    if args.normalize_view_deltas:
        view_delta_mean, view_delta_std = compute_view_delta_stats(train_set)
        train_set = ViewDeltaNormalizedDataset(train_set, view_delta_mean, view_delta_std)
        val_set = ViewDeltaNormalizedDataset(val_set, view_delta_mean, view_delta_std)
        print(
            "view_delta_normalization:",
            f"mean={view_delta_mean.tolist()}",
            f"std={view_delta_std.tolist()}",
        )
    train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    world_model = build_lewm_model(args)

    total_steps = args.max_steps
    warmup_steps = (
        0
        if total_steps <= 1
        else min(max(1, int(0.01 * total_steps)), total_steps - 1)
    )
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": {
                "type": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": total_steps,
            },
            "interval": "step",
        },
    }

    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj),
        forward=partial(
            lejepa_forward,
            history_size=args.history_size,
            num_preds=args.num_preds,
            sigreg_weight=args.sigreg_weight,
            grad_accum_steps=args.accumulate_grad_batches,
        ),
        optim=optimizers,
        hparams=vars(args),
    )
    data_module = spt.data.DataModule(train=train, val=val)
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        default_root_dir=run_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=None,
        val_check_interval=max(1, min(50, args.max_steps)),
        callbacks=[MetricsCsvCallback(run_dir / "metrics.csv")],
    )
    trainer.fit(module, datamodule=data_module)

    checkpoint_path = run_dir / "final.pt"
    torch.save(
        {
            "model": world_model.state_dict(),
            "args": vars(args),
            "global_step": trainer.global_step,
            "view_delta_mean": view_delta_mean,
            "view_delta_std": view_delta_std,
        },
        checkpoint_path,
    )
    print(f"checkpoint: {checkpoint_path}")
    print(f"metrics: {run_dir / 'metrics.csv'}")


def _item(value) -> float | str:
    if value is None:
        return ""
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _reset_cuda_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_allocated_mb() -> float | str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.max_memory_allocated() / 1024**2


def _cuda_peak_reserved_mb() -> float | str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.max_memory_reserved() / 1024**2


if __name__ == "__main__":
    main()
