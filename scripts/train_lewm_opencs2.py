from __future__ import annotations

import argparse
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from les1mple.data import LocalTensorSequenceDataset
from les1mple.data.transforms import (
    ViewDeltaNormalizedDataset,
    compute_view_delta_stats,
)
from les1mple.models import build_lewm_model
from les1mple.training import LeWMLightningModule, split_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LeWM on cached OpenCS2 tensors.")
    parser.add_argument("--data-dir", default="/tmp/les1mple-opencs2-full-batch-128")
    parser.add_argument("--run-dir", default="runs/opencs2-lewm")
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument(
        "--sigreg-frame-stride",
        type=int,
        default=1,
        help="Compute SIGReg on every Nth encoded frame.",
    )
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
    parser.add_argument(
        "--pixel-dtype",
        choices=("float32", "uint8"),
        default="uint8",
        help="Keep cached uint8 pixels uint8 until they are moved to the model.",
    )
    parser.add_argument(
        "--mmap-load",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--limit-val-batches", type=int, default=4)
    parser.add_argument("--num-sanity-val-steps", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--lightning-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write full Lightning .ckpt files with optimizer/scheduler state.",
    )
    parser.add_argument(
        "--checkpoint-every-n-steps",
        type=int,
        default=0,
        help="Save full Lightning checkpoints every N optimizer steps; 0 uses Lightning defaults.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="les1mple")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--normalize-view-deltas",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--view-delta-stats-cache", default=None)
    args = parser.parse_args()

    if args.accumulate_grad_batches < 1:
        raise SystemExit("--accumulate-grad-batches must be positive")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    if args.history_size < 1:
        raise SystemExit("--history-size must be positive")
    if args.sigreg_frame_stride < 1:
        raise SystemExit("--sigreg-frame-stride must be positive")
    if args.limit_val_batches < 1:
        raise SystemExit("--limit-val-batches must be positive")
    if args.checkpoint_every_n_steps < 0:
        raise SystemExit("--checkpoint-every-n-steps must be non-negative")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    pl.seed_everything(args.seed, workers=True)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = LocalTensorSequenceDataset(
        args.data_dir,
        pixel_dtype=args.pixel_dtype,
        mmap_load=args.mmap_load,
    )
    args.sequence_length = _dataset_sequence_length(dataset)
    if args.history_size >= args.sequence_length:
        raise SystemExit(
            "--history-size must be smaller than the cached sequence length "
            f"({args.sequence_length})"
        )
    args.target_horizon = args.sequence_length - args.history_size
    print(
        "temporal_split:",
        f"sequence_length={args.sequence_length}",
        f"history_size={args.history_size}",
        f"target_horizon={args.target_horizon}",
    )

    train_set, val_set = split_dataset(dataset, args.val_split, args.seed)
    view_delta_mean = None
    view_delta_std = None
    if args.normalize_view_deltas:
        view_delta_mean, view_delta_std = _load_or_compute_view_delta_stats(
            train_set,
            Path(args.view_delta_stats_cache) if args.view_delta_stats_cache else None,
        )
        train_set = ViewDeltaNormalizedDataset(
            train_set,
            view_delta_mean,
            view_delta_std,
            copy_action=False,
        )
        val_set = ViewDeltaNormalizedDataset(
            val_set,
            view_delta_mean,
            view_delta_std,
            copy_action=False,
        )
        print(
            "view_delta_normalization:",
            f"mean={view_delta_mean.tolist()}",
            f"std={view_delta_std.tolist()}",
        )

    prefetch_factor = 1 if args.num_workers > 0 else None
    persistent_workers = args.num_workers > 0
    train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    world_model = build_lewm_model(args)
    total_steps = args.max_steps
    warmup_steps = (
        0 if total_steps <= 1 else min(max(1, int(0.01 * total_steps)), total_steps - 1)
    )
    module = LeWMLightningModule(
        model=world_model,
        history_size=args.history_size,
        target_horizon=args.target_horizon,
        sigreg_weight=args.sigreg_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        sigreg_frame_stride=args.sigreg_frame_stride,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adamw_fused="auto",
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        hparams=vars(args),
    )
    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name or run_dir.name,
        save_dir=run_dir,
        log_model=False,
    )
    callbacks = []
    enable_checkpointing = args.lightning_checkpoints
    if args.lightning_checkpoints and args.checkpoint_every_n_steps:
        callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                filename="step-{step:06d}",
                every_n_train_steps=args.checkpoint_every_n_steps,
                save_top_k=-1,
                save_last=True,
                monitor=None,
            )
        )

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        default_root_dir=run_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
        num_sanity_val_steps=args.num_sanity_val_steps,
        check_val_every_n_epoch=None,
        val_check_interval=max(1, min(50, args.max_steps)),
        limit_val_batches=args.limit_val_batches,
        callbacks=callbacks,
    )
    trainer.fit(
        module,
        train_dataloaders=train,
        val_dataloaders=val,
        ckpt_path=args.resume_from_checkpoint,
    )

    checkpoint_path = run_dir / "final.pt"
    _save_model_checkpoint(
        checkpoint_path,
        model=world_model,
        args=args,
        global_step=trainer.global_step,
        view_delta_mean=view_delta_mean,
        view_delta_std=view_delta_std,
    )
    print(f"checkpoint: {checkpoint_path}")


def _save_model_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    args: argparse.Namespace,
    global_step: int,
    view_delta_mean: torch.Tensor | None,
    view_delta_std: torch.Tensor | None,
) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "global_step": global_step,
            "view_delta_mean": view_delta_mean,
            "view_delta_std": view_delta_std,
        },
        tmp_path,
    )
    tmp_path.replace(path)


def _dataset_sequence_length(dataset: LocalTensorSequenceDataset) -> int:
    sample = dataset[0]
    pixels = sample["pixels"]
    if not torch.is_tensor(pixels):
        raise SystemExit("dataset sample pixels must be a tensor")
    return int(pixels.shape[0])


def _load_or_compute_view_delta_stats(dataset, cache_path: Path | None):
    if cache_path is not None and cache_path.exists():
        stats = torch.load(cache_path, map_location="cpu", weights_only=True)
        return stats["mean"], stats["std"]

    mean, std = compute_view_delta_stats(dataset)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean, "std": std}, cache_path)
    return mean, std


if __name__ == "__main__":
    main()
