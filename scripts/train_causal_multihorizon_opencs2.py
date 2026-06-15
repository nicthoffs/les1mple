from __future__ import annotations

import argparse
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from les1mple.data import LocalTensorSequenceDataset, action_dim
from les1mple.data.transforms import ViewDeltaNormalizedDataset
from les1mple.models import build_causal_multi_horizon_lewm, build_lewm_model
from les1mple.training import CausalMultiHorizonLeWMLightningModule, split_dataset
from opencs2_train_utils import (
    check_checkpoint_path,
    configure_torch,
    copy_low_shape_args,
    dataset_sequence_length,
    fill_missing_args,
    load_low_checkpoint,
    load_or_compute_view_delta_stats,
    parameter_counts,
    parse_horizons,
    save_model_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a causal seq-to-seq multi-horizon latent predictor. For each "
            "horizon h, token t predicts z[t+h] from causal prefix z[:t+1] "
            "and action window a[t:t+h]."
        )
    )
    parser.add_argument("--data-dir", default="/tmp/les1mple-opencs2-full-batch-128")
    parser.add_argument("--run-dir", default="runs/opencs2-causal-multihorizon-lewm")
    parser.add_argument("--low-checkpoint", required=True)

    parser.add_argument("--horizons", default="1,2,4,8,16,32,50")
    parser.add_argument(
        "--num-positions",
        type=int,
        default=0,
        help=(
            "Number of causal output positions per clip. "
            "0 uses sequence_length - max(horizons)."
        ),
    )
    parser.add_argument("--loss-type", choices=("mse", "l1"), default="mse")
    parser.add_argument("--sigreg-weight", type=float, default=0.0)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--sigreg-frame-stride", type=int, default=1)

    parser.add_argument(
        "--causal-action-encoder",
        choices=("mean-pool", "transformer"),
        default="transformer",
    )
    parser.add_argument("--action-hidden-dim", type=int, default=None)
    parser.add_argument("--action-encoder-depth", type=int, default=2)
    parser.add_argument("--action-encoder-heads", type=int, default=4)
    parser.add_argument("--action-encoder-mlp-dim", type=int, default=512)
    parser.add_argument("--action-dropout", type=float, default=0.0)
    parser.add_argument("--causal-predictor-depth", type=int, default=4)
    parser.add_argument("--causal-predictor-heads", type=int, default=4)
    parser.add_argument("--causal-predictor-mlp-dim", type=int, default=512)
    parser.add_argument("--causal-predictor-dim-head", type=int, default=48)
    parser.add_argument("--causal-predictor-dropout", type=float, default=0.1)
    parser.add_argument("--causal-predictor-emb-dropout", type=float, default=0.0)
    parser.add_argument(
        "--freeze-low",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--freeze-low-unused",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--pixel-dtype",
        choices=("float32", "uint8"),
        default="uint8",
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
    )
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="les1mple")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--normalize-view-deltas",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--view-delta-stats-cache", default=None)

    # Low-level shape defaults. The checkpoint overrides these.
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

    args = parser.parse_args()
    args.horizons_tuple = parse_horizons(args.horizons)
    _validate_args(args)
    configure_torch()
    pl.seed_everything(args.seed, workers=True)

    check_checkpoint_path(Path(args.low_checkpoint))
    loaded_low = load_low_checkpoint(Path(args.low_checkpoint))
    low_args = loaded_low["args"]
    fill_missing_args(low_args, args)
    copy_low_shape_args(args, low_args)

    dataset = LocalTensorSequenceDataset(
        args.data_dir,
        pixel_dtype=args.pixel_dtype,
        mmap_load=args.mmap_load,
    )
    args.sequence_length = dataset_sequence_length(dataset)
    args.max_horizon = max(args.horizons_tuple)
    if args.num_positions == 0:
        args.num_positions = args.sequence_length - args.max_horizon
    if args.num_positions + args.max_horizon > args.sequence_length:
        raise SystemExit(
            "--num-positions + max(--horizons) must fit in the sequence "
            f"({args.sequence_length}); got num_positions={args.num_positions} "
            f"max_horizon={args.max_horizon}"
        )
    print(
        "causal_multihorizon_temporal_split:",
        f"sequence_length={args.sequence_length}",
        f"num_positions={args.num_positions}",
        f"horizons={','.join(str(h) for h in args.horizons_tuple)}",
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set = split_dataset(dataset, args.val_split, args.seed)
    view_delta_mean = None
    view_delta_std = None
    if args.normalize_view_deltas:
        view_delta_mean, view_delta_std = load_or_compute_view_delta_stats(
            train_set,
            Path(args.view_delta_stats_cache) if args.view_delta_stats_cache else None,
            checkpoint_mean=loaded_low["view_delta_mean"],
            checkpoint_std=loaded_low["view_delta_std"],
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

    low_model = build_lewm_model(low_args)
    low_model.load_state_dict(loaded_low["state_dict"], strict=True)
    world_model = build_causal_multi_horizon_lewm(
        low=low_model,
        action_dim=action_dim(),
        emb_dim=args.emb_dim,
        num_frames=args.num_positions,
        max_horizon=args.max_horizon,
        action_encoder_type=args.causal_action_encoder,
        action_hidden_dim=args.action_hidden_dim,
        action_encoder_depth=args.action_encoder_depth,
        action_encoder_heads=args.action_encoder_heads,
        action_encoder_mlp_dim=args.action_encoder_mlp_dim,
        action_dropout=args.action_dropout,
        predictor_depth=args.causal_predictor_depth,
        predictor_heads=args.causal_predictor_heads,
        predictor_mlp_dim=args.causal_predictor_mlp_dim,
        predictor_dim_head=args.causal_predictor_dim_head,
        predictor_dropout=args.causal_predictor_dropout,
        predictor_emb_dropout=args.causal_predictor_emb_dropout,
        freeze_low=args.freeze_low,
        freeze_low_unused=args.freeze_low_unused,
    )
    trainable, total = parameter_counts(world_model)
    print(f"parameters: trainable={trainable:,} total={total:,}")

    total_steps = args.max_steps
    warmup_steps = (
        0 if total_steps <= 1 else min(max(1, int(0.01 * total_steps)), total_steps - 1)
    )
    module = CausalMultiHorizonLeWMLightningModule(
        model=world_model,
        horizons=args.horizons_tuple,
        num_positions=args.num_positions,
        loss_type=args.loss_type,
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
    save_model_checkpoint(
        checkpoint_path,
        model=world_model,
        args=args,
        global_step=trainer.global_step,
        view_delta_mean=view_delta_mean,
        view_delta_std=view_delta_std,
    )
    print(f"checkpoint: {checkpoint_path}")


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.horizons_tuple) < 1:
        raise SystemExit("--horizons must be positive")
    if args.num_positions < 0:
        raise SystemExit("--num-positions must be non-negative")
    if args.sigreg_weight < 0:
        raise SystemExit("--sigreg-weight must be non-negative")
    if args.sigreg_frame_stride < 1:
        raise SystemExit("--sigreg-frame-stride must be positive")
    if args.action_hidden_dim is not None and args.action_hidden_dim < 1:
        raise SystemExit("--action-hidden-dim must be positive")
    if args.action_encoder_depth < 1:
        raise SystemExit("--action-encoder-depth must be positive")
    if args.action_encoder_heads < 1:
        raise SystemExit("--action-encoder-heads must be positive")
    if args.action_encoder_mlp_dim < 1:
        raise SystemExit("--action-encoder-mlp-dim must be positive")
    if args.action_dropout < 0:
        raise SystemExit("--action-dropout must be non-negative")
    if args.causal_predictor_depth < 1:
        raise SystemExit("--causal-predictor-depth must be positive")
    if args.causal_predictor_heads < 1:
        raise SystemExit("--causal-predictor-heads must be positive")
    if args.causal_predictor_mlp_dim < 1:
        raise SystemExit("--causal-predictor-mlp-dim must be positive")
    if args.causal_predictor_dim_head < 1:
        raise SystemExit("--causal-predictor-dim-head must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.accumulate_grad_batches < 1:
        raise SystemExit("--accumulate-grad-batches must be positive")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    if args.limit_val_batches < 1:
        raise SystemExit("--limit-val-batches must be positive")
    if args.checkpoint_every_n_steps < 0:
        raise SystemExit("--checkpoint-every-n-steps must be non-negative")


if __name__ == "__main__":
    main()
