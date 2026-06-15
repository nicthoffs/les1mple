from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from les1mple.data import LocalTensorSequenceDataset, action_dim
from les1mple.data.transforms import ViewDeltaNormalizedDataset
from les1mple.models import build_causal_multi_horizon_lewm, build_lewm_model
from les1mple.training import split_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate causal multi-horizon LeWM latent prediction baselines."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pixel-dtype", choices=("float32", "uint8"), default=None)
    parser.add_argument(
        "--mmap-load",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint, train_args = _load_checkpoint(checkpoint_path)
    if args.data_dir is not None:
        train_args.data_dir = args.data_dir
    if args.pixel_dtype is not None:
        train_args.pixel_dtype = args.pixel_dtype
    if args.mmap_load is not None:
        train_args.mmap_load = args.mmap_load

    default_output_base = (
        checkpoint_path.parent.parent
        if checkpoint_path.parent.name == "checkpoints"
        else checkpoint_path.parent
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_base / f"eval-step{checkpoint['global_step']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    low = build_lewm_model(train_args)
    model = build_causal_multi_horizon_lewm(
        low=low,
        action_dim=action_dim(),
        emb_dim=train_args.emb_dim,
        num_frames=train_args.num_positions,
        max_horizon=max(train_args.horizons_tuple),
        action_encoder_type=getattr(train_args, "causal_action_encoder", "mean-pool"),
        action_hidden_dim=getattr(train_args, "action_hidden_dim", None),
        action_encoder_depth=getattr(train_args, "action_encoder_depth", 2),
        action_encoder_heads=getattr(train_args, "action_encoder_heads", 4),
        action_encoder_mlp_dim=getattr(train_args, "action_encoder_mlp_dim", 512),
        action_dropout=getattr(train_args, "action_dropout", 0.0),
        predictor_depth=getattr(train_args, "causal_predictor_depth", 4),
        predictor_heads=getattr(train_args, "causal_predictor_heads", 4),
        predictor_mlp_dim=getattr(train_args, "causal_predictor_mlp_dim", 512),
        predictor_dim_head=getattr(train_args, "causal_predictor_dim_head", 48),
        predictor_dropout=getattr(train_args, "causal_predictor_dropout", 0.1),
        predictor_emb_dropout=getattr(train_args, "causal_predictor_emb_dropout", 0.0),
        freeze_low=getattr(train_args, "freeze_low", True),
        freeze_low_unused=getattr(train_args, "freeze_low_unused", True),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    dataset = LocalTensorSequenceDataset(
        train_args.data_dir,
        pixel_dtype=getattr(train_args, "pixel_dtype", "uint8"),
        mmap_load=getattr(train_args, "mmap_load", False),
    )
    _, val_set = split_dataset(dataset, train_args.val_split, train_args.seed)
    view_delta_mean, view_delta_std = _view_delta_stats(checkpoint, train_args)
    if getattr(train_args, "normalize_view_deltas", False):
        if view_delta_mean is None or view_delta_std is None:
            raise SystemExit("checkpoint is missing view-delta normalization stats")
        val_set = ViewDeltaNormalizedDataset(
            val_set,
            view_delta_mean,
            view_delta_std,
            copy_action=False,
        )

    num_samples = min(args.num_samples, len(val_set))
    loader = DataLoader(
        Subset(val_set, range(num_samples)),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.shuffle_seed)

    report = _evaluate(
        model,
        loader,
        device=device,
        horizons=train_args.horizons_tuple,
        num_positions=train_args.num_positions,
        max_horizon=max(train_args.horizons_tuple),
        loss_type=train_args.loss_type,
        generator=generator,
    )
    report["summary"].update(
        {
            "checkpoint": str(checkpoint_path),
            "global_step": checkpoint.get("global_step"),
            "num_clips": num_samples,
            "num_positions": train_args.num_positions,
            "horizons": list(train_args.horizons_tuple),
            "loss_type": train_args.loss_type,
        }
    )

    output_path = output_dir / "causal_multihorizon_eval.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"causal_multihorizon_eval: {output_path}")


@torch.inference_mode()
def _evaluate(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    horizons: tuple[int, ...],
    num_positions: int,
    max_horizon: int,
    loss_type: str,
    generator: torch.Generator,
) -> dict[str, Any]:
    per_horizon_totals = {horizon: _new_totals() for horizon in horizons}
    totals = _new_totals()
    total_points = 0

    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True).nan_to_num_(0.0)
        if num_positions + max_horizon > pixels.shape[1]:
            raise ValueError(
                "num positions plus max horizon does not fit in batch sequence: "
                f"num_positions={num_positions} max_horizon={max_horizon} "
                f"sequence_length={pixels.shape[1]}"
            )

        z_all = model.encode_latents(pixels)
        z_prefix = z_all[:, :num_positions]
        copy_pred = z_prefix

        for horizon in horizons:
            z_target = z_all[:, horizon : horizon + num_positions]
            action_windows = _causal_action_windows(
                action,
                num_positions=num_positions,
                horizon=horizon,
            )
            real_pred = model.predict_horizon(z_prefix, action_windows, horizon)
            shuffled_pred = model.predict_horizon(
                z_prefix,
                _shuffled_actions(action_windows, generator=generator),
                horizon,
            )
            zero_pred = model.predict_horizon(
                z_prefix,
                torch.zeros_like(action_windows),
                horizon,
            )

            real_loss = _latent_loss_per_step(real_pred, z_target, loss_type)
            shuffled_loss = _latent_loss_per_step(shuffled_pred, z_target, loss_type)
            zero_loss = _latent_loss_per_step(zero_pred, z_target, loss_type)
            copy_loss = _latent_loss_per_step(copy_pred, z_target, loss_type)
            action_sensitivity = (real_pred - shuffled_pred).pow(2).mean(dim=-1)
            zero_sensitivity = (real_pred - zero_pred).pow(2).mean(dim=-1)

            _add_losses(
                per_horizon_totals[horizon],
                real_loss,
                shuffled_loss,
                zero_loss,
                copy_loss,
                action_sensitivity,
                zero_sensitivity,
            )
            _add_losses(
                totals,
                real_loss,
                shuffled_loss,
                zero_loss,
                copy_loss,
                action_sensitivity,
                zero_sensitivity,
            )
            total_points += int(real_loss.numel())

    summary = _summarize_totals(totals, total_points)
    per_horizon = {
        f"h{horizon}": _summarize_totals(values, values["count"])
        for horizon, values in per_horizon_totals.items()
    }
    return {"summary": summary, "per_horizon": per_horizon}


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], SimpleNamespace]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in raw and "args" in raw:
        checkpoint = raw
    elif "state_dict" in raw and "hyper_parameters" in raw:
        checkpoint = {
            "model": {
                key.removeprefix("model."): value
                for key, value in raw["state_dict"].items()
                if key.startswith("model.")
            },
            "args": dict(raw["hyper_parameters"]),
            "global_step": raw.get("global_step"),
            "view_delta_mean": raw.get("view_delta_mean"),
            "view_delta_std": raw.get("view_delta_std"),
        }
    else:
        raise ValueError(f"{path} is not a causal multi-horizon LeWM checkpoint")

    args = SimpleNamespace(**checkpoint["args"])
    if not hasattr(args, "horizons_tuple"):
        args.horizons_tuple = _parse_horizons(args.horizons)
    else:
        args.horizons_tuple = tuple(args.horizons_tuple)
    _fill_missing_from_state(args, checkpoint["model"])
    return checkpoint, args


def _parse_horizons(raw: str) -> tuple[int, ...]:
    return tuple(sorted({int(piece) for piece in raw.split(",") if piece.strip()}))


def _fill_missing_from_state(
    args: SimpleNamespace,
    state_dict: dict[str, torch.Tensor],
) -> None:
    if not hasattr(args, "history_size"):
        low_pos_embedding = state_dict.get("low.predictor.pos_embedding")
        if low_pos_embedding is None:
            raise ValueError("checkpoint args are missing history_size")
        args.history_size = int(low_pos_embedding.shape[1])
    if not hasattr(args, "num_positions"):
        pos_embedding = state_dict.get("predictor.pos_embedding")
        if pos_embedding is None:
            raise ValueError("checkpoint args are missing num_positions")
        args.num_positions = int(pos_embedding.shape[1])
    if not hasattr(args, "emb_dim"):
        pos_embedding = state_dict.get("predictor.pos_embedding")
        if pos_embedding is None:
            raise ValueError("checkpoint args are missing emb_dim")
        args.emb_dim = int(pos_embedding.shape[-1])
    if not hasattr(args, "max_horizon"):
        horizon_embedding = state_dict.get("horizon_embedding.weight")
        if horizon_embedding is None:
            raise ValueError("checkpoint args are missing max_horizon")
        args.max_horizon = int(horizon_embedding.shape[0] - 1)
    if not hasattr(args, "action_dim"):
        args.action_dim = action_dim()
    if not hasattr(args, "causal_action_encoder"):
        args.causal_action_encoder = (
            "transformer"
            if "action_encoder.cls_token" in state_dict
            else "mean-pool"
        )


def _view_delta_stats(
    checkpoint: dict[str, Any],
    train_args: SimpleNamespace,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if checkpoint.get("view_delta_mean") is not None:
        return checkpoint["view_delta_mean"], checkpoint["view_delta_std"]
    stats_path = getattr(train_args, "view_delta_stats_cache", None)
    if stats_path and Path(stats_path).exists():
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        return stats["mean"], stats["std"]
    low_checkpoint = getattr(train_args, "low_checkpoint", None)
    if low_checkpoint and Path(low_checkpoint).exists():
        raw = torch.load(low_checkpoint, map_location="cpu", weights_only=False)
        if raw.get("view_delta_mean") is not None:
            return raw["view_delta_mean"], raw["view_delta_std"]
    return None, None


def _causal_action_windows(
    action: torch.Tensor,
    *,
    num_positions: int,
    horizon: int,
) -> torch.Tensor:
    return torch.stack(
        [
            action[:, offset : offset + num_positions]
            for offset in range(horizon)
        ],
        dim=2,
    )


def _shuffled_actions(
    action_windows: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size = action_windows.shape[0]
    if batch_size <= 1:
        return action_windows
    permutation = torch.randperm(batch_size, generator=generator, device="cpu")
    if torch.equal(permutation, torch.arange(batch_size)):
        permutation = permutation.roll(1)
    return action_windows.index_select(0, permutation.to(action_windows.device))


def _latent_loss_per_step(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "l1":
        return (pred - target).abs().mean(dim=-1)
    if loss_type == "mse":
        return (pred - target).pow(2).mean(dim=-1)
    raise ValueError(f"unknown loss type {loss_type!r}")


def _new_totals() -> dict[str, float]:
    return {
        "pred": 0.0,
        "shuffled_action": 0.0,
        "zero_action": 0.0,
        "copy_last": 0.0,
        "action_sensitivity": 0.0,
        "zero_action_sensitivity": 0.0,
        "count": 0.0,
    }


def _add_losses(
    totals: dict[str, float],
    real_loss: torch.Tensor,
    shuffled_loss: torch.Tensor,
    zero_loss: torch.Tensor,
    copy_loss: torch.Tensor,
    action_sensitivity: torch.Tensor,
    zero_action_sensitivity: torch.Tensor,
) -> None:
    totals["pred"] += _sum(real_loss)
    totals["shuffled_action"] += _sum(shuffled_loss)
    totals["zero_action"] += _sum(zero_loss)
    totals["copy_last"] += _sum(copy_loss)
    totals["action_sensitivity"] += _sum(action_sensitivity)
    totals["zero_action_sensitivity"] += _sum(zero_action_sensitivity)
    totals["count"] += int(real_loss.numel())


def _summarize_totals(totals: dict[str, float], count: int | float) -> dict[str, float]:
    count = max(float(count), 1.0)
    pred = totals["pred"] / count
    shuffled = totals["shuffled_action"] / count
    zero = totals["zero_action"] / count
    copy = totals["copy_last"] / count
    return {
        "pred_mse": pred,
        "copy_last_mse": copy,
        "shuffled_action_mse": shuffled,
        "zero_action_mse": zero,
        "pred_vs_copy_mse_improvement": _relative_improvement(pred, copy),
        "real_vs_shuffled_action_mse_delta": shuffled - pred,
        "real_vs_zero_action_mse_delta": zero - pred,
        "action_sensitivity_mse": totals["action_sensitivity"] / count,
        "zero_action_sensitivity_mse": totals["zero_action_sensitivity"] / count,
    }


def _sum(value: torch.Tensor) -> float:
    return float(value.detach().sum().cpu())


def _relative_improvement(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0
    return 1.0 - candidate / baseline


if __name__ == "__main__":
    main()
