from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from les1mple.data import LocalTensorSequenceDataset, action_dim
from les1mple.data.transforms import ViewDeltaNormalizedDataset
from les1mple.models import build_causal_multi_horizon_lewm, build_lewm_model
from les1mple.training import split_dataset
from scripts.eval_causal_multihorizon import _load_checkpoint, _view_delta_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detailed diagnostics for causal multi-horizon latent prediction."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pixel-dtype", choices=("float32", "uint8"), default="uint8")
    parser.add_argument("--mmap-load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-target-horizon", type=int, default=50)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint, train_args = _load_checkpoint(checkpoint_path)
    train_args.data_dir = args.data_dir
    train_args.pixel_dtype = args.pixel_dtype
    train_args.mmap_load = args.mmap_load

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    horizons = tuple(train_args.horizons_tuple)
    max_target_horizon = min(args.max_target_horizon, 150 - train_args.num_positions)
    target_horizons = tuple(range(1, max_target_horizon + 1))

    low = build_lewm_model(train_args)
    model = build_causal_multi_horizon_lewm(
        low=low,
        action_dim=action_dim(),
        emb_dim=train_args.emb_dim,
        num_frames=train_args.num_positions,
        max_horizon=max(horizons),
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
        args.data_dir,
        pixel_dtype=args.pixel_dtype,
        mmap_load=args.mmap_load,
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

    loader = DataLoader(
        Subset(val_set, range(min(args.num_samples, len(val_set)))),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.shuffle_seed)

    state = {
        horizon: _new_horizon_state(target_horizons)
        for horizon in horizons
    }
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixels"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True).nan_to_num_(0.0)
            z_all = model.encode_latents(pixels)
            z_prefix = z_all[:, : train_args.num_positions]
            copy = z_prefix
            shuffled_context = _shuffle_batch(z_prefix, generator)

            for horizon in horizons:
                action_windows = torch.stack(
                    [
                        action[:, offset : offset + train_args.num_positions]
                        for offset in range(horizon)
                    ],
                    dim=2,
                )
                shuffled_actions = _shuffle_batch(action_windows, generator)
                pred = model.predict_horizon(z_prefix, action_windows, horizon)
                pred_zero = model.predict_horizon(
                    z_prefix,
                    torch.zeros_like(action_windows),
                    horizon,
                )
                pred_shuffled_actions = model.predict_horizon(
                    z_prefix,
                    shuffled_actions,
                    horizon,
                )
                pred_shuffled_context = model.predict_horizon(
                    shuffled_context,
                    action_windows,
                    horizon,
                )
                target = z_all[:, horizon : horizon + train_args.num_positions]
                _update_state(
                    state[horizon],
                    pred=pred,
                    pred_zero=pred_zero,
                    pred_shuffled_actions=pred_shuffled_actions,
                    pred_shuffled_context=pred_shuffled_context,
                    copy=copy,
                    target=target,
                    z_all=z_all,
                    target_horizons=target_horizons,
                    num_positions=train_args.num_positions,
                )

    report = {
        "checkpoint": str(checkpoint_path),
        "global_step": checkpoint.get("global_step"),
        "action_encoder": getattr(train_args, "causal_action_encoder", "mean-pool"),
        "num_samples": min(args.num_samples, len(val_set)),
        "num_positions": train_args.num_positions,
        "horizons": list(horizons),
        "target_horizon_search": list(target_horizons),
        "diagnostics": {
            f"h{horizon}": _summarize_state(state[horizon], target_horizons)
            for horizon in horizons
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(_compact_report(report), indent=2, sort_keys=True))
    print(f"diagnostics: {output_path}")


def _new_horizon_state(target_horizons: tuple[int, ...]) -> dict[str, Any]:
    return {
        "pred_loss": 0.0,
        "copy_loss": 0.0,
        "zero_loss": 0.0,
        "shuffled_action_loss": 0.0,
        "shuffled_context_loss": 0.0,
        "pred_copy_loss": 0.0,
        "pred_zero_distance": 0.0,
        "pred_shuffled_action_distance": 0.0,
        "pred_shuffled_context_distance": 0.0,
        "count": 0,
        "elem_count": 0,
        "delta_dot": 0.0,
        "pred_delta_sq": 0.0,
        "target_delta_sq": 0.0,
        "pred_sum": None,
        "pred_sq_sum": None,
        "target_sum": None,
        "target_sq_sum": None,
        "copy_sum": None,
        "copy_sq_sum": None,
        "target_search": {h: 0.0 for h in target_horizons},
    }


def _update_state(
    state: dict[str, Any],
    *,
    pred: torch.Tensor,
    pred_zero: torch.Tensor,
    pred_shuffled_actions: torch.Tensor,
    pred_shuffled_context: torch.Tensor,
    copy: torch.Tensor,
    target: torch.Tensor,
    z_all: torch.Tensor,
    target_horizons: tuple[int, ...],
    num_positions: int,
) -> None:
    pred_loss = (pred - target).pow(2).mean(dim=-1)
    copy_loss = (copy - target).pow(2).mean(dim=-1)
    zero_loss = (pred_zero - target).pow(2).mean(dim=-1)
    shuffled_action_loss = (pred_shuffled_actions - target).pow(2).mean(dim=-1)
    shuffled_context_loss = (pred_shuffled_context - target).pow(2).mean(dim=-1)
    state["pred_loss"] += _sum(pred_loss)
    state["copy_loss"] += _sum(copy_loss)
    state["zero_loss"] += _sum(zero_loss)
    state["shuffled_action_loss"] += _sum(shuffled_action_loss)
    state["shuffled_context_loss"] += _sum(shuffled_context_loss)
    state["pred_copy_loss"] += _sum((pred - copy).pow(2).mean(dim=-1))
    state["pred_zero_distance"] += _sum((pred - pred_zero).pow(2).mean(dim=-1))
    state["pred_shuffled_action_distance"] += _sum(
        (pred - pred_shuffled_actions).pow(2).mean(dim=-1)
    )
    state["pred_shuffled_context_distance"] += _sum(
        (pred - pred_shuffled_context).pow(2).mean(dim=-1)
    )
    state["count"] += int(pred_loss.numel())

    pred_delta = pred - copy
    target_delta = target - copy
    state["delta_dot"] += float((pred_delta * target_delta).sum().cpu())
    state["pred_delta_sq"] += float(pred_delta.pow(2).sum().cpu())
    state["target_delta_sq"] += float(target_delta.pow(2).sum().cpu())
    state["elem_count"] += int(pred_delta.numel())

    _update_moments(state, "pred", pred)
    _update_moments(state, "target", target)
    _update_moments(state, "copy", copy)

    for target_horizon in target_horizons:
        candidate = z_all[:, target_horizon : target_horizon + num_positions]
        state["target_search"][target_horizon] += _sum(
            (pred - candidate).pow(2).mean(dim=-1)
        )


def _update_moments(
    state: dict[str, Any],
    prefix: str,
    value: torch.Tensor,
) -> None:
    flat = value.reshape(-1, value.shape[-1])
    sum_key = f"{prefix}_sum"
    sq_key = f"{prefix}_sq_sum"
    value_sum = flat.sum(dim=0).detach().cpu()
    value_sq_sum = flat.pow(2).sum(dim=0).detach().cpu()
    if state[sum_key] is None:
        state[sum_key] = value_sum
        state[sq_key] = value_sq_sum
    else:
        state[sum_key] += value_sum
        state[sq_key] += value_sq_sum


def _summarize_state(
    state: dict[str, Any],
    target_horizons: tuple[int, ...],
) -> dict[str, Any]:
    count = max(state["count"], 1)
    elem_count = max(state["elem_count"], 1)
    pred_mse = state["pred_loss"] / count
    copy_mse = state["copy_loss"] / count
    pred_delta_sq = state["pred_delta_sq"]
    target_delta_sq = state["target_delta_sq"]
    dot = state["delta_dot"]
    alpha = dot / pred_delta_sq if pred_delta_sq > 0.0 else 0.0
    alpha_clamped = max(0.0, min(1.0, alpha))
    blend_mse = (
        target_delta_sq
        + alpha_clamped * alpha_clamped * pred_delta_sq
        - 2.0 * alpha_clamped * dot
    ) / elem_count
    target_search = {
        str(h): state["target_search"][h] / count
        for h in target_horizons
    }
    best_target_horizon = min(target_horizons, key=lambda h: target_search[str(h)])

    return {
        "pred_mse": pred_mse,
        "copy_mse": copy_mse,
        "pred_vs_copy_improvement": _relative_improvement(pred_mse, copy_mse),
        "zero_action_mse": state["zero_loss"] / count,
        "shuffled_action_mse": state["shuffled_action_loss"] / count,
        "shuffled_context_mse": state["shuffled_context_loss"] / count,
        "real_vs_zero_delta": state["zero_loss"] / count - pred_mse,
        "real_vs_shuffled_action_delta": state["shuffled_action_loss"] / count
        - pred_mse,
        "real_vs_shuffled_context_delta": state["shuffled_context_loss"] / count
        - pred_mse,
        "pred_to_copy_mse": state["pred_copy_loss"] / count,
        "pred_to_zero_action_pred_mse": state["pred_zero_distance"] / count,
        "pred_to_shuffled_action_pred_mse": state["pred_shuffled_action_distance"]
        / count,
        "pred_to_shuffled_context_pred_mse": state[
            "pred_shuffled_context_distance"
        ]
        / count,
        "pred_delta_to_target_delta_cosine": dot
        / math.sqrt(max(pred_delta_sq * target_delta_sq, 1e-24)),
        "pred_delta_norm_over_target_delta_norm": math.sqrt(
            pred_delta_sq / max(target_delta_sq, 1e-24)
        ),
        "best_copy_to_pred_blend_alpha": alpha,
        "best_clamped_blend_mse": blend_mse,
        "best_clamped_blend_improvement_vs_copy": _relative_improvement(
            blend_mse,
            copy_mse,
        ),
        "pred_variance": _variance(state, "pred"),
        "target_variance": _variance(state, "target"),
        "copy_variance": _variance(state, "copy"),
        "best_target_horizon": best_target_horizon,
        "best_target_horizon_mse": target_search[str(best_target_horizon)],
        "target_horizon_mse": target_search,
    }


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "checkpoint": report["checkpoint"],
        "global_step": report["global_step"],
        "action_encoder": report["action_encoder"],
        "horizons": {},
    }
    for horizon, values in report["diagnostics"].items():
        compact["horizons"][horizon] = {
            "pred_mse": values["pred_mse"],
            "copy_mse": values["copy_mse"],
            "pred_vs_copy_improvement": values["pred_vs_copy_improvement"],
            "best_target_horizon": values["best_target_horizon"],
            "delta_cosine": values["pred_delta_to_target_delta_cosine"],
            "delta_norm_ratio": values["pred_delta_norm_over_target_delta_norm"],
            "best_blend_alpha": values["best_copy_to_pred_blend_alpha"],
            "best_blend_improvement": values[
                "best_clamped_blend_improvement_vs_copy"
            ],
            "shuffled_action_delta": values["real_vs_shuffled_action_delta"],
            "shuffled_context_delta": values["real_vs_shuffled_context_delta"],
            "pred_variance": values["pred_variance"],
            "target_variance": values["target_variance"],
        }
    return compact


def _shuffle_batch(value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    batch_size = value.shape[0]
    if batch_size <= 1:
        return value
    permutation = torch.randperm(batch_size, generator=generator, device="cpu")
    if torch.equal(permutation, torch.arange(batch_size)):
        permutation = permutation.roll(1)
    return value.index_select(0, permutation.to(value.device))


def _sum(value: torch.Tensor) -> float:
    return float(value.detach().sum().cpu())


def _variance(state: dict[str, Any], prefix: str) -> float:
    count = max(state["count"], 1)
    value_sum = state[f"{prefix}_sum"]
    value_sq_sum = state[f"{prefix}_sq_sum"]
    variance = value_sq_sum / count - (value_sum / count).pow(2)
    return float(variance.mean())


def _relative_improvement(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0
    return 1.0 - candidate / baseline


if __name__ == "__main__":
    main()
