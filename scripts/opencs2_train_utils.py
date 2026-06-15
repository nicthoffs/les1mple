from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from les1mple.data import LocalTensorSequenceDataset
from les1mple.data.transforms import compute_view_delta_stats


def configure_torch() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def check_checkpoint_path(path: Path) -> None:
    if path.exists():
        return
    candidates = sorted(path.parent.glob(f"{path.name}*.pt"))
    candidates.extend(sorted(path.parent.glob(f"{path.name}*.ckpt")))
    if candidates:
        choices = "\n".join(f"  {candidate}" for candidate in candidates[:8])
        raise SystemExit(
            f"checkpoint does not exist: {path}\n"
            f"Did you mean one of these?\n{choices}"
        )
    raise SystemExit(f"checkpoint does not exist: {path}")


def load_low_checkpoint(path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in raw and "args" in raw:
        return {
            "state_dict": raw["model"],
            "args": SimpleNamespace(**raw["args"]),
            "view_delta_mean": raw.get("view_delta_mean"),
            "view_delta_std": raw.get("view_delta_std"),
        }

    if "state_dict" in raw and "hyper_parameters" in raw:
        state_dict = {
            key.removeprefix("model."): value
            for key, value in raw["state_dict"].items()
            if key.startswith("model.")
        }
        hparams = dict(raw["hyper_parameters"])
        mean = None
        std = None
        stats_path = hparams.get("view_delta_stats_cache")
        if stats_path and Path(stats_path).exists():
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            mean = stats["mean"]
            std = stats["std"]
        return {
            "state_dict": state_dict,
            "args": SimpleNamespace(**hparams),
            "view_delta_mean": mean,
            "view_delta_std": std,
        }

    raise SystemExit(f"unsupported low-level checkpoint format: {path}")


def fill_missing_args(target: SimpleNamespace, defaults: argparse.Namespace) -> None:
    for key, value in vars(defaults).items():
        if not hasattr(target, key):
            setattr(target, key, value)


def copy_low_shape_args(args: argparse.Namespace, low_args: SimpleNamespace) -> None:
    for key in (
        "encoder",
        "vit_size",
        "patch_size",
        "img_size",
        "timm_model",
        "emb_dim",
        "encoder_chunk_size",
        "checkpoint_encoder",
        "action_smoothed_dim",
        "predictor_type",
        "predictor_depth",
        "predictor_heads",
        "predictor_mlp_dim",
        "predictor_dim_head",
        "predictor_dropout",
        "predictor_emb_dropout",
        "mamba_d_state",
        "mamba_headdim",
        "projection_hidden_dim",
        "channels_last",
    ):
        if hasattr(low_args, key):
            setattr(args, key, getattr(low_args, key))


def parse_horizons(raw: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(piece) for piece in raw.split(",") if piece.strip()}))
    if not horizons:
        raise SystemExit("--horizons must contain at least one integer")
    return horizons


def dataset_sequence_length(dataset: LocalTensorSequenceDataset) -> int:
    sample = dataset[0]
    pixels = sample["pixels"]
    if not torch.is_tensor(pixels):
        raise SystemExit("dataset sample pixels must be a tensor")
    return int(pixels.shape[0])


def load_or_compute_view_delta_stats(
    dataset,
    cache_path: Path | None,
    *,
    checkpoint_mean: torch.Tensor | None,
    checkpoint_std: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path is not None and cache_path.exists():
        stats = torch.load(cache_path, map_location="cpu", weights_only=True)
        return stats["mean"], stats["std"]

    if checkpoint_mean is not None and checkpoint_std is not None:
        mean = checkpoint_mean.cpu()
        std = checkpoint_std.cpu()
    else:
        mean, std = compute_view_delta_stats(dataset)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean, "std": std}, cache_path)
    return mean, std


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def save_model_checkpoint(
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
