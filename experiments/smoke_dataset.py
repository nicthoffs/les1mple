from __future__ import annotations

import argparse

from les1mple.data import LocalTensorSequenceDataset, SyntheticOpenCS2Dataset, action_dim


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the OpenCS2-to-LeWM dataset contract."
    )
    parser.add_argument("--source", choices=("synthetic", "local"), default="synthetic")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    torch = _require_torch()
    from torch.utils.data import DataLoader

    if args.source == "synthetic":
        dataset = SyntheticOpenCS2Dataset(
            sequence_length=args.sequence_length,
            image_size=args.image_size,
        )
    else:
        if args.data_dir is None:
            raise SystemExit("--data-dir is required when --source=local")
        dataset = LocalTensorSequenceDataset(args.data_dir)

    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size)))
    pixels = batch["pixels"]
    action = batch["action"]

    expected_pixels = 5
    expected_action = 3
    if pixels.ndim != expected_pixels:
        raise SystemExit(f"pixels should be B,T,C,H,W, got {tuple(pixels.shape)}")
    if action.ndim != expected_action:
        raise SystemExit(f"action should be B,T,A, got {tuple(action.shape)}")
    if pixels.shape[0] != action.shape[0] or pixels.shape[1] != action.shape[1]:
        raise SystemExit("pixels and action must agree on batch and sequence dimensions")
    if action.shape[-1] != action_dim():
        raise SystemExit(f"expected action width {action_dim()}, got {action.shape[-1]}")
    if not torch.is_floating_point(pixels) or not torch.is_floating_point(action):
        raise SystemExit("pixels and action must be floating point tensors")

    print(f"pixels: {tuple(pixels.shape)}")
    print(f"action: {tuple(action.shape)}")
    print(f"sample_ids: {list(batch['sample_id'])}")


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install project dependencies before running this smoke test."
        ) from exc

    return torch


if __name__ == "__main__":
    main()
