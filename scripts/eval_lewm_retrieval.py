from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from les1mple.data import LocalTensorSequenceDataset
from les1mple.data.transforms import ViewDeltaNormalizedDataset
from les1mple.models import build_lewm_model
from les1mple.training import split_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LeWM final-step latent retrieval on cached OpenCS2 clips."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-viz", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = SimpleNamespace(**checkpoint["args"])
    target_horizon = getattr(train_args, "target_horizon", None)
    if target_horizon is None:
        target_horizon = train_args.num_preds
    if args.data_dir is not None:
        train_args.data_dir = args.data_dir

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset = LocalTensorSequenceDataset(train_args.data_dir)
    _, val_set = split_dataset(dataset, train_args.val_split, train_args.seed)
    if checkpoint.get("view_delta_mean") is not None:
        val_set = ViewDeltaNormalizedDataset(
            val_set,
            checkpoint["view_delta_mean"],
            checkpoint["view_delta_std"],
        )

    model = build_lewm_model(train_args)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    records = []
    max_samples = min(args.num_samples, len(val_set))
    loader = DataLoader(
        torch.utils.data.Subset(val_set, range(max_samples)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    with torch.inference_mode():
        for batch in loader:
            batch = _to_device(batch, device)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            output = model.encode(batch)
            emb = output["emb"]
            act_emb = output["act_emb"]
            ctx_emb = emb[:, : train_args.history_size]
            ctx_act = act_emb[:, : train_args.history_size]
            pred = model.predict(ctx_emb, ctx_act)[:, -1]
            copy = ctx_emb[:, -1]
            target = emb[:, train_args.history_size - 1 + target_horizon]

            pixels = batch["pixels"].detach().cpu()
            sample_ids = list(batch["sample_id"])
            for index in range(pred.size(0)):
                records.append(
                    {
                        "sample_id": sample_ids[index],
                        "pred": pred[index].detach().cpu(),
                        "copy": copy[index].detach().cpu(),
                        "target": target[index].detach().cpu(),
                        "context_frame": pixels[index, train_args.history_size - 1],
                        "target_frame": pixels[
                            index, train_args.history_size - 1 + target_horizon
                        ],
                    }
                )

    metrics, retrieval = _compute_metrics(records)
    metrics_path = output_dir / "retrieval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    grid_path = output_dir / "retrieval_grid.png"
    _save_retrieval_grid(records, retrieval, grid_path, num_viz=args.num_viz)

    print(json.dumps(metrics, indent=2))
    print(f"metrics: {metrics_path}")
    print(f"grid: {grid_path}")


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _compute_metrics(records: list[dict[str, object]]) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    pred = torch.stack([record["pred"] for record in records])
    copy = torch.stack([record["copy"] for record in records])
    target = torch.stack([record["target"] for record in records])

    pred_scores = _cosine_scores(pred, target)
    copy_scores = _cosine_scores(copy, target)
    pred_ranks = _true_ranks(pred_scores)
    copy_ranks = _true_ranks(copy_scores)

    metrics = {
        "num_samples": len(records),
        "pred_mse": float((pred - target).pow(2).mean()),
        "copy_last_mse": float((copy - target).pow(2).mean()),
        "pred_mean_rank": float(pred_ranks.float().mean()),
        "copy_last_mean_rank": float(copy_ranks.float().mean()),
        "pred_top1": float((pred_ranks <= 1).float().mean()),
        "copy_last_top1": float((copy_ranks <= 1).float().mean()),
        "pred_top5": float((pred_ranks <= 5).float().mean()),
        "copy_last_top5": float((copy_ranks <= 5).float().mean()),
    }
    retrieval = {
        "pred_nn": pred_scores.argmax(dim=1),
        "copy_nn": copy_scores.argmax(dim=1),
        "pred_ranks": pred_ranks,
        "copy_ranks": copy_ranks,
    }
    return metrics, retrieval


def _cosine_scores(query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    query = torch.nn.functional.normalize(query, dim=-1)
    target = torch.nn.functional.normalize(target, dim=-1)
    return query @ target.T


def _true_ranks(scores: torch.Tensor) -> torch.Tensor:
    true_scores = scores.diag()[:, None]
    return (scores > true_scores).sum(dim=1) + 1


def _save_retrieval_grid(
    records: list[dict[str, object]],
    retrieval: dict[str, torch.Tensor],
    path: Path,
    *,
    num_viz: int,
) -> None:
    count = min(num_viz, len(records))
    fig, axes = plt.subplots(count, 4, figsize=(8, 2 * count))
    if count == 1:
        axes = axes[None, :]

    columns = ("context", "true future", "pred NN", "copy NN")
    for axis, column in zip(axes[0], columns, strict=True):
        axis.set_title(column)

    for row in range(count):
        pred_index = int(retrieval["pred_nn"][row])
        copy_index = int(retrieval["copy_nn"][row])
        images = (
            records[row]["context_frame"],
            records[row]["target_frame"],
            records[pred_index]["target_frame"],
            records[copy_index]["target_frame"],
        )
        for col, image in enumerate(images):
            axes[row, col].imshow(_image_to_numpy(image))
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(
            f"rank p/c: {int(retrieval['pred_ranks'][row])}/"
            f"{int(retrieval['copy_ranks'][row])}",
            rotation=0,
            labelpad=36,
            va="center",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _image_to_numpy(image: torch.Tensor):
    image = image.detach().cpu().clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


if __name__ == "__main__":
    main()
