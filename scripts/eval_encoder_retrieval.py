from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from les1mple.data import LocalTensorSequenceDataset, action_dim
from les1mple.models import build_lewm_model
from les1mple.training import split_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained encoder geometry and nearest-neighbor retrieval."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="assets/eval")
    parser.add_argument("--num-clips", type=int, default=64)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-viz", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = SimpleNamespace(**checkpoint["args"])
    if args.data_dir is not None:
        train_args.data_dir = args.data_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LocalTensorSequenceDataset(train_args.data_dir)
    _, val_set = split_dataset(dataset, train_args.val_split, train_args.seed)
    max_clips = min(args.num_clips, len(val_set))

    model = build_lewm_model(train_args)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    records = _collect_records(
        model=model,
        dataset=Subset(val_set, range(max_clips)),
        frames_per_clip=args.frames_per_clip,
        batch_size=args.batch_size,
        device=torch.device(args.device),
    )
    metrics, retrieval = _compute_metrics(records)

    metrics_path = output_dir / "encoder_metrics.json"
    grid_path = output_dir / "encoder_neighbors.png"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    _save_neighbor_grid(records, retrieval, grid_path, num_viz=args.num_viz)

    print(json.dumps(metrics, indent=2))
    print(f"metrics: {metrics_path}")
    print(f"grid: {grid_path}")


def _collect_records(
    *,
    model,
    dataset: Subset,
    frames_per_clip: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    records: list[dict[str, object]] = []
    clip_offset = 0

    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixels"]
            selected = _frame_positions(pixels.size(1), frames_per_clip)
            pixels = pixels[:, selected].to(device)
            action = torch.zeros(
                pixels.size(0),
                pixels.size(1),
                action_dim(),
                dtype=pixels.dtype,
                device=device,
            )
            output = model.encode({"pixels": pixels, "action": action})
            emb = output["emb"].detach().cpu()
            pixels = pixels.detach().cpu()
            sample_ids = list(batch["sample_id"])

            for batch_index, sample_id in enumerate(sample_ids):
                for frame_index, frame_position in enumerate(selected):
                    records.append(
                        {
                            "clip_index": clip_offset + batch_index,
                            "sample_id": sample_id,
                            "frame_position": frame_position,
                            "emb": emb[batch_index, frame_index],
                            "image": pixels[batch_index, frame_index],
                        }
                    )
            clip_offset += len(sample_ids)

    return records


def _frame_positions(sequence_length: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("frames_per_clip must be positive")
    if count == 1:
        return [sequence_length // 2]
    positions = torch.linspace(0, sequence_length - 1, count).round().long().tolist()
    return sorted(set(int(position) for position in positions))


def _compute_metrics(
    records: list[dict[str, object]],
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    emb = torch.stack([record["emb"] for record in records])
    scores = _cosine_scores(emb, emb)
    scores.fill_diagonal_(-float("inf"))
    nn_indices = scores.argsort(dim=1, descending=True)

    clip_ids = torch.tensor([int(record["clip_index"]) for record in records])
    same_clip = clip_ids[:, None] == clip_ids[None, :]
    same_clip.fill_diagonal_(False)
    same_clip_top1 = same_clip.gather(1, nn_indices[:, :1]).any(dim=1)
    same_clip_top5 = same_clip.gather(1, nn_indices[:, :5]).any(dim=1)

    centered = emb - emb.mean(dim=0, keepdim=True)
    std = centered.std(dim=0)
    cov = centered.T @ centered / max(1, centered.size(0) - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
    eig_probs = eigvals / eigvals.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(eig_probs * eig_probs.clamp_min(1e-12).log()).sum()
    )

    metrics = {
        "num_frames": len(records),
        "num_clips": int(clip_ids.unique().numel()),
        "embedding_dim": emb.size(1),
        "latent_abs_mean": float(emb.abs().mean()),
        "latent_std_mean": float(std.mean()),
        "latent_std_min": float(std.min()),
        "latent_std_max": float(std.max()),
        "effective_rank": float(effective_rank),
        "top1_same_clip": float(same_clip_top1.float().mean()),
        "top5_same_clip": float(same_clip_top5.float().mean()),
        "mean_top1_cosine": float(scores.max(dim=1).values.mean()),
    }
    retrieval = {"nn_indices": nn_indices}
    return metrics, retrieval


def _cosine_scores(query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    query = torch.nn.functional.normalize(query, dim=-1)
    target = torch.nn.functional.normalize(target, dim=-1)
    return query @ target.T


def _save_neighbor_grid(
    records: list[dict[str, object]],
    retrieval: dict[str, torch.Tensor],
    path: Path,
    *,
    num_viz: int,
) -> None:
    count = min(num_viz, len(records))
    fig, axes = plt.subplots(count, 4, figsize=(8, 1.9 * count))
    if count == 1:
        axes = axes[None, :]

    for axis, title in zip(axes[0], ("query", "NN 1", "NN 2", "NN 3"), strict=True):
        axis.set_title(title)

    nn_indices = retrieval["nn_indices"]
    for row in range(count):
        indices = [row] + [int(index) for index in nn_indices[row, :3]]
        query = records[row]
        for col, index in enumerate(indices):
            record = records[index]
            axes[row, col].imshow(_image_to_numpy(record["image"]))
            axes[row, col].axis("off")
            same = "same" if record["clip_index"] == query["clip_index"] else "diff"
            axes[row, col].set_xlabel(f"{same} f{record['frame_position']}", fontsize=7)
        axes[row, 0].set_ylabel(f"clip {query['clip_index']}", rotation=0, labelpad=28)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _image_to_numpy(image: torch.Tensor):
    image = image.detach().cpu().clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


if __name__ == "__main__":
    main()
