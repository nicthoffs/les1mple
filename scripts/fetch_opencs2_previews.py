from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import pandas as pd
from huggingface_hub import hf_hub_download

from les1mple.data import ACTION_BUTTONS, build_local_sequence


DATASET_ID = "blanchon/opencs2_dataset"
INTERESTING_BUTTONS = set(ACTION_BUTTONS) - {"score"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch small OpenCS2 preview clips and build LeWM tensor samples."
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", default="/tmp/les1mple-real-opencs2")
    parser.add_argument("--cache-dir", default="/tmp/les1mple-hf-cache")
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--min-duration-s", type=float, default=2.0)
    parser.add_argument("--max-preview-bytes", type=int, default=5_000_000)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sampling", choices=("raw", "interesting"), default="raw")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = hf_hub_download(
        DATASET_ID,
        "index/pov_rounds.parquet",
        repo_type="dataset",
        cache_dir=args.cache_dir,
    )
    pov_rounds = pd.read_parquet(index_path)
    candidates = pov_rounds[
        (pov_rounds["preview_video_bytes"] <= args.max_preview_bytes)
        & (pov_rounds["duration_s"] >= args.min_duration_s)
    ]
    if args.sampling == "raw":
        candidates = candidates.sample(frac=1, random_state=args.seed)
    else:
        candidates = candidates.sort_values("preview_video_bytes")

    built = 0
    checked = 0
    for _, row in candidates.iterrows():
        if built >= args.num_samples or checked >= args.max_candidates:
            break
        checked += 1

        output_path = output_dir / f"{row['media_id']}-preview.pt"
        if output_path.exists():
            built += 1
            print(f"{built}/{args.num_samples} reuse {output_path.name}", flush=True)
            continue

        print(
            f"checking {checked}/{args.max_candidates} {row['media_id']} "
            f"bytes={row['preview_video_bytes']}",
            flush=True,
        )
        ticks_repo_path = _repo_path(row["ticks_parquet_path"])
        ticks_path = hf_hub_download(
            DATASET_ID,
            ticks_repo_path,
            repo_type="dataset",
            cache_dir=args.cache_dir,
        )
        video_repo_path = _repo_path(row["preview_video"]["path"])
        video_path = hf_hub_download(
            DATASET_ID,
            video_repo_path,
            repo_type="dataset",
            cache_dir=args.cache_dir,
        )
        if args.sampling == "interesting":
            ticks = pd.read_parquet(ticks_path)
            first_time = _first_interesting_time(ticks)
            if first_time is None:
                continue
            start_frame = _start_frame_at_time(
                video_path,
                first_time,
                sequence_length=args.sequence_length,
                frame_step=args.frame_step,
            )
        else:
            start_frame = _random_start_frame(
                video_path,
                rng=rng,
                sequence_length=args.sequence_length,
                frame_step=args.frame_step,
            )
        try:
            sample = build_local_sequence(
                video_path=video_path,
                ticks_path=ticks_path,
                output_path=output_path,
                start_frame=start_frame,
                sequence_length=args.sequence_length,
                frame_step=args.frame_step,
                image_size=args.image_size,
            )
        except ValueError as exc:
            print(f"skip {row['media_id']}: {exc}", flush=True)
            continue

        built += 1
        print(
            f"{built}/{args.num_samples} {output_path.name} "
            f"pixels={tuple(sample['pixels'].shape)} dtype={sample['pixels'].dtype} "
            f"action={tuple(sample['action'].shape)}",
            flush=True,
        )

    if built < args.num_samples:
        raise SystemExit(
            f"built {built} samples after checking {checked} candidates; "
            "raise --max-candidates or relax filters"
        )


def _repo_path(hf_path: str) -> str:
    return hf_path.split("@main/", 1)[1] if "@main/" in hf_path else hf_path


def _first_interesting_time(ticks: pd.DataFrame) -> float | None:
    if "active" not in ticks or "t" not in ticks:
        return None

    for _, row in ticks.iterrows():
        buttons = {_normalize_button(button) for button in row["active"]}
        if buttons & INTERESTING_BUTTONS:
            return float(row["t"])
    return None


def _normalize_button(button: object) -> str:
    return str(button).lower().lstrip("+").replace("_", "")


def _start_frame_at_time(
    video_path: str,
    first_time: float,
    *,
    sequence_length: int,
    frame_step: int,
) -> int:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"could not open video {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    if fps <= 0:
        raise ValueError(f"could not read FPS for {video_path}")

    last_needed = (sequence_length - 1) * frame_step
    max_start = max(0, frame_count - 1 - last_needed)
    return max(0, min(int(first_time * fps), max_start))


def _random_start_frame(
    video_path: str,
    *,
    rng: random.Random,
    sequence_length: int,
    frame_step: int,
) -> int:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"could not open video {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    last_needed = (sequence_length - 1) * frame_step
    max_start = max(0, frame_count - 1 - last_needed)
    return rng.randint(0, max_start) if max_start > 0 else 0


if __name__ == "__main__":
    main()
