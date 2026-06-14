from __future__ import annotations

import argparse
from io import BytesIO
import random
import tempfile
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers

from les1mple.data import ACTION_BUTTONS, build_local_sequence


DATASET_ID = "blanchon/opencs2_dataset_wds"
INTERESTING_BUTTONS = set(ACTION_BUTTONS) - {"score"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch OpenCS2 POV videos and build LeWM tensor samples."
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", default="/tmp/les1mple-real-opencs2")
    parser.add_argument("--cache-dir", default="/tmp/les1mple-hf-cache")
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--min-duration-s", type=float, default=2.0)
    parser.add_argument("--max-media-bytes", type=int, default=100_000_000)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--frame-step", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sampling", choices=("raw", "interesting"), default="raw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = {path.stem for path in output_dir.glob("*.pt")}

    index_path = hf_hub_download(
        DATASET_ID,
        "index/wds_samples.parquet",
        repo_type="dataset",
        cache_dir=args.cache_dir,
    )
    samples = pd.read_parquet(index_path)
    candidates = samples[
        (samples["mp4_size"] <= args.max_media_bytes)
        & (samples["duration_s"] >= args.min_duration_s)
    ]
    if existing_ids:
        candidates = candidates[~candidates["media_id"].isin(existing_ids)]

    if args.sampling == "raw":
        candidates = candidates.sample(frac=1, random_state=args.seed)
    else:
        candidates = candidates.sort_values("mp4_size")

    candidates = candidates.head(args.max_candidates)

    if existing_ids:
        print(
            f"skipping {len(existing_ids)} existing samples in {output_dir}", flush=True
        )

    session = requests.Session()
    session.headers.update(build_hf_headers())
    shard_urls: dict[str, str] = {}
    built = 0
    checked = 0
    for shard_path, shard_rows in candidates.groupby("shard_path", sort=False):
        if built >= args.num_samples:
            break

        shard_url = shard_urls.setdefault(
            shard_path,
            hf_hub_url(DATASET_ID, shard_path, repo_type="dataset"),
        )
        print(f"shard {shard_path} candidates={len(shard_rows)}", flush=True)

        for _, row in shard_rows.iterrows():
            if built >= args.num_samples:
                break
            checked += 1

            output_path = output_dir / f"{row['media_id']}.pt"
            print(
                f"checking {checked}/{len(candidates)} {row['media_id']} "
                f"bytes={row['mp4_size']}",
                flush=True,
            )

            try:
                with tempfile.TemporaryDirectory(prefix="les1mple-wds-") as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    ticks_path = tmpdir_path / row["ticks_member"]
                    video_path = tmpdir_path / row["mp4_member"]
                    _fetch_range_to_path(
                        session=session,
                        url=shard_url,
                        offset=int(row["ticks_offset"]),
                        size=int(row["ticks_size"]),
                        output_path=ticks_path,
                        timeout_s=args.timeout_s,
                    )

                    if args.sampling == "interesting":
                        ticks_bytes = ticks_path.read_bytes()
                        ticks = pd.read_parquet(BytesIO(ticks_bytes))
                        first_time = _first_interesting_time(ticks)
                        if first_time is None:
                            continue
                        start_frame = _start_frame_at_time(
                            first_time,
                            frame_count=int(row["frames"]),
                            fps=float(row["frames"]) / float(row["duration_s"]),
                            sequence_length=args.sequence_length,
                            frame_step=args.frame_step,
                        )
                    else:
                        start_frame = _random_start_frame(
                            rng=rng,
                            frame_count=int(row["frames"]),
                            sequence_length=args.sequence_length,
                            frame_step=args.frame_step,
                        )

                    _fetch_range_to_path(
                        session=session,
                        url=shard_url,
                        offset=int(row["mp4_offset"]),
                        size=int(row["mp4_size"]),
                        output_path=video_path,
                        timeout_s=args.timeout_s,
                    )
                    sample = build_local_sequence(
                        video_path=video_path,
                        ticks_path=ticks_path,
                        output_path=output_path,
                        start_frame=start_frame,
                        sequence_length=args.sequence_length,
                        frame_step=args.frame_step,
                        image_size=args.image_size,
                    )
            except (requests.RequestException, ValueError) as exc:
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


def _fetch_range_to_path(
    *,
    session: requests.Session,
    url: str,
    offset: int,
    size: int,
    output_path: Path,
    timeout_s: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    end = offset + size - 1
    response = session.get(
        url,
        headers={"Range": f"bytes={offset}-{end}"},
        timeout=timeout_s,
        stream=True,
    )
    with response:
        response.raise_for_status()
        if response.status_code != 206:
            raise ValueError(
                f"expected HTTP 206 range response, got {response.status_code}"
            )

        bytes_written = 0
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                bytes_written += len(chunk)

    if bytes_written != size:
        raise ValueError(f"expected {size} bytes from {url}, got {bytes_written}")


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
    first_time: float,
    *,
    frame_count: int,
    fps: float,
    sequence_length: int,
    frame_step: int,
) -> int:
    if fps <= 0:
        raise ValueError("could not infer video FPS from WDS index")

    last_needed = (sequence_length - 1) * frame_step
    max_start = max(0, frame_count - 1 - last_needed)
    return max(0, min(int(first_time * fps), max_start))


def _random_start_frame(
    *,
    rng: random.Random,
    frame_count: int,
    sequence_length: int,
    frame_step: int,
) -> int:
    last_needed = (sequence_length - 1) * frame_step
    max_start = max(0, frame_count - 1 - last_needed)
    return rng.randint(0, max_start) if max_start > 0 else 0


if __name__ == "__main__":
    main()
