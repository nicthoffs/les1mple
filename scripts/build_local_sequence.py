from __future__ import annotations

import argparse

from les1mple.data import build_local_sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one LeWM-shaped tensor sequence from local OpenCS2 files."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--ticks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--buttons-column", default=None)
    args = parser.parse_args()

    sample = build_local_sequence(
        video_path=args.video,
        ticks_path=args.ticks,
        output_path=args.output,
        start_frame=args.start_frame,
        sequence_length=args.sequence_length,
        frame_step=args.frame_step,
        image_size=args.image_size,
        time_column=args.time_column,
        buttons_column=args.buttons_column,
    )

    print(f"wrote: {args.output}")
    print(f"pixels: {tuple(sample['pixels'].shape)} dtype={sample['pixels'].dtype}")
    print(f"action: {tuple(sample['action'].shape)}")


if __name__ == "__main__":
    main()
