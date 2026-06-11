# AGENTS.md

## Project

LeS1mple is a hierarchical JEPA world model for long-horizon Counter-Strike dynamics. It uses OpenCS2 preview clips and actions to train a latent predictor for POV gameplay.

The encoder sees only the current POV frame. Do not expect it to encode hidden long-term memory. The predictor is responsible for memory, dynamics, and counterfactual futures.

## Repository Boundaries

- `src/les1mple/`: reusable package code.
- `scripts/`: maintained entrypoints that are expected to keep working.
- `experiments/`: quick checks, smoke tests, and disposable probes.
- `data/`, `.cache/`, run directories, checkpoints, and logs are local artifacts.

## Running

Use `uv` from the repo root:

```bash
uv --cache-dir /tmp/uv-cache run --extra train ...
```

Smoke checks:

```bash
uv --cache-dir /tmp/uv-cache run --extra train python experiments/smoke_dataset.py
uv --cache-dir /tmp/uv-cache run --extra train python experiments/train_lewm_smoke.py --source synthetic --steps 1
```

## Important Paths

- Current 10k cache target: `/home/nic/Work/les1mple/data/opencs2-preview-raw10000-seq150-step4`
- Completed 1.2k cache: `/home/nic/Work/les1mple/data/opencs2-preview-raw1024-seq150-step4`
- Short-horizon `sigreg=0.05` run: `/tmp/les1mple-runs/opencs2-lewm-mamba3-seq150-step4-h146-b128-adaln-sigreg0.05-1000step`

## Current Clip Setup

- OpenCS2 preview videos are 20 FPS.
- `frame_step=4` gives 5 Hz.
- `sequence_length=150` gives 30 second clips.
- New cached clips should store pixels as `uint8`; `LocalTensorSequenceDataset` normalizes them to float32 on load.
- `history_size=146`, `num_preds=4` is 29.2 seconds of context and only a 0.8 second target offset.
- A better long-horizon target is `history_size=100`, `num_preds=50`, which is 20 seconds of context and a 10 second target offset.

## Data Notes

- The 5 MB preview cap with 30 second clips only yielded 583 candidates.
- Use `--max-preview-bytes 8000000` for the 10k cache.
- Old float32 cached clips are large: expect roughly 800-900 GB for 10k clips.
- New caches always store pixels as `uint8`; shard/compress later if needed.

## Common Commands

```bash
find /home/nic/Work/les1mple/data/opencs2-preview-raw10000-seq150-step4 -name '*.pt' | wc -l
du -sh /home/nic/Work/les1mple/data/opencs2-preview-raw10000-seq150-step4
tail -n 30 /tmp/les1mple-runs/opencs2-lewm-mamba3-seq150-step4-h146-b128-adaln-sigreg0.05-1000step/metrics.csv
```

## Coding Style

- Prefer simple, surgical changes.
- Do not add abstractions for single-use code.
- Keep maintained commands in `scripts/`, not `experiments/`.
- Match the existing local style.
- Do not clean up unrelated code while making a targeted change.
- For bug fixes, prefer a reproducing check before the fix.
- Run the smallest meaningful verification command before calling work done.

## Notes

- The manifest stuff should be left to smoke tests, debug paths, or maybe eval.
- Possible future direction: train an omniscient world model first, then use it to densely supervise a player model and reduce dependence on CEM planning.
- When checking background work, trust metrics, checkpoints, and data counts over terminal session lists.
