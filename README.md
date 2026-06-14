# LeS1mple

LeS1mple is a world model project for Counter-Strike. It trains predictive latent state-action models on the [OpenCS2 Dataset](https://huggingface.co/datasets/blanchon/opencs2_dataset), which provides tick-aligned POV video, audio, input actions, and world state.

The goal is to learn a compact representation of gameplay dynamics that can support long-horizon prediction and planning in a partially observable, multi-agent environment.

## Architecture

The current model follows a LeWM-style JEPA setup:

- The encoder observes the current POV frame and produces a latent embedding of current visual evidence.
- The predictor consumes a history of visual latents and action embeddings, then predicts future visual latents.
- Actions condition the predictor through AdaLN-style modulation.
- SIGReg regularizes the latent space during training.

The encoder is intentionally smaller than the predictor. Memory, dynamics, and counterfactual futures should mostly live in the predictor rather than in a single-frame encoder.

## Repository Layout

```text
src/les1mple/data/        dataset loading, cached sequence format, data transforms
src/les1mple/models/      encoder, predictor, and LeWM construction
src/les1mple/training/    shared training objective and dataset split helpers
scripts/                  maintained data, train, and eval commands
experiments/              quick smoke checks and scratch experiments
```

## Setup

Use `uv` from the repo root. The training extra installs the stable-worldmodel stack:

```bash
uv --cache-dir /tmp/uv-cache run --extra train python experiments/smoke_dataset.py
```

## Data

The current cached OpenCS2 clip format is:

- 150 sampled frames
- `frame_step=6` from 32 FPS full POV videos, so about 5.33 Hz
- about 28 seconds per clip
- 224x224 RGB
- uint8 cached pixels, normalized to float32 by the dataset at load time

Build or resume the full-video cache:

```bash
uv --cache-dir /tmp/uv-cache run --extra train python scripts/fetch_opencs2_videos.py \
  --num-samples 10000 \
  --output-dir /home/nic/Work/les1mple/data/opencs2-full-raw10000-seq150-step6 \
  --cache-dir /home/nic/Work/les1mple/.cache/hf \
  --max-candidates 50000 \
  --min-duration-s 30 \
  --max-media-bytes 100000000 \
  --sequence-length 150 \
  --frame-step 6 \
  --image-size 224 \
  --sampling raw \
  --seed 0
```

Cached pixels are stored as `uint8` to keep disk usage manageable. The dataset normalizes them to float32 at load time.
The fetcher uses `blanchon/opencs2_dataset_wds` and reads selected MP4/tick members by byte range from tar shards, so source videos are not cached as loose files. Existing `.pt` files in the output directory are skipped, so rerunning this command with the same output directory adds new samples instead of overwriting or duplicating existing media IDs.

## Training

Short-horizon stability run:

```bash
uv --cache-dir /tmp/uv-cache run --extra train python scripts/train_lewm_opencs2.py \
  --predictor-type mamba3 \
  --data-dir /home/nic/Work/les1mple/data/opencs2-preview-raw1024-seq150-step4 \
  --run-dir /tmp/les1mple-runs/opencs2-lewm-mamba3-seq150-step4-h146-b128-adaln-sigreg0.05-1000step \
  --batch-size 128 \
  --accumulate-grad-batches 1 \
  --max-steps 1000 \
  --history-size 146 \
  --encoder-chunk-size 128 \
  --checkpoint-encoder \
  --sigreg-weight 0.05
```

Longer-horizon run candidate:

```bash
uv --cache-dir /tmp/uv-cache run --extra train python scripts/train_lewm_opencs2.py \
  --predictor-type mamba3 \
  --data-dir /home/nic/Work/les1mple/data/opencs2-full-raw10000-seq150-step6 \
  --run-dir /tmp/les1mple-runs/opencs2-lewm-mamba3-seq150-step6-h100-p50-b128-sigreg0.05 \
  --batch-size 128 \
  --accumulate-grad-batches 1 \
  --max-steps 5000 \
  --history-size 100 \
  --encoder-chunk-size 128 \
  --checkpoint-encoder \
  --sigreg-weight 0.05
```

The target horizon is derived as `sequence_length - history_size`. At about 5.33 Hz with `sequence_length=150`, `history_size=100` gives 18.75 seconds of context and `target_horizon=50` gives a 9.375 second target offset.

Maintained transformer overnight run:

```bash
./scripts/run_transformer_h149_p1_overnight.sh
```

The maintained script defaults to the current fast data path: cached pixels stay `uint8` through DataLoader, `torch.load(..., mmap=True)` is used for cached `.pt` files, train workers use `num_workers=4,prefetch=1`, validation is limited to 4 batches, W&B logs go under the run directory, and run outputs default to repo-local `runs/`.

## Evaluation

Predictor retrieval compares predicted future latents against copy-last in latent space:

```bash
uv --cache-dir /tmp/uv-cache run --extra train python scripts/eval_lewm_retrieval.py \
  --checkpoint /tmp/les1mple-runs/opencs2-lewm-mamba3-seq150-step4-h146-b128-adaln-sigreg0.05-1000step/final.pt \
  --data-dir /home/nic/Work/les1mple/data/opencs2-preview-raw1024-seq150-step4 \
  --output-dir /tmp/les1mple-eval/sigreg0.05_predictor \
  --num-samples 122 \
  --batch-size 16 \
  --num-viz 12 \
  --device cuda
```

Encoder retrieval checks latent geometry and nearest neighbors:

```bash
uv --cache-dir /tmp/uv-cache run --extra train python scripts/eval_encoder_retrieval.py \
  --checkpoint /tmp/les1mple-runs/opencs2-lewm-mamba3-seq150-step4-h146-b128-adaln-sigreg0.05-1000step/final.pt \
  --data-dir /home/nic/Work/les1mple/data/opencs2-preview-raw1024-seq150-step4 \
  --output-dir /tmp/les1mple-eval/encoder \
  --num-clips 64 \
  --frames-per-clip 4 \
  --batch-size 4 \
  --num-viz 12 \
  --device cpu
```
