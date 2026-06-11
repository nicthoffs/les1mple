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
- `frame_step=4` from 20 FPS preview videos, so 5 Hz
- 30 seconds per clip
- 224x224 RGB
- uint8 cached pixels, normalized to float32 by the dataset at load time

Build or resume the preview cache:

```bash
uv --cache-dir /tmp/uv-cache run --extra train python scripts/fetch_opencs2_previews.py \
  --num-samples 10000 \
  --output-dir /home/nic/Work/les1mple/data/opencs2-preview-raw10000-seq150-step4 \
  --cache-dir /home/nic/Work/les1mple/.cache/hf \
  --max-candidates 50000 \
  --min-duration-s 30 \
  --max-preview-bytes 8000000 \
  --sequence-length 150 \
  --frame-step 4 \
  --image-size 224 \
  --sampling raw \
  --seed 0
```

Cached pixels are stored as `uint8` to keep disk usage manageable. The dataset normalizes them to float32 at load time.

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
  --num-preds 4 \
  --encoder-chunk-size 128 \
  --checkpoint-encoder \
  --sigreg-weight 0.05
```

Longer-horizon run candidate:

```bash
uv --cache-dir /tmp/uv-cache run --extra train python scripts/train_lewm_opencs2.py \
  --predictor-type mamba3 \
  --data-dir /home/nic/Work/les1mple/data/opencs2-preview-raw10000-seq150-step4 \
  --run-dir /tmp/les1mple-runs/opencs2-lewm-mamba3-seq150-step4-h100-p50-b128-sigreg0.05 \
  --batch-size 128 \
  --accumulate-grad-batches 1 \
  --max-steps 5000 \
  --history-size 100 \
  --num-preds 50 \
  --encoder-chunk-size 128 \
  --checkpoint-encoder \
  --sigreg-weight 0.05
```

At 5 Hz, `history_size=100` gives 20 seconds of context and `num_preds=50` gives a 10 second target offset.

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
