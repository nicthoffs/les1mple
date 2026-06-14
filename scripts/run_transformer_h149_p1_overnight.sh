#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/opencs2-full-raw10000-seq150-step6}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/uv}"
TMP_DIR="${TMP_DIR:-$ROOT_DIR/.cache/tmp}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs}"

RUN_NAME="opencs2-lewm-transformer-small-seq150-step6-h149-p1-b32x4-sigreg0.05-${STAMP}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
VIEW_DELTA_STATS="$RUN_ROOT/view-delta-stats-full-seq150-step6-seed0-vsplit0.1.pt"

mkdir -p "$CACHE_DIR" "$TMP_DIR" "$RUN_ROOT"

echo "run_dir: $RUN_DIR"
echo "wandb_name: $RUN_NAME"

cd "$ROOT_DIR"

TMPDIR="$TMP_DIR" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv --cache-dir "$CACHE_DIR" run --extra train python scripts/train_lewm_opencs2.py \
  --data-dir "$DATA_DIR" \
  --run-dir "$RUN_DIR" \
  --predictor-type transformer \
  --history-size 149 \
  --batch-size 32 \
  --accumulate-grad-batches 4 \
  --predictor-depth 4 \
  --predictor-heads 4 \
  --predictor-dim-head 48 \
  --predictor-mlp-dim 512 \
  --max-steps 4000 \
  --sigreg-weight 0.05 \
  --encoder-chunk-size 64 \
  --checkpoint-encoder \
  --mmap-load \
  --pixel-dtype uint8 \
  --precision bf16-mixed \
  --channels-last \
  --num-workers 4 \
  --limit-val-batches 4 \
  --num-sanity-val-steps 0 \
  --no-lightning-checkpoints \
  --view-delta-stats-cache "$VIEW_DELTA_STATS" \
  --wandb-name "$RUN_NAME"
