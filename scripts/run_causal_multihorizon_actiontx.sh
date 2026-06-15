#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/opencs2-full-raw10000-seq150-step6}"
CACHE_DIR="${CACHE_DIR:-/tmp/uv-cache}"
TMP_DIR="${TMP_DIR:-$ROOT_DIR/.cache/tmp}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs}"

LOW_RUN_NAME="${LOW_RUN_NAME:-opencs2-lewm-transformer-small-seq150-step6-h149-p1-b32x4-s4000-ckpt1000-sigreg0.05-20260613_211529}"
LOW_CHECKPOINT="${LOW_CHECKPOINT:-$RUN_ROOT/$LOW_RUN_NAME/eval-step4000.pt}"
RUN_NAME="${RUN_NAME:-opencs2-causal-multihorizon-lewm-h100-p1_2_4_8_16_32_50-actiontx}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
VIEW_DELTA_STATS="$RUN_ROOT/view-delta-stats-full-seq150-step6-seed0-vsplit0.1.pt"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
MAX_STEPS="${MAX_STEPS:-12000}"

mkdir -p "$CACHE_DIR" "$TMP_DIR" "$RUN_ROOT"

echo "run_dir: $RUN_DIR"
echo "wandb_name: $RUN_NAME"
echo "low_checkpoint: $LOW_CHECKPOINT"
echo "max_steps: $MAX_STEPS"
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  echo "resume_from_checkpoint: $RESUME_FROM_CHECKPOINT"
fi

cd "$ROOT_DIR"

CMD=(
  uv --cache-dir "$CACHE_DIR" run --extra train python scripts/train_causal_multihorizon_opencs2.py
  --data-dir "$DATA_DIR"
  --run-dir "$RUN_DIR"
  --low-checkpoint "$LOW_CHECKPOINT"
  --horizons 1,2,4,8,16,32,50
  --num-positions 100
  --causal-action-encoder transformer
  --action-encoder-depth 2
  --action-encoder-heads 4
  --action-encoder-mlp-dim 512
  --batch-size 32
  --accumulate-grad-batches 4
  --max-steps "$MAX_STEPS"
  --lr 5e-5
  --weight-decay 1e-3
  --mmap-load
  --pixel-dtype uint8
  --precision 32-true
  --channels-last
  --num-workers 0
  --limit-val-batches 4
  --num-sanity-val-steps 0
  --lightning-checkpoints
  --checkpoint-every-n-steps 1000
  --view-delta-stats-cache "$VIEW_DELTA_STATS"
  --wandb-name "$RUN_NAME"
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  CMD+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

TMPDIR="$TMP_DIR" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${CMD[@]}"
