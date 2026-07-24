#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/qwen3_6_27b_dapo_tinker}"
STEPS="${STEPS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-$STEPS}"
PROJECT="${PROJECT:-qwen3_6_dapo_lora}"
RUN_NAME="${RUN_NAME:-qwen27_dapo_tinker_e2e}"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY must be set" >&2
  exit 1
fi

base_url_args=()
if [[ -n "${TINKER_BASE_URL:-}" ]]; then
  base_url_args=(--base-url "$TINKER_BASE_URL")
fi

uv run --extra tinker python examples/tinker/dapo/dapo_tinker.py \
  --train-file "$DATA_DIR/dapo-math-17k-cleaned.parquet" \
  --eval-file "$DATA_DIR/aime-2024-eval.parquet" \
  --eval-file "$DATA_DIR/aime-2026-eval.parquet" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --eval-interval "$EVAL_INTERVAL" \
  --project "$PROJECT" \
  --run-name "$RUN_NAME" \
  "${base_url_args[@]}" \
  "$@"
