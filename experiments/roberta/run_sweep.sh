#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 PRIVATE_DATASET.jsonl OUTPUT_ROOT" >&2
  exit 2
fi

ROBERTA_DATASET=$1
ROBERTA_OUTPUT_ROOT=$2
ROBERTA_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROBERTA_PYTHON=${ROBERTA_PYTHON:-/home/michal/roberta-segmentation/bin/python}
ROBERTA_CUDA_TOOLS=${ROBERTA_CUDA_TOOLS:-/home/michal/.conda/pkgs/cuda-nvcc-tools-13.3.73-hecca717_1/bin}
HF_HOME=${HF_HOME:-/tmp/hf-cache-roberta}
export HF_HOME
export PATH="${ROBERTA_CUDA_TOOLS}:${PATH}"
umask 077

if [[ ! -f "$ROBERTA_DATASET" ]]; then
  echo "Private dataset not found: $ROBERTA_DATASET" >&2
  exit 2
fi

mkdir -p "$ROBERTA_OUTPUT_ROOT"
chmod 700 "$ROBERTA_OUTPUT_ROOT"

ROBERTA_CONFIGS=(
  "seed42-lr2e5-w1 2e-5 1.0 42"
  "seed42-lr2e5-w4 2e-5 4.0 42"
  "seed42-lr3e5-w1 3e-5 1.0 42"
  "seed42-lr3e5-w4 3e-5 4.0 42"
  "seed42-lr5e5-w1 5e-5 1.0 42"
  "seed42-lr5e5-w4 5e-5 4.0 42"
  "seed123-lr5e5-w1 5e-5 1.0 123"
  "seed123-lr5e5-w4 5e-5 4.0 123"
)
ROBERTA_RUN_DIRS=()

for ROBERTA_CONFIG in "${ROBERTA_CONFIGS[@]}"; do
  read -r ROBERTA_NAME ROBERTA_LR ROBERTA_ENTITY_WEIGHT ROBERTA_SEED <<< "$ROBERTA_CONFIG"
  ROBERTA_RUN_DIR="${ROBERTA_OUTPUT_ROOT}/${ROBERTA_NAME}"
  ROBERTA_RUN_DIRS+=("$ROBERTA_RUN_DIR")
  "$ROBERTA_PYTHON" "$ROBERTA_SCRIPT_DIR/train_eval.py" \
    --dataset "$ROBERTA_DATASET" \
    --output-dir "${ROBERTA_RUN_DIR}/models" \
    --epochs 8 --batch-size 8 \
    --learning-rate "$ROBERTA_LR" \
    --entity-weight "$ROBERTA_ENTITY_WEIGHT" \
    --seed "$ROBERTA_SEED" \
    --save-dtype bf16 \
    --skip-test-eval
done

ROBERTA_SELECTED_DIR="${ROBERTA_OUTPUT_ROOT}/selected-validation"
"$ROBERTA_PYTHON" "$ROBERTA_SCRIPT_DIR/select_best.py" \
  --validation-only \
  --runs "${ROBERTA_RUN_DIRS[@]}" \
  --output-dir "$ROBERTA_SELECTED_DIR"

"$ROBERTA_PYTHON" "$ROBERTA_SCRIPT_DIR/evaluate_saved.py" \
  --dataset "$ROBERTA_DATASET" \
  --model-dir "${ROBERTA_SELECTED_DIR}/models" \
  --output "${ROBERTA_SELECTED_DIR}/recheck.json"
