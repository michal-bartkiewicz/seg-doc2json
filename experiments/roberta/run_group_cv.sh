#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 PRIVATE_FOLD_DIR OUTPUT_ROOT" >&2
  exit 2
fi

ROBERTA_FOLD_DIR=$1
ROBERTA_OUTPUT_ROOT=$2
ROBERTA_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROBERTA_PYTHON=${ROBERTA_PYTHON:-/home/michal/roberta-segmentation/bin/python}
ROBERTA_CONTEXT_VARIANT=legacy
if [[ -f "${ROBERTA_FOLD_DIR}/fold-summary.json" ]]; then
  ROBERTA_CONTEXT_VARIANT=$(
    "$ROBERTA_PYTHON" -c 'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); print(data.get("context_variant", "legacy") if isinstance(data, dict) else "legacy")' \
      "${ROBERTA_FOLD_DIR}/fold-summary.json"
  )
fi
ROBERTA_CUDA_TOOLS=${ROBERTA_CUDA_TOOLS:-/home/michal/.conda/pkgs/cuda-nvcc-tools-13.3.73-hecca717_1/bin}
ROBERTA_WINDOW_SIZE=${ROBERTA_WINDOW_SIZE:-512}
ROBERTA_STRIDE=${ROBERTA_STRIDE:-96}
ROBERTA_EVAL_BATCH_SIZE=${ROBERTA_EVAL_BATCH_SIZE:-32}
ROBERTA_VALIDATION_DTYPE=${ROBERTA_VALIDATION_DTYPE:-bf16}
ROBERTA_WEIGHTS_ODWOLANIE=${ROBERTA_WEIGHTS_ODWOLANIE:-${ROBERTA_WEIGHTS:-4}}
ROBERTA_WEIGHTS_PELNOMOCNICTWO=${ROBERTA_WEIGHTS_PELNOMOCNICTWO:-${ROBERTA_WEIGHTS:-1}}
ROBERTA_WEIGHTS_PRZELEW=${ROBERTA_WEIGHTS_PRZELEW:-${ROBERTA_WEIGHTS:-4}}
ROBERTA_DOCUMENT_TYPES=${ROBERTA_DOCUMENT_TYPES:-"odwolanie pelnomocnictwo przelew"}
read -r -a ROBERTA_DOCUMENT_TYPE_LIST <<< "$ROBERTA_DOCUMENT_TYPES"
ROBERTA_EPOCHS=${ROBERTA_EPOCHS:-24}
ROBERTA_EARLY_STOPPING_PATIENCE=${ROBERTA_EARLY_STOPPING_PATIENCE:-4}
ROBERTA_BATCH_SIZE=${ROBERTA_BATCH_SIZE:-16}
ROBERTA_LEARNING_RATE=${ROBERTA_LEARNING_RATE:-7e-5}
HF_HOME=${HF_HOME:-/tmp/hf-cache-roberta}
export HF_HOME
export PATH="${ROBERTA_CUDA_TOOLS}:${PATH}"
umask 077

if [[ ! -d "$ROBERTA_FOLD_DIR" ]]; then
  echo "Private CV fold directory not found: $ROBERTA_FOLD_DIR" >&2
  exit 2
fi

mkdir -p "$ROBERTA_OUTPUT_ROOT"
chmod 700 "$ROBERTA_OUTPUT_ROOT"

for ROBERTA_FOLD_INDEX in 1 2 3; do
  ROBERTA_FOLD_NAME=$(printf 'fold-%02d' "$ROBERTA_FOLD_INDEX")
  ROBERTA_DATASET="${ROBERTA_FOLD_DIR}/${ROBERTA_FOLD_NAME}.jsonl"
  if [[ ! -f "$ROBERTA_DATASET" ]]; then
    echo "Missing fold dataset: $ROBERTA_DATASET" >&2
    exit 2
  fi

  for ROBERTA_TYPE in "${ROBERTA_DOCUMENT_TYPE_LIST[@]}"; do
    case "$ROBERTA_TYPE" in
      odwolanie) ROBERTA_TYPE_WEIGHTS=$ROBERTA_WEIGHTS_ODWOLANIE ;;
      pelnomocnictwo) ROBERTA_TYPE_WEIGHTS=$ROBERTA_WEIGHTS_PELNOMOCNICTWO ;;
      przelew) ROBERTA_TYPE_WEIGHTS=$ROBERTA_WEIGHTS_PRZELEW ;;
    esac
    read -r -a ROBERTA_WEIGHT_LIST <<< "$ROBERTA_TYPE_WEIGHTS"
    for ROBERTA_WEIGHT in "${ROBERTA_WEIGHT_LIST[@]}"; do
      ROBERTA_RUN_DIR="${ROBERTA_OUTPUT_ROOT}/${ROBERTA_FOLD_NAME}/${ROBERTA_TYPE}-w${ROBERTA_WEIGHT}"
      "$ROBERTA_PYTHON" "$ROBERTA_SCRIPT_DIR/train_eval.py" \
        --dataset "$ROBERTA_DATASET" \
        --output-dir "${ROBERTA_RUN_DIR}/models" \
        --epochs "$ROBERTA_EPOCHS" \
        --early-stopping-patience "$ROBERTA_EARLY_STOPPING_PATIENCE" \
        --batch-size "$ROBERTA_BATCH_SIZE" \
        --eval-batch-size "$ROBERTA_EVAL_BATCH_SIZE" \
        --validation-dtype "$ROBERTA_VALIDATION_DTYPE" \
        --learning-rate "$ROBERTA_LEARNING_RATE" \
        --window-size "$ROBERTA_WINDOW_SIZE" \
        --stride "$ROBERTA_STRIDE" \
        --entity-weight "$ROBERTA_WEIGHT" \
        --seed "$((41 + ROBERTA_FOLD_INDEX))" \
        --save-dtype bf16 \
        --document-types "$ROBERTA_TYPE" \
        --skip-test-eval

      "$ROBERTA_PYTHON" "$ROBERTA_SCRIPT_DIR/tune_validation_threshold.py" \
        --dataset "$ROBERTA_DATASET" \
        --model-dir "${ROBERTA_RUN_DIR}/models" \
        --output "${ROBERTA_RUN_DIR}/thresholds.json" \
        --dtype bf16 \
        --window-size "$ROBERTA_WINDOW_SIZE" \
        --stride "$ROBERTA_STRIDE" \
        --eval-batch-size "$ROBERTA_EVAL_BATCH_SIZE" \
        --document-types "$ROBERTA_TYPE"

      # Keep fold metrics only; weights are retrained for the final full fit.
      rm -rf -- "${ROBERTA_RUN_DIR}/models/${ROBERTA_TYPE}"
    done
  done
done

"$ROBERTA_PYTHON" "$ROBERTA_SCRIPT_DIR/summarize_group_cv.py" \
  --cv-root "$ROBERTA_OUTPUT_ROOT" \
  --output "${ROBERTA_OUTPUT_ROOT}/summary.json" \
  --document-types "${ROBERTA_DOCUMENT_TYPE_LIST[@]}" \
  --context-variant "$ROBERTA_CONTEXT_VARIANT"
