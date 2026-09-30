#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
DATASET="${DATASET:-bindingdb}"
case "$DATASET" in bindingdb|kiba) ;; *) echo 'DATASET must be bindingdb or kiba' >&2; exit 2 ;; esac
args=(--config "$ROOT/configs/$DATASET.json" --device "${DEVICE:-cuda}"
      --output-dir "${OUTPUT_DIR:-$ROOT/runs}" --micro-batch-size "${MICRO_BATCH_SIZE:-16}"
      --num-workers "${NUM_WORKERS:-4}")
[[ -n "${DATA_DIR:-}" ]] && args+=(--data-dir "$DATA_DIR")
[[ -n "${POCKET_CONTRACT:-}" ]] && args+=(--pocket-contract "$POCKET_CONTRACT")
[[ -n "${COMPONENT_SEQUENCES:-}" ]] && args+=(--component-sequences "$COMPONENT_SEQUENCES")
[[ -n "${ESMC6B_EMBEDDINGS:-}" ]] && args+=(--esmc6b-embeddings "$ESMC6B_EMBEDDINGS")
exec "${PYTHON_BIN:-python}" -m model.train "${args[@]}" "$@"
