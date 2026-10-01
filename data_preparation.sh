#!/usr/bin/env bash
# End-to-end data pipeline for DPO meta-learning:
#   1. Download raw Concept16k v1/v2 datasets
#   2. Convert them into training-ready preference JSONL
#      under training_ready_data/data/
#   3. Delete all raw data to reclaim disk space
#
# Usage:
#   bash data_preparation.sh                # full pipeline
#   KEEP_RAW=1 bash data_preparation.sh     # skip the cleanup step
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/data_raw"
PREP_DIR="$SCRIPT_DIR/training_ready_data"
OUT_DIR="$PREP_DIR/data"

# Optional: set HF_HOME to a large disk if the default cache is too small.
# export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== [1/3] Downloading raw data ==="
python "$RAW_DIR/download.py"

echo "=== [2/3] Preparing training-ready data ==="
python "$PREP_DIR/concept16k_v1.py"
python "$PREP_DIR/concept16k_v2.py"

echo "=== [3/3] Cleaning up raw data ==="
if [[ "${KEEP_RAW:-0}" == "1" ]]; then
    echo "KEEP_RAW=1 set, skipping cleanup."
else
    # data_raw/data is a symlink to scratch; clear its contents, keep the dir.
    if [[ -d "$RAW_DIR/data" ]]; then
        find "$RAW_DIR/data" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    fi
    # Drop HF cache copies of the raw dataset repos as well.
    HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    for repo in datasets--pyvene--axbench-concept16k \
                datasets--pyvene--axbench-concept16k_v2; do
        rm -rf "$HF_HOME/hub/$repo"
    done
    echo "Raw data deleted."
fi

echo "=== Pipeline finished. Training-ready files: ==="
ls -lh "$OUT_DIR"
