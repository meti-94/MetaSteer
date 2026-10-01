#!/usr/bin/env bash
# Download the SteerBoost dataset from the Hugging Face Hub into data/steerboost/,
# alongside the concept16k datasets downloaded by download.py.
#
# The dataset ships:
#   cache/                 # feature .npz + judgment .json caches per (method, model, layer)
#   result__*.tar.zst      # per (method, model) tarball of steered generations + GPT labels
#   training/              # GPT-generated positive/negative example banks per concept
#
# Hidden-state .pt files are NOT distributed (>500 GB on disk).
#
# Usage:
#   bash download.sh                       # everything
#   bash download.sh --no-result           # skip the 7.5GB result tarballs
#   bash download.sh --no-extract          # keep tarballs compressed
#   bash download.sh --repo-id ORG/NAME    # override default HF repo
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ID="Fcr09/SteerBoost-data"
DATA_DIR="$SCRIPT_DIR/data/steerboost"
NO_RESULT=0
NO_EXTRACT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)    REPO_ID="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --no-result)  NO_RESULT=1; shift ;;
        --no-extract) NO_EXTRACT=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Requires the Hugging Face CLI (`hf`) on PATH.
# Optional: set HF_HOME to a large disk if the default cache is too small.
# export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$DATA_DIR"

INCLUDE_ARGS=(--include "cache/*" --include "training/*" --include "concepts.json" --include "alpaca_eval.json")
if [[ $NO_RESULT -eq 0 ]]; then
    INCLUDE_ARGS+=(--include "result__*.tar.zst")
fi

echo "[hf] download repo_id=$REPO_ID -> $DATA_DIR"
hf download "$REPO_ID" \
    --type dataset \
    --local-dir "$DATA_DIR" \
    "${INCLUDE_ARGS[@]}"

if [[ $NO_RESULT -eq 0 && $NO_EXTRACT -eq 0 ]]; then
    shopt -s nullglob
    for tarball in "$DATA_DIR"/result__*.tar.zst; do
        echo "[extract] $(basename "$tarball") -> $DATA_DIR/result/"
        tar --use-compress-program=unzstd -xf "$tarball" -C "$DATA_DIR"
    done
    shopt -u nullglob
fi

echo "[done]"
