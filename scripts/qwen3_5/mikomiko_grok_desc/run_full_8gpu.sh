#!/usr/bin/env bash
# Full 1-epoch run on the 20260721 description dataset. 8x H200, ZeRO-3, 9B full-param.
#
# Everything here comes from the yaml; this script only pins the GPUs and detaches the
# process. Detaching is not optional -- the run is ~15h and a closed terminal SIGTERMs it.
#
#   bash scripts/qwen3_5/mikomiko_desc/run_full_8gpu.sh
#
# First launch spends ~30-60min tokenizing 1,340,392 rows into
# saves/qwen3.5-9b/mikomiko/tokenized_desc_0721 before step 1. Re-launches reuse it.
# Changing dataset / template / cutoff_len REQUIRES deleting that directory -- the cache
# keys on raw text and does not validate the dataset name, so a stale hit is silent.
set -euo pipefail

source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"
cd "$LF_ROOT"
source "${LF_VENV}/bin/activate"

export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"

mkdir -p logs
LOG="logs/train_9b_grok_desc_v0.log"

exec llamafactory-cli train examples/train_full/qwen3_5_9b_mikomiko_grok_desc.yaml \
  >> "$LOG" 2>&1
