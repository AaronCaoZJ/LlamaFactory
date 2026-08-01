#!/usr/bin/env bash
# InternVL3.5-2B LoRA on ms0717_blockpap_follow_wood -- the 方案 D BlockPAP follower dataset
# regenerated on a WOOD tabletop (RLinf texture 006) instead of the featureless white one.
# Same seeds / tracks / cameras / letterbox as ms0717_blockpap_follow, so the pair isolates
# the tabletop texture, and the hyper-params match the other ms0717 scripts for comparison.
#
#   bash scripts/internvl/train/ms0717_blockpap_follow_wood_2b.sh          # defaults (GPUs 4,6)
#   GPUS=6 FORCE_TORCHRUN=1 bash scripts/internvl/train/ms0717_blockpap_follow_wood_2b.sh
#
# The config keeps ds_z2, and llamafactory-cli only switches to torchrun when it sees more
# than one GPU -- so a SINGLE-GPU run needs FORCE_TORCHRUN=1 or it aborts with
# "Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training".
#
# Outputs -> saves/internvl3.5-2b/robot/ms0717_blockpap_follow_wood
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export CONFIG="${CONFIG:-examples/train_lora/internvl/internvl3_5_2b_ms0717_blockpap_follow_wood.yaml}"
export MODEL_PATH="${MODEL_PATH:-/workspace1/zechen/hf_download/InternVL3_5-2B-HF}"

exec bash "${SCRIPT_DIR}/repro_qwen_data.sh"
