#!/usr/bin/env bash
# InternVL3.5-2B LoRA on ms0717_blockpap_oracle_wide -- the ManiSkill BlockPAP real2sim
# dataset (100 episodes / 5338 pairs, fully randomised block + coaster, one 2 cm single-axis
# move per MV_* token). Same driver/hyper-params as the fk-pp reproduction scripts; only the
# CONFIG differs.
#
#   bash scripts/internvl/train/ms0717_oracle_wide_2b.sh          # defaults (GPUs 4,6)
#   GPUS=0,1 bash scripts/internvl/train/ms0717_oracle_wide_2b.sh # pick GPUs
#
# The config keeps ds_z2, and llamafactory-cli only switches to torchrun when it sees more
# than one GPU -- so a SINGLE-GPU run needs FORCE_TORCHRUN=1 or it aborts with
# "Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training":
#   GPUS=4 FORCE_TORCHRUN=1 bash scripts/internvl/train/ms0717_oracle_wide_2b.sh
#
# Outputs -> saves/internvl3.5-2b/robot/ms0717_blockpap_oracle_wide
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export CONFIG="${CONFIG:-examples/train_lora/internvl/internvl3_5_2b_ms0717_blockpap_oracle_wide.yaml}"
export MODEL_PATH="${MODEL_PATH:-/workspace1/zechen/hf_download/InternVL3_5-2B-HF}"

exec bash "${SCRIPT_DIR}/repro_qwen_data.sh"
