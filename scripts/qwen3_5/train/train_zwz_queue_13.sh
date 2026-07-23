#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SH="${SCRIPT_DIR}/train_zwz_2.sh"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

echo "GPUS=${GPUS}"
echo "TRAIN_SH=${TRAIN_SH}"

echo "================================================================================"
echo "[queue] 1/2 task_aug_13"
echo "================================================================================"
GPUS="${GPUS}" bash "${TRAIN_SH}" \
  "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token_task_aug_plus_robovqa_clean_ans6k_under500_13.yaml"

echo "================================================================================"
echo "[queue] 2/2 exchange_token_13"
echo "================================================================================"
GPUS="${GPUS}" bash "${TRAIN_SH}" \
  "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token_13.yaml"
