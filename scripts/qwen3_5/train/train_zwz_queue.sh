#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SH="${SCRIPT_DIR}/train_zwz_2.sh"

GPUS="${GPUS:-6,7}"

echo "GPUS=${GPUS}"
echo "TRAIN_SH=${TRAIN_SH}"

# 1) base
# echo "================================================================================"
# echo "[queue] 1/6 base"
# echo "================================================================================"
# GPUS="${GPUS}" bash "${TRAIN_SH}" \
#   "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token_base.yaml"

# 2) task_aug + RoboVQA clean answer-only reasoning
# echo "================================================================================"
# echo "[queue] 2/6 task_aug + robovqa clean ans6k under500"
# echo "================================================================================"
# GPUS="${GPUS}" bash "${TRAIN_SH}" \
#   "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token_task_aug_plus_robovqa_clean_ans6k_under500.yaml"

# 3) task_aug
# echo "================================================================================"
# echo "[queue] 3/6 task_aug"
# echo "================================================================================"
# GPUS="${GPUS}" bash "${TRAIN_SH}" \
#   "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token_task_aug.yaml"

# 4) task_aug + zwz_0723
echo "================================================================================"
echo "[queue] 4/6 task_aug + zwz_0723"
echo "================================================================================"
GPUS="${GPUS}" bash "${TRAIN_SH}" \
  "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token_task_aug_plus_zwz_0723.yaml"

# 5) z2
# echo "================================================================================"
# echo "[queue] 5/6 z2"
# echo "================================================================================"
# GPUS="${GPUS}" bash "${TRAIN_SH}" \
#   "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token_z2.yaml"

# # 6) exchange_token
# echo "================================================================================"
# echo "[queue] 6/6 exchange_token"
# echo "================================================================================"
# GPUS="${GPUS}" bash "${TRAIN_SH}" \
#   "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token.yaml"
