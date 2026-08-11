#!/usr/bin/env bash
# Visual-prompt experiment queue (9B). Every sample carries THREE images -- agentview, wrist,
# and <rollout>/visual_prompt.png (first agentview frame, green box = object to pick,
# red cross = release spot). 12011 samples / 233 rollouts, no _horizon_flip, no 0802 long collect.
#
# Inventory: data/agentrobot/MVTOKEN/mix_22-06_fk-pp/02_exchange_token/zwz_visual_prompt/SAMPLES.md
# Rebuild:   python .../zwz_visual_prompt/build_visual_prompt_dataset.py --write
#
# Uncomment exactly the block(s) you want to run; leave the finished ones commented so this
# file stays the history of what has been trained.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SH="${SCRIPT_DIR}/train_zwz_2.sh"   # 9B driver (MODEL_PATH defaults to Qwen3.5-9B)

GPUS="${GPUS:-4,5}"

echo "GPUS=${GPUS}"
echo "TRAIN_SH=${TRAIN_SH}"

# 1) notask -- the actual visual-prompt setting.
#    The "Task:" line is ONE constant sentence ("pick up the object marked by the green box
#    and release it at the red cross"), byte-identical in all 12011 samples, so which object
#    to pick and where it goes can only be read off the third image.
#    Inference: VARIANT=notask bash run_real_mvtoken_visual_prompt.sh  (prompts/v3_visual_prompt_notask/)
echo "================================================================================"
echo "[queue] visual prompt 1/1  notask (constant task line; the marks are the instruction)"
echo "================================================================================"
GPUS="${GPUS}" bash "${TRAIN_SH}" \
  "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token_zwz_visual_prompt_notask.yaml"

# 2) task -- control run, same samples and same three images, but the real per-episode task
#    sentence is also sent. The text alone already says what to do, so the model can solve most
#    of the episode without ever reading the green box; only worth running as the comparison
#    against 1). Inference: VARIANT=task bash run_real_mvtoken_visual_prompt.sh
# echo "================================================================================"
# echo "[queue] visual prompt  task (real task text; control)"
# echo "================================================================================"
# GPUS="${GPUS}" bash "${TRAIN_SH}" \
#   "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token_zwz_visual_prompt.yaml"
