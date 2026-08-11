#!/usr/bin/env bash
# 0807 chazuo (plug-into-socket) post-training queue (9B).
#
# Data: data/agentrobot/MVTOKEN/zwz_0807_chazuo only -- 11 rollouts / 513 samples, single
# constant task line "Insert the plug into the socket.", normal 2-image v3 prompt (NOT the
# visual-prompt setting). The trailing MV_UP retreat is stripped from every episode per
# 说明.txt, so the synthesized DONE lands on the frame where the plug is seated.
#   Rebuild: python data/agentrobot/zwz_process_code/0807/build_0807_chazuo_dataset.py
#   Stats:   data/agentrobot/MVTOKEN/zwz_0807_chazuo/rollout_lite_chazuo.json.stats.json
#
# 513 samples would overfit from scratch, so the starting point is the already-trained
# saves/qwen3.5-9b/robot/mix_22-06_fk-pp/02_exchange_token_base adapter, not bare Qwen3.5-9B.
# 10 epochs (not 20) -> ~160 steps.
#
# Uncomment exactly the block(s) you want to run; leave the finished ones commented so this
# file stays the history of what has been trained.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SH="${SCRIPT_DIR}/train_zwz_2.sh"   # 9B driver (MODEL_PATH defaults to Qwen3.5-9B)

# 默认吃满 8 张（2026-08-11 复查时 0-7 全空）。想少占几张就 GPUS=0,1,2,3 bash 本脚本。
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# 等效 batch size 必须恒为 32（yaml 里按 8 卡写死 per_device=4 / grad_acc=1），换卡数时
# 这里重算 grad_acc 再覆盖进去，免得 4 卡跑出 eff_bs=16、LR schedule 和步数全变。
EFF_BS=32
PER_DEV=4
NGPU="$(awk -F, '{print NF}' <<<"${GPUS}")"
if (( EFF_BS % (NGPU * PER_DEV) != 0 )); then
  echo "ERROR: ${NGPU} 卡 x per_device ${PER_DEV} 除不尽 eff_bs ${EFF_BS}；用 1/2/4/8 卡。" >&2
  exit 1
fi
GRAD_ACC=$(( EFF_BS / (NGPU * PER_DEV) ))

echo "GPUS=${GPUS}  (${NGPU} 卡)"
echo "eff_bs=${EFF_BS} = ${NGPU} x per_device ${PER_DEV} x grad_acc ${GRAD_ACC}"
echo "TRAIN_SH=${TRAIN_SH}"

# 1) continue -- keep training 02_exchange_token_base's own LoRA weights.
#    Output is ONE self-contained adapter on top of bare Qwen3.5-9B: nothing to merge, just add
#    it to start_vllm_server_2_zwz.sh and pick it with --model. This is the default.
echo "================================================================================"
echo "[queue] chazuo 1/2  continue (post-train 02_exchange_token_base in place, 10ep)"
echo "================================================================================"
GPUS="${GPUS}" bash "${TRAIN_SH}" \
  "examples/train_lora/qwen3_5_9b/zwz_0807_chazuo/qwen3_5_9b_chazuo_posttrain_continue.yaml" \
  "per_device_train_batch_size=${PER_DEV}" "gradient_accumulation_steps=${GRAD_ACC}"

# 2) new_adapter -- freeze 02_exchange_token_base (merged into the weights) and train a SECOND
#    LoRA on top. Reversible, but the resulting adapter is only valid on the merged model, so
#    before serving it you must run the export once:
#      llamafactory-cli export examples/merge_lora/qwen3_5_9b_merge_02_exchange_token_base.yaml
#    and then use scripts/qwen3_5/eval/start_vllm_server_chazuo.sh instead of the shared server.
echo "================================================================================"
echo "[queue] chazuo 2/2  new_adapter (second LoRA stacked on the frozen base, 10ep)"
echo "================================================================================"
GPUS="${GPUS}" bash "${TRAIN_SH}" \
  "examples/train_lora/qwen3_5_9b/zwz_0807_chazuo/qwen3_5_9b_chazuo_posttrain_new_adapter.yaml" \
  "per_device_train_batch_size=${PER_DEV}" "gradient_accumulation_steps=${GRAD_ACC}"
