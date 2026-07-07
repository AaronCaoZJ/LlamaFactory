#!/usr/bin/env bash
# vLLM OpenAI server: Gemma-4-12B + LoRA adapters (default :8104, foreground / Ctrl-C 停).
# ⚠️ gemma4 是新架构，当前 vLLM 未必支持；跑不起来就改用 start_hf_server.sh。
# 覆盖项: CUDA_VISIBLE_DEVICES PORT GPU_UTIL MAX_LEN MAX_NUM_SEQS ENFORCE_EAGER
set -euo pipefail
# ═══ GPU / runtime knobs (edit here) ═══
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

PORT="${PORT:-8104}"
GPU_UTIL="${GPU_UTIL:-0.7}"
MAX_LEN="${MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# resolve machine paths: locate & source scripts/workspace_dir.sh (sets LF_ROOT, MODELS_DIR, LF_VENV, VLLM_VENV, AGENTROBOT_ROOT, HF_HOME)
_wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
source "$_wsd/scripts/workspace_dir.sh"

VENV="${VLLM_VENV}"
BASE_MODEL="${MODELS_DIR}/gemma4-12B-it"
SAVES="${LF_ROOT}/saves/gemma4-12b/robot"

# name=path，全部挂在同一个 --lora-modules 下（多个 --lora-modules 只会保留最后一个）。
LORA_MODULES=(
  # mix 训练在 3400 步被中止，未跑完，顶层无 adapter；指向已保存的 checkpoint。
  "gemma4_12b_mix_22_27_v3=${SAVES}/mix_22_27_v3/checkpoint-3400"
  "gemma4_12b_overfit=${SAVES}/overfit"
)

# gcc-12 on this node lacks cc1plus; use gcc-11 for CUDA JIT.
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"

source "${VENV}/bin/activate"

SEP="================================================================================"
echo "Starting vllm server on http://0.0.0.0:${PORT}"
echo "  GPU                 : ${CUDA_VISIBLE_DEVICES}"
echo "  GPU util            : ${GPU_UTIL}"
echo "  Max seq len         : ${MAX_LEN}"
echo "  Max num seqs        : ${MAX_NUM_SEQS}"
echo "  Enforce eager       : ${ENFORCE_EAGER}"
echo "${SEP}"
echo "  Base model          : ${BASE_MODEL}"
echo "${SEP}"
for m in "${LORA_MODULES[@]}"; do printf "  %-22s: %s\n" "${m%%=*}" "${m#*=}"; done
echo "${SEP}"

CMD=(
  vllm serve "${BASE_MODEL}"
  --dtype bfloat16
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --enable-lora --max-lora-rank 64
  --lora-modules "${LORA_MODULES[@]}"
  --trust-remote-code
  --port "${PORT}"
)
[ "${ENFORCE_EAGER}" = "1" ] && CMD+=(--enforce-eager)
exec "${CMD[@]}"
