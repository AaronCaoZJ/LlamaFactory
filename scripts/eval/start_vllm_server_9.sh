#!/usr/bin/env bash
# vLLM OpenAI server: Qwen3.5-27B + MVTOKEN LoRA adapters (default :8101, foreground / Ctrl-C 停).
# 覆盖项: CUDA_VISIBLE_DEVICES PORT GPU_UTIL MAX_LEN MAX_NUM_SEQS ENFORCE_EAGER
set -euo pipefail

VENV="/workspace1/zhijun/AgentRobot/.venv-vllm"
BASE_MODEL="/workspace1/zhijun/hf_download/models/Qwen3.5-9B"
SAVES="/workspace1/zhijun/LlamaFactory/saves/qwen3.5-9b/robot"

# name=path，全部挂在同一个 --lora-modules 下（多个 --lora-modules 只会保留最后一个）。
# 注意: adapter 的 base 必须是上面的 27B；9B 的 adapter 要单独起一个 9B-base server。
# v2 = 仅 0622 rollout（3808 条，lite prompt）。
LORA_MODULES=(
  "mvtoken_0622_v2_9=${SAVES}/mvtoken_0622_v2"
)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
PORT="${PORT:-8109}"; GPU_UTIL="${GPU_UTIL:-0.7}"; MAX_LEN="${MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"; ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

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
for m in "${LORA_MODULES[@]}"; do p="${m#*=}"; printf "  %-18s: %s\n" "${m%%=*}" "saves/${p#*/saves/}"; done
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
