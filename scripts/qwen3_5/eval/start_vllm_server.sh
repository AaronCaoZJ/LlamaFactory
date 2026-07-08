#!/usr/bin/env bash
# vLLM OpenAI server: Qwen3.5-27B + MVTOKEN LoRA adapters (default :8101, foreground / Ctrl-C 停).
# 覆盖项: CUDA_VISIBLE_DEVICES PORT GPU_UTIL MAX_LEN MAX_NUM_SEQS TEMPERATURE ENFORCE_EAGER
set -euo pipefail

# ============================================================
#! GPU / runtime knobs (edit here)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
PORT="${PORT:-8101}"
GPU_UTIL="${GPU_UTIL:-0.7}"
MAX_LEN="${MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
TEMPERATURE="${TEMPERATURE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# ============================================================
#! Paths (machine-agnostic; see scripts/workspace_dir.sh)
# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(
  d="$(dirname "${BASH_SOURCE[0]}")"
  until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done
  echo "$d"
)/scripts/workspace_dir.sh"
VLLM_VENV="${VLLM_VENV}"
BASE_MODEL="${MODELS_DIR}/Qwen3.5-27B"
SAVES="${LF_ROOT}/saves/qwen3.5-27b/robot"
# name=path，全部挂在同一个 --lora-modules 下（多个 --lora-modules 只会保留最后一个）。
LORA_MODULES=(
  "mvtoken_0622_v0=${SAVES}/mvtoken_0622_v0"
  "mvtoken_0622_v1=${SAVES}/mvtoken_0622_v1"
  "mix_22_27_v3=${SAVES}/mix_22_27_v3"
)

# ============================================================
#! CUDA JIT compiler (machine-adaptive)
# Use env_setup's validated .cc-shim only if the default compiler can't build C++;
# otherwise leave the system default alone.
_shim="${LF_ROOT}/.cc-shim"
if [ -x "${_shim}/g++" ] && echo 'int main(){return 0;}' | "${_shim}/g++" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export CC="${_shim}/gcc" CXX="${_shim}/g++" CUDAHOSTCXX="${_shim}/g++" NVCC_PREPEND_FLAGS="-ccbin ${_shim}/g++"
fi

source "${VLLM_VENV}/bin/activate"

# ============================================================
#! Launch
SEP="================================================================================"
echo "Starting vllm server on http://0.0.0.0:${PORT}"
echo "  GPU                 : ${CUDA_VISIBLE_DEVICES}"
echo "  GPU util            : ${GPU_UTIL}"
echo "  Max seq len         : ${MAX_LEN}"
echo "  Max num seqs        : ${MAX_NUM_SEQS}"
echo "  Temperature         : ${TEMPERATURE}"
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
  --override-generation-config "{\"temperature\": ${TEMPERATURE}, \"top_p\": 1.0, \"top_k\": -1}"
  --trust-remote-code
  --port "${PORT}"
)
[ "${ENFORCE_EAGER}" = "1" ] && CMD+=(--enforce-eager)
exec "${CMD[@]}"
