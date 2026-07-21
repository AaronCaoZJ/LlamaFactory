#!/usr/bin/env bash
# vLLM OpenAI server: Gemma-4-12B + LoRA adapters (default :8104).
set -euo pipefail

# ================================================================================
# Paths (machine-agnostic; see scripts/workspace_dir.sh)
#* Exports: LF_ROOT | MODELS_DIR | LF_VENV | VLLM_VENV | HF_HOME | AGENTROBOT_ROOT
source "$(
  d="$(dirname "${BASH_SOURCE[0]}")"
  until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done
  echo "$d"
)/scripts/workspace_dir.sh"

# ================================================================================
#! Cuda device / runtime knobs (edit here)
GPU="${GPU:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

# ================================================================================
#! Args (server knobs / model / LoRA)
#* Overrides: GPU | PORT | GPU_UTIL | TEMPERATURE
PORT="${PORT:-8104}"
GPU_UTIL="${GPU_UTIL:-0.6}"
TEMPERATURE="${TEMPERATURE:-0}"

MAX_LEN=8192
MAX_NUM_SEQS=256
ENFORCE_EAGER=0

# BASE_MODEL="${MODELS_DIR}/gemma4-12B-it"
# SAVES="${LF_ROOT}/saves/gemma4-12b/robot"

BASE_MODEL="${MODELS_DIR}/gemma4-E4B-it"
SAVES="${LF_ROOT}/saves/gemma4-e4b/robot"

# LORA_MODULES=(
#   # mix 训练在 3400 步被中止，顶层无 adapter；指向已保存的 checkpoint。
#   "gemma4_12b_mix_22_27_v3=${SAVES}/mix_22_27_v3"
#   "gemma4_12b_overfit=${SAVES}/overfit"
# )
LORA_MODULES=(
  # e4b 训练已跑满 30 epoch（step 3900），顶层目录即最终 adapter。
  "gemma4_e4b_mix_22_27_v3=${SAVES}/mix_22_27_v3"
)

# gemma4 官方 chat_template 与 LF `gemma4n` 训练模板不一致（缺 system turn + 空 thought 段），
# 直接用会导致 prompt 分布失配 -> 必须用复刻训练渲染的模板。
CHAT_TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/chat_template_gemma4n_lf.jinja"

# ================================================================================
# CUDA JIT compiler (machine-adaptive)
_shim="${LF_ROOT}/.cc-shim"
if [ -x "${_shim}/g++" ] && echo 'int main(){return 0;}' | "${_shim}/g++" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export CC="${_shim}/gcc" CXX="${_shim}/g++" CUDAHOSTCXX="${_shim}/g++" NVCC_PREPEND_FLAGS="-ccbin ${_shim}/g++"
fi

# ================================================================================
#! Source venv
VLLM_VENV="${VLLM_VENV}"
source "${VLLM_VENV}/bin/activate"

# ================================================================================
#! Launch
SEP="================================================================================"
echo "Starting vllm server on http://0.0.0.0:${PORT}"
echo "  GPU                 : ${GPU}"
echo "  GPU util            : ${GPU_UTIL}"
echo "  Temperature         : ${TEMPERATURE}"
echo "  Max seq len         : ${MAX_LEN}"
echo "  Max num seqs        : ${MAX_NUM_SEQS}"
echo "  Enforce eager       : ${ENFORCE_EAGER}"
echo "${SEP}"
echo "  Base model          : ${BASE_MODEL}"
echo "${SEP}"
for m in "${LORA_MODULES[@]}"; do printf "  %-22s: %s\n" "${m%%=*}" "${m#*=}"; done
echo "${SEP}"
echo "  Chat template       : ${CHAT_TEMPLATE}"
echo "${SEP}"

CMD=(
  vllm serve "${BASE_MODEL}"
  --dtype bfloat16
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --chat-template "${CHAT_TEMPLATE}"
  --enable-lora --max-lora-rank 64
  --lora-modules "${LORA_MODULES[@]}"
  --override-generation-config "{\"temperature\": ${TEMPERATURE}, \"top_p\": 1.0, \"top_k\": -1}"
  --trust-remote-code
  --port "${PORT}"
)
[ "${ENFORCE_EAGER}" = "1" ] && CMD+=(--enforce-eager)

exec "${CMD[@]}"
