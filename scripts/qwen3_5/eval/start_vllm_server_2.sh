#!/usr/bin/env bash
# vLLM OpenAI server: Qwen3.5-2B + MVTOKEN LoRA adapters (default :8102).
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
PORT="${PORT:-8102}"
GPU_UTIL="${GPU_UTIL:-0.5}"
TEMPERATURE="${TEMPERATURE:-0}"

MAX_LEN=8192
MAX_NUM_SEQS=256
ENFORCE_EAGER=0

BASE_MODEL="${MODELS_DIR}/Qwen3.5-2B"
SAVES="${LF_ROOT}/saves/qwen3.5-2b/robot"

LORA_MODULES=(
  "mix_22_27_v3_2=${SAVES}/mix_22_27_v3"
  "mix_22-06_fk-pp_02_2=${SAVES}/mix_22-06_fk-pp/02_exchange_token"
  "zechen_repro=/workspace1/zechen/finetune/lora/Qwen3.5-2b"
)

# LF 对齐的 chat template（必需）。Qwen3.5 官方模板即使 enable_thinking=false 也会在
# '<|im_start|>assistant\n' 后插一个空 think 块 '<think>\n\n</think>\n\n'，而 LF 的
# qwen3_5_nothink 什么都不插 —— 不挂这个文件，prompt 与训练分布差 4 个 token（HANDOFF §4.2）。
# 对 image 布局和 video 槽位都适用；已用 /tokenize 逐 token 比对验证过。
CHAT_TEMPLATE="${LF_ROOT}/scripts/qwen3_5/eval/chat_template_qwen3_5_lf.jinja"

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
  --enable-lora --max-lora-rank 64
  --lora-modules "${LORA_MODULES[@]}"
  --chat-template "${CHAT_TEMPLATE}"
  --override-generation-config "{\"temperature\": ${TEMPERATURE}, \"top_p\": 1.0, \"top_k\": -1}"
  --trust-remote-code
  --port "${PORT}"
)
[ "${ENFORCE_EAGER}" = "1" ] && CMD+=(--enforce-eager)

exec "${CMD[@]}"
