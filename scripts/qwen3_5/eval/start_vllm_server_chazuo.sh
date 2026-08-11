#!/usr/bin/env bash
# vLLM OpenAI server for the TWO-LoRA chazuo path (default :8110).
#
# Only needed if you trained with qwen3_5_9b_chazuo_posttrain_new_adapter.yaml. That run merged
# 02_exchange_token_base into the weights and trained a FRESH LoRA on top, so its adapter is a
# delta on the merged model, not on bare Qwen3.5-9B -- and vLLM applies at most one adapter per
# request, so the first LoRA has to be baked in. Run the export once:
#
#   llamafactory-cli export examples/merge_lora/qwen3_5_9b_merge_02_exchange_token_base.yaml
#
# If you trained with ..._continue.yaml instead, you do NOT need this file: that output is one
# self-contained adapter on bare 9B -- uncomment its line in start_vllm_server_2_zwz.sh (:8103).
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
# train_queue_chazuo.sh 现在默认吃满 0-7，所以训练在跑的时候这个 server 起不来：要么等训完，
# 要么开训时留卡（GPUS=0,1,2,3 bash train_queue_chazuo.sh，再 GPU=6 起这个）。
GPU="${GPU:-5}"
export CUDA_VISIBLE_DEVICES="${GPU}"
if [ -x /usr/local/cuda-12.9/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda-12.9
  export CUDA_PATH="${CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

# ================================================================================
#! Args (server knobs / model / LoRA)
#* Overrides: GPU | PORT | GPU_UTIL
PORT="${PORT:-8110}"
GPU_UTIL="${GPU_UTIL:-0.25}"

MAX_LEN=8192
MAX_NUM_SEQS=256
ENFORCE_EAGER=0

SAVES="${LF_ROOT}/saves/qwen3.5-9b/robot"

# The BASE here is not stock Qwen3.5-9B -- it already contains 02_exchange_token_base.
BASE_MODEL="${SAVES}/merged/02_exchange_token_base_merged"

LORA_MODULES=(
  "chazuo_delta_9=${SAVES}/zwz_0807_chazuo/chazuo_posttrain_new_adapter"
)

if [ ! -f "${BASE_MODEL}/config.json" ]; then
  echo "ERROR: merged base model not found at ${BASE_MODEL}" >&2
  echo "  run: llamafactory-cli export examples/merge_lora/qwen3_5_9b_merge_02_exchange_token_base.yaml" >&2
  exit 1
fi

# 启动前自检：vLLM 是先把 base model 全部加载完、再去解析 --lora-modules 的，所以一个还没
# 训完（或路径写错）的 adapter 会让你白等几分钟再看到崩溃。这里提前失败，并指出是哪一个。
MISSING=()
for m in "${LORA_MODULES[@]}"; do
  [ -f "${m#*=}/adapter_model.safetensors" ] || MISSING+=("${m%%=*}  ->  ${m#*=}")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: 这些 LoRA 还没有 adapter_model.safetensors（训练未完成 / 路径错）：" >&2
  printf '  %s\n' "${MISSING[@]}" >&2
  exit 1
fi

# LF 对齐的 chat template（必需）—— 同 start_vllm_server_2_zwz.sh，理由见那里的注释。
CHAT_TEMPLATE="${LF_ROOT}/scripts/qwen3_5/eval/chat_template_qwen3_5_lf.jinja"

# ================================================================================
# CUDA JIT compiler (machine-adaptive)
_shim="${LF_ROOT}/.cc-shim"
if [ -x "${_shim}/g++" ] && echo 'int main(){return 0;}' | "${_shim}/g++" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export CC="${_shim}/gcc" CXX="${_shim}/g++" CUDAHOSTCXX="${_shim}/g++" NVCC_PREPEND_FLAGS="-ccbin ${_shim}/g++"
fi

# ================================================================================
#! Source venv
source "${VLLM_VENV}/bin/activate"

# ================================================================================
#! Launch
SEP="================================================================================"
echo "Starting vllm server on http://0.0.0.0:${PORT}"
echo "  GPU                 : ${GPU}"
echo "  GPU util            : ${GPU_UTIL}"
echo "  Max seq len         : ${MAX_LEN}"
echo "  Max num seqs        : ${MAX_NUM_SEQS}"
echo "${SEP}"
echo "  Base model (MERGED) : ${BASE_MODEL}"
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
  --trust-remote-code
  --port "${PORT}"
)
[ "${ENFORCE_EAGER}" = "1" ] && CMD+=(--enforce-eager)

exec "${CMD[@]}"
