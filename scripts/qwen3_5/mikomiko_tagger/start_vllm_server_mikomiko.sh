#!/usr/bin/env bash
# vLLM OpenAI server for the mikomiko image->tag FULL checkpoint (NOT a LoRA adapter).
# Serves saves/qwen3.5-2b/mikomiko/full_v0/checkpoint-<CKPT_STEP> as model name "mikomiko" (default :8110).
set -euo pipefail

# ================================================================================
# Paths (machine-agnostic; see scripts/workspace_dir.sh)
#* Exports: LF_ROOT | MODELS_DIR | LF_VENV | VLLM_VENV | HF_HOME | AGENTROBOT_ROOT
# Walk up from the ABSOLUTE script dir (dirname "." == "." would loop forever when the script is
# invoked as `bash start_vllm_server_mikomiko.sh` from its own directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="${SCRIPT_DIR}"
until [ -e "${_d}/scripts/workspace_dir.sh" ] || [ "${_d}" = / ]; do _d="$(dirname "${_d}")"; done
[ -e "${_d}/scripts/workspace_dir.sh" ] || { echo "ERROR: repo root not found above ${SCRIPT_DIR}" >&2; exit 1; }
source "${_d}/scripts/workspace_dir.sh"

# ================================================================================
#! Cuda device / runtime knobs (edit here)
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="${GPU}"

# ================================================================================
#! Args (ckpt step / server knobs / model)
#* Overrides: GPU | PORT | GPU_UTIL | TEMPERATURE | CKPT_STEP
PORT="${PORT:-8110}"
GPU_UTIL="${GPU_UTIL:-0.7}"
TEMPERATURE="${TEMPERATURE:-0}"    # 0 = greedy/deterministic (eval parity); >0 to sample

MAX_LEN=4096
MAX_NUM_SEQS=64
MAX_PIXELS=262144 # match training image_max_pixels for eval parity
ENFORCE_EAGER=0

BASE_MODEL="${MODELS_DIR}/Qwen3.5-2B"

# CKPT config
CKPT_ROOT="${LF_ROOT}/saves/qwen3.5-2b/mikomiko/full_v0"
CKPT_STEP="${CKPT_STEP:-11530}"
CKPT="${CKPT_ROOT}/checkpoint-${CKPT_STEP}"

# ================================================================================
# CUDA JIT compiler (machine-adaptive)
_shim="${LF_ROOT}/.cc-shim"
if [ -x "${_shim}/g++" ] && echo 'int main(){return 0;}' | "${_shim}/g++" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export CC="${_shim}/gcc" CXX="${_shim}/g++" CUDAHOSTCXX="${_shim}/g++" NVCC_PREPEND_FLAGS="-ccbin ${_shim}/g++"
fi

# ================================================================================
# Checkpoint prep
if [ ! -f "${CKPT}/model.safetensors" ]; then
  echo "[server] ERROR: ${CKPT}/model.safetensors not found." >&2
  exit 1
fi
# LlamaFactory sometimes omits VL processor files from a checkpoint; supplement from base if missing.
for f in preprocessor_config.json video_preprocessor_config.json merges.txt vocab.json chat_template.jinja; do
  if [ ! -e "${CKPT}/${f}" ] && [ -e "${BASE_MODEL}/${f}" ]; then
    cp "${BASE_MODEL}/${f}" "${CKPT}/${f}"; echo "[server] copied ${f} from base model"
  fi
done

# ================================================================================
#! Source venv
VLLM_VENV="${VLLM_VENV}"
source "${VLLM_VENV}/bin/activate"

# ================================================================================
#! Launch
SEP="================================================================================"
echo "Starting vllm mikomiko server on http://0.0.0.0:${PORT}"
echo "  GPU                 : ${GPU}"
echo "  Served model name   : mikomiko"
echo "  Checkpoint          : ${CKPT}"
echo "${SEP}"
echo "  Temperature         : ${TEMPERATURE}"
echo "  GPU util / max len  : ${GPU_UTIL} / ${MAX_LEN}"
echo "  Max num seqs        : ${MAX_NUM_SEQS}"
echo "  Image max pixels    : ${MAX_PIXELS}"
echo "${SEP}"

# Training parity: the checkpoint's own jinja injects an empty "<think>\n\n</think>\n\n" block
# after "assistant\n", which LlamaFactory's qwen3_5_nothink (the SFT template) never emitted.
# Those 4 tokens cost 1.2pt microF1 and inflate composite over-generation, so serve the patched
# template instead. Both render token-for-token identical to the training prompt.
CHAT_TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/chat_template_train_parity.jinja"
[ -f "${CHAT_TEMPLATE}" ] || { echo "[server] ERROR: ${CHAT_TEMPLATE} missing" >&2; exit 1; }

CMD=(
  vllm serve "${CKPT}"
  --served-model-name mikomiko
  --dtype bfloat16
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --limit-mm-per-prompt '{"image": 1}'
  --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}"
  --override-generation-config "{\"temperature\": ${TEMPERATURE}, \"top_p\": 1.0, \"top_k\": -1}"
  --trust-remote-code
  --chat-template "${CHAT_TEMPLATE}"
  --port "${PORT}"
)
[ "${ENFORCE_EAGER}" = "1" ] && CMD+=(--enforce-eager)

exec "${CMD[@]}"
