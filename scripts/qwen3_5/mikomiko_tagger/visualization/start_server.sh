#!/usr/bin/env bash
# vLLM server for the review-page pipeline (step 2/3) — serves the RELEASED tagger checkpoint
# ${MODELS_DIR}/Mikomiko_pornpic_tagger/checkpoint-${CKPT_STEP} as model "mikomiko".
# Same eval-parity knobs as ../start_vllm_server_mikomiko.sh (greedy, max_pixels 262144), but
# defaults tuned for a lightweight ad-hoc run on a busy GPU (GPU_UTIL 0.35, 16 seqs, port 8111).
#
# build_html.sh starts and stops this for you; run it by hand only to keep a server warm across
# several invocations (then pass API=http://localhost:8111 so build_html.sh reuses it).
# Env:    GPU | PORT | GPU_UTIL | CKPT_STEP | TEMPERATURE
set -euo pipefail

# Walk up from the ABSOLUTE script dir (dirname "." == "." would loop forever otherwise).
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="${_HERE}"
until [ -e "${_d}/scripts/workspace_dir.sh" ] || [ "${_d}" = / ]; do _d="$(dirname "${_d}")"; done
[ -e "${_d}/scripts/workspace_dir.sh" ] || { echo "ERROR: repo root not found above ${_HERE}" >&2; exit 1; }
source "${_d}/scripts/workspace_dir.sh"

GPU="${GPU:-6}"
export CUDA_VISIBLE_DEVICES="${GPU}"
PORT="${PORT:-8111}"
GPU_UTIL="${GPU_UTIL:-0.35}"
TEMPERATURE="${TEMPERATURE:-0}"
CKPT_STEP="${CKPT_STEP:-17296}"
MAX_PIXELS=262144   # match training image_max_pixels for eval parity

CKPT="${MODELS_DIR}/Mikomiko_pornpic_tagger/checkpoint-${CKPT_STEP}"
if [ ! -f "${CKPT}/model.safetensors" ]; then
  echo "[server] ERROR: ${CKPT}/model.safetensors not found." >&2
  exit 1
fi

# Training parity: the ckpt jinja injects an empty <think></think> block after "assistant\n" that
# the SFT template (qwen3_5_nothink) never emitted; those 4 tokens cost 1.2pt microF1. Serve the
# patched template, which renders token-for-token identical to the training prompt.
CHAT_TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/chat_template_train_parity.jinja"
[ -f "${CHAT_TEMPLATE}" ] || { echo "[server] ERROR: ${CHAT_TEMPLATE} missing" >&2; exit 1; }

# CUDA JIT compiler shim (machine-adaptive, same as sibling scripts)
_shim="${LF_ROOT}/.cc-shim"
if [ -x "${_shim}/g++" ] && echo 'int main(){return 0;}' | "${_shim}/g++" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export CC="${_shim}/gcc" CXX="${_shim}/g++" CUDAHOSTCXX="${_shim}/g++" NVCC_PREPEND_FLAGS="-ccbin ${_shim}/g++"
fi

source "${VLLM_VENV}/bin/activate"

echo "[server] mikomiko review server :${PORT}  GPU=${GPU}  util=${GPU_UTIL}  ckpt=${CKPT}"
exec vllm serve "${CKPT}" \
  --served-model-name mikomiko \
  --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}" \
  --override-generation-config "{\"temperature\": ${TEMPERATURE}, \"top_p\": 1.0, \"top_k\": -1}" \
  --trust-remote-code \
  --chat-template "${CHAT_TEMPLATE}" \
  --port "${PORT}"
