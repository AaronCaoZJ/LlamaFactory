#!/usr/bin/env bash
# vLLM server for the grok_desc review page (step 2/3). Serves ONE model dir, named by SERVED.
#
# Used twice by build_html.sh: once for the SFT checkpoint, once for the untuned base. Both are
# served with the SAME chat template (chat_template_qwen3_5_lf.jinja) on purpose -- base and ckpt
# ship byte-identical chat_template.jinja files, so serving both through the LF template makes the
# prompt token-for-token identical between the two runs and the ONLY difference is the weights.
# Serving the base with its own template instead would inject an empty <think></think> block and
# hand the base a different prompt than the SFT model got, which is not a comparison.
#
#   MODEL=/path/to/model SERVED=desc_sft PORT=8121 GPU=0 bash start_server.sh
#
# Env: MODEL (required) | SERVED | PORT | GPU | GPU_UTIL | MAX_LEN | TEMPERATURE
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="${_HERE}"
until [ -e "${_d}/scripts/workspace_dir.sh" ] || [ "${_d}" = / ]; do _d="$(dirname "${_d}")"; done
[ -e "${_d}/scripts/workspace_dir.sh" ] || { echo "ERROR: repo root not found above ${_HERE}" >&2; exit 1; }
source "${_d}/scripts/workspace_dir.sh"

MODEL="${MODEL:?set MODEL to the model directory}"
SERVED="${SERVED:-desc}"
PORT="${PORT:-8121}"
GPU="${GPU:-0}"
GPU_UTIL="${GPU_UTIL:-0.60}"
MAX_LEN="${MAX_LEN:-4096}"          # 470 prompt + 256 image + 1536 output, with headroom
TEMPERATURE="${TEMPERATURE:-0}"
MAX_PIXELS=262144                   # == training image_max_pixels; anything else is a different model input

export CUDA_VISIBLE_DEVICES="${GPU}"

ls "${MODEL}"/*.safetensors >/dev/null 2>&1 || {
  echo "[server] ERROR: no *.safetensors in ${MODEL}" >&2; exit 1; }

CHAT_TEMPLATE="${LF_ROOT}/scripts/qwen3_5/mikomiko_tagger/chat_template_qwen3_5_lf.jinja"
[ -f "${CHAT_TEMPLATE}" ] || { echo "[server] ERROR: ${CHAT_TEMPLATE} missing" >&2; exit 1; }

# CUDA JIT compiler shim (machine-adaptive, same as sibling scripts)
_shim="${LF_ROOT}/.cc-shim"
if [ -x "${_shim}/g++" ] && echo 'int main(){return 0;}' | "${_shim}/g++" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export CC="${_shim}/gcc" CXX="${_shim}/g++" CUDAHOSTCXX="${_shim}/g++" NVCC_PREPEND_FLAGS="-ccbin ${_shim}/g++"
fi

source "${VLLM_VENV}/bin/activate"

echo "[server] ${SERVED} :${PORT}  GPU=${GPU}  util=${GPU_UTIL}  model=${MODEL}"
exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}" \
  --override-generation-config "{\"temperature\": ${TEMPERATURE}, \"top_p\": 1.0, \"top_k\": -1}" \
  --trust-remote-code \
  --chat-template "${CHAT_TEMPLATE}" \
  --port "${PORT}"
