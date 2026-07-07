#!/usr/bin/env bash
# vLLM OpenAI server for the mikomiko image->tag FULL checkpoint (NOT a LoRA adapter).
# Serves saves/qwen3.5-2b/mikomiko/full_v0/checkpoint-<STEP> directly as model name "mikomiko".
# Foreground / Ctrl-C to stop.  Default: GPU 1, port 8110.
#
# Usage:  bash scripts/qwen3_5/eval/start_vllm_server_mikomiko.sh [STEP]
# Env override: CUDA_VISIBLE_DEVICES PORT GPU_UTIL MAX_LEN MAX_NUM_SEQS MAX_PIXELS ENFORCE_EAGER
set -euo pipefail
# ═══ GPU / runtime knobs (edit here) ═══
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PORT="${PORT:-8110}"
GPU_UTIL="${GPU_UTIL:-0.7}"
MAX_LEN="${MAX_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
MAX_PIXELS="${MAX_PIXELS:-262144}"   # match training image_max_pixels for eval parity


# resolve machine paths: locate & source scripts/workspace_dir.sh (sets LF_ROOT, MODELS_DIR, LF_VENV, VLLM_VENV, AGENTROBOT_ROOT, HF_HOME)
_wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
source "$_wsd/scripts/workspace_dir.sh"
VENV="${VLLM_VENV}"
BASE_MODEL="${MODELS_DIR}/Qwen3.5-2B"

STEP="${1:-11530}"
CKPT="${LF_ROOT}/saves/qwen3.5-2b/mikomiko/full_v0/checkpoint-${STEP}"


# gcc-12 on this node lacks cc1plus; use gcc-11 for CUDA JIT (same as the robot servers).
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"

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

source "${VENV}/bin/activate"

SEP="================================================================================"
echo "Starting vllm mikomiko server on http://0.0.0.0:${PORT}"
echo "  GPU                 : ${CUDA_VISIBLE_DEVICES}"
echo "  Served model name   : mikomiko"
echo "  Checkpoint          : saves/qwen3.5-2b/mikomiko/full_v0/checkpoint-${STEP}"
echo "  GPU util / max len  : ${GPU_UTIL} / ${MAX_LEN}"
echo "  Max num seqs        : ${MAX_NUM_SEQS}"
echo "  Image max pixels    : ${MAX_PIXELS}"
echo "${SEP}"

CMD=(
  vllm serve "${CKPT}"
  --served-model-name mikomiko
  --dtype bfloat16
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --limit-mm-per-prompt '{"image": 1}'
  --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}"
  --trust-remote-code
  --port "${PORT}"
)
[ "${ENFORCE_EAGER}" = "1" ] && CMD+=(--enforce-eager)
exec "${CMD[@]}"
