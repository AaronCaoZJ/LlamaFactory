#!/usr/bin/env bash
# One-shot OnlyFans review page: list images -> inference -> self-contained HTML.
#
# Same machinery as build_html.sh (../infer_mikomiko.py for generation, start_server.sh for the
# vllm backend), but the images have NO gold tags, so build_html.py switches to its unscored mode:
# no F1, instead a distribution check (tags/atoms/composites per image + out-of-vocabulary tags).
#
# Default is INCREMENTAL: if WORK_DIR/samples_pred.json exists, inference is skipped and only the
# HTML is rebuilt (no GPU). FORCE=1 re-runs everything.
#
# Usage:
#   FORCE=1 GPU=0 bash build_onlyfans.sh              # all 480 images, vllm/bf16
#   FORCE=1 GPU=0 N=32 bash build_onlyfans.sh         # smoke test on 32 images
#   FORCE=1 GPU=0 BACKEND=hf DTYPE=fp32 bash build_onlyfans.sh
#   bash build_onlyfans.sh                            # rebuild HTML from existing predictions
#
# Env: FORCE | N | IMG_DIR | BACKEND(vllm|hf) | GPU | PORT | GPU_UTIL | API | DTYPE | BATCH_SIZE
#      CKPT_STEP | EPOCHS | WORK_DIR | OUT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="${SCRIPT_DIR}"
until [ -e "${_d}/scripts/workspace_dir.sh" ] || [ "${_d}" = / ]; do _d="$(dirname "${_d}")"; done
[ -e "${_d}/scripts/workspace_dir.sh" ] || { echo "ERROR: repo root not found above ${SCRIPT_DIR}" >&2; exit 1; }
source "${_d}/scripts/workspace_dir.sh"

FORCE="${FORCE:-0}"
N="${N:-}"                       # empty = all images
IMG_DIR="${IMG_DIR:-${LF_ROOT}/data/mikomiko_tag/onlyfans}"
BACKEND="${BACKEND:-vllm}"
GPU="${GPU:-0}"
PORT="${PORT:-8112}"             # not 8111: avoid clashing with a warm seen/unseen server
GPU_UTIL="${GPU_UTIL:-0.35}"
DTYPE="${DTYPE:-bf16}"           # hf backend only; vllm serves bf16
BATCH_SIZE="${BATCH_SIZE:-8}"
CKPT_STEP="${CKPT_STEP:-17296}"
EPOCHS="${EPOCHS:-2.0}"
WORK_DIR="${WORK_DIR:-${LF_ROOT}/saves/qwen3.5-2b/mikomiko/viz_onlyfans}"
OUT="${OUT:-${WORK_DIR}/mikomiko_tagger_onlyfans_review_$(date +%Y%m%d).html}"
MODEL_NAME="${MODEL_NAME:-QWEN 3.5 2B}"
SUBTITLE="${SUBTITLE:-Full SFT · ${EPOCHS} epochs · ${CKPT_STEP} steps · temperature 0.0}"
CKPT="${MODELS_DIR}/Mikomiko_pornpic_tagger/checkpoint-${CKPT_STEP}"
SERVER_PID=""
SERVER_LOG="${WORK_DIR}/vllm_server.log"

cleanup() {
  [ -n "${SERVER_PID}" ] && kill "${SERVER_PID}" 2>/dev/null && echo "[pipeline] vllm server stopped" || true
}
trap cleanup EXIT

cd "${SCRIPT_DIR}"
mkdir -p "${WORK_DIR}"

if [ "${FORCE}" = "1" ] || [ ! -f "${WORK_DIR}/samples_pred.json" ]; then
  [ -f "${CKPT}/model.safetensors" ] || { echo "[pipeline] ERROR: no ckpt at ${CKPT}" >&2; exit 1; }

  echo "[pipeline] step 1/3 list images from ${IMG_DIR}"
  python3 -u sample_onlyfans.py --img-dir "${IMG_DIR}" --work-dir "${WORK_DIR}" ${N:+--n "${N}"}

  if [ "${BACKEND}" = "vllm" ]; then
    API="${API:-http://localhost:${PORT}}"
    if curl -sf -m 3 "${API}/v1/models" >/dev/null 2>&1; then
      echo "[pipeline] step 2/3 reusing vllm server at ${API}"
    else
      echo "[pipeline] step 2/3 starting vllm server: GPU=${GPU} port=${PORT} (~90s) -> ${SERVER_LOG}"
      GPU="${GPU}" PORT="${PORT}" GPU_UTIL="${GPU_UTIL}" CKPT_STEP="${CKPT_STEP}" \
        bash start_server.sh >"${SERVER_LOG}" 2>&1 &
      SERVER_PID=$!
      for i in $(seq 1 120); do            # 10 min ceiling
        curl -sf -m 3 "${API}/v1/models" >/dev/null 2>&1 && break
        kill -0 "${SERVER_PID}" 2>/dev/null || {
          echo "[pipeline] ERROR: vllm server died. Tail of ${SERVER_LOG}:" >&2
          tail -25 "${SERVER_LOG}" >&2; exit 1; }
        [ "$i" = 120 ] && { echo "[pipeline] ERROR: server not up after 10min, see ${SERVER_LOG}" >&2; exit 1; }
        printf '.'; sleep 5
      done
      echo " up (${SECONDS}s)"
    fi
    INFER_ARGS=(--backend vllm --api "${API}")
  else
    NEED=$([ "${DTYPE}" = "fp32" ] && echo 14000 || echo 8000)
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}")
    [ "${FREE}" -ge "${NEED}" ] || {
      echo "[pipeline] ERROR: GPU ${GPU} has ${FREE} MiB free, need ~${NEED} MiB for ${DTYPE}." >&2
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >&2; exit 1; }
    echo "[pipeline] step 2/3 infer (hf): GPU=${GPU} bs=${BATCH_SIZE} dtype=${DTYPE}"
    INFER_ARGS=(--backend hf --ckpt "${CKPT}" --batch-size "${BATCH_SIZE}" --dtype "${DTYPE}")
  fi

  source "${LF_VENV}/bin/activate"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 -u "${SCRIPT_DIR}/../infer_mikomiko.py" \
    --input "${WORK_DIR}/samples.json" --output "${WORK_DIR}/samples_pred.json" "${INFER_ARGS[@]}"
  cleanup; SERVER_PID=""
else
  echo "[pipeline] reuse ${WORK_DIR}/samples_pred.json (FORCE=1 to re-run)"
fi

echo "[pipeline] step 3/3 build html (thumbnail base64 encoding, ~80s for 480 images)"
python3 -u build_html.py --work-dir "${WORK_DIR}" --out "${OUT}" \
  --model-name "${MODEL_NAME}" --subtitle "${SUBTITLE}"
echo "[pipeline] DONE (${SECONDS}s) -> ${OUT}"
