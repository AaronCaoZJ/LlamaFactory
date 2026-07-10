#!/usr/bin/env bash
# One-shot seen/unseen review page: sample -> inference -> self-contained HTML.
#
# Generation goes through the single entry point ../infer_mikomiko.py. Default backend is vllm
# (bf16, the same engine that serves the model in production); this script starts the server,
# waits for it, runs inference, and shuts it down. BACKEND=hf skips the server and drives
# transformers directly. Scoring is ../metrics_mikomiko.py either way.
#
# Default is INCREMENTAL: if WORK_DIR/samples_pred.json already exists, sample+infer are skipped
# and only the HTML is rebuilt (fast, no GPU). FORCE=1 re-runs the whole pipeline.
#
# Usage:
#   bash build_html.sh                         # rebuild HTML from existing predictions
#   FORCE=1 N=200 GPU=4 bash build_html.sh     # the full 400-image page (200 seen + 200 unseen)
#   FORCE=1 BACKEND=hf DTYPE=fp32 bash build_html.sh   # transformers, batch-invariant greedy
#
# Timings (400 images, H200): sample 3s | vllm server up ~90s | infer ~40s | thumbnails+html ~70s.
#
# Env overrides:
#   FORCE     re-run sample+infer even if predictions exist (default 0)
#   N         samples per split (default 200 -> 400 images)   SEED       sampling seed (default 42)
#   BACKEND   vllm (default) | hf                             GPU        CUDA device (default 6)
#   PORT      vllm port (default 8111)                        GPU_UTIL   vllm mem frac (default 0.35)
#   API       reuse an already-running server at this url (skips start/stop)
#   DTYPE     bf16 (default) | fp32   [hf backend only]       BATCH_SIZE hf gen batch (default 8)
#   CKPT_STEP checkpoint step (default 17296)                 EPOCHS     sidebar subtitle (default 2.0)
#   WORK_DIR  intermediates (default saves/qwen3.5-2b/mikomiko/viz_review)
#   OUT       output html (default WORK_DIR/mikomiko_tagger_seen_unseen_review_<today>.html)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Walk up from the ABSOLUTE script dir. Starting from a relative dirname loops forever when the
# script is invoked as `bash build_html.sh` from its own directory, because dirname "." == ".".
_d="${SCRIPT_DIR}"
until [ -e "${_d}/scripts/workspace_dir.sh" ] || [ "${_d}" = / ]; do _d="$(dirname "${_d}")"; done
[ -e "${_d}/scripts/workspace_dir.sh" ] || { echo "ERROR: repo root not found above ${SCRIPT_DIR}" >&2; exit 1; }
source "${_d}/scripts/workspace_dir.sh"

FORCE="${FORCE:-0}"
N="${N:-200}"
SEED="${SEED:-42}"
BACKEND="${BACKEND:-vllm}"
GPU="${GPU:-6}"
PORT="${PORT:-8111}"
GPU_UTIL="${GPU_UTIL:-0.35}"
DTYPE="${DTYPE:-bf16}"           # hf backend only; vllm serves bf16
BATCH_SIZE="${BATCH_SIZE:-8}"
CKPT_STEP="${CKPT_STEP:-17296}"
EPOCHS="${EPOCHS:-2.0}"
WORK_DIR="${WORK_DIR:-${LF_ROOT}/saves/qwen3.5-2b/mikomiko/viz_review}"
OUT="${OUT:-${WORK_DIR}/mikomiko_tagger_seen_unseen_review_$(date +%Y%m%d).html}"
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

# ── sample + infer (skipped when predictions exist and FORCE!=1) ──────────────────────────────
if [ "${FORCE}" = "1" ] || [ ! -f "${WORK_DIR}/samples_pred.json" ]; then
  [ -f "${CKPT}/model.safetensors" ] || { echo "[pipeline] ERROR: no ckpt at ${CKPT}" >&2; exit 1; }

  echo "[pipeline] step 1/3 sample: N=${N} per split, seed=${SEED}"
  python3 -u sample_data.py --n "${N}" --seed "${SEED}" --work-dir "${WORK_DIR}"

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
    # fp32 weights need ~11 GiB (bf16 ~6). Fail here instead of OOM-ing 40s into model loading.
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
  echo "[pipeline] reuse ${WORK_DIR}/samples_pred.json (FORCE=1 to re-run sample+infer)"
fi

# ── build the page ───────────────────────────────────────────────────────────────────────────
echo "[pipeline] step 3/3 build html (thumbnail base64 encoding, ~70s for 400 images)"
python3 -u build_html.py --work-dir "${WORK_DIR}" --out "${OUT}" \
  --model-name "${MODEL_NAME}" --subtitle "${SUBTITLE}"
echo "[pipeline] DONE (${SECONDS}s) -> ${OUT}"
