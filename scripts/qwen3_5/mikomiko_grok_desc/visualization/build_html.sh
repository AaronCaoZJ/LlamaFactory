#!/usr/bin/env bash
# One-shot grok_desc review page: sample -> SFT inference -> base inference -> self-contained HTML.
#
#   bash build_html.sh                      # rebuild the HTML from existing predictions (no GPU)
#   FORCE=1 bash build_html.sh              # the whole pipeline (~6 min on 2 idle H200s)
#   FORCE=1 WITH_BASE=0 bash build_html.sh  # skip the untuned-base column (halves the GPU time)
#
# The two models are served through the SAME chat template on purpose (see start_server.sh):
# identical prompt tokens, so the only difference between the two columns is the weights.
#
# Timings (120 images, H200): sample ~90s (streams the 7.3 GB train.jsonl) | server up ~110s each
# | inference ~40s each | thumbnails + page ~40s.
#
# Env: FORCE | N | SEED | WITH_BASE | GPU_SFT | GPU_BASE | PORT_SFT | PORT_BASE | CKPT | WORK_DIR | OUT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="${SCRIPT_DIR}"
until [ -e "${_d}/scripts/workspace_dir.sh" ] || [ "${_d}" = / ]; do _d="$(dirname "${_d}")"; done
[ -e "${_d}/scripts/workspace_dir.sh" ] || { echo "ERROR: repo root not found above ${SCRIPT_DIR}" >&2; exit 1; }
source "${_d}/scripts/workspace_dir.sh"

FORCE="${FORCE:-0}"
N="${N:-20}"                       # per language per split -> 6 * N images
SEED="${SEED:-42}"
WITH_BASE="${WITH_BASE:-1}"
GPU_SFT="${GPU_SFT:-0}"; PORT_SFT="${PORT_SFT:-8121}"
GPU_BASE="${GPU_BASE:-1}"; PORT_BASE="${PORT_BASE:-8122}"
MAX_NEW="${MAX_NEW:-1536}"         # gold p99 is ~1100 output tokens; SFT peaked at 904 here
CKPT="${CKPT:-${LF_ROOT}/saves/qwen3.5-9b/mikomiko/grok_desc_v0}"
BASE_MODEL="${BASE_MODEL:-${MODELS_DIR}/Qwen3.5-9B}"
WORK_DIR="${WORK_DIR:-${LF_ROOT}/saves/qwen3.5-9b/mikomiko/viz_desc_0721}"
OUT="${OUT:-${WORK_DIR}/mikomiko_grok_desc_review_$(date +%Y%m%d).html}"
SUBTITLE="${SUBTITLE:-Qwen3.5-9B 全参 SFT · 1 epoch · 13963 步 · eval_loss 0.4257 · 贪心解码}"

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "${p}" ] && kill "${p}" 2>/dev/null || true; done; }
trap cleanup EXIT

cd "${SCRIPT_DIR}"
mkdir -p "${WORK_DIR}"

# Wait for an OpenAI-compatible server, failing loudly if the process dies instead of hanging.
wait_up() {  # $1=url $2=pid $3=log
  for i in $(seq 1 90); do
    curl -sf -m 3 "$1/v1/models" >/dev/null 2>&1 && { echo " up (${SECONDS}s)"; return 0; }
    kill -0 "$2" 2>/dev/null || { echo "ERROR: server died, tail of $3:" >&2; tail -25 "$3" >&2; exit 1; }
    printf '.'; sleep 5
  done
  echo "ERROR: server not up after 7.5min, see $3" >&2; exit 1
}

run_model() {  # $1=tag $2=model_dir $3=served $4=gpu $5=port
  local log="${WORK_DIR}/vllm_$1.log" api="http://localhost:$5"
  if curl -sf -m 3 "${api}/v1/models" >/dev/null 2>&1; then
    echo "[pipeline] reusing server at ${api}"
  else
    echo -n "[pipeline] starting $1 server: GPU=$4 port=$5 -> ${log} "
    MODEL="$2" SERVED="$3" PORT="$5" GPU="$4" bash start_server.sh >"${log}" 2>&1 &
    PIDS+=($!); wait_up "${api}" "$!" "${log}"
  fi
  python3 -u infer_desc.py --input "${WORK_DIR}/samples_pred.json" \
    --output "${WORK_DIR}/samples_pred.json" --api "${api}" --model "$3" \
    --tag "$1" --max-new-tokens "${MAX_NEW}"
}

if [ "${FORCE}" = "1" ] || [ ! -f "${WORK_DIR}/samples_pred.json" ]; then
  ls "${CKPT}"/*.safetensors >/dev/null 2>&1 || { echo "[pipeline] ERROR: no ckpt at ${CKPT}" >&2; exit 1; }

  echo "[pipeline] step 1/3 sample: ${N} per language per split, seed=${SEED}"
  python3 -u sample_data.py --n "${N}" --seed "${SEED}" --work-dir "${WORK_DIR}"
  cp "${WORK_DIR}/samples.json" "${WORK_DIR}/samples_pred.json"   # preds accumulate into this file

  source "${LF_VENV}/bin/activate"
  echo "[pipeline] step 2/3 inference"
  run_model sft "${CKPT}" desc_sft "${GPU_SFT}" "${PORT_SFT}"
  [ "${WITH_BASE}" = "1" ] && run_model base "${BASE_MODEL}" desc_base "${GPU_BASE}" "${PORT_BASE}"
  cleanup; PIDS=()
else
  echo "[pipeline] reuse ${WORK_DIR}/samples_pred.json (FORCE=1 to re-run sample+infer)"
  source "${LF_VENV}/bin/activate"
fi

echo "[pipeline] step 3/3 build html"
python3 -u build_html.py --work-dir "${WORK_DIR}" --out "${OUT}" --subtitle "${SUBTITLE}" \
  --note "每语言 seen/unseen 各 ${N} 张；同一 prompt、同一 chat 模板下，微调后与未微调基座的输出并排。"
echo "[pipeline] DONE (${SECONDS}s) -> ${OUT}"
