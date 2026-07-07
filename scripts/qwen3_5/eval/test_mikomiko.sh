#!/usr/bin/env bash
set -euo pipefail
# Large-scale eval of a training checkpoint on the mikomiko image->tag task (hf-predict path).
# For the faster vLLM path see start_vllm_server_mikomiko.sh + eval_vllm_mikomiko.py.
#
# Eval set = the two EXISTING mini test sets, reused directly:
#   test_unseen_mini.jsonl (200)  +  test_stratified_mini.jsonl (200)  = 400 samples.
# They are concatenated into eval_mini.jsonl with an added `_src` field so the scorer can split
# unseen vs stratified. Built once and reused, so every checkpoint is scored on the SAME 400 images.
#
# Scoring is delegated to metrics_mikomiko.py (shared with the vLLM path so metrics are identical).
#   KEPT   : micro P/R/F1, macro F1, atomF1, compEx, compSub, tokF1 + over-generation diagnostic.
#   DROPPED: BLEU-4 / ROUGE-* (LlamaFactory seq2seq auto-metrics, meaningless for unordered tags).
#
# Usage:  bash scripts/qwen3_5/eval/test_mikomiko.sh [STEP] [GPU]
#   STEP : checkpoint step to test (default 11530). Waits until that checkpoint is fully saved.
#   GPU  : CUDA device for inference (default 1). Pick a GPU NOT used by training.
# Env override: EVAL_BS (default 16) per-device eval batch size.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# machine-agnostic paths via the central workspace_dir.sh (no hardcoded /workspace1)
_d="$SCRIPT_DIR"; while [ "$_d" != "/" ] && [ ! -f "$_d/scripts/workspace_dir.sh" ]; do _d="$(dirname "$_d")"; done
source "$_d/scripts/workspace_dir.sh"
LLAMA_FACTORY_ROOT="${LF_ROOT}"
VENV_PATH="${LF_VENV}"
BASE_MODEL="${MODELS_DIR}/Qwen3.5-2B"

STEP="${1:-11530}"
GPU="${2:-1}"
EVAL_BS="${EVAL_BS:-16}"

CKPT="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-2b/mikomiko/full_v0/checkpoint-${STEP}"
CONFIG_TMPL="${LLAMA_FACTORY_ROOT}/examples/inference/qwen3_5_2b_full_mikomiko.yaml"
CONFIG_RUN="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-2b/mikomiko/predict_sanity/_run_evalmini_${STEP}.yaml"
PRED_DIR="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-2b/mikomiko/predict_sanity"
JSONL_DIR="${LLAMA_FACTORY_ROOT}/data/mikomiko_tag/jsonl"
META="${JSONL_DIR}/eval_mini.jsonl"     # 200 unseen + 200 strat, carries _src for grouping; also the dataset file

source "${VENV_PATH}/bin/activate"
export DISABLE_VERSION_CHECK=1
_SHIM="${LLAMA_FACTORY_ROOT}/.cc-shim"
if [ -x "${_SHIM}/gcc" ] && echo 'int main(){return 0;}' | "${_SHIM}/gcc" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}"
fi

cd "${LLAMA_FACTORY_ROOT}"

# ── 0. build eval_mini.jsonl by REUSING the two existing mini test sets (+ _src for grouping) ────
if [ ! -f "${META}" ]; then
  echo "[eval] building ${META} by reusing test_unseen_mini + test_stratified_mini (+_src) ..."
  python3 - "${JSONL_DIR}" "${META}" <<'PY'
import json, sys
jdir, out = sys.argv[1], sys.argv[2]
def load(path, src):
    rows = []
    for l in open(f"{jdir}/{path}"):
        l = l.strip()
        if not l:
            continue
        d = json.loads(l); d["_src"] = src; rows.append(d)
    return rows
rows = load("test_unseen_mini.jsonl", "unseen") + load("test_stratified_mini.jsonl", "strat")
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[eval] wrote {len(rows)} rows -> {out}")
PY
else
  echo "[eval] reusing existing ${META} ($(wc -l < "${META}") rows)"
fi

# ── 1. wait until checkpoint-STEP is fully written (dir + weights + trainer_state) ─────────────
echo "[eval] waiting for ${CKPT} ..."
until [ -f "${CKPT}/model.safetensors" ] && [ -f "${CKPT}/trainer_state.json" ]; do
  sleep 30
done
sleep 5   # let the final files flush
echo "[eval] checkpoint ready."

# ── 2. supplement the VL processor files LlamaFactory doesn't save into the checkpoint ─────────
for f in preprocessor_config.json video_preprocessor_config.json merges.txt vocab.json; do
  if [ ! -e "${CKPT}/${f}" ] && [ -e "${BASE_MODEL}/${f}" ]; then
    cp "${BASE_MODEL}/${f}" "${CKPT}/${f}"
    echo "[eval] copied ${f} from base model"
  fi
done

# ── 3. write a run config: point at this checkpoint + the 400-sample eval set, then predict ─────
mkdir -p "${PRED_DIR}"
sed -e "s#^model_name_or_path:.*#model_name_or_path: ${CKPT}#" \
    -e "s#^eval_dataset:.*#eval_dataset: mikomiko_tag_eval_mini#" \
    -e "s#^per_device_eval_batch_size:.*#per_device_eval_batch_size: ${EVAL_BS}#" \
    "${CONFIG_TMPL}" > "${CONFIG_RUN}"
echo "[eval] running prediction on GPU ${GPU} (bs=${EVAL_BS}, 400 samples) ..."
env CUDA_VISIBLE_DEVICES="${GPU}" llamafactory-cli train "${CONFIG_RUN}"

# ── 4. archive this run and score with the shared scorer (appends to evalmini_history.tsv) ──────
RUN_DIR="${PRED_DIR}/runs/evalmini_step_${STEP}"
HISTORY="${PRED_DIR}/evalmini_history.tsv"
mkdir -p "${RUN_DIR}"
cp "${PRED_DIR}/generated_predictions.jsonl" "${RUN_DIR}/predictions.jsonl"
cp "${CONFIG_RUN}" "${RUN_DIR}/config.yaml" 2>/dev/null || true

python3 "${SCRIPT_DIR}/metrics_mikomiko.py" \
  "${RUN_DIR}/predictions.jsonl" "${META}" "${STEP}" "${HISTORY}" "${RUN_DIR}/metrics.json" \
  | tee "${RUN_DIR}/report.txt"

echo "[eval] saved -> ${RUN_DIR}/ (predictions.jsonl, report.txt, metrics.json)"
echo "[eval] history -> ${HISTORY}"
