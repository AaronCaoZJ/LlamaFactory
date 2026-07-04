#!/usr/bin/env bash
set -euo pipefail
# Sanity-check a training checkpoint: run the trained model on 5 test_unseen + 5 test_stratified
# images, score them (normalized, order-invariant tag P/R/F1 + atomic/composite/token F1), and
# ARCHIVE every run under predict_sanity/runs/step_<STEP>/ plus a growing sanity_history.tsv.
#
# Usage:  bash scripts/qwen3_5/eval/instruction_follow_check_mikomiko.sh [STEP] [GPU]
#   STEP : checkpoint step to test (default 11530). Waits until that checkpoint is fully saved.
#   GPU  : CUDA device for inference (default 6). Pick a GPU NOT used by training.

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-/workspace1/zhijun/LlamaFactory}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"
BASE_MODEL="/workspace1/zhijun/hf_download/models/Qwen3.5-2B"

STEP="${1:-11530}"
GPU="${2:-6}"
CKPT="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-2b/mikomiko/full_v0/checkpoint-${STEP}"
CONFIG_TMPL="${LLAMA_FACTORY_ROOT}/examples/inference/qwen3_5_2b_full_mikomiko.yaml"
CONFIG_RUN="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-2b/mikomiko/predict_sanity/_run_${STEP}.yaml"
PRED_DIR="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-2b/mikomiko/predict_sanity"
META="${LLAMA_FACTORY_ROOT}/data/mikomiko_tag/jsonl/sanity_10.jsonl"

source "${VENV_PATH}/bin/activate"
export DISABLE_VERSION_CHECK=1
_SHIM="${LLAMA_FACTORY_ROOT}/.cc-shim"
if [ -x "${_SHIM}/gcc" ] && echo 'int main(){return 0;}' | "${_SHIM}/gcc" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}"
fi

# ── 1. wait until checkpoint-STEP is fully written (dir + weights + trainer_state) ─────────────
echo "[sanity] waiting for ${CKPT} ..."
until [ -f "${CKPT}/model.safetensors" ] && [ -f "${CKPT}/trainer_state.json" ]; do
  sleep 30
done
sleep 5   # let the final files flush
echo "[sanity] checkpoint ready."

# ── 2. supplement the VL processor files LlamaFactory doesn't save into the checkpoint ─────────
for f in preprocessor_config.json video_preprocessor_config.json merges.txt vocab.json; do
  if [ ! -e "${CKPT}/${f}" ] && [ -e "${BASE_MODEL}/${f}" ]; then
    cp "${BASE_MODEL}/${f}" "${CKPT}/${f}"
    echo "[sanity] copied ${f} from base model"
  fi
done

# ── 3. write a run config pointing at this checkpoint, then predict on the chosen GPU ──────────
mkdir -p "${PRED_DIR}"
sed "s#^model_name_or_path:.*#model_name_or_path: ${CKPT}#" "${CONFIG_TMPL}" > "${CONFIG_RUN}"
echo "[sanity] running prediction on GPU ${GPU} ..."
cd "${LLAMA_FACTORY_ROOT}"
env CUDA_VISIBLE_DEVICES="${GPU}" llamafactory-cli train "${CONFIG_RUN}"

# ── 4. archive this run (predictions + config + report + metrics) and append to history ─────────
RUN_DIR="${PRED_DIR}/runs/step_${STEP}"
HISTORY="${PRED_DIR}/sanity_history.tsv"
mkdir -p "${RUN_DIR}"
cp "${PRED_DIR}/generated_predictions.jsonl" "${RUN_DIR}/predictions.jsonl"
cp "${CONFIG_RUN}" "${RUN_DIR}/config.yaml" 2>/dev/null || true

python3 - "${RUN_DIR}/predictions.jsonl" "${META}" "${STEP}" "${HISTORY}" "${RUN_DIR}/metrics.json" <<'PY' | tee "${RUN_DIR}/report.txt"
import json, sys, re, os
from datetime import datetime

pred_path, meta_path, step, history_file, metrics_out = sys.argv[1:6]
preds = [json.loads(l) for l in open(pred_path, encoding="utf-8")]
try:
    meta = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
except Exception:
    meta = []
use_meta = len(meta) == len(preds)   # meta only aligns for the 10-sample sanity set

def norm(t):
    """lowercase, drop punctuation, collapse whitespace -> canonical tag (order/spacing/case-proof)."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t.strip().lower()).split())

def tagset(s):
    return {n for t in s.split(",") if (n := norm(t))}

def toks(ts):                         # word-level token set (credits near/substring tags)
    out = set()
    for t in ts:
        out.update(t.split())
    return out

def prf(tp, npred, ngt):
    p = tp / npred if npred else 0.0
    r = tp / ngt   if ngt   else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)

rows = []
for i, pr in enumerate(preds):
    g, p = tagset(pr.get("label", "")), tagset(pr.get("predict", ""))
    ga, pa = {t for t in g if " " not in t}, {t for t in p if " " not in t}   # atomic (1 word)
    gc, pc = {t for t in g if " " in t},     {t for t in p if " " in t}       # composite (>=2 words)
    gtk, ptk = toks(g), toks(p)
    rows.append(dict(
        img=(meta[i]["images"][0].split("/")[-1] if use_meta else f"#{i+1}"),
        src=(meta[i].get("_src", "?") if use_meta else "?"),
        tp=len(g & p), npred=len(p), ngt=len(g),
        atp=len(ga & pa), anp=len(pa), ang=len(ga),
        ctp=len(gc & pc), cnp=len(pc), cng=len(gc),
        ttp=len(gtk & ptk), tnp=len(ptk), tng=len(gtk),
        gt=pr.get("label", ""), pd=pr.get("predict", ""),
    ))

print("\n" + "=" * 100)
for i, r in enumerate(rows[:20]):
    p_, r_, f_ = prf(r["tp"], r["npred"], r["ngt"])
    print(f"[{i+1}] {r['img']} ({r['src']})  P={p_*100:.0f}%  R={r_*100:.0f}%  F1={f_*100:.0f}%")
    print(f"    GT  : {r['gt']}")
    print(f"    PRED: {r['pd']}")
    print("-" * 100)
if len(rows) > 20:
    print(f"... {len(rows) - 20} more rows omitted from per-image detail\n")

def agg(sel):
    sub = [r for r in rows if sel(r)]
    if not sub:
        return None
    def mf(a, b, c):
        return prf(sum(r[a] for r in sub), sum(r[b] for r in sub), sum(r[c] for r in sub))
    miP, miR, miF = mf("tp", "npred", "ngt")
    maF = sum(prf(r["tp"], r["npred"], r["ngt"])[2] for r in sub) / len(sub)
    return dict(n=len(sub), microP=miP, microR=miR, microF1=miF, macroF1=maF,
                atomicF1=mf("atp", "anp", "ang")[2], compositeF1=mf("ctp", "cnp", "cng")[2],
                tokenF1=mf("ttp", "tnp", "tng")[2])

groups = {"ALL": lambda r: True,
          "unseen": lambda r: r["src"] == "unseen",
          "stratified": lambda r: r["src"] == "strat"}
results = {name: agg(sel) for name, sel in groups.items()}

print("=" * 100)
print(f"{'group':<12}{'n':>5}{'microP':>8}{'microR':>8}{'microF1':>8}{'macroF1':>8}{'atomF1':>8}{'compF1':>8}{'tokF1':>8}")
for name in groups:
    a = results[name]
    if a:
        print(f"{name:<12}{a['n']:>5}{a['microP']*100:>7.1f}{a['microR']*100:>8.1f}{a['microF1']*100:>8.1f}"
              f"{a['macroF1']*100:>8.1f}{a['atomicF1']*100:>8.1f}{a['compositeF1']*100:>8.1f}{a['tokenF1']*100:>8.1f}")
print("strict=exact normalized tag set (order-invariant) | atom/comp=single vs multi-word | tok=word-level")

# ── persist ──
ts = datetime.now().isoformat(timespec="seconds")
json.dump({"step": step, "time": ts, "n": len(rows), "groups": results,
           "per_image": [{k: r[k] for k in ("img", "src", "tp", "npred", "ngt")} for r in rows]},
          open(metrics_out, "w"), ensure_ascii=False, indent=2)

new = not os.path.exists(history_file)
A, U, S = results["ALL"], results.get("unseen"), results.get("stratified")
with open(history_file, "a") as f:
    if new:
        f.write("step\ttime\tn\tmicroF1\tmacroF1\tatomF1\tcompF1\ttokF1\tunseen_microF1\tstrat_microF1\n")
    f.write(f"{step}\t{ts}\t{A['n']}\t{A['microF1']*100:.1f}\t{A['macroF1']*100:.1f}\t{A['atomicF1']*100:.1f}"
            f"\t{A['compositeF1']*100:.1f}\t{A['tokenF1']*100:.1f}"
            f"\t{(U['microF1']*100 if U else 0):.1f}\t{(S['microF1']*100 if S else 0):.1f}\n")
print(f"\n[metrics] history appended -> {history_file}")
PY

echo "[sanity] saved -> ${RUN_DIR}/ (predictions.jsonl, report.txt, metrics.json)"
echo "[sanity] history -> ${HISTORY}"
