#!/usr/bin/env bash
# Seen-probe eval: score a checkpoint on 200 RANDOM samples FROM THE TRAINING SET
# (data/mikomiko_tag/jsonl/train_seen_mini.jsonl, seed=42, _src=train_seen).
#
# Purpose: 记忆探针。模型对"原样训过"的图重打 tag 的能力上限:
#   - train_seen F1 >> unseen F1  -> 拟合-泛化 gap 真实存在,加数据多样性有空间
#   - train_seen F1 ≈  unseen F1  -> 标签自相矛盾(gold 不一致),堆步数无提升空间
# 三层探针: train_seen(原图见过) / stratified(同 post 不同张,近似见过) / unseen(post 级全新)。
#
# Usage:  bash scripts/qwen3_5/mikomiko_tagger/test_mikomiko_seen.sh [STEP] [GPU]
# 结果落盘 predict_sanity/runs/seen_step_<STEP>/;history 单独记 seen_history.tsv
# (不写主 evalmini_history.tsv,因为本集无 unseen/strat 分组,混记会出现 0 值行)。
set -euo pipefail

GPU="${2:-1}"
CKPT_STEP="${1:-${CKPT_STEP:-17296}}"
EVAL_BS="${EVAL_BS:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(
  d="$(dirname "${BASH_SOURCE[0]}")"
  until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done
  echo "$d"
)/scripts/workspace_dir.sh"

VENV_PATH="${LF_VENV}"
BASE_MODEL="${MODELS_DIR}/Qwen3.5-2B"
CKPT="${LF_ROOT}/saves/qwen3.5-2b/mikomiko/full_v0_resume11530/checkpoint-${CKPT_STEP}"
# CKPT="${MODELS_DIR}/Mikomiko_pornpic_tagger/checkpoint-${CKPT_STEP}"
CONFIG_TMPL="${LF_ROOT}/examples/inference/qwen3_5_2b_full_mikomiko.yaml"
CONFIG_RUN="${LF_ROOT}/saves/qwen3.5-2b/mikomiko/predict_sanity/_run_seen_${CKPT_STEP}.yaml"
PRED_DIR="${LF_ROOT}/saves/qwen3.5-2b/mikomiko/predict_sanity"
JSONL_DIR="${LF_ROOT}/data/mikomiko_tag/jsonl"
META="${JSONL_DIR}/train_seen_mini.jsonl"   # 200 random train rows, _src=train_seen

source "${VENV_PATH}/bin/activate"
export DISABLE_VERSION_CHECK=1

_SHIM="${LF_ROOT}/.cc-shim"
if [ -x "${_SHIM}/gcc" ] && echo 'int main(){return 0;}' | "${_SHIM}/gcc" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}"
fi

cd "${LF_ROOT}"

[ -f "${META}" ] || { echo "[seen] missing ${META} (rebuild: random.sample(train.jsonl, 200), seed=42)"; exit 1; }

echo "[seen] waiting for ${CKPT} ..."
until [ -f "${CKPT}/model.safetensors" ] && [ -f "${CKPT}/trainer_state.json" ]; do
  sleep 30
done
sleep 5
echo "[seen] checkpoint ready."

for f in preprocessor_config.json video_preprocessor_config.json merges.txt vocab.json; do
  if [ ! -e "${CKPT}/${f}" ] && [ -e "${BASE_MODEL}/${f}" ]; then
    cp "${BASE_MODEL}/${f}" "${CKPT}/${f}"
    echo "[seen] copied ${f} from base model"
  fi
done

mkdir -p "${PRED_DIR}"
sed -e "s#^model_name_or_path:.*#model_name_or_path: ${CKPT}#" \
    -e "s#^eval_dataset:.*#eval_dataset: mikomiko_tag_train_seen_mini#" \
    -e "s#^per_device_eval_batch_size:.*#per_device_eval_batch_size: ${EVAL_BS}#" \
    "${CONFIG_TMPL}" > "${CONFIG_RUN}"
echo "[seen] running prediction on GPU ${GPU} (bs=${EVAL_BS}, 200 train-seen samples) ..."
env CUDA_VISIBLE_DEVICES="${GPU}" llamafactory-cli train "${CONFIG_RUN}"

RUN_DIR="${PRED_DIR}/runs/seen_step_${CKPT_STEP}"
HISTORY="${PRED_DIR}/seen_history.tsv"
mkdir -p "${RUN_DIR}"
cp "${PRED_DIR}/generated_predictions.jsonl" "${RUN_DIR}/predictions.jsonl"
cp "${CONFIG_RUN}" "${RUN_DIR}/config.yaml" 2>/dev/null || true

python3 "${SCRIPT_DIR}/metrics_mikomiko.py" \
  "${RUN_DIR}/predictions.jsonl" "${META}" "${CKPT_STEP}" "${HISTORY}" "${RUN_DIR}/metrics.json" \
  | tee "${RUN_DIR}/report.txt"

echo "[seen] saved -> ${RUN_DIR}/  (ALL 行即 train_seen 的 200 条;unseen/strat 组为空属预期)"
echo "[seen] 对照读法: train_seen(此处 ALL) vs evalmini 的 unseen —— gap 大=有学习空间, gap≈0=标签天花板"
