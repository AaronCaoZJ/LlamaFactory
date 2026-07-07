#!/usr/bin/env bash
set -euo pipefail
# ═══ GPU / runtime knobs (edit here) ═══
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export API_URL="${API_URL:-http://localhost:8101}"

# resolve machine paths: locate & source scripts/workspace_dir.sh (sets LF_ROOT, MODELS_DIR, LF_VENV, VLLM_VENV, AGENTROBOT_ROOT, HF_HOME)
_wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
source "$_wsd/scripts/workspace_dir.sh"


# :8101 上 serve 的三个模型名（见 start_vllm_server.sh）：
# export MODEL_NAME="${MODEL_NAME:-${MODELS_DIR}/Qwen3.5-27B}"
export MODEL_NAME="${MODEL_NAME:-mvtoken_0622_v2}"
# export MODEL_NAME="${MODEL_NAME:-mvtoken_0622_v1}"
# export MODEL_NAME="${MODEL_NAME:-mvtoken_0622_v0}"

# OOD 测试集（留空则用 infer.py 默认数据集；传入则覆盖）
EVALSET="${LF_ROOT}/scripts/eval/ood_sample/v2/rollout_lite.json"
EVALSET_ARG="${EVALSET:+--evalset ${EVALSET}}"

cd ${LF_ROOT}
source .venv/bin/activate

VQA_IMAGE="${LF_ROOT}/data/agentrobot/overfit_test/rollout_000/agentview/0000.png"

# ── 日志设置 ──────────────────────────────────────────────────────────────────
RESULTS_DIR="${LF_ROOT}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULTS_DIR}/${TIMESTAMP}.txt"

# ── 批量评估：准确率 ──────────────────────────────────────────────────────────
# EVAL_CMD="python scripts/eval/infer.py eval -n 100 ${EVALSET_ARG}"

# ── 批量评估：显示完整回复（验证指令遵从，原 eval_MVTOKEN.sh）─────────────────
EVAL_CMD="python scripts/eval/eval_mvtoken.py eval --raw --no-stage -n 100 ${EVALSET_ARG}"

# ── 单条 VQA（原 eval_VQA.sh）───────────────────────────────────────────────
# EVAL_CMD="python scripts/eval/infer.py single 'Describe the image in detail' --image '${VQA_IMAGE}'"

# ── 写入日志头 ────────────────────────────────────────────────────────────────
{
    echo "=== Eval Log ==="
    echo "Timestamp : ${TIMESTAMP}"
    echo "Model     : ${MODEL_NAME}"
    echo "API_URL   : ${API_URL}"
    echo "Evalset   : ${EVALSET:-<infer.py default>}"
    echo "Command   : ${EVAL_CMD}"
    echo "================"
} | tee "${LOG_FILE}"

# ── 执行并同时写入日志 ────────────────────────────────────────────────────────
eval "${EVAL_CMD}" 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "Log saved to: ${LOG_FILE}"
