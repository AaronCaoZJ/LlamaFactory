#!/usr/bin/env bash
set -euo pipefail
# ═══ GPU / runtime knobs (edit here) ═══
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

# resolve machine paths: locate & source scripts/workspace_dir.sh (sets LF_ROOT, MODELS_DIR, LF_VENV, VLLM_VENV, AGENTROBOT_ROOT, HF_HOME)
_wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
source "$_wsd/scripts/workspace_dir.sh"

# HF server 默认 8114；若用 vllm server 改成 8104。
export API_URL="${API_URL:-http://localhost:8104}"
export MODEL_NAME="${MODEL_NAME:-gemma4_12b_mix_22_27_v3}"

# OOD 测试集（留空则用 infer.py 默认数据集；这里复用 qwen3_5 的共享评测样本）
EVALSET="${LF_ROOT}/data/agentrobot/ood_sample/v3/rollout_lite.json"
EVALSET_ARG="${EVALSET:+--evalset ${EVALSET}}"

cd ${LF_ROOT}
source .venv-gemma4/bin/activate
export DISABLE_VERSION_CHECK=1  # gemma4 需 transformers>=5.10，绕过 LF 硬编码上限

# ── 日志设置 ──────────────────────────────────────────────────────────────────
RESULTS_DIR="${LF_ROOT}/results/gemma4"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULTS_DIR}/${TIMESTAMP}.txt"

# ── 批量评估：显示完整回复（验证指令遵从）────────────────────────────────────
EVAL_CMD="python scripts/gemma4/eval/infer.py eval --raw --no-stage -n 100 ${EVALSET_ARG}"

# ── 备选：只看 token 级准确率 ────────────────────────────────────────────────
# EVAL_CMD="python scripts/gemma4/eval/infer.py eval -n 100 ${EVALSET_ARG}"

# ── 备选：单条 VQA ───────────────────────────────────────────────────────────
# VQA_IMAGE="${LF_ROOT}/data/agentrobot/overfit_test/rollout_000/agentview/0000.png"
# EVAL_CMD="python scripts/gemma4/eval/infer.py single 'Describe the image in detail' --image '${VQA_IMAGE}'"

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
