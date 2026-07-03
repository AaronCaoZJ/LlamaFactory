#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
# HF server 默认 8114；若用 vllm server 改成 8104。
export API_URL="${API_URL:-http://localhost:8104}"
export MODEL_NAME="${MODEL_NAME:-gemma4_12b_mix_22_27_v3}"

# OOD 测试集（留空则用 infer.py 默认数据集；这里复用 qwen3_5 的共享评测样本）
EVALSET="/workspace1/zhijun/LlamaFactory/data/agentrobot/ood_sample/v3/rollout_lite.json"
EVALSET_ARG="${EVALSET:+--evalset ${EVALSET}}"

cd /workspace1/zhijun/LlamaFactory
source .venv-gemma4/bin/activate
export DISABLE_VERSION_CHECK=1  # gemma4 需 transformers>=5.10，绕过 LF 硬编码上限

# ── 日志设置 ──────────────────────────────────────────────────────────────────
RESULTS_DIR="/workspace1/zhijun/LlamaFactory/results/gemma4"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULTS_DIR}/${TIMESTAMP}.txt"

# ── 批量评估：显示完整回复（验证指令遵从）────────────────────────────────────
EVAL_CMD="python scripts/gemma4/eval/infer.py eval --raw --no-stage -n 100 ${EVALSET_ARG}"

# ── 备选：只看 token 级准确率 ────────────────────────────────────────────────
# EVAL_CMD="python scripts/gemma4/eval/infer.py eval -n 100 ${EVALSET_ARG}"

# ── 备选：单条 VQA ───────────────────────────────────────────────────────────
# VQA_IMAGE="/workspace1/zhijun/LlamaFactory/data/agentrobot/overfit_test/rollout_000/agentview/0000.png"
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
