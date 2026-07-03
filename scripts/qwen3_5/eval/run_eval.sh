#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export API_URL="${API_URL:-http://localhost:8101}"

# :8101 上 serve 的三个模型名（见 start_vllm_server.sh）：
# export MODEL_NAME="${MODEL_NAME:-/workspace1/zhijun/hf_download/models/Qwen3.5-27B}"
export MODEL_NAME="${MODEL_NAME:-mvtoken_0622_v2}"
# export MODEL_NAME="${MODEL_NAME:-mvtoken_0622_v1}"
# export MODEL_NAME="${MODEL_NAME:-mvtoken_0622_v0}"

# OOD 测试集（留空则用 infer.py 默认数据集；传入则覆盖）
EVALSET="/workspace1/zhijun/LlamaFactory/scripts/eval/ood_sample/v2/rollout_lite.json"
EVALSET_ARG="${EVALSET:+--evalset ${EVALSET}}"

cd /workspace1/zhijun/LlamaFactory
source .venv/bin/activate

VQA_IMAGE="/workspace1/zhijun/LlamaFactory/data/agentrobot/overfit_test/rollout_000/agentview/0000.png"

# ── 日志设置 ──────────────────────────────────────────────────────────────────
RESULTS_DIR="/workspace1/zhijun/LlamaFactory/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULTS_DIR}/${TIMESTAMP}.txt"

# ── 批量评估：准确率 ──────────────────────────────────────────────────────────
# EVAL_CMD="python scripts/eval/infer.py eval -n 100 ${EVALSET_ARG}"

# ── 批量评估：显示完整回复（验证指令遵从，原 eval_MVTOKEN.sh）─────────────────
EVAL_CMD="python scripts/eval/infer.py eval --raw --no-stage -n 100 ${EVALSET_ARG}"

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
