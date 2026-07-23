#!/usr/bin/env bash
# InternVL3.5 MVTOKEN 评测的日志包装（真正的逻辑在 scripts/internvl/eval/infer.py，
# 通用核心在 scripts/eval_common/mvtoken_client.py；用法见 scripts/eval_common/README.md）。
set -euo pipefail

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

# :8202 = InternVL3.5-2B / :8201 = InternVL3.5-1B（见 start_vllm_server.sh 的 --lora-modules）。
export API_URL="${API_URL:-http://localhost:8202}"
export MODEL_NAME="${MODEL_NAME:-internvl3.5-2b}"

# ⚠️ prompt 版本必须与该 LoRA 的训练数据一致，否则置信度全是废的（映射表见 eval_common/README.md）。
#   internvl3.5-2b / internvl3.5-1b -> v3（franka+piper mix，Franka 数据直接评）
#   History2* / VideoSlot / ms0717_* -> 不适用 ood_sample（6 图契约 / video 槽位 / ManiSkill 仿真）
EVALSET="${EVALSET:-${LF_ROOT}/data/agentrobot/ood_sample/v3/rollout_lite.json}"

cd "${LF_ROOT}"
source .venv/bin/activate

# ── 日志设置 ──────────────────────────────────────────────────────────────────
RESULTS_DIR="${LF_ROOT}/results/internvl"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULTS_DIR}/${TIMESTAMP}.txt"

# ── 批量评测 + logprobs（推荐；JSONL 落到 results/logprobs/）────────────────────
EVAL_CMD="python scripts/internvl/eval/infer.py eval -e ${EVALSET} -n 50 --logprobs"

# ── 备选：只看 token 级准确率 ────────────────────────────────────────────────
# EVAL_CMD="python scripts/internvl/eval/infer.py eval -e ${EVALSET} -n 50"

# ── 备选：显示完整回复（验证指令遵从）/ Stage 消融（只有 v0 有 Stage 行）──────
# EVAL_CMD="python scripts/internvl/eval/infer.py eval -e ${EVALSET} -n 50 --raw --no-stage"

# ── 备选：单条 VQA ───────────────────────────────────────────────────────────
# VQA_IMAGE="${LF_ROOT}/data/agentrobot/ood_sample/agentview/0000.png"
# EVAL_CMD="python scripts/internvl/eval/infer.py single 'Describe the image in detail' --image '${VQA_IMAGE}'"

# ── 写入日志头 ────────────────────────────────────────────────────────────────
{
    echo "=== Eval Log ==="
    echo "Timestamp : ${TIMESTAMP}"
    echo "Model     : ${MODEL_NAME}"
    echo "API_URL   : ${API_URL}"
    echo "Evalset   : ${EVALSET}"
    echo "Command   : ${EVAL_CMD}"
    echo "================"
} | tee "${LOG_FILE}"

# ── 执行并同时写入日志 ────────────────────────────────────────────────────────
eval "${EVAL_CMD}" 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "Log saved to: ${LOG_FILE}"
