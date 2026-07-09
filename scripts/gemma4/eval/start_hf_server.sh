#!/usr/bin/env bash
# 启动 LlamaFactory HuggingFace API server — Gemma-4-12B + LoRA
# 模型常驻显存，后续推理直接打 HTTP API，无需重载。
#
# gemma4 是新架构，vLLM 未必支持；HF backend 最稳，优先用这个。
#
# 用法:
#   bash scripts/gemma4/eval/start_hf_server.sh       # 前台（Ctrl-C 停）
#   bash scripts/gemma4/eval/start_hf_server.sh &     # 后台
# 推理:
#   API_URL=http://localhost:8110 python scripts/gemma4/eval/infer.py eval -n 100 --raw

set -euo pipefail··
# ═══ GPU / runtime knobs (edit here) ═══
GPU="${GPU:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${LF_ROOT}"   # repo root from workspace_dir.sh (was SCRIPT_DIR/../../..)

export API_PORT="${API_PORT:-8114}"
export SAFE_MEDIA_PATH="${SAFE_MEDIA_PATH:-$(dirname "${LF_ROOT}")}"  # 允许本地图片路径

INFER_CONFIG="${REPO_ROOT}/examples/inference/gemma4_12b_lora.yaml"

echo "Starting LlamaFactory HF API server on http://0.0.0.0:${API_PORT}"
echo "  GPU     : ${GPU}"
echo "  Port    : ${API_PORT}"
echo "  Config  : ${INFER_CONFIG}"

cd "${REPO_ROOT}"
source .venv-gemma4/bin/activate
export DISABLE_VERSION_CHECK=1  # gemma4 需 transformers>=5.10，绕过 LF 硬编码上限

exec llamafactory-cli api "${INFER_CONFIG}"
