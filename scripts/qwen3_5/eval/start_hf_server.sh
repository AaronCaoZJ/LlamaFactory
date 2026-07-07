#!/usr/bin/env bash
# 启动 LlamaFactory HuggingFace API server — Qwen3.5-27B + LoRA
# 模型常驻显存，后续推理直接打 HTTP API，无需重载。
#
# 用法:
#   bash scripts/eval/start_server.sh              # 前台运行（Ctrl-C 停止）
#   bash scripts/eval/start_server.sh &            # 后台运行
#
# 推理:
#   python scripts/eval/eval.py --api-url http://localhost:8100
#   python scripts/eval/infer.py "描述图中场景" --image /path/to/img.png --url http://localhost:8100
#
# 注：HuggingFace backend 会全量加载模型，无法像 vllm 一样按比例限制显存。
#     如需显存控制，改用 start_vllm_server.sh（需要 vllm 支持 Qwen3.5）。

set -euo pipefail
# ═══ GPU / runtime knobs (edit here) ═══
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

# resolve machine paths: locate & source scripts/workspace_dir.sh (sets LF_ROOT, MODELS_DIR, LF_VENV, VLLM_VENV, AGENTROBOT_ROOT, HF_HOME)
_wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
source "$_wsd/scripts/workspace_dir.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${LF_ROOT}"   # was SCRIPT_DIR/../.. (only 2 levels -> pointed at scripts/qwen3_5, a bug); LF_ROOT is the repo root

BASE_MODEL="${MODELS_DIR}/Qwen3.5-27B"
LORA_DIR="${LF_ROOT}/saves/qwen3.5-27b/robot/overfit"

export API_PORT="${API_PORT:-8111}"
export SAFE_MEDIA_PATH="${SAFE_MEDIA_PATH:-$(dirname "${LF_ROOT}")}"  # 允许本地图片路径

echo "Starting LlamaFactory HF API server on http://0.0.0.0:${API_PORT}"
echo "  GPU            : ${CUDA_VISIBLE_DEVICES}"
echo "  Port           : ${API_PORT}"
echo "  Base model     : ${BASE_MODEL}"
echo "  LoRA           : ${LORA_DIR}"

cd "${REPO_ROOT}"
source .venv/bin/activate

exec llamafactory-cli api "${REPO_ROOT}/examples/inference/qwen35_27b_hf_api.yaml"
