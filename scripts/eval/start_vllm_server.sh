#!/usr/bin/env bash
# 启动 vllm OpenAI-compatible server — Qwen3.5-27B + LoRA
# 模型常驻显存，后续推理直接打 HTTP API，无需重载。
#
# 用法:
#   bash scripts/eval/start_vllm_server.sh          # 前台运行（Ctrl-C 停止）
#   bash scripts/eval/start_vllm_server.sh &        # 后台运行
#
# 环境变量:
#   CUDA_VISIBLE_DEVICES   GPU 编号（默认 5）
#   PORT                   监听端口（默认 8101）
#   GPU_UTIL               显存占用比例 0~1（默认 0.6，约 84 GB）
#   MAX_LEN                最大序列长度（默认 8192）
#   ENFORCE_EAGER          1=禁用 CUDA graph（默认 0）
#
# 推理:
#   python scripts/eval/infer.py "描述图中场景" --image /path/to/img.png
#   python scripts/eval/eval.py --api-url http://localhost:8101

set -euo pipefail

VLLM_VENV_PATH="/workspace1/zhijun/AgentRobot/.venv-vllm"
BASE_MODEL="/workspace1/zhijun/hf_download/models/Qwen3.5-27B"
OVERFIT_LORA_DIR="/workspace1/zhijun/LlamaFactory/saves/qwen3.5-27b/robot/overfit"
MVTOKEN_0622_v0_LORA_DIR="/workspace1/zhijun/LlamaFactory/saves/qwen3.5-27b/robot/mvtoken_0622_v0"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
PORT="${PORT:-8101}"
GPU_UTIL="${GPU_UTIL:-0.7}"
MAX_LEN="${MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"   # Mamba cache 受显存限制，需低于可用 blocks 数
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

if [ ! -f "${VLLM_VENV_PATH}/bin/activate" ]; then
  echo "vLLM venv not found at ${VLLM_VENV_PATH}." >&2
  exit 1
fi

if [ ! -d "${BASE_MODEL}" ]; then
  echo "Model path does not exist: ${BASE_MODEL}" >&2
  exit 1
fi

# gcc-12 on this node lacks cc1plus; use gcc-11 for CUDA JIT compilation.
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin /usr/bin/g++-11}"

echo "Starting vllm server on http://0.0.0.0:${PORT}"
echo "  GPU            : ${CUDA_VISIBLE_DEVICES}"
echo "  GPU util       : ${GPU_UTIL}"
echo "  Max seq len    : ${MAX_LEN}"
echo "  Max num seqs   : ${MAX_NUM_SEQS}"
echo "  Enforce eager  : ${ENFORCE_EAGER}"
echo "  Base model     : ${BASE_MODEL}"
echo "  LoRA_000       : ${OVERFIT_LORA_DIR}"
echo "  LoRA_001       : ${MVTOKEN_0622_v0_LORA_DIR}"

source "${VLLM_VENV_PATH}/bin/activate"

VLLM_CMD=(
  vllm serve "${BASE_MODEL}"
  --dtype bfloat16
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --enable-lora
  --max-lora-rank 64
  --lora-modules "OVERFIT=${OVERFIT_LORA_DIR}"
  --lora-modules "mvtoken_0622_v0=${MVTOKEN_0622_v0_LORA_DIR}"
  --trust-remote-code
  --port "${PORT}"
)

if [ "${ENFORCE_EAGER}" = "1" ]; then
  VLLM_CMD+=(--enforce-eager)
fi

exec "${VLLM_CMD[@]}"
