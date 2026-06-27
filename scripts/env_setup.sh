#!/usr/bin/env bash
set -euo pipefail

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-/workspace1/zhijun/LlamaFactory}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv, then rerun this script." >&2
  echo "See: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# Put uv cache under workspace to avoid permission issues with system cache dirs
export UV_CACHE_DIR="${LLAMA_FACTORY_ROOT}/.uv-cache"
# Use copy instead of hardlink (cross-filesystem safe)
export UV_LINK_MODE=copy

uv venv --python "${PYTHON_VERSION}" "${VENV_PATH}" --prompt "llamafactory"
source "${VENV_PATH}/bin/activate"

# Install PyTorch with CUDA 12.9 support first (flash-attn needs torch headers)
uv pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129

# Install build tools required by extensions with C/CUDA code
uv pip install setuptools wheel packaging "hatchling>=1.18.0" editables

# Install LLaMA-Factory as editable package (pulls all core deps from pyproject.toml)
uv pip install --no-build-isolation -e "${LLAMA_FACTORY_ROOT}"

# Install flash-attn (must come after torch so it can find torch headers)
uv pip install flash-attn --no-build-isolation

# --- Optional extras (uncomment as needed) ---

# Liger kernel (fused ops, speeds up training)
uv pip install -r "${LLAMA_FACTORY_ROOT}/requirements/liger-kernel.txt"

# DeepSpeed (multi-GPU ZeRO training)
uv pip install -r "${LLAMA_FACTORY_ROOT}/requirements/deepspeed.txt"

# BitsAndBytes (4-bit/8-bit quantization)
# uv pip install -r "${LLAMA_FACTORY_ROOT}/requirements/bitsandbytes.txt"

# Transformer Engine (FP8 training, requires Hopper GPU + NCCL headers)
# uv pip install --no-build-isolation -r "${LLAMA_FACTORY_ROOT}/requirements/fp8-te.txt"

# Evaluation metrics (BLEU, ROUGE)
uv pip install -r "${LLAMA_FACTORY_ROOT}/requirements/metrics.txt"

# vLLM (fast inference / RLHF rollout)
uv pip install -r "${LLAMA_FACTORY_ROOT}/requirements/vllm.txt"

uv pip install wandb
# uv pip install flash-linear-attention
uv pip install transformers==5.6.1
# uv pip install tilelang

cat <<EOF

LLaMA-Factory environment ready.

Activate with:
  source ${VENV_PATH}/bin/activate

Quick test:
  llamafactory-cli version

Train example:
  llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

Run WebUI:
  llamafactory-cli webui
EOF
