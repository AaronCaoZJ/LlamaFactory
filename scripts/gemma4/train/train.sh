#!/usr/bin/env bash
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-/workspace1/zhijun/LlamaFactory}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv-gemma4}"

DATA_DIR="${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/0622"
TRAIN_CONFIG="${LLAMA_FACTORY_ROOT}/examples/train_lora/gemma4_12b_mix_22_27_v3.yaml"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

# ── Activate venv ────────────────────────────────────────────────────────────
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv not found at ${VENV_PATH}." >&2
  echo "Run first: bash ${LLAMA_FACTORY_ROOT}/env_setup.sh" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"
export DISABLE_VERSION_CHECK=1  # gemma4 需 transformers>=5.10，绕过 LF 硬编码上限

# ── Launch training ───────────────────────────────────────────────────────────
echo "Starting training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ..."
cd "${LLAMA_FACTORY_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  llamafactory-cli train "${TRAIN_CONFIG}"
