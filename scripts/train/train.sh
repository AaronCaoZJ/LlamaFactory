#!/usr/bin/env bash
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-/workspace1/zhijun/LlamaFactory}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"

DATA_DIR="${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/0622"
TRAIN_CONFIG="${LLAMA_FACTORY_ROOT}/examples/train_lora/qwen3_5_27b_mvtoken_0622_v1.yaml"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

# TASK_MAP=(
#   "banana=pick up the banana and place it on the blue plate"
#   "yellow_cup=pick up the yellow cup and place it on the green coaster"
#   "mango=pick up the mango and place it on the blue plate"
#   "white_bowl=pick up the white bowl and stack it on the pink bowl"
# )

# ── Activate venv ────────────────────────────────────────────────────────────
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv not found at ${VENV_PATH}." >&2
  echo "Run first: bash ${LLAMA_FACTORY_ROOT}/env_setup.sh" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"

# ── Generate dataset ─────────────────────────────────────────────────────────
# echo "Generating dataset from ${DATA_DIR} ..."
# python "${LLAMA_FACTORY_ROOT}/data/agentrobot/rollout_to_llamafactory.py" \
#   "${DATA_DIR}/banana" \
#   "${DATA_DIR}/yellow_cup" \
#   "${DATA_DIR}/mango" \
#   "${DATA_DIR}/white_bowl" \
#   --task-map "${TASK_MAP[@]}" \
#   --output "${DATA_DIR}/rollout.json"

# ── Launch training ───────────────────────────────────────────────────────────
echo "Starting training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ..."
cd "${LLAMA_FACTORY_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  llamafactory-cli train "${TRAIN_CONFIG}"
