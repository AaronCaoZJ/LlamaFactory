#!/usr/bin/env bash
set -euo pipefail
# ═══ GPU / runtime knobs (edit here) ═══
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

# ── Paths ────────────────────────────────────────────────────────────────────
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
AGENTROBOT_ROOT="${AGENTROBOT_ROOT:-${AGENTROBOT_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"

ROLLOUT_DIR="${LLAMA_FACTORY_ROOT}/data/agentrobot/overfit_test/rollout_000"
DATASET_JSON="${LLAMA_FACTORY_ROOT}/data/agentrobot/overfit_test/rollout_000.json"
TRAIN_CONFIG="${LLAMA_FACTORY_ROOT}/examples/train_lora/qwen3_5_27b_overfit.yaml"


# ── Activate venv ────────────────────────────────────────────────────────────
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv not found at ${VENV_PATH}." >&2
  echo "Run first: bash ${LLAMA_FACTORY_ROOT}/env_setup.sh" >&2
  exit 1
fi
# in short, just run `llama`
source "${VENV_PATH}/bin/activate"

# ── Generate dataset ─────────────────────────────────────────────────────────
echo "Generating dataset from ${ROLLOUT_DIR} ..."
python "${LLAMA_FACTORY_ROOT}/data/agentrobot/rollout_to_llamafactory.py" \
  "${ROLLOUT_DIR}" \
  --output "${DATASET_JSON}"

# ── Launch training ───────────────────────────────────────────────────────────
echo "Starting overfit test on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ..."
cd "${LLAMA_FACTORY_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  llamafactory-cli train "${TRAIN_CONFIG}"
