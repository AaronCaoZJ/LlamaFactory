#!/usr/bin/env bash
set -euo pipefail
# ═══ GPU / runtime knobs (edit here) ═══
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# resolve machine paths: locate & source scripts/workspace_dir.sh (sets LF_ROOT, MODELS_DIR, LF_VENV, VLLM_VENV, AGENTROBOT_ROOT, HF_HOME)
_wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
source "$_wsd/scripts/workspace_dir.sh"

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
