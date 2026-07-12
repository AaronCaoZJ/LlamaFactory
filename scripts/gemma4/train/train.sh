#!/usr/bin/env bash
set -euo pipefail
# ═══ GPU / runtime knobs (edit here) ═══
GPU="${GPU:-3,4}"
MODEL="${MODEL:-e4b}"   # 12b | e4b

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

# ── Paths ────────────────────────────────────────────────────────────────────
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv-gemma4}"

DATA_DIR="${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/0622"

case "${MODEL}" in
  12b) TRAIN_CONFIG="${LLAMA_FACTORY_ROOT}/examples/train_lora/gemma4/gemma4_12b_mix_22_27_v3.yaml" ;;
  e4b) TRAIN_CONFIG="${LLAMA_FACTORY_ROOT}/examples/train_lora/gemma4/gemma4_e4b_mix_22_27_v3.yaml" ;;
  *)   echo "ERROR: unknown MODEL=${MODEL} (expected 12b or e4b)" >&2; exit 1 ;;
esac

if [ ! -f "${TRAIN_CONFIG}" ]; then
  echo "ERROR: train config not found: ${TRAIN_CONFIG}" >&2
  exit 1
fi


# ── Activate venv ────────────────────────────────────────────────────────────
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv not found at ${VENV_PATH}." >&2
  echo "Run first: bash ${LLAMA_FACTORY_ROOT}/env_setup.sh" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"
export DISABLE_VERSION_CHECK=1  # gemma4 需 transformers>=5.10，绕过 LF 硬编码上限

# ── Launch training ───────────────────────────────────────────────────────────
echo "Starting training: MODEL=${MODEL} GPU=${GPU} CONFIG=${TRAIN_CONFIG}"
cd "${LLAMA_FACTORY_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${GPU}" \
  llamafactory-cli train "${TRAIN_CONFIG}"
