#!/usr/bin/env bash
set -euo pipefail

# resolve machine paths: locate & source scripts/workspace_dir.sh (sets LF_ROOT, MODELS_DIR, LF_VENV, VLLM_VENV, AGENTROBOT_ROOT, HF_HOME)
_wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
source "$_wsd/scripts/workspace_dir.sh"

LLAMA_FACTORY_ROOT="${LF_ROOT}"
VLLM_VENV_PATH="${LF_ROOT}/.venv-vllm"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

cd "${LLAMA_FACTORY_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv, then rerun this script." >&2
  echo "See: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv venv --clear --python "${PYTHON_VERSION}" "${VLLM_VENV_PATH}"
source "${VLLM_VENV_PATH}/bin/activate"

uv pip install -r "${LLAMA_FACTORY_ROOT}/requirements/requirements-vllm.txt"