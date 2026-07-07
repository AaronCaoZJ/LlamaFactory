#!/usr/bin/env bash
set -euo pipefail

# machine-agnostic paths via workspace_dir.sh -> .env.paths (create .env.paths first; see .env.paths.example)
_d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_d" != "/" ] && [ ! -f "$_d/scripts/workspace_dir.sh" ]; do _d="$(dirname "$_d")"; done
source "$_d/scripts/workspace_dir.sh"
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