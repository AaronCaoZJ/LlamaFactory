#!/usr/bin/env bash
set -euo pipefail

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

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