#!/usr/bin/env bash
set -euo pipefail

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

LLAMA_FACTORY_ROOT="${LF_ROOT}"
VENV_PATH="${LF_VENV}"
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

# 防 uv 意外重装本环境：`uv run` 会隐式 sync，把 pyproject 里声明的包换成 uv.lock 锁定的版本
# （不在 pyproject 里的手工包如 fla-core/tilelang 实测不受影响）。2026-07-22 就是这样把
# torch 2.8.0+cu129 换成 2.13.0，导致 flash-attn 与 vllm 的全部 C++ 扩展 undefined symbol，
# 静默坏了 6 天。写进 activate 而非 ~/.bashrc，作用域仅限本 venv，不影响其他项目。
cat >> "${VENV_PATH}/bin/activate" <<'ACTIVATE_EOF'

export UV_NO_SYNC=1
ACTIVATE_EOF

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
uv pip install transformers==5.6.1
# Qwen3.5 (gated-delta-net) needs fla; LF only uses fla-core, flash-linear-attention adds layers/models.
# --no-deps keeps them from upgrading transformers past 5.6.1 (would break LF's Qwen3.5 patch).
uv pip install "fla-core==0.5.1" "flash-linear-attention==0.5.1" --no-deps
# On Hopper (H200) + Triton>=3.4 fla's triton GDN backward is broken -> fla requires the tilelang backend.
# But tilelang 0.1.11 + apache-tvm-ffi 0.1.12 crash on import (tvm-ffi double registration);
# pin apache-tvm-ffi to 0.1.11. Nothing else in this venv depends on apache-tvm-ffi.
uv pip install "tilelang==0.1.11" "apache-tvm-ffi==0.1.11"

# tilelang JIT-compiles kernels by calling `gcc`/`g++` directly (it ignores CC/CXX). It needs a
# compiler whose cc1plus is actually installed. Rather than hardcode a version (this box's default
# gcc-12 is missing cc1plus while gcc-11 works, but another machine differs), AUTO-DETECT: use the
# default g++ if it compiles, else pick the newest working g++-N and shim gcc/g++ -> it on PATH.
# The train scripts prepend ${LLAMA_FACTORY_ROOT}/.cc-shim to PATH only if it validates.
# IMPORTANT: tilelang invokes `gcc -x c++` (the C driver on C++ source), so the probe must test
# exactly that — NOT `g++`. On this box `g++` is v11 (works) but `gcc` is v12 whose cc1plus is
# missing, so a g++-based probe would wrongly pass while tilelang still fails.
SHIM_DIR="${LLAMA_FACTORY_ROOT}/.cc-shim"
rm -rf "${SHIM_DIR}"   # drop any stale shim carried over from another machine
_cc_probe() { echo 'int main(){return 0;}' | "$1" -x c++ - -o /dev/null >/dev/null 2>&1; }
if _cc_probe gcc; then
  echo "Default gcc compiles C++; no compiler shim needed for tilelang."
else
  _cc=""
  for cand in gcc-13 gcc-12 gcc-11 gcc-10 gcc-9; do
    if command -v "${cand}" >/dev/null 2>&1 && _cc_probe "${cand}"; then _cc="${cand}"; break; fi
  done
  if [ -n "${_cc}" ]; then
    _cxx="${_cc/gcc/g++}"
    mkdir -p "${SHIM_DIR}"
    ln -sf "$(command -v "${_cc}")"                        "${SHIM_DIR}/gcc"
    ln -sf "$(command -v "${_cxx}" || command -v "${_cc}")" "${SHIM_DIR}/g++"
    ln -sf "$(command -v "${_cc}")"                        "${SHIM_DIR}/cc"
    ln -sf "$(command -v "${_cxx}" || command -v "${_cc}")" "${SHIM_DIR}/c++"
    echo "Default gcc can't compile C++; shimmed gcc/g++ -> ${_cc} at ${SHIM_DIR} (for tilelang JIT)."
  else
    echo "WARNING: no gcc that compiles C++ found (default + gcc-9..13 all failed). tilelang JIT" >&2
    echo "         will fail; install a CUDA-compatible gcc/g++ (e.g. 'apt install g++-11') & rerun." >&2
  fi
fi

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
