#!/usr/bin/env bash
set -euo pipefail
# ═══ GPU / runtime knobs (edit here) ═══
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,7}"


# ── Paths (machine-agnostic; see scripts/workspace_dir.sh) ─────────────────────────────
# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"
LLAMA_FACTORY_ROOT="${LF_ROOT}"
VENV_PATH="${LF_VENV}"

DATA_DIR="${LLAMA_FACTORY_ROOT}/data/mikomiko_tag"
BUILDER="${DATA_DIR}/dataset_builder.py"
TRAIN_CONFIG="${LLAMA_FACTORY_ROOT}/examples/train_full/qwen3_5_2b_mikomiko_tag.yaml"


export DISABLE_VERSION_CHECK=1  # transformers 5.6.1 > LF 硬编码上限 5.6.0；Qwen3.5 需新版，绕过版本闸

# Qwen3.5 的 GDN 反向内核在 Hopper 上用 tilelang(JIT)，需要能用的 g++。env_setup.sh 会按机器自动探测
# 并在 .cc-shim 建 gcc/g++ 垫片（默认编译器好用时则不建）。只有垫片实际能编译才前置，避免换机器后
# 悬空/错误的垫片挡住系统好编译器。
_SHIM="${LLAMA_FACTORY_ROOT}/.cc-shim"
if [ -x "${_SHIM}/gcc" ] && echo 'int main(){return 0;}' | "${_SHIM}/gcc" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}"
fi

# ── Activate venv ────────────────────────────────────────────────────────────
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv not found at ${VENV_PATH}." >&2
  echo "Run first: bash ${LLAMA_FACTORY_ROOT}/env_setup.sh" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"

# ── Build dataset (plan -> download) if the jsonl files are missing ───────────
if [ ! -f "${DATA_DIR}/train.jsonl" ] || [ ! -f "${DATA_DIR}/test_unseen.jsonl" ] \
   || [ ! -f "${DATA_DIR}/test_stratified.jsonl" ]; then
  echo "Building mikomiko tag dataset (train + two test sets) ..."
  python "${BUILDER}" --plan
  python "${BUILDER}" --download
fi

# ── Launch training ───────────────────────────────────────────────────────────
echo "Starting training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ..."
cd "${LLAMA_FACTORY_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  llamafactory-cli train "${TRAIN_CONFIG}" model_name_or_path="${MODELS_DIR}/Qwen3.5-2B"
