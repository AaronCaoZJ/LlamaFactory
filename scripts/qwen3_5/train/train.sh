#!/usr/bin/env bash
set -euo pipefail
# ═══ GPU / runtime knobs (edit here) ═══
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,7}"

# resolve machine paths: locate & source scripts/workspace_dir.sh (sets LF_ROOT, MODELS_DIR, LF_VENV, VLLM_VENV, AGENTROBOT_ROOT, HF_HOME)
_wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
source "$_wsd/scripts/workspace_dir.sh"

# ── Paths ────────────────────────────────────────────────────────────────────
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"

DATA_DIR="${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/0622"
TRAIN_CONFIG="${LLAMA_FACTORY_ROOT}/examples/train_lora/qwen3_5_9b_piper_0705_v4.yaml"


export DISABLE_VERSION_CHECK=1  # transformers 5.6.1 > LF 硬编码上限 5.6.0；Qwen3.5 需新版，绕过版本闸

# Qwen3.5 的 GDN 反向用 tilelang JIT，nvcc 会用 gcc 当宿主编译器；系统默认 gcc-12 缺 cc1plus
# (报 "cannot execute cc1plus")。用 env_setup.sh 建的 gcc-11 垫片，且仅在其确实能编译时前置 PATH。
_SHIM="${LLAMA_FACTORY_ROOT}/.cc-shim"
if [ -x "${_SHIM}/gcc" ] && echo 'int main(){return 0;}' | "${_SHIM}/gcc" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}"
fi

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
