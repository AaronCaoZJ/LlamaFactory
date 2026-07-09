#!/usr/bin/env bash
#* 覆盖项: GPU | CKPT_ROOT | CKPT_STEP | LR | EPOCHS | RUN_NAME | OUT_DIR

# Full-parameter SFT of Qwen3.5-2B on the mikomiko image->tag task, RESUMED from a prior checkpoint.
# Builds the dataset first if its jsonl files are missing, then launches training. Ctrl-C to stop.
set -euo pipefail

# ============================================================
#! GPU / runtime knobs (edit here)
GPU="${GPU:-4,5,6,7}"  # 4-GPU training (e.g. 4x H100 80GB; use 0,1,2,3 for the first 4)

# ============================================================
#! Paths (machine-agnostic; see scripts/workspace_dir.sh)
# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(
  d="$(dirname "${BASH_SOURCE[0]}")"
  until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done
  echo "$d"
)/scripts/workspace_dir.sh"
VENV_PATH="${LF_VENV}"

export DISABLE_VERSION_CHECK=1  # transformers 5.6.1 > LF 硬编码上限 5.6.0；Qwen3.5 需新版，绕过版本闸

# ============================================================
#! CUDA JIT compiler (machine-adaptive)
# Qwen3.5 的 GDN 反向内核在 Hopper 上用 tilelang(JIT)，需要能用的 g++。env_setup.sh 会按机器自动探测
# 并在 .cc-shim 建 gcc/g++ 垫片（默认编译器好用时则不建）。只有垫片实际能编译才前置，避免换机器后
# 悬空/错误的垫片挡住系统好编译器。
_SHIM="${LF_ROOT}/.cc-shim"
if [ -x "${_SHIM}/gcc" ] && echo 'int main(){return 0;}' | "${_SHIM}/gcc" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}"
fi

# ============================================================
#! Activate venv
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv not found at ${VENV_PATH}." >&2
  echo "Run first: bash ${LF_ROOT}/env_setup.sh" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"

# ============================================================
#! Build dataset (plan -> download) if the jsonl files are missing
DATA_DIR="${LF_ROOT}/data/mikomiko_tag"
BUILDER="${DATA_DIR}/dataset_builder.py"

if [ ! -f "${DATA_DIR}/train.jsonl" ] || [ ! -f "${DATA_DIR}/test_unseen.jsonl" ] \
   || [ ! -f "${DATA_DIR}/test_stratified.jsonl" ]; then
  echo "Building mikomiko tag dataset (train + two test sets) ..."
  python "${BUILDER}" --plan
  python "${BUILDER}" --download
fi

# ============================================================
#! Launch training (resume from a prior checkpoint under ${MODELS_DIR})
TRAIN_CONFIG="${LF_ROOT}/examples/train_full/qwen3_5_2b_mikomiko_tag.yaml"
CKPT_STEP="${CKPT_STEP:-11530}"  # resume from this step
CKPT="${CKPT:-${MODELS_DIR}/Mikomiko_pornpic_tagger/checkpoint-${CKPT_STEP}}"

LR="${LR:-5e-6}"
EPOCHS="${EPOCHS:-2.0}"
RUN_NAME="${RUN_NAME:-qwen3.5-2b-mikomiko-resume${CKPT_STEP}-lr${LR}}"
OUT_DIR="${OUT_DIR:-saves/qwen3.5-2b/mikomiko/full_v0_resume${CKPT_STEP}}"

echo "Starting training on GPU=${GPU} (resume ${CKPT}) ..."
cd "${LF_ROOT}"

# Fresh full_v0 training (no resume): uncomment to train from the base model instead.
# exec env CUDA_VISIBLE_DEVICES="${GPU}" \
#   llamafactory-cli train "${TRAIN_CONFIG}" model_name_or_path="${MODELS_DIR}/Qwen3.5-2B"

exec env CUDA_VISIBLE_DEVICES="${GPU}" \
  llamafactory-cli train "${TRAIN_CONFIG}" \
    model_name_or_path="${CKPT}" \
    resume_from_checkpoint="${CKPT}" \
    learning_rate="${LR}" \
    run_name="${RUN_NAME}" \
    output_dir="${OUT_DIR}" \
    num_train_epochs="${EPOCHS}" \
    overwrite_output_dir=false
