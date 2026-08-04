#!/usr/bin/env bash
# Qwen3.5-9B LoRA on chess_mixup_filtered -- the single-arm Chinese Xiangqi task, two views
# (agentview + wrist) per step, one action token per sample, text move history in the prompt.
#
# 1690 samples (11 rollouts / 1714 source steps, 25 steps dropped by zechen's review pass).
#
#   bash scripts/qwen3_5/train/chess_mixup_train.sh              # default: TWO GPUs (3,4)
#   GPUS=2,3 bash scripts/qwen3_5/train/chess_mixup_train.sh     # pick the pair
#   GPUS=3   bash scripts/qwen3_5/train/chess_mixup_train.sh     # one card -- see below
#   FOREGROUND=1 bash scripts/qwen3_5/train/chess_mixup_train.sh # stream to the terminal
#
# The yaml is written for this two-card default: batch=4 x grad_acc=4 x 2 cards -> eff_bs=32,
# ~52 steps/epoch, 40 epochs ≈ 2110 steps. world_size multiplies the effective batch, so on
# ONE card raise gradient_accumulation_steps to 8 in the yaml to keep eff_bs=32.
#
# A 9B LoRA under ZeRO-2 wants ~30-45GB free per card on an H200 (frozen vision tower,
# seq ~1.0k, bs=4) -- check `nvidia-smi` first, this box is shared and a co-resident job costs
# SM time even when the memory fits. The 3,4 default is only what was freest when the script
# was written; re-check before launching.
#
# Outputs -> saves/qwen3.5-9b/robot/chess/mixup_filtered
set -uo pipefail
# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"
CONFIG="${CONFIG:-${LLAMA_FACTORY_ROOT}/examples/train_lora/qwen3_5_9b/chess/qwen3_5_9b_chess_mixup_filtered.yaml}"
LOG_DIR="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-9b/robot/chess/logs"
GPUS="${GPUS:-3,4}"
# Distinct from dual_cloth's 29521-29523 and dual_drawer's 29531 so a concurrent job never
# collides on rendezvous.
MASTER_PORT="${MASTER_PORT:-29541}"

export DISABLE_VERSION_CHECK=1  # transformers 5.7.0 > LF 硬编码上限 5.6.0；绕过版本闸
export FORCE_TORCHRUN=1         # 单卡 deepspeed 必需；多卡时本来就走 torchrun，设了无副作用

# 不设的话 torch/OpenMP 按核数(384)开线程，每个 preprocessing worker 都拉 128~220 线程：
# "Running tokenizer" 那步(其实是解码+归一化每条样本的 2 张图)会从 ~10s 退化到几分钟，
# 整机 load 被推到 600+。这台机器是共享的，别把别人一起拖下水。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# gcc-11 垫片（Qwen3.5 GDN 反向 tilelang JIT 需 cc1plus；系统 gcc-12 缺）
_SHIM="${LLAMA_FACTORY_ROOT}/.cc-shim"
if [ -x "${_SHIM}/gcc" ] && echo 'int main(){return 0;}' | "${_SHIM}/gcc" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}"
fi

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv not found at ${VENV_PATH}." >&2; exit 1
fi
source "${VENV_PATH}/bin/activate"
cd "${LLAMA_FACTORY_ROOT}"

if [ ! -f "${CONFIG}" ]; then echo "ERROR: missing config ${CONFIG}" >&2; exit 1; fi

# The json here is zechen's rollout_lite.json with the image prefix rewritten onto the
# chess/mixup symlink (his copy points at /users/zechen/..., unreadable from this account).
# Fail early and point at the importer instead of letting llamafactory-cli die on a missing
# file -- or worse, on 3380 unreadable images halfway through preprocessing.
_JSON="${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/chess/mixup_filtered/rollout_lite.json"
if [ ! -f "${_JSON}" ] || [ ! -d "${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/chess/mixup/rollout_000" ]; then
  echo "ERROR: missing dataset ${_JSON} (or the chess/mixup image symlink is broken)" >&2
  echo "       rebuild with:" >&2
  echo "         ln -sfn /workspace1/zechen/DATA/chess/mixup \\" >&2
  echo "             ${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/chess/mixup" >&2
  echo "         python data/agentrobot/MVTOKEN/chess/relocate_images.py" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/mixup_filtered.log"

echo "config : ${CONFIG}"
echo "GPUs   : ${GPUS}   MASTER_PORT=${MASTER_PORT}"

if [ -n "${FOREGROUND:-}" ]; then
  exec env CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
    llamafactory-cli train "${CONFIG}"
fi

CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
  nohup llamafactory-cli train "${CONFIG}" > "${LOG}" 2>&1 &
echo "pid    : $!"
echo "log    : ${LOG}"
echo
echo "follow with:  tail -f ${LOG}"
