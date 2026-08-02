#!/usr/bin/env bash
# Qwen3.5-9B LoRA on dual_drawer_v4_once -- the dual-arm piper drawer task packaged with
# v4 --dual --once: three views (agentview + wrist_left + wrist_right) per step, one sample
# per step whose output is "<left> <right>", so one image forward supervises both arms.
#
# 1223 samples (21 rollouts / 1207 steps + 21 synthesized DONE, minus 5 in-place regrasp
# samples cleaned by data/agentrobot/MVTOKEN/dual_drawer/v4/clean_once_regrasp.py).
#
#   bash scripts/qwen3_5/train/dual_drawer_train.sh              # default: TWO GPUs (4,5)
#   GPUS=2,3 bash scripts/qwen3_5/train/dual_drawer_train.sh     # pick the pair
#   GPUS=4   bash scripts/qwen3_5/train/dual_drawer_train.sh     # one card -- see below
#   FOREGROUND=1 bash scripts/qwen3_5/train/dual_drawer_train.sh # stream to the terminal
#
# The yaml is written for this two-card default: batch=4 x grad_acc=4 x 2 cards -> eff_bs=32,
# ~39 steps/epoch, 40 epochs ≈ 1560 steps. world_size multiplies the effective batch, so on
# ONE card raise gradient_accumulation_steps to 8 in the yaml to keep eff_bs=32.
#
# A 9B LoRA under ZeRO-2 wants ~30-45GB free per card on an H200 (frozen vision tower,
# seq ~1.1k, bs=4) -- check `nvidia-smi` first, a co-resident job costs SM time even when the
# memory fits.
#
# Outputs -> saves/qwen3.5-9b/robot/dual_drawer/once
set -uo pipefail
# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"
CONFIG="${CONFIG:-${LLAMA_FACTORY_ROOT}/examples/train_lora/qwen3_5_9b/dual_drawer/qwen3_5_9b_dual_once.yaml}"
LOG_DIR="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-9b/robot/dual_drawer/logs"
GPUS="${GPUS:-4,5}"
# Distinct from dual_cloth's 29521-29523 so a concurrent job never collides on rendezvous.
MASTER_PORT="${MASTER_PORT:-29531}"

export DISABLE_VERSION_CHECK=1  # transformers 5.6.1 > LF 硬编码上限 5.6.0；绕过版本闸
export FORCE_TORCHRUN=1         # 单卡 deepspeed 必需；多卡时本来就走 torchrun，设了无副作用

# 不设的话 torch/OpenMP 按核数(384)开线程，每个 preprocessing worker 都拉 128~220 线程：
# "Running tokenizer" 那步(其实是解码+归一化每条样本的 3 张图)会从 ~10s 退化到 5+ 分钟，
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

# The dataset is produced by rollout_to_llamafactory.py + the cleaning pass; fail early and
# point at the generator instead of letting llamafactory-cli die on a missing file.
_JSON="${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/dual_drawer/v4/rollout_dual_once.json"
if [ ! -f "${_JSON}" ]; then
  echo "ERROR: missing dataset ${_JSON}" >&2
  echo "       rebuild with:" >&2
  echo "         python data/agentrobot/rollout_to_llamafactory.py \\" >&2
  echo "             data/agentrobot/MVTOKEN/dual_drawer --version v4 --dual --once \\" >&2
  echo "             --task \"open the drawer, put the blue cup into the drawer, then close the drawer\" \\" >&2
  echo "             --output data/agentrobot/MVTOKEN/dual_drawer/v4/rollout_dual_once.json" >&2
  echo "         python data/agentrobot/MVTOKEN/dual_drawer/v4/clean_once_regrasp.py" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/once.log"

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
