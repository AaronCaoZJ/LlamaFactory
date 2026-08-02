#!/usr/bin/env bash
# InternVL3.5-2B LoRA on dual_drawer_v4_once -- the dual-arm piper drawer task packaged with
# v4 --dual --once: three views (agentview + wrist_left + wrist_right) per step, one sample
# per step whose output is "<left> <right>", so one image forward supervises both arms.
#
# 1223 samples (21 rollouts / 1207 steps + 21 synthesized DONE, minus 5 in-place regrasp
# samples cleaned by data/agentrobot/MVTOKEN/dual_drawer/v4/clean_once_regrasp.py).
# eff_bs=16 -> ~76 steps/epoch, 30 epochs ≈ 2290 steps.
#
#   bash scripts/internvl/train/dual_drawer_once_2b.sh            # default: TWO GPUs (4,5)
#   GPUS=2,3 bash scripts/internvl/train/dual_drawer_once_2b.sh   # pick the pair
#   GPUS=4   bash scripts/internvl/train/dual_drawer_once_2b.sh   # one card -- see below
#
# The yaml is written for this two-card default: batch=4 x grad_acc=2 x 2 cards -> eff_bs=16,
# ~77 steps/epoch, 30 epochs ≈ 2310 steps. world_size multiplies the effective batch, so if
# you drop to ONE card set gradient_accumulation_steps back to 4 (or pass it on the CLI):
#   GPUS=4 bash scripts/internvl/train/dual_drawer_once_2b.sh   # then eff_bs=8, 2x the steps
# Going to four cards without lowering grad_acc would give eff_bs=32 and halve the optimizer
# steps on an already small 1223-sample set.
#
# The config keeps ds_z2, and llamafactory-cli only switches to torchrun when it sees more
# than one GPU, so a single-GPU deepspeed run needs FORCE_TORCHRUN=1 or it aborts with
# "Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training". It is exported below, and is
# harmless for the multi-GPU case.
#
# Outputs -> saves/internvl3.5-2b/robot/dual_drawer_once
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export CONFIG="${CONFIG:-examples/train_lora/internvl/internvl3_5_2b_dual_drawer_once.yaml}"
export MODEL_PATH="${MODEL_PATH:-/workspace1/zechen/hf_download/InternVL3_5-2B-HF}"
export GPUS="${GPUS:-4,5}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"

# Otherwise torch/OpenMP spawn threads by core count (384 here) inside every preprocessing
# worker; this box is shared, so cap it the way the qwen dual scripts do.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# The dataset is produced by data/agentrobot/process_data.sh + the cleaning pass; fail early
# and point at the generator instead of letting llamafactory-cli die on a missing file.
_JSON="${LF_ROOT}/data/agentrobot/MVTOKEN/dual_drawer/v4/rollout_dual_once.json"
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

exec bash "${SCRIPT_DIR}/repro_qwen_data.sh"
