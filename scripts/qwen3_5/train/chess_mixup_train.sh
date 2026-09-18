#!/usr/bin/env bash
# Qwen3.5-9B LoRA on the chess (Chinese Xiangqi) MVTOKEN rollouts -- single arm, two views
# (agentview + wrist) per step, one action token per sample, text move history in the prompt.
#
# Two variants, both over the SAME 11 rollouts / 1679 frames:
#
#   long_horizon (default)  chess_mixup_long_horizon + chess_mixup_single_task, mix_strategy
#                           concat -> 3413 samples. The long-horizon prompt lists all 4
#                           subtasks (DONE only at the end); the single-task prompt names one
#                           subtask (DONE at every boundary). 33 boundary frames carry
#                           conflicting labels ON PURPOSE -- DONE becomes conditional on the
#                           task scope in the prompt. Base model is the mix-fk-pp MERGED
#                           policy, not stock Qwen3.5-9B.
#   filtered                chess_mixup_filtered alone -> 1690 samples, stock Qwen3.5-9B.
#                           The earlier run, kept so it stays reproducible.
#
#   bash scripts/qwen3_5/train/chess_mixup_train.sh                    # long_horizon on GPU 2
#   GPUS=2,3 bash scripts/qwen3_5/train/chess_mixup_train.sh           # two cards
#   VARIANT=filtered bash scripts/qwen3_5/train/chess_mixup_train.sh   # the older run
#   FOREGROUND=1 bash scripts/qwen3_5/train/chess_mixup_train.sh       # stream to the terminal
#   DRY_RUN=1 bash scripts/qwen3_5/train/chess_mixup_train.sh          # print the command only
#
# Effective batch is pinned at 32 regardless of card count: world_size multiplies it, so
# gradient_accumulation_steps is DERIVED here (1 card -> 8, 2 cards -> 4, 4 cards -> 2) and
# passed on the command line, overriding the yaml. Editing the yaml's grad_acc by hand is what
# silently changes the lr curve between a one-card and a two-card launch; don't.
#
# A 9B LoRA under ZeRO-2 wants ~30-45GB free per card on an H200 (frozen vision tower,
# seq ~0.5k, bs=4) -- check `nvidia-smi` first, this box is shared and a co-resident job costs
# SM time even when the memory fits. GPUS defaults to 2 because that was the only idle card on
# 2026-08-11; re-check before launching.
#
# Outputs -> saves/qwen3.5-9b/robot/chess/<variant>
set -uo pipefail
# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"
VARIANT="${VARIANT:-long_horizon}"
LOG_DIR="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-9b/robot/chess/logs"
GPUS="${GPUS:-2}"
# Distinct from dual_cloth's 29521-29523 and dual_drawer's 29531 so a concurrent job never
# collides on rendezvous.
MASTER_PORT="${MASTER_PORT:-29541}"
EFF_BS="${EFF_BS:-32}"

case "${VARIANT}" in
  long_horizon)
    _CONFIG_DEFAULT="${LLAMA_FACTORY_ROOT}/examples/train_lora/qwen3_5_9b/chess/qwen3_5_9b_chess_mixup_long_horizon.yaml"
    # both jsons must be present: the run concats them
    _JSONS=("mixup_long_horizon" "mixup_single_task")
    ;;
  filtered)
    _CONFIG_DEFAULT="${LLAMA_FACTORY_ROOT}/examples/train_lora/qwen3_5_9b/chess/qwen3_5_9b_chess_mixup_filtered.yaml"
    _JSONS=("mixup_filtered")
    ;;
  *)
    echo "ERROR: unknown VARIANT='${VARIANT}' (expected: long_horizon | filtered)" >&2; exit 1 ;;
esac
CONFIG="${CONFIG:-${_CONFIG_DEFAULT}}"

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

# The jsons here are zechen's rollout_lite.json with the image prefix rewritten onto the
# chess/mixup symlink (his copies point at /users/zechen/..., unreadable from this account).
# All variants share that one image tree. Fail early and point at the importer instead of
# letting llamafactory-cli die on a missing file -- or worse, on 3380 unreadable images
# halfway through preprocessing.
for _name in "${_JSONS[@]}"; do
  _JSON="${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/chess/${_name}/rollout_lite.json"
  if [ ! -f "${_JSON}" ] || [ ! -d "${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/chess/mixup/rollout_000" ]; then
    echo "ERROR: missing dataset ${_JSON} (or the chess/mixup image symlink is broken)" >&2
    echo "       rebuild with:" >&2
    echo "         ln -sfn /workspace1/zechen/DATA/chess/mixup \\" >&2
    echo "             ${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/chess/mixup" >&2
    echo "         python3 data/agentrobot/MVTOKEN/chess/relocate_images.py" >&2
    exit 1
  fi
done

# A full-width comma (GPUS=2，4, trivially easy to type with a Chinese IME) is NOT a separator:
# the card count comes out as 1 AND CUDA rejects the malformed device list with
# "CUDA-capable device(s) is/are busy or unavailable" -- which reads like a busy GPU and sends
# you looking at nvidia-smi instead of at the comma. Fail here with the real reason.
if ! [[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "ERROR: GPUS='${GPUS}' is not a comma-separated list of indices (e.g. GPUS=2,4)." >&2
  case "${GPUS}" in
    *，*) echo "       it contains a FULL-WIDTH comma '，' -- retype it as a half-width ','." >&2 ;;
    *" "*) echo "       it contains a space -- write GPUS=2,4 with no spaces." >&2 ;;
  esac
  exit 1
fi

# Keep eff_bs = per_device_bs x grad_acc x world_size constant across card counts.
NGPU="$(awk -F',' '{print NF}' <<< "${GPUS}")"
MICRO_BS="$(awk '/^per_device_train_batch_size:/{print $2; exit}' "${CONFIG}")"
if [ -z "${MICRO_BS}" ]; then
  echo "ERROR: could not read per_device_train_batch_size from ${CONFIG}" >&2; exit 1
fi
if (( EFF_BS % (MICRO_BS * NGPU) != 0 )); then
  echo "ERROR: eff_bs ${EFF_BS} is not divisible by per_device_bs ${MICRO_BS} x ${NGPU} card(s)." >&2
  echo "       set EFF_BS / GPUS so the accumulation steps come out whole." >&2
  exit 1
fi
ACCUM=$(( EFF_BS / (MICRO_BS * NGPU) ))

mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${VARIANT}.log"

SUMMARY="variant: ${VARIANT}
config : ${CONFIG}
GPUs   : ${GPUS} (${NGPU} card(s))   MASTER_PORT=${MASTER_PORT}
batch  : ${MICRO_BS} x accum ${ACCUM} x ${NGPU} = eff_bs ${EFF_BS}"
echo "${SUMMARY}"

if [ -n "${DRY_RUN:-}" ]; then
  echo
  echo "would run:"
  echo "  CUDA_VISIBLE_DEVICES=${GPUS} MASTER_PORT=${MASTER_PORT} \\"
  echo "    llamafactory-cli train ${CONFIG} gradient_accumulation_steps=${ACCUM}"
  exit 0
fi

# A card can read as completely idle in nvidia-smi (0 MiB, 0% util) and still refuse every
# allocation with "CUDA-capable device(s) is/are busy or unavailable" -- that is what a stale
# CUDA context held by someone else's process looks like from the outside, and GPU 2 on this
# box was exactly that on 2026-08-11. nvidia-smi is therefore NOT sufficient to pick a card.
# Probe for real, so a bad card fails in seconds instead of after the model has loaded.
# SKIP_GPU_CHECK=1 to bypass (takes a few seconds per card).
if [ -z "${SKIP_GPU_CHECK:-}" ]; then
  echo "probing ${NGPU} card(s) ..."
  for _g in ${GPUS//,/ }; do
    _probe="$(CUDA_VISIBLE_DEVICES="${_g}" python3 -c "
import torch
if torch.cuda.device_count() < 1:
    print('NOCARD'); raise SystemExit
try:
    torch.zeros(256, 1024, 1024, device='cuda:0')  # 1GB
    free, _ = torch.cuda.mem_get_info()
    print(f'OK {free / 2**30:.0f}')
except Exception as e:
    print('FAIL', str(e).split(chr(10))[0][:70])
" 2>/dev/null | tail -1)"
    case "${_probe}" in
      OK*) echo "  GPU ${_g}: usable, $(awk '{print $2}' <<< "${_probe}") GB free" ;;
      NOCARD)
        echo "ERROR: GPU ${_g} does not exist on this host (indices 0..$(( $(nvidia-smi -L | wc -l) - 1 )))." >&2
        echo "       'GPUS=45' means card FORTY-FIVE, not cards 4 and 5 -- use GPUS=4,5." >&2
        exit 1 ;;
      *)
        echo "ERROR: GPU ${_g} is not usable: ${_probe#FAIL }" >&2
        echo "       nvidia-smi may still show it as idle; pick another card." >&2
        exit 1 ;;
    esac
  done
fi

if [ -n "${FOREGROUND:-}" ]; then
  exec env CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
    llamafactory-cli train "${CONFIG}" "gradient_accumulation_steps=${ACCUM}"
fi

# Header first: without it the log never records WHICH cards ran, and a wrong-GPU launch is
# indistinguishable from a broken one when you come back to the log later.
printf '=== %s ===\n%s\n\n' "$(date '+%F %T')" "${SUMMARY}" > "${LOG}"
CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
  nohup llamafactory-cli train "${CONFIG}" "gradient_accumulation_steps=${ACCUM}" >> "${LOG}" 2>&1 &
echo "pid    : $!"
echo "log    : ${LOG}"
echo
echo "follow with:  tail -f ${LOG}"
