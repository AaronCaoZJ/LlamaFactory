#!/usr/bin/env bash
# Launch the three dual_cloth schemes concurrently, each on ONE GPU (single-card).
#
#   scheme   samples  steps/epoch  config
#   twice     4364      ~136       qwen3_5_9b_dual_twice.yaml   (2 calls/step, Alpaca)
#   once      2182       ~68       qwen3_5_9b_dual_once.yaml    (1 call/step,  Alpaca)
#   chain     2182       ~68       qwen3_5_9b_dual_chain.yaml   (1 fwd/2 turns, ShareGPT)
#
# All three train off the SAME 26 rollouts (dual_cloth, "fold the black t-shirt") -- only the
# packaging differs, so a like-for-like comparison is by EPOCH, not by step. Each scheme
# supervises both arms of every frame once per epoch.
#
# Pick which schemes to run and which GPU each gets (paired by position, one GPU per scheme):
#
#   SCHEMES="twice,once"   GPUS="4,6"   bash scripts/qwen3_5/train/dual_cloth_train.sh
#   SCHEMES="chain"        GPUS="4"     bash scripts/qwen3_5/train/dual_cloth_train.sh
#   (default: all three)
#
# Single-card deepspeed needs FORCE_TORCHRUN (set below); each scheme has a pinned distinct
# MASTER_PORT so concurrent torchrun rendezvous do not collide.
set -uo pipefail
# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"
CFG_DIR="${LLAMA_FACTORY_ROOT}/examples/train_lora/qwen3_5_9b/dual_cloth"
LOG_DIR="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-9b/robot/dual_cloth/logs"
mkdir -p "${LOG_DIR}"

# Which schemes to run, and one GPU each (paired by position). A 9B LoRA under ZeRO-2 wants
# ~30-45GB free on an H200 (frozen vision tower, seq ~550, bs=4); check `nvidia-smi` first --
# a co-resident job on the same card costs SM time even when the memory fits.
SCHEMES="${SCHEMES:-twice,once,chain}"
GPUS="${GPUS:-0,1,2}"
IFS="," read -r -a SCHEME_ARR <<< "${SCHEMES}"
IFS="," read -r -a GPU_ARR <<< "${GPUS}"
if [ "${#SCHEME_ARR[@]}" -ne "${#GPU_ARR[@]}" ]; then
  echo "ERROR: SCHEMES and GPUS must have the same length (one GPU per scheme)." >&2
  echo "       SCHEMES=${SCHEMES} (${#SCHEME_ARR[@]})  GPUS=${GPUS} (${#GPU_ARR[@]})" >&2
  exit 1
fi

# Pinned per-scheme rendezvous port, so re-running a subset never collides with a live job.
declare -A PORTS=( [twice]=29521 [once]=29522 [chain]=29523 )

export DISABLE_VERSION_CHECK=1  # transformers 5.6.1 > LF 硬编码上限 5.6.0；绕过版本闸
export FORCE_TORCHRUN=1

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

# Validate every requested scheme up front: its dataset (process_data.sh) and its config must
# both exist, so a typo fails before any job is backgrounded.
for name in "${SCHEME_ARR[@]}"; do
  if [ -z "${PORTS[${name}]:-}" ]; then
    echo "ERROR: unknown scheme '${name}' (expected twice / once / chain)." >&2; exit 1
  fi
  json="${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/dual_cloth/v4/rollout_dual_${name}.json"
  if [ ! -f "${json}" ]; then
    echo "ERROR: missing dataset ${json}" >&2
    echo "       run first: bash data/agentrobot/process_data.sh" >&2
    exit 1
  fi
  cfg="${CFG_DIR}/qwen3_5_9b_dual_${name}.yaml"
  if [ ! -f "${cfg}" ]; then echo "ERROR: missing config ${cfg}" >&2; exit 1; fi
done

for i in "${!SCHEME_ARR[@]}"; do
  name="${SCHEME_ARR[$i]}"
  gpu="${GPU_ARR[$i]}"
  port="${PORTS[${name}]}"
  cfg="${CFG_DIR}/qwen3_5_9b_dual_${name}.yaml"
  log="${LOG_DIR}/${name}.log"
  echo "[launch] ${name}  GPU=${gpu}  port=${port}  -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" MASTER_PORT="${port}" \
    nohup llamafactory-cli train "${cfg}" > "${log}" 2>&1 &
  echo "         pid=$!"
done

echo
echo "Launched: ${SCHEMES} on GPUs ${GPUS}. Follow logs with:"
for name in "${SCHEME_ARR[@]}"; do
  echo "  tail -f ${LOG_DIR}/${name}.log"
done
