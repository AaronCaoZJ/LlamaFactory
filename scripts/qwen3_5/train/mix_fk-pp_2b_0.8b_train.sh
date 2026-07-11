#!/usr/bin/env bash
# Launch the mix_22-06_fk-pp route 02_exchange_token for the two small Qwen3.5 sizes,
# each on ONE GPU (single-card), 30 epochs.
#
#   size   GPU   config
#   2b     3     qwen3_5_2b/mix_22-06_fk-pp/qwen3_5_2b_02_exchange_token.yaml
#   0.8b   4     qwen3_5_0_8b/mix_22-06_fk-pp/qwen3_5_0_8b_02_exchange_token.yaml
#
# NOTE: use ../hf_download/models/Qwen3.5-0.8B, NOT the sibling dir named "Qwen3.5-0.5B" —
# that dir is MISNAMED and actually holds the 0.8B weights (identical sha256, 873M params).
#
# GPUs 1,5,6,7 are used by other workloads and are intentionally avoided. Single-card deepspeed
# needs FORCE_TORCHRUN (set below); each job gets a pinned distinct MASTER_PORT so the two
# torchrun rendezvous do not collide.
set -uo pipefail
# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"
CFG_DIR="${LLAMA_FACTORY_ROOT}/examples/train_lora"
LOG_DIR="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-small/robot/mix_22-06_fk-pp/logs"
mkdir -p "${LOG_DIR}"

export DISABLE_VERSION_CHECK=1  # transformers 5.6.1 > LF 硬编码上限 5.6.0；绕过版本闸
export FORCE_TORCHRUN=1

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

# size : gpu : master_port : config
JOBS=(
  "2b:3:29521:${CFG_DIR}/qwen3_5_2b/mix_22-06_fk-pp/qwen3_5_2b_02_exchange_token.yaml"
  "0.8b:4:29522:${CFG_DIR}/qwen3_5_0_8b/mix_22-06_fk-pp/qwen3_5_0_8b_02_exchange_token.yaml"
)

for job in "${JOBS[@]}"; do
  IFS=":" read -r name gpus port cfg <<< "${job}"
  if [ ! -f "${cfg}" ]; then echo "ERROR: missing config ${cfg}" >&2; exit 1; fi
  log="${LOG_DIR}/${name}_02_exchange_token.log"
  echo "[launch] ${name}  GPUs=${gpus}  port=${port}  -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpus}" MASTER_PORT="${port}" \
    nohup llamafactory-cli train "${cfg}" > "${log}" 2>&1 &
  echo "         pid=$!"
done

echo
echo "Both trainings launched. Follow logs with:"
echo "  tail -f ${LOG_DIR}/2b_02_exchange_token.log"
echo "  tail -f ${LOG_DIR}/0.8b_02_exchange_token.log"
