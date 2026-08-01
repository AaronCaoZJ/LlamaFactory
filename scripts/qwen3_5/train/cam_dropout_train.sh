#!/usr/bin/env bash
# mix_22-06_fk-pp route 02_exchange_token, retrained with CAMERA DROPOUT.
#
#   size   GPU   config
#   9b     4     qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token_cam_dropout.yaml
#   2b     7     qwen3_5_2b/mix_22-06_fk-pp/qwen3_5_2b_02_exchange_token_cam_dropout.yaml
#
# Each config is byte-identical to its 02_exchange_token baseline except for
# `camera_dropout: 0.15` (and the output/run names), so the pair is a clean ablation.
# Dropout blanks each of the two views independently with p=0.15 -- one black frame of the
# same size, hence the same token count -- and always keeps at least one view. Applied in
# the collator, so the mask is redrawn every epoch; inference is unaffected.
#
# GPUs 0-3,5,6 carry other workloads (incl. two vLLM eval servers) and are avoided.
# Single-card deepspeed needs FORCE_TORCHRUN (set below); each job gets a distinct
# MASTER_PORT so the two torchrun rendezvous do not collide.
set -uo pipefail
# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"
CFG_DIR="${LLAMA_FACTORY_ROOT}/examples/train_lora"
LOG_DIR="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-cam-dropout/logs"
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
  "9b:4:29521:${CFG_DIR}/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_02_exchange_token_cam_dropout.yaml"
  "2b:7:29522:${CFG_DIR}/qwen3_5_2b/mix_22-06_fk-pp/qwen3_5_2b_02_exchange_token_cam_dropout.yaml"
)

# ONLY=2b restarts just that size (e.g. after fixing one config); default runs both.
for job in "${JOBS[@]}"; do
  IFS=":" read -r name gpus port cfg <<< "${job}"
  if [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ]; then continue; fi
  if [ ! -f "${cfg}" ]; then echo "ERROR: missing config ${cfg}" >&2; exit 1; fi
  log="${LOG_DIR}/${name}_cam_dropout.log"
  echo "[launch] ${name}  GPU=${gpus}  port=${port}  -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpus}" MASTER_PORT="${port}" \
    nohup llamafactory-cli train "${cfg}" > "${log}" 2>&1 &
  echo "         pid=$!"
done

echo
echo "Launched. Follow logs with:"
echo "  tail -f ${LOG_DIR}/9b_cam_dropout.log"
echo "  tail -f ${LOG_DIR}/2b_cam_dropout.log"
