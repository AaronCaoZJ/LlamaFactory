#!/usr/bin/env bash
# Launch the three mix_22-06_fk-pp routes concurrently, each on ONE GPU (single-card).
#
#   route             GPU   config
#   01_flip_img       3     qwen3_5_9b_01_flip_img.yaml
#   02_exchange_token 4     qwen3_5_9b_02_exchange_token.yaml
#   03_just_mix       5     qwen3_5_9b_03_just_mix.yaml
#
# GPUs 0,1,2 are used by other workloads and are intentionally avoided. Single-card deepspeed
# needs FORCE_TORCHRUN (set below); each job gets a pinned distinct MASTER_PORT so the three
# torchrun rendezvous do not collide.
set -uo pipefail
# resolve machine paths: locate & source scripts/workspace_dir.sh (sets LF_ROOT, MODELS_DIR, LF_VENV, VLLM_VENV, AGENTROBOT_ROOT, HF_HOME)
_wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
source "$_wsd/scripts/workspace_dir.sh"

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"
CFG_DIR="${LLAMA_FACTORY_ROOT}/examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp"
LOG_DIR="${LLAMA_FACTORY_ROOT}/saves/qwen3.5-9b/robot/mix_22-06_fk-pp/logs"
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

# route : gpu : master_port : config
JOBS=(
  "01_flip_img:3:29511:${CFG_DIR}/qwen3_5_9b_01_flip_img.yaml"
  "02_exchange_token:4:29512:${CFG_DIR}/qwen3_5_9b_02_exchange_token.yaml"
  "03_just_mix:5:29513:${CFG_DIR}/qwen3_5_9b_03_just_mix.yaml"
)

for job in "${JOBS[@]}"; do
  IFS=":" read -r name gpus port cfg <<< "${job}"
  if [ ! -f "${cfg}" ]; then echo "ERROR: missing config ${cfg}" >&2; exit 1; fi
  log="${LOG_DIR}/${name}.log"
  echo "[launch] ${name}  GPUs=${gpus}  port=${port}  -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpus}" MASTER_PORT="${port}" \
    nohup llamafactory-cli train "${cfg}" > "${log}" 2>&1 &
  echo "         pid=$!"
done

echo
echo "All three trainings launched. Follow logs with:"
echo "  tail -f ${LOG_DIR}/01_flip_img.log"
echo "  tail -f ${LOG_DIR}/02_exchange_token.log"
echo "  tail -f ${LOG_DIR}/03_just_mix.log"
