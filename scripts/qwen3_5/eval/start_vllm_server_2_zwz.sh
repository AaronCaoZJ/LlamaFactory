#!/usr/bin/env bash
# vLLM OpenAI server: Qwen3.5-9B + MVTOKEN LoRA adapters (default :8103).
set -euo pipefail

# ================================================================================
# Paths (machine-agnostic; see scripts/workspace_dir.sh)
#* Exports: LF_ROOT | MODELS_DIR | LF_VENV | VLLM_VENV | HF_HOME | AGENTROBOT_ROOT
source "$(
  d="$(dirname "${BASH_SOURCE[0]}")"
  until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done
  echo "$d"
)/scripts/workspace_dir.sh"

# ================================================================================
#! Cuda device / runtime knobs (edit here)
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
if [ -x /usr/local/cuda-12.9/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda-12.9
  export CUDA_PATH="${CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

# ================================================================================
#! Args (server knobs / model / LoRA)
#* Overrides: GPU | PORT | GPU_UTIL | TEMPERATURE
PORT="${PORT:-8103}"
GPU_UTIL="${GPU_UTIL:-0.35}"
TEMPERATURE="${TEMPERATURE:-0}"

MAX_LEN=8192
MAX_NUM_SEQS=256
ENFORCE_EAGER=0

BASE_MODEL="${MODELS_DIR}/Qwen3.5-9B"
SAVES="${LF_ROOT}/saves/qwen3.5-9b/robot"

LORA_MODULES=(
  "franka_only=${SAVES}/mix_22_27_04_v3_zwz_new_prompt"
  "02_exchange_franka_prompt=${SAVES}/mix_22-06_fk-pp/02_exchange_token_zwz_new_prompt"
  "02_mix_22-06_fk-pp_03=${SAVES}/mix_22-06_fk-pp/02_exchange_token"
  "02_exchange_task_aug=${SAVES}/mix_22-06_fk-pp/02_exchange_token_task_aug"
  "02_exchange_base=${SAVES}/mix_22-06_fk-pp/02_exchange_token_base"
  "02_exchange_base_z0=${SAVES}/mix_22-06_fk-pp/02_exchange_token_base_z0"
  "02_exchange_2_schedule=${SAVES}/mix_22-06_fk-pp/02_exchange_token_2_schedule"
  "02_exchange_2_schedule_z2=${SAVES}/mix_22-06_fk-pp/02_exchange_token_2_schedule_z2"
  "02_exchange_task_aug_2_schedule=${SAVES}/mix_22-06_fk-pp/02_exchange_token_task_aug_2_schedule"
  "02_exchange_2_schedule_13=${SAVES}/mix_22-06_fk-pp/02_exchange_token_2_schedule_13"
  "02_exchange_task_aug_robovqa_2_schedule_13=${SAVES}/mix_22-06_fk-pp/02_exchange_token_task_aug_plus_robovqa_clean_ans6k_under500_2_schedule_13"
  "02_exchange_task_aug_zwz_0723=${SAVES}/mix_22-06_fk-pp/02_exchange_token_task_aug_plus_zwz_0723"
  "03_just_mix_tailed_prompt=${SAVES}/mix_22-06_fk-pp/03_just_mix_zwz_new_prompt"
  "03_just_mix_tailed_prompt_spatial_aug=${SAVES}/mix_22-06_fk-pp/03_just_mix_zwz_new_prompt_add_horizon_flip"
  "03_just_mix_tailed_prompt_spatial_aug_rule_aug=${SAVES}/mix_22-06_fk-pp/03_just_mix_zwz_new_prompt_add_horizon_flip_add_augment_rules"
  "03_just_mix_tailed_prompt_spatial_aug_2_schedule=${SAVES}/mix_22-06_fk-pp/03_just_mix_zwz_new_prompt_add_horizon_flip_2_training_schedule"
  "02_exchange_0724_easy_spatial=${SAVES}/mix_22-06_fk-pp/02_exchange_token_plus_zwz_0724_easy_spatial"
  "02_exchange_0723_0724=${SAVES}/mix_22-06_fk-pp/02_exchange_token_plus_zwz_0723_0724"
  "02_exchange_0723_0724_v3_prompt_13=${SAVES}/mix_22-06_fk-pp/02_exchange_token_plus_zwz_0723_0724_v3_prompt_13"
  "02_exchange_0723_0724_v3_prompt_robovqa=${SAVES}/mix_22-06_fk-pp/02_exchange_token_plus_zwz_0723_0724_v3_prompt_plus_robovqa_clean_ans6k_under500"
  "03_just_mix_0723_0724_v2_prompt_13=${SAVES}/mix_22-06_fk-pp/03_just_mix_plus_zwz_0723_0724_v2_prompt_13"
  "03_just_mix_0723_0724_v2_prompt_robovqa=${SAVES}/mix_22-06_fk-pp/03_just_mix_plus_zwz_0723_0724_v2_prompt_plus_robovqa_clean_ans6k_under500"
)

# 启动前自检：vLLM 是先把 base model 全部加载完、再去解析 --lora-modules 的，所以一个还没
# 训完（或路径写错）的 adapter 会让你白等几分钟再看到崩溃。这里提前失败，并指出是哪一个。
MISSING=()
for m in "${LORA_MODULES[@]}"; do
  [ -f "${m#*=}/adapter_model.safetensors" ] || MISSING+=("${m%%=*}  ->  ${m#*=}")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: 这些 LoRA 还没有 adapter_model.safetensors（训练未完成 / 路径错）：" >&2
  printf '  %s\n' "${MISSING[@]}" >&2
  echo "  训练完成前请把它们从 LORA_MODULES 注释掉，或等训练结束。" >&2
  exit 1
fi

# LF 对齐的 chat template（必需）。Qwen3.5 官方模板即使 enable_thinking=false 也会在
# '<|im_start|>assistant\n' 后插一个空 think 块 '<think>\n\n</think>\n\n'，而 LF 的
# qwen3_5_nothink 什么都不插 —— 不挂这个文件，prompt 与训练分布差 4 个 token（HANDOFF §4.2）。
# 对 image 布局和 video 槽位都适用；已用 /tokenize 逐 token 比对验证过。
CHAT_TEMPLATE="${LF_ROOT}/scripts/qwen3_5/eval/chat_template_qwen3_5_lf.jinja"

# ================================================================================
# CUDA JIT compiler (machine-adaptive)
_shim="${LF_ROOT}/.cc-shim"
if [ -x "${_shim}/g++" ] && echo 'int main(){return 0;}' | "${_shim}/g++" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export CC="${_shim}/gcc" CXX="${_shim}/g++" CUDAHOSTCXX="${_shim}/g++" NVCC_PREPEND_FLAGS="-ccbin ${_shim}/g++"
fi

# ================================================================================
#! Source venv
VLLM_VENV="${VLLM_VENV}"
source "${VLLM_VENV}/bin/activate"

# ================================================================================
#! Launch
SEP="================================================================================"
echo "Starting vllm server on http://0.0.0.0:${PORT}"
echo "  GPU                 : ${GPU}"
echo "  GPU util            : ${GPU_UTIL}"
# echo "  Temperature         : ${TEMPERATURE}"
echo "  Max seq len         : ${MAX_LEN}"
echo "  Max num seqs        : ${MAX_NUM_SEQS}"
echo "  Enforce eager       : ${ENFORCE_EAGER}"
echo "${SEP}"
echo "  Base model          : ${BASE_MODEL}"
echo "${SEP}"
for m in "${LORA_MODULES[@]}"; do printf "  %-22s: %s\n" "${m%%=*}" "${m#*=}"; done
echo "${SEP}"
echo "  Chat template       : ${CHAT_TEMPLATE}"
echo "${SEP}"

CMD=(
  vllm serve "${BASE_MODEL}"
  --dtype bfloat16
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --enable-lora --max-lora-rank 64
  --lora-modules "${LORA_MODULES[@]}"
  --chat-template "${CHAT_TEMPLATE}"
  # --override-generation-config "{\"temperature\": ${TEMPERATURE}, \"top_p\": 1.0, \"top_k\": -1}"
  --trust-remote-code
  --port "${PORT}"
)
[ "${ENFORCE_EAGER}" = "1" ] && CMD+=(--enforce-eager)

exec "${CMD[@]}"
