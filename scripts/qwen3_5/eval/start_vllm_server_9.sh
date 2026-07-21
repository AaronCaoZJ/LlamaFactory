#!/usr/bin/env bash
# vLLM OpenAI server: Qwen3.5-9B + MVTOKEN LoRA adapters (default :8109).
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
GPU="${GPU:-6}"
export CUDA_VISIBLE_DEVICES="${GPU}"

# ================================================================================
#! Args (server knobs / model / LoRA)
#* Overrides: GPU | PORT | GPU_UTIL | TEMPERATURE
PORT="${PORT:-8109}"
GPU_UTIL="${GPU_UTIL:-0.7}"
TEMPERATURE="${TEMPERATURE:-0}"

MAX_LEN=8192
MAX_NUM_SEQS=256
ENFORCE_EAGER=0

BASE_MODEL="${MODELS_DIR}/Qwen3.5-9B"
SAVES="${LF_ROOT}/saves/qwen3.5-9b/robot"

LORA_MODULES=(
  "mix_22_27_v3_9=${SAVES}/mix_22_27_v3"
  "mix_22_27_04_v3_9=${SAVES}/mix_22_27_04_v3"
  "piper_0705_v4_9=${SAVES}/piper_0705_v4"

  # mix franka and piper, 三种处理方式（反转 piper 图像、对掉 piper FWD 和 BACK，直接混合）
  # "mix_22-06_fk-pp_01=${SAVES}/mix_22-06_fk-pp/01_flip_img"
  "mix_22-06_fk-pp_02=${SAVES}/mix_22-06_fk-pp/02_exchange_token" # ✅
  # "mix_22-06_fk-pp_03=${SAVES}/mix_22-06_fk-pp/03_just_mix"

  # 方案 B（video 槽位）——训练跑完后解注释；vLLM 加载不存在的 LoRA 路径会直接启动失败。
  "mix_22_27_v3_9_video=${SAVES}/mix_22_27_v3_video"

  # dual_cloth（双臂折衣，三种"一次推理出两个 token"的契约）。
  # scheme 和 LoRA 一一对应，AgentRobot 端用 run_real_dual_mvtoken.sh 的 SCHEME 选，
  # 混搭必然解析失败（prompt 和请求形状是同一份契约的两半）。
  "dual_cloth_twice=${SAVES}/dual_cloth/twice"
  "dual_cloth_once=${SAVES}/dual_cloth/once"
  "dual_cloth_chain=${SAVES}/dual_cloth/chain"
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
