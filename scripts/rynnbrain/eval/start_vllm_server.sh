#!/usr/bin/env bash
# vLLM OpenAI server: RynnBrain1.1-2B + MVTOKEN LoRA adapter (default :8107).
#
# RynnBrain1.1-2B 与 Qwen3.5-2B 同构，已逐项核对（见下方 "同构核对" 注释块），
# 因此本脚本是 scripts/rynnbrain/eval/start_vllm_server_2.sh 的同构改写：
# 只换基座 / LoRA / 端口 / 卡，chat template 与所有运行时旋钮原样复用。
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
# GPU 7 而非 2 号脚本的 4：本机 GPU4 常年被占到 137/143 GB，0.5 的 util 会直接 OOM。
GPU="${GPU:-7}"
export CUDA_VISIBLE_DEVICES="${GPU}"

# ================================================================================
#! Args (server knobs / model / LoRA)
#* Overrides: GPU | PORT | GPU_UTIL | TEMPERATURE | BASE_MODEL | LORA_DIR
PORT="${PORT:-8107}"
GPU_UTIL="${GPU_UTIL:-0.5}"
TEMPERATURE="${TEMPERATURE:-0}"

MAX_LEN=8192
MAX_NUM_SEQS=256
ENFORCE_EAGER=0

# 基座与 LoRA 都在 zechen 的目录下（组内可读），不在本机 MODELS_DIR/SAVES 里 ——
# 所以这里是绝对路径而非 ${MODELS_DIR} 拼接。模型搬进 MODELS_DIR 后覆盖 BASE_MODEL 即可。
# BASE_MODEL="${BASE_MODEL:-/workspace1/zechen/hf_download/RynnBrain1.1-2B}"
# LORA_MODULES=(
#   "rynnbrain1.1-2b=/workspace1/zechen/finetune/lora/RynnBrain1.1-2b"
#   "rynnbrain1.1-2b-5k=/workspace1/zechen/finetune/lora/RynnBrain1.1-2b-5k"
# )

BASE_MODEL="${BASE_MODEL:-/workspace1/zechen/hf_download/RynnBrain1.1-9B}"
LORA_MODULES=(
  "rynnbrain1.1-9b=/workspace1/zechen/finetune/lora/RynnBrain1.1-9b"
)

# LF 对齐的 chat template（必需）。Qwen3.5 官方模板即使 enable_thinking=false 也会在
# '<|im_start|>assistant\n' 后插一个空 think 块 '<think>\n\n</think>\n\n'，而 LF 的
# qwen3_5_nothink 什么都不插 —— 不挂这个文件，prompt 与训练分布差 4 个 token（HANDOFF §4.2）。
# 对 image 布局和 video 槽位都适用；已用 /tokenize 逐 token 比对验证过。
#
# 这里直接复用 qwen3_5 那份而不是复制一份到本目录：RynnBrain 基座自带的
# chat_template.jinja 与 Qwen3.5-2B 官方模板 md5 完全相同（3dd635d8…），同一个坑、
# 同一个修法；复制会让两份将来各自漂移。
CHAT_TEMPLATE="${LF_ROOT}/scripts/qwen3_5/eval/chat_template_qwen3_5_lf.jinja"

# ================================================================================
# 同构核对（2026-07-23，逐项比对 RynnBrain1.1-2B vs Qwen3.5-2B，全部通过）
#   - 架构      : Qwen3_5ForConditionalGeneration / model_type qwen3_5，24 层
#                 linear+full 混合注意力，hidden 2048、head_dim 256 —— 完全一致。
#                 config 仅两处差异：partial_rotary_factor=0.25 被显式写出（Qwen 侧
#                 靠默认值），以及 transformers_version（5.2.0 vs 4.57）。
#   - tokenizer : model.vocab 逐条相同；merges 247587 条归一化后完全相同（差异只是
#                 transformers 5.x 存成 ["a","b"] 而 4.x 存成 "a b"）。RynnBrain 多注册
#                 7 个 audio/tts added_token（id 248070-248076），机器人 prompt 用不到。
#                 => action token 的 id 不会错位。
#   - 图像预处理: image_mean/std 均 [0.5,0.5,0.5]、patch 16、merge 2、temporal 2、
#                 size {65536, 16777216} —— 与 Qwen3.5 的 preprocessor_config 一致。
#                 RynnBrain 多写的 do_*/resample/rescale_factor 都是 HF 默认值显式化。
#                 => 没有 InternVL 那种训推归一化失配的风险。
#   - LoRA      : r=64 / alpha=128 / target_modules 12 个（含 GDN 的 in_proj_{a,b,z,qkv}、
#                 out_proj），与本仓已跑通的 Qwen3.5 LoRA 逐项相同 => max-lora-rank 64 够。
#   - 训练来源  : README tags 含 llama-factory，run_name =
#                 rynnbrain1.1-2b-robot-mix_22-06_fk-pp_02_exchange_token，
#                 即与 Qwen3.5 侧 mix_22-06_fk-pp/02_exchange_token 同一份数据、同一框架。
#                 => 沿用 LF 对齐模板是对的。
#   - 运行环境  : VLLM_VENV = vllm 0.24.0 + transformers 5.12.1，能读 RynnBrain 的
#                 TokenizersBackend 新格式 tokenizer_config。
# 首次起服后仍建议用 /tokenize 与一条训练样本逐 token 比对一次再跑评测。

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
echo "  Temperature         : ${TEMPERATURE}"
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
  --override-generation-config "{\"temperature\": ${TEMPERATURE}, \"top_p\": 1.0, \"top_k\": -1}"
  --trust-remote-code
  --port "${PORT}"
)
[ "${ENFORCE_EAGER}" = "1" ] && CMD+=(--enforce-eager)

exec "${CMD[@]}"
