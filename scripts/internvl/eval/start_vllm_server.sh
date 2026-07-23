#!/usr/bin/env bash
# vLLM OpenAI server: Qwen3.5-27B + MVTOKEN LoRA adapters (default :8101).
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
GPU="${GPU:-5}"
export CUDA_VISIBLE_DEVICES="${GPU}"

# ================================================================================
#! Args (server knobs / model / LoRA)
#* Overrides: GPU | PORT | GPU_UTIL | TEMPERATURE
GPU_UTIL="${GPU_UTIL:-0.7}"
TEMPERATURE="${TEMPERATURE:-0}"

MAX_LEN=8192
MAX_NUM_SEQS=256
ENFORCE_EAGER=0

ZECHEN="/workspace1/zechen/finetune/lora"
SAVES="${LF_ROOT}/saves/internvl3.5-2b/robot"

#* Internvl3.5-8b
# PORT="${PORT:-8208}"
# BASE_MODEL="/workspace1/zechen/hf_download/InternVL3_5-8B-HF"
# LORA_MODULES=(
#   "internvl3.5-8b=${SAVES}/InternVL3.5-8b"
# )

#* Internvl3.5-4b
# PORT="${PORT:-8204}"
# BASE_MODEL="/workspace1/zechen/hf_download/InternVL3_5-4B-HF"
# LORA_MODULES=(
#   "internvl3.5-4b=${SAVES}/InternVL3.5-4b"
# )

#* Internvl3.5-2b
PORT="${PORT:-8202}"
BASE_MODEL="/workspace1/zechen/hf_download/InternVL3_5-2B-HF"
LORA_MODULES=(
  "internvl3.5-2b=${ZECHEN}/InternVL3.5-2b"
  "internvl3.5-2b-History2=${ZECHEN}/InternVL3.5-2b-History2"
  "internvl3.5-2b-History2-VideoSlot=${ZECHEN}/InternVL3.5-2b-History2-VideoSlot"
  "internvl3.5-2b-History2-PlainPrompt=${ZECHEN}/InternVL3.5-2b-History2-PlainPrompt"
  # "internvl3.5-2b-ms0717_blockpap=${SAVES}/ms0717_blockpap_oracle_wide"
  "internvl3.5-2b-ms0717_blockpap_follow=${SAVES}/ms0717_blockpap_follow"
  "internvl3.5-2b-ms0717_stackcube_follow=${SAVES}/ms0717_stackcube_follow"
)

#* Internvl3.5-1b
# PORT="${PORT:-8201}"
# BASE_MODEL="/workspace1/zechen/hf_download/InternVL3_5-1B-HF"
# LORA_MODULES=(
#   "internvl3.5-1b=${SAVES}/InternVL3.5-1b"
# )

# LF 对齐的 chat template（必需）。InternVL3.5-HF 自带的 chat_template.jinja 与 LF
# 'template: intern_vl' 的训练分布差 35 个 token：(1) 它完全不注入 system，而 LF 在数据集
# 没有 system 列时总会补上 default_system（书生·万象那句，~31 token）；(2) 它在每个图像
# 占位符后面硬加一个 '\n'，而 LF 是就地替换 <image>，本数据集是 '<image><image>You are...'
# 紧贴无换行（2 张图 → 2 个多余 \n）。已用训练侧编码器逐 token 对拍：756 == 756。
CHAT_TEMPLATE="${LF_ROOT}/scripts/internvl/eval/chat_template_internvl_lf.jinja"
# 必须配 openai：content-format 若被判成 string，vLLM 会自己用 '\n' 拼图像占位符
# （chat_utils.py:_get_full_multimodal_text_prompt），绕过模板把 (2) 又加回来。
CONTENT_FORMAT="openai"

#! 【必需，否则模型输出乱码】强制 untie lm_head。
# InternVL3_5-*-HF 的 config.json 里根本没有 tie_word_embeddings 字段，而两边的默认值相反：
#   - transformers 的 Qwen3Config 默认 False -> 正确加载 checkpoint 里独立的 language_model.lm_head
#   - vLLM 走 PretrainedConfig 默认 True，并传播进 text_config -> 直接丢弃 lm_head，拿
#     embed_tokens 当输出层（实测 lm_head.data_ptr() == embed_tokens.data_ptr()）
# 后果：输出层用错权重矩阵，语法流畅但语义崩坏且疯狂重复（"I'm Sophia Zhang Zhou Zhou Zhou..."），
# 且永远吐不出 <|im_end|>（rank 掉到 16）所以不会停。这与 LoRA / 图像 / chat template 都无关，
# base model 纯文本就能复现。加上本行后 lm_head.abs_mean 恢复 0.024792（= checkpoint 真值）。
HF_OVERRIDES='{"tie_word_embeddings": false, "text_config": {"tie_word_embeddings": false}}'

#! 【服务 <video> 槽位的 LoRA 时必需】把 video 帧的归一化拉回 ImageNet。
# InternVL 的 image / video 两条预处理分支用的是不同常数，而 LF 训练 <video> 时（mm_plugin.py
# :882-883）根本没走 video_processor，是拿 image_processor 处理每一帧的：
#   LF 训练   448×448 + ImageNet mean/std (.485/.456/.406, .229/.224/.225)
#   vLLM 推理 448×448（interns1.py:184-189 已强制对齐 size）+ CLIP mean/std ← 只有归一化没对齐
# 后果是静默的：token 数完全对得上，不报错，但 ViT 拿到的输入分布压缩到 84%，逐元素偏差
# 0.1934，而这些 history 模型要读的真实帧间信号只有 0.3136 —— 失配是信号的 62%。
# 这行同时打到 image / video 两个 processor（interns1.py 把 **kwargs 转发给两者），但 image
# 分支的 baseline 本来就是这两个值，所以对它是空操作，同 server 上的 <image> LoRA 不受影响。
# 实测：加上后 video 分支的 pixel_values 与 LF 训练侧逐元素 0 差异（max|Δ|=0.000000）。
MM_PROCESSOR_KWARGS='{"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}'

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
echo "  Chat template       : ${CHAT_TEMPLATE:-<model default>}"
echo "  Content format      : ${CONTENT_FORMAT}"
echo "  HF overrides        : ${HF_OVERRIDES}"
echo "  MM processor kwargs : ${MM_PROCESSOR_KWARGS}"
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
  --chat-template-content-format "${CONTENT_FORMAT}"
  --hf-overrides "${HF_OVERRIDES}"
  --mm-processor-kwargs "${MM_PROCESSOR_KWARGS}"
  --override-generation-config "{\"temperature\": ${TEMPERATURE}, \"top_p\": 1.0, \"top_k\": -1}"
  --trust-remote-code
  --port "${PORT}"
)
[ "${ENFORCE_EAGER}" = "1" ] && CMD+=(--enforce-eager)

exec "${CMD[@]}"
