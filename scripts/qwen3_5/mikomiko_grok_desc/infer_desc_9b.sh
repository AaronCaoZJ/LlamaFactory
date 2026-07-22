#!/usr/bin/env bash
# INFER + VIZ — mikomiko 图 -> 四段描述(Qwen3.5-9B)。起服务和出审阅页都在这。
#
#   bash infer_desc_9b.sh serve              起 vLLM 服务 :8121(前台,Ctrl-C 停)
#   bash infer_desc_9b.sh viz                复用已有预测重建 HTML(不占 GPU,4 秒)
#   FORCE=1 bash infer_desc_9b.sh viz        完整流水线(2 张空闲 H200 约 6 分钟)
#   FORCE=1 WITH_BASE=0 bash infer_desc_9b.sh viz    只跑微调模型,省一半 GPU 时间
#
# 这里**没有 eval 段**,是有意的:描述是自由散文,措辞完全不同也可以完全正确,BLEU/ROUGE 主要
# 测用词运气还容易被当成准确率读。能机械判定的只有"形状对不对"(语种、4 段齐不齐、字数、
# 重复度、有没有撞 token 上限),那些判据在 visualization/metrics_desc.py 里,由 viz 调用并直接
# 画进页面。内容质量要人看图判断 —— 这正是这一页要放原图的原因。
set -euo pipefail

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_DIR="${LF_ROOT}/saves/qwen3.5-9b/mikomiko"
CKPT="${CKPT:-${SAVE_DIR}/grok_desc_v0}"
BASE_MODEL="${BASE_MODEL:-${MODELS_DIR}/Qwen3.5-9B}"
MAX_PIXELS=262144                    # 必须等于训练的 image_max_pixels,否则就是另一个输入
# 用 tag 任务那份 chat 模板,不用模型自带的 —— 原因见下面 serve_vllm 的注释
CHAT_TEMPLATE="${LF_ROOT}/scripts/qwen3_5/mikomiko_tagger/chat_template_qwen3_5_lf.jinja"

CMD="${1:-}"; shift || true

usage() {
  cat <<'EOF'
usage: bash infer_desc_9b.sh <command> [args]

  serve                  起 vLLM 服务 :8121(前台)
  viz                    抽样 -> 微调+基座推理 -> 自包含 HTML 审阅页
EOF
  exit 1
}
case "${CMD}" in serve|viz) ;; *) usage ;; esac

# 起 vLLM。$1=模型目录 $2=服务名 $3=端口 $4=GPU $5=显存占比。前台跑,exec 掉当前进程。
#
# 微调 ckpt 和未微调基座**用同一份 chat 模板**是刻意的:两边自带的 chat_template.jinja 逐字节
# 相同,都走 LF 这份模板 serve,prompt token 就完全一致,两栏的差异只来自权重。给基座用它自带
# 模板会多注入一个空的 <think></think>,那就不是对照了。
serve_vllm() {
  local model="$1" served="$2" port="$3" gpu="$4" util="$5"
  local temp="${TEMPERATURE:-0}"     # 0 = 贪心;>0 才采样
  ls "${model}"/*.safetensors >/dev/null 2>&1 || {
    echo "[serve] ERROR: ${model} 下没有 *.safetensors" >&2; exit 1; }
  [ -f "${CHAT_TEMPLATE}" ] || { echo "[serve] ERROR: 缺 ${CHAT_TEMPLATE}" >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="${gpu}"
  source "${VLLM_VENV}/bin/activate"
  echo "[serve] ${served} :${port}  GPU=${gpu}  util=${util}  temp=${temp}"
  echo "[serve] model=${model}"
  exec vllm serve "${model}" \
    --served-model-name "${served}" \
    --dtype bfloat16 \
    --gpu-memory-utilization "${util}" \
    --max-model-len "${MAX_LEN:-4096}" \
    --max-num-seqs "${MAX_NUM_SEQS:-64}" \
    --limit-mm-per-prompt '{"image": 1}' \
    --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}" \
    --override-generation-config "{\"temperature\": ${temp}, \"top_p\": 1.0, \"top_k\": -1}" \
    --chat-template "${CHAT_TEMPLATE}" \
    --trust-remote-code \
    --port "${port}"
}

case "${CMD}" in

# ═══ serve ═════════════════════════════════════════════════════════════════════════════════════
# 起一个 vLLM 服务。viz 会自己起并在跑完关掉,只有想跨多次调用保持服务温着时才手动跑这个
# (然后 viz 会自动复用同端口上已在跑的服务,且跑完不关它)。
# Env: MODEL (默认 CKPT) | SERVED (desc_sft) | PORT (8121) | GPU (0) | GPU_UTIL (0.60) | MAX_LEN
serve)
  serve_vllm "${MODEL:-${CKPT}}" "${SERVED:-desc_sft}" "${PORT:-8121}" "${GPU:-0}" "${GPU_UTIL:-0.60}"
  ;;

# ═══ viz ═══════════════════════════════════════════════════════════════════════════════════════
# 审阅页:按 seen/unseen × en/ja/zh 抽样,微调 checkpoint 和未微调基座各跑一遍,出一个自包含
# HTML,gold / 微调后 / 基座三栏并排。
#
# 抽样按语言等量,不是随机:数据 80/10/10 en/zh/ja 且每张图只有一种语言,随机抽 60 条会落到
# 约 48 en / 6 zh / 6 ja,对 ja 什么都说不了。
#
# 耗时(120 张,H200):抽样 ~90s(要流式扫 7.3 GB 的 train.jsonl)| 每个服务起 ~110s |
# 每次推理 ~40s | 缩略图+出页 ~40s。
#
# Env: FORCE | N (每语言每 split 20 张 -> 6N 张) | SEED (42) | WITH_BASE (1)
#      GPU_SFT (0) | GPU_BASE (1) | PORT_SFT (8121) | PORT_BASE (8122) | MAX_NEW (1536)
#      CKPT | BASE_MODEL | WORK_DIR | OUT
viz)
  N="${N:-20}"
  WORK_DIR="${WORK_DIR:-${SAVE_DIR}/viz_desc_0721}"
  OUT="${OUT:-${WORK_DIR}/mikomiko_grok_desc_review_$(date +%Y%m%d).html}"
  mkdir -p "${WORK_DIR}"
  PIDS=()
  # 只关自己起的服务:复用别人已在跑的服务时不会往 PIDS 里加,也就不会误杀
  cleanup() { for p in "${PIDS[@]:-}"; do [ -n "${p}" ] && kill "${p}" 2>/dev/null || true; done; PIDS=(); }
  trap cleanup EXIT
  cd "${HERE}/visualization"

  # 跑一个模型:没服务就起一个、等就绪,然后推理。$1=标签 $2=模型目录 $3=服务名 $4=GPU $5=端口
  run_model() {
    local tag="$1" model="$2" served="$3" gpu="$4" port="$5"
    local api="http://localhost:${port}" log="${WORK_DIR}/vllm_${tag}.log"
    if curl -sf -m 3 "${api}/v1/models" >/dev/null 2>&1; then
      echo "[viz] 复用 ${api} 上已在跑的服务"
    else
      echo -n "[viz] 起 ${tag} 服务:GPU=${gpu} port=${port} -> ${log} "
      # 子 shell 里跑,serve_vllm 结尾的 exec 只会替换掉那个子进程
      ( serve_vllm "${model}" "${served}" "${port}" "${gpu}" "${GPU_UTIL:-0.60}" ) >"${log}" 2>&1 &
      local pid=$!; PIDS+=("${pid}")
      for i in $(seq 1 90); do          # 7.5 分钟上限
        curl -sf -m 3 "${api}/v1/models" >/dev/null 2>&1 && { echo " 起来了 (${SECONDS}s)"; break; }
        # 进程死了要带着日志报错,不然就在这儿干等到超时
        kill -0 "${pid}" 2>/dev/null || {
          echo; echo "[viz] ERROR: ${tag} 服务挂了,${log} 末尾:" >&2; tail -25 "${log}" >&2; exit 1; }
        [ "$i" = 90 ] && { echo; echo "[viz] ERROR: 7.5 分钟还没起来,看 ${log}" >&2; exit 1; }
        printf '.'; sleep 5
      done
    fi
    # infer_desc.py 比 infer_mikomiko.py 多留 finish_reason / completion_tokens:30 token 的标签
    # 列表没人关心这个,500-950 token 的描述里"是自己停的还是撞了上限"却是完整与截断之别。
    python3 -u infer_desc.py --input "${WORK_DIR}/samples_pred.json" \
      --output "${WORK_DIR}/samples_pred.json" --api "${api}" --model "${served}" \
      --tag "${tag}" --max-new-tokens "${MAX_NEW:-1536}"
  }

  if [ "${FORCE:-0}" = "1" ] || [ ! -f "${WORK_DIR}/samples_pred.json" ]; then
    ls "${CKPT}"/*.safetensors >/dev/null 2>&1 || { echo "[viz] ERROR: ${CKPT} 下没有 ckpt" >&2; exit 1; }
    echo "[viz] 1/3 抽样:每语言每 split ${N} 张,seed=${SEED:-42}"
    python3 -u sample_data.py --n "${N}" --seed "${SEED:-42}" --work-dir "${WORK_DIR}"
    cp "${WORK_DIR}/samples.json" "${WORK_DIR}/samples_pred.json"   # 两次预测累加进同一个文件

    source "${LF_VENV}/bin/activate"
    export DISABLE_VERSION_CHECK=1
    echo "[viz] 2/3 推理"
    run_model sft "${CKPT}" desc_sft "${GPU_SFT:-0}" "${PORT_SFT:-8121}"
    [ "${WITH_BASE:-1}" = "1" ] && run_model base "${BASE_MODEL}" desc_base "${GPU_BASE:-1}" "${PORT_BASE:-8122}"
    cleanup
  else
    echo "[viz] 复用 ${WORK_DIR}/samples_pred.json(要重跑加 FORCE=1)"
    source "${LF_VENV}/bin/activate"
    export DISABLE_VERSION_CHECK=1
  fi

  echo "[viz] 3/3 出页"
  python3 -u build_html.py --work-dir "${WORK_DIR}" --out "${OUT}" \
    --subtitle "${SUBTITLE:-Qwen3.5-9B 全参 SFT · 1 epoch · 13963 步 · eval_loss 0.4257 · 贪心解码}" \
    --note "每语言 seen/unseen 各 ${N} 张；同一 prompt、同一 chat 模板下，微调后与未微调基座的输出并排。"
  echo "[viz] DONE (${SECONDS}s) -> ${OUT}"
  ;;
esac
