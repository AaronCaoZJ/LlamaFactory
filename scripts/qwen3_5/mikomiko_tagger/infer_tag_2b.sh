#!/usr/bin/env bash
# INFER + VIZ — mikomiko 图 -> tag(Qwen3.5-2B)。评测、起服务、出审阅页都在这。
#
#   bash infer_tag_2b.sh serve                  起 vLLM 服务 :8110(前台,Ctrl-C 停)
#   bash infer_tag_2b.sh eval  [STEP] [GPU]     固定 400 张评测 -> evalmini_history.tsv
#   bash infer_tag_2b.sh seen  [STEP] [GPU]     200 张训练集图,记忆探针 -> seen_history.tsv
#   bash infer_tag_2b.sh eval-vllm [STEP]       同样 400 张,打已起的服务(快很多)
#   FORCE=1 N=200 GPU=4 bash infer_tag_2b.sh viz          审阅页
#   FORCE=1 GPU=0 bash infer_tag_2b.sh viz-onlyfans       onlyfans 图(无 gold)
#
# 推理只有 infer_mikomiko.py 一份实现,打分只有 metrics_mikomiko.py 一份 —— hf / vllm / 审阅页
# 三条路都走它们,这样三边的数字才可比。
#
# 通用 env:GPU | CKPT_STEP (17296) | CKPT。各段自己的 env 写在对应段落里。
set -euo pipefail

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${BASE_MODEL:-${MODELS_DIR}/Qwen3.5-2B}"
CKPT_STEP="${CKPT_STEP:-17296}"                 # 发布版 checkpoint(2 epochs)
CKPT="${CKPT:-${MODELS_DIR}/Mikomiko_pornpic_tagger/checkpoint-${CKPT_STEP}}"
SAVE_DIR="${LF_ROOT}/saves/qwen3.5-2b/mikomiko"
JSONL_DIR="${LF_ROOT}/data/mikomiko_tag/jsonl"
MAX_PIXELS=262144                               # 必须等于训练的 image_max_pixels,否则就是另一个输入
CHAT_TEMPLATE="${HERE}/chat_template_qwen3_5_lf.jinja"

CMD="${1:-}"; shift || true

usage() {
  cat <<'EOF'
usage: bash infer_tag_2b.sh <command> [args]

  serve                  起 vLLM 服务 :8110(前台)
  eval  [STEP] [GPU]     固定 400 张评测 -> evalmini_history.tsv
  seen  [STEP] [GPU]     200 张训练集图,记忆探针 -> seen_history.tsv
  eval-vllm [STEP]       同样 400 张,打已起的服务
  viz                    抽样 -> 推理 -> 自包含 HTML 审阅页
  viz-onlyfans           同上,跑 onlyfans/(无 gold,走无评分模式)
EOF
  exit 1
}
case "${CMD}" in serve|eval|seen|eval-vllm|viz|viz-onlyfans) ;; *) usage ;; esac

# ── 小工具 ─────────────────────────────────────────────────────────────────────────────────────

# LlamaFactory 存 checkpoint 时会漏掉几个 VL processor 文件,从基座补过去,ckpt 才能单独加载。
copy_vl_files() {
  for f in preprocessor_config.json video_preprocessor_config.json merges.txt vocab.json chat_template.jinja; do
    if [ ! -e "${CKPT}/${f}" ] && [ -e "${BASE_MODEL}/${f}" ]; then
      cp "${BASE_MODEL}/${f}" "${CKPT}/${f}"; echo "[infer] 从基座补了 ${f}"
    fi
  done
}

# 起 vLLM。$1=端口 $2=GPU $3=显存占比。前台跑,exec 掉当前进程。
#
# 永远发 chat_template_qwen3_5_lf.jinja,不发 ckpt 自带的 jinja:后者会在 "assistant\n" 之后插
# 一个空的 "<think>\n\n</think>\n\n",而训练模板 qwen3_5_nothink 从不发这个块。这 4 个 token
# 值 1.2pt microF1,还会把每图复合 tag 从 5.6 抬到 6.1。
serve_vllm() {
  local port="$1" gpu="$2" util="$3"
  local temp="${TEMPERATURE:-0}"                # 0 = 贪心(与评测对齐);>0 才采样
  [ -f "${CKPT}/model.safetensors" ] || { echo "[serve] ERROR: ${CKPT} 下没有 ckpt" >&2; exit 1; }
  [ -f "${CHAT_TEMPLATE}" ] || { echo "[serve] ERROR: 缺 ${CHAT_TEMPLATE}" >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="${gpu}"
  source "${VLLM_VENV}/bin/activate"
  echo "[serve] mikomiko :${port}  GPU=${gpu}  util=${util}  temp=${temp}"
  echo "[serve] ckpt=${CKPT}"
  exec vllm serve "${CKPT}" \
    --served-model-name mikomiko \
    --dtype bfloat16 \
    --gpu-memory-utilization "${util}" \
    --max-model-len 4096 \
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
# 起 vLLM OpenAI 服务,伺服一个**全参 checkpoint**(不是 LoRA adapter),模型名 "mikomiko"。
# 前台运行。viz 会自己起服务(:8111),不需要先跑这个。
# Env: GPU (1) | PORT (8110) | GPU_UTIL (0.7) | TEMPERATURE (0 = 贪心) | CKPT_STEP | CKPT
serve)
  copy_vl_files
  serve_vllm "${PORT:-8110}" "${GPU:-1}" "${GPU_UTIL:-0.7}"
  ;;

# ═══ eval / seen ═══════════════════════════════════════════════════════════════════════════════
# 两段共用下面同一套 hf-predict 流程,只差"评哪批行"和"history 写到哪"。
#
# eval:每次都评**同一批 400 张**(200 test_unseen_mini + 200 test_stratified_mini),一次性拼成
#   eval_mini.jsonl 并带上 _src 字段,打分时才能分开报 unseen / stratified。
# seen:从**训练集**里随机取的 200 行(seed 42),记忆探针。要对着 eval 的 unseen 读:
#   seen F1 >> unseen F1  -> 拟合-泛化 gap 真实存在,加数据多样性有空间
#   seen F1 ≈  unseen F1  -> 标签自相矛盾,堆步数没有提升空间
#   它的 history 单独记一份 —— 这个集没有 unseen/strat 分组,混记会写出一堆 0 值行。
#
# 用法:bash infer_tag_2b.sh eval [STEP] [GPU]     Env: EVAL_BS (16) | CKPT
eval|seen)
  [ -n "${1:-}" ] && CKPT_STEP="$1" && CKPT="${CKPT:-${MODELS_DIR}/Mikomiko_pornpic_tagger/checkpoint-${CKPT_STEP}}"
  GPU="${2:-1}"

  if [ "${CMD}" = "eval" ]; then
    TAG=evalmini
    META="${JSONL_DIR}/eval_mini.jsonl"
    DATASET=mikomiko_tag_eval_mini
    HISTORY="${SAVE_DIR}/predict_sanity/evalmini_history.tsv"
    if [ ! -f "${META}" ]; then
      echo "[eval] 由 test_unseen_mini + test_stratified_mini 拼 ${META} (+_src) ..."
      python3 - "${JSONL_DIR}" "${META}" <<'PY'
import json, sys
jdir, out = sys.argv[1], sys.argv[2]
rows = []
for name, src in (("test_unseen_mini.jsonl", "unseen"), ("test_stratified_mini.jsonl", "strat")):
    for line in open(f"{jdir}/{name}"):
        if line.strip():
            d = json.loads(line); d["_src"] = src; rows.append(d)
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[eval] wrote {len(rows)} rows -> {out}")
PY
    fi
  else
    TAG=seen
    META="${JSONL_DIR}/train_seen_mini.jsonl"
    DATASET=mikomiko_tag_train_seen_mini
    HISTORY="${SAVE_DIR}/predict_sanity/seen_history.tsv"
    [ -f "${META}" ] || { echo "[seen] 缺 ${META}(重建方式:random.sample(train.jsonl, 200), seed=42)" >&2; exit 1; }
  fi

  source "${LF_VENV}/bin/activate"
  export DISABLE_VERSION_CHECK=1
  cd "${LF_ROOT}"

  # 等 checkpoint 写完 —— 可以先把评测排上,等训练跑到那一步
  echo "[${TAG}] 等 ${CKPT} ..."
  until [ -f "${CKPT}/model.safetensors" ] && [ -f "${CKPT}/trainer_state.json" ]; do sleep 30; done
  sleep 5     # 让最后几个文件落盘
  copy_vl_files

  PRED_DIR="${SAVE_DIR}/predict_sanity"
  RUN_DIR="${PRED_DIR}/runs/${TAG}_step_${CKPT_STEP}"
  CFG="${PRED_DIR}/_run_${TAG}_${CKPT_STEP}.yaml"
  mkdir -p "${RUN_DIR}"

  sed -e "s#^model_name_or_path:.*#model_name_or_path: ${CKPT}#" \
      -e "s#^eval_dataset:.*#eval_dataset: ${DATASET}#" \
      -e "s#^per_device_eval_batch_size:.*#per_device_eval_batch_size: ${EVAL_BS:-16}#" \
      "${LF_ROOT}/examples/inference/qwen3_5_2b_full_mikomiko.yaml" > "${CFG}"

  echo "[${TAG}] GPU ${GPU} 上预测(bs=${EVAL_BS:-16},$(wc -l < "${META}") 条) ..."
  env CUDA_VISIBLE_DEVICES="${GPU}" llamafactory-cli train "${CFG}"

  cp "${PRED_DIR}/generated_predictions.jsonl" "${RUN_DIR}/predictions.jsonl"
  cp "${CFG}" "${RUN_DIR}/config.yaml" 2>/dev/null || true

  # 保留:micro P/R/F1、macro F1、atomF1、compEx、compSub、tokF1 + 过度生成诊断。
  # 丢弃:LlamaFactory 自带的 BLEU-4 / ROUGE-* —— 对无序标签集没有意义。
  python3 "${HERE}/metrics_mikomiko.py" \
    "${RUN_DIR}/predictions.jsonl" "${META}" "${CKPT_STEP}" "${HISTORY}" "${RUN_DIR}/metrics.json" \
    | tee "${RUN_DIR}/report.txt"
  echo "[${TAG}] 落盘 -> ${RUN_DIR}/   history -> ${HISTORY}"
  ;;

# ═══ eval-vllm ═════════════════════════════════════════════════════════════════════════════════
# 同样 400 张,但打的是你用 `serve` 起好的服务。比 hf-predict 快很多;数字完全一致,因为两条路
# 都走 infer_mikomiko.py + metrics_mikomiko.py。
# Env: API (http://localhost:8110) | MODEL_NAME (mikomiko) | 以及 eval_vllm_mikomiko.py 的任意 flag
eval-vllm)
  source "${LF_VENV}/bin/activate"
  export DISABLE_VERSION_CHECK=1
  cd "${HERE}"
  exec python3 eval_vllm_mikomiko.py --step "${1:-${CKPT_STEP}}" \
    --api "${API:-http://localhost:8110}" --model "${MODEL_NAME:-mikomiko}" "${@:2}"
  ;;

# ═══ viz / viz-onlyfans ════════════════════════════════════════════════════════════════════════
# 审阅页:抽样 -> 推理 -> 一个自包含 HTML(图片 base64 内嵌,可直接发)。一行四卡,每卡列出
# post tag / 分类标签 / gemini gold / pred,并给每图 tag 级 + 词级 P/R/F1。
#
# 默认**增量**:samples_pred.json 已存在就只重建 HTML(不占 GPU,~70s)。FORCE=1 重跑抽样+推理。
# 400 张耗时:服务启动 ~90s | 推理 ~40s | 出页 ~70s。
#
# viz-onlyfans 是同一条流水线跑 data/mikomiko_tag/onlyfans/,那批图**没有 gold**,build_html.py
# 会切到无评分模式(不出 F1,改出每图 tag/atom/复合数 + 词表外 tag)。它默认用 8112 端口,
# 好跟一个温着的 8111 并存。
#
#   bash infer_tag_2b.sh viz                                  复用已有预测重建页面
#   FORCE=1 N=200 GPU=4 bash infer_tag_2b.sh viz              完整 400 张
#   API=http://localhost:8110 FORCE=1 bash infer_tag_2b.sh viz    复用已起的服务(跑完不关它)
#   FORCE=1 BACKEND=hf DTYPE=fp32 bash infer_tag_2b.sh viz    走 transformers,逐位可复现
#
# Env: FORCE | N (200/split) | SEED (42) | BACKEND (vllm|hf) | GPU (6) | PORT (8111/8112)
#      GPU_UTIL (0.35) | API | DTYPE (bf16|fp32,仅 hf) | BATCH_SIZE (8,仅 hf)
#      CKPT_STEP | WORK_DIR | OUT | IMG_DIR(仅 viz-onlyfans)
viz|viz-onlyfans)
  GPU="${GPU:-6}"
  if [ "${CMD}" = "viz" ]; then
    WORK_DIR="${WORK_DIR:-${SAVE_DIR}/viz_review}"
    OUT="${OUT:-${WORK_DIR}/mikomiko_tagger_seen_unseen_review_$(date +%Y%m%d).html}"
    PORT="${PORT:-8111}"
    SAMPLE=(python3 -u sample_data.py --n "${N:-200}" --seed "${SEED:-42}" --work-dir "${WORK_DIR}")
    STEP1="抽样:每 split ${N:-200} 张,seed=${SEED:-42}"
  else
    WORK_DIR="${WORK_DIR:-${SAVE_DIR}/viz_onlyfans}"
    OUT="${OUT:-${WORK_DIR}/mikomiko_tagger_onlyfans_review_$(date +%Y%m%d).html}"
    PORT="${PORT:-8112}"
    IMG_DIR="${IMG_DIR:-${LF_ROOT}/data/mikomiko_tag/onlyfans}"
    SAMPLE=(python3 -u sample_onlyfans.py --img-dir "${IMG_DIR}" --work-dir "${WORK_DIR}")
    [ -n "${N:-}" ] && SAMPLE+=(--n "${N}")
    STEP1="列图:${IMG_DIR}"
  fi
  mkdir -p "${WORK_DIR}"
  SERVER_PID=""
  # 只关自己起的服务:复用别人已在跑的服务时 SERVER_PID 为空,不会误杀
  cleanup() { [ -n "${SERVER_PID}" ] && kill "${SERVER_PID}" 2>/dev/null && echo "[viz] 已停掉服务" || true; }
  trap cleanup EXIT
  cd "${HERE}/visualization"

  if [ "${FORCE:-0}" = "1" ] || [ ! -f "${WORK_DIR}/samples_pred.json" ]; then
    [ -f "${CKPT}/model.safetensors" ] || { echo "[viz] ERROR: ${CKPT} 下没有 ckpt" >&2; exit 1; }
    echo "[viz] 1/3 ${STEP1}"
    "${SAMPLE[@]}"

    if [ "${BACKEND:-vllm}" = "vllm" ]; then
      API="${API:-http://localhost:${PORT}}"
      LOG="${WORK_DIR}/vllm_server.log"
      if curl -sf -m 3 "${API}/v1/models" >/dev/null 2>&1; then
        echo "[viz] 2/3 复用 ${API} 上已在跑的服务"
      else
        echo -n "[viz] 2/3 起 vLLM:GPU=${GPU} port=${PORT}(~90s)-> ${LOG} "
        # 子 shell 里跑,serve_vllm 结尾的 exec 只会替换掉那个子进程
        ( serve_vllm "${PORT}" "${GPU}" "${GPU_UTIL:-0.35}" ) >"${LOG}" 2>&1 &
        SERVER_PID=$!
        for i in $(seq 1 120); do        # 10 分钟上限
          curl -sf -m 3 "${API}/v1/models" >/dev/null 2>&1 && { echo " 起来了 (${SECONDS}s)"; break; }
          # 进程死了要带着日志报错,不然就在这儿干等到超时
          kill -0 "${SERVER_PID}" 2>/dev/null || {
            echo; echo "[viz] ERROR: 服务挂了,${LOG} 末尾:" >&2; tail -25 "${LOG}" >&2; exit 1; }
          [ "$i" = 120 ] && { echo; echo "[viz] ERROR: 10 分钟还没起来,看 ${LOG}" >&2; exit 1; }
          printf '.'; sleep 5
        done
      fi
      INFER_ARGS=(--backend vllm --api "${API}")
    else
      # fp32 权重要 ~11 GiB(bf16 ~6)。在这里失败,好过加载到一半才 OOM。
      NEED=$([ "${DTYPE:-bf16}" = "fp32" ] && echo 14000 || echo 8000)
      FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}")
      [ "${FREE}" -ge "${NEED}" ] || {
        echo "[viz] ERROR: GPU ${GPU} 只剩 ${FREE} MiB,${DTYPE:-bf16} 需要 ~${NEED} MiB" >&2
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >&2; exit 1; }
      echo "[viz] 2/3 推理(hf):GPU=${GPU} bs=${BATCH_SIZE:-8} dtype=${DTYPE:-bf16}"
      INFER_ARGS=(--backend hf --ckpt "${CKPT}" --batch-size "${BATCH_SIZE:-8}" --dtype "${DTYPE:-bf16}")
    fi

    source "${LF_VENV}/bin/activate"
    export DISABLE_VERSION_CHECK=1
    CUDA_VISIBLE_DEVICES="${GPU}" python3 -u "${HERE}/infer_mikomiko.py" \
      --input "${WORK_DIR}/samples.json" --output "${WORK_DIR}/samples_pred.json" "${INFER_ARGS[@]}"
    cleanup; SERVER_PID=""
  else
    echo "[viz] 复用 ${WORK_DIR}/samples_pred.json(要重跑加 FORCE=1)"
    source "${LF_VENV}/bin/activate"
    export DISABLE_VERSION_CHECK=1
  fi

  echo "[viz] 3/3 出页(base64 缩略图,400 张约 70s)"
  python3 -u build_html.py --work-dir "${WORK_DIR}" --out "${OUT}" \
    --model-name "${MODEL_NAME:-QWEN 3.5 2B}" \
    --subtitle "${SUBTITLE:-Full SFT · ${EPOCHS:-2.0} epochs · ${CKPT_STEP} steps · temperature 0.0}"
  echo "[viz] DONE (${SECONDS}s) -> ${OUT}"
  ;;
esac
