#!/usr/bin/env bash
# TRAIN — mikomiko 图 -> 四段描述(en/ja/zh),Qwen3.5-9B 全参 SFT,8x H200 ZeRO-3。
# 数据 = 0721 交付,由 data/mikomiko_tag/process_data.sh desc0721 构建。
#
#   bash train_desc_9b.sh smoke            # 这个配置能不能训起来(4 卡,不写 ckpt)
#   bash train_desc_9b.sh probe 8 12 16    # 显存能塞下多大 per_device batch
#   nohup bash train_desc_9b.sh full &     # 真正的 1 epoch 训练,~15h,必须 detach
#
# 与 ../mikomiko_tagger/ 是**两个不同任务**:那个输出 ~30 token 的标签列表,这个输出 500-950
# token 的散文。除 model / deepspeed 外几乎每项超参都不同,不要互相 copy 数值。
set -euo pipefail

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

YAML="examples/train_full/qwen3_5_9b_mikomiko_grok_desc.yaml"
SAVE_DIR="${LF_ROOT}/saves/qwen3.5-9b/mikomiko"
CMD="${1:-}"; shift || true

usage() {
  cat <<'EOF'
usage: bash train_desc_9b.sh <command> [args]

  smoke                  4 卡跑通性检查,不写 checkpoint
  probe <bs> [bs ...]    在真实卡数上找最大 per_device batch
  full                   正式 1 epoch 训练(8x H200,~15h,记得 detach)
EOF
  exit 1
}

case "${CMD}" in smoke|probe|full) ;; *) usage ;; esac

source "${LF_VENV}/bin/activate"
export DISABLE_VERSION_CHECK=1     # transformers 5.6.1 > LF 硬编码上限 5.6.0;Qwen3.5 需要新版

# Qwen3.5 的 GDN 反向内核在 Hopper 上走 tilelang(JIT),需要能用的 g++;env_setup.sh 会按机器
# 建 .cc-shim 垫片。只有垫片真能编译才前置,免得换机器后悬空的垫片挡住系统里好用的编译器。
_SHIM="${LF_ROOT}/.cc-shim"
if [ -x "${_SHIM}/g++" ] && echo 'int main(){return 0;}' | "${_SHIM}/g++" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}" CC="${_SHIM}/gcc" CXX="${_SHIM}/g++" CUDAHOSTCXX="${_SHIM}/g++"
fi

cd "${LF_ROOT}"

# smoke 与 probe 共用的那套覆盖:2,400 行三语分层子集 + 它自己的 tokenized 缓存(所以两者都
# 不会去建正式训练需要的 1.34M 行缓存)+ 不写 checkpoint。
#
# 注意 '"no"' 的引号。CLI 覆盖走 OmegaConf.from_cli,每个值都会被 YAML 解析一遍 —— 裸的 no
# 会变成布尔 False,transformers 随即报 "False is not a valid IntervalStrategy"。任何
# yes/no/on/off/y/n 形态的值都得带内层引号。
overrides() {   # $1 bs  $2 steps  $3 输出子目录  -> 填充 OVERRIDES 数组
  OVERRIDES=(
    dataset=mikomiko_desc_0721_smoke
    tokenized_path="${SAVE_DIR}/tokenized_smoke_0721"
    output_dir="${SAVE_DIR}/$3"
    per_device_train_batch_size="$1"
    gradient_accumulation_steps=1
    max_steps="$2"
    save_strategy='"no"'
    eval_strategy='"no"'
    logging_steps=1
    plot_loss=false
    report_to=none
    num_train_epochs=1.0
  )
}

case "${CMD}" in

# ═══ smoke ═════════════════════════════════════════════════════════════════════════════════════
# 只回答"这个配置能不能训起来":数据能读、图能解码、模板能套上、loss 有限且在动、一步多少钱。
# 不写 checkpoint。
#
# 4 卡量出来的显存是 8 卡的**下界**:ZeRO-3 下 rank 越多,每 rank 的参数/优化器分片越小,
# 这里塞得下的,8 卡一定塞得下。
#
#   bash train_desc_9b.sh smoke              per_device 2,30 步
#   BS=8 STEPS=12 bash train_desc_9b.sh smoke
# Env: BS (2) | STEPS (30) | GPUS (0,1,2,3)
smoke)
  export WANDB_DISABLED=true          # 冒烟跑不该往 wandb 项目里灌垃圾
  export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
  BS="${BS:-2}"
  overrides "${BS}" "${STEPS:-30}" "smoke_bs${BS}"
  exec llamafactory-cli train "${YAML}" "${OVERRIDES[@]}" "$@"
  ;;

# ═══ probe ═════════════════════════════════════════════════════════════════════════════════════
# 找能塞下的最大 per_device_train_batch_size,**必须用正式训练的卡数**跑。卡数是有意义的:
# ZeRO-3 下分片是 (模型状态 / world_size),4 卡塞得下的 batch 在 8 卡上还有余量 —— 拿 4 卡的
# 结论去训 8 卡等于白扔显存。
#
# 峰值出现在装着最长序列的那个 batch 上,所以步数要够看到几个 batch:STEPS=15 是下限,不是形式。
# OOM 在这里是**结果不是失败**,脚本会继续探下一档。
#
#   bash train_desc_9b.sh probe 8 12 16 24     Env: STEPS (15) | GPUS (0..7)
probe)
  [ $# -gt 0 ] || { echo "usage: bash train_desc_9b.sh probe <bs> [bs ...]" >&2; exit 1; }
  export WANDB_DISABLED=true
  export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
  NGPU=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  mkdir -p logs
  set +e
  for BS in "$@"; do
    LOG="logs/probe_bs${BS}_${NGPU}gpu.log"
    echo "=== probing per_device=${BS} on ${NGPU} GPUs -> ${LOG} ==="
    # 边跑边采样显存:nvidia-smi 报的是分配器的 reservation,那才是真正要塞下的量,不只是活张量。
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 4; done ) \
      > "logs/.mem_bs${BS}" 2>/dev/null &
    MEMPID=$!
    overrides "${BS}" "${STEPS:-15}" "probe_bs${BS}"
    llamafactory-cli train "${YAML}" "${OVERRIDES[@]}" > "${LOG}" 2>&1
    RC=$?
    kill ${MEMPID} 2>/dev/null; wait ${MEMPID} 2>/dev/null
    PEAK=$(sort -n "logs/.mem_bs${BS}" 2>/dev/null | tail -1); rm -f "logs/.mem_bs${BS}"
    if [ ${RC} -ne 0 ]; then
      grep -qi "out of memory" "${LOG}" \
        && echo "  bs=${BS} -> OOM (峰值 ${PEAK:-?} MiB)" \
        || echo "  bs=${BS} -> FAILED rc=${RC}(不是 OOM,看 ${LOG})"
    else
      SPS=$(grep -oE "'train_samples_per_second': '[0-9.]+'" "${LOG}" | tail -1 | grep -oE "[0-9.]+")
      echo "  bs=${BS} -> OK  峰值=${PEAK:-?} MiB  samples/s(含 warmup)=${SPS:-?}"
    fi
  done
  ;;

# ═══ full ══════════════════════════════════════════════════════════════════════════════════════
# 正式训练:1 epoch,8x H200,ZeRO-3,9B 全参。超参全在 yaml 里,这里只钉卡和重定向日志。
#
# 首次启动会先花 ~30-60min 把 1,340,392 行 tokenize 进 saves/.../tokenized_desc_0721 才到第 1 步;
# 之后重启复用。改了 dataset / template / cutoff_len **必须删掉那个目录** —— 缓存按原文本键、
# 不校验 dataset 名,命中旧 token 是静默的。
#
# 跑 ~15h,关终端会 SIGTERM 掉它,所以要 detach:
#   nohup bash train_desc_9b.sh full &        Env: GPUS (0..7) | LOG
full)
  export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
  mkdir -p logs
  LOG="${LOG:-logs/train_9b_grok_desc_v0.log}"
  echo "[train-desc] GPUS=${CUDA_VISIBLE_DEVICES} -> ${LOG}"
  exec llamafactory-cli train "${YAML}" "$@" >> "${LOG}" 2>&1
  ;;
esac
