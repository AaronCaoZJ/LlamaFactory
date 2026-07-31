#!/usr/bin/env bash
# TRAIN — mikomiko 图 -> 四段描述(en/ja/zh),Qwen3.5-9B 全参 SFT,8x H200 ZeRO-3。
#
# 数据 = 20260730 交付,4 个来源合并等权(每行等权,不做按来源的 interleave 加权):
#   pornpics 1,625,718 (49.1%)  av 805,995 (24.3%)  of 748,234 (22.6%)  oneione 132,156 (4.0%)
#   = 3,312,103 行,语言 en 60.9% / ja 27.5% / zh 11.7%
# 由 data/mikomiko_tag/ 下四个 dataset_builder_desc_*_0730.py 分别构建,再 mix_desc_0730.py
# --merge 外部洗牌合并。eval 是 4 个来源各自的 200 行 unseen mini,yaml 里开了
# eval_on_each_dataset,所以 wandb 会有 4 条独立的 eval_*_loss 曲线。
#
#   bash train_desc_9b_0730.sh smoke            # 这个配置能不能训起来(不写 ckpt)
#   bash train_desc_9b_0730.sh probe 6 8 12     # 显存/吞吐曲线
#   nohup bash train_desc_9b_0730.sh full &     # 真正的 1 epoch 训练,~38h,必须 detach
#
# 2026-07-30 8 卡实测(热态,见 yaml 的 train 段):bs=6/8/12 = 22.0/23.2/24.2 samp/s,
# 显存 50.9%/61.9%/94.8%。yaml 取 per_device 12 x grad_accum 1 = eff 96(最快档)。
# ⚠ 冷态下同样的配置会量出 8.8/13.6/6.9,慢 3 倍且顺序相关 —— 见 yaml 注释,别被开头几百步骗到。
#
# 与 ../mikomiko_tagger/ 是**两个不同任务**:那个输出 ~30 token 的标签列表,这个输出 500-950
# token 的散文。除 model / deepspeed 外几乎每项超参都不同,不要互相 copy 数值。
set -euo pipefail

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

YAML="examples/train_full/qwen3_5_9b_mikomiko_grok_desc_0730.yaml"
SAVE_DIR="${LF_ROOT}/saves/qwen3.5-9b/mikomiko"
CMD="${1:-}"; shift || true

usage() {
  cat <<'EOF'
usage: bash train_desc_9b_0730.sh <command> [args]

  smoke                  4 卡跑通性检查,不写 checkpoint
  probe <bs> [bs ...]    在真实卡数上找最大 per_device batch
  full                   正式 1 epoch 训练(8x H200,~38h,记得 detach)
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

# Triton 的 autotune 缓存目录不存在时,每次启动都要重新调优内核 —— node0 上就是这样,整轮
# probe 始终停在 ~9 samp/s 没热起来,把人误导到"吞吐随 batch 增大不升反降"的错误结论。
# 建一下是零成本的,顺带消掉日志里那条 `df: /root/.triton/autotune: No such file or directory`。
mkdir -p "${HOME}/.triton/autotune" "${HOME}/.tilelang"

# smoke 与 probe 共用的那套覆盖:2,400 行的 mikomiko_desc_0730_smoke + 它自己的 tokenized 缓存
# (所以两者都不会去建正式训练需要的 3.31M 行缓存)+ 不写 checkpoint。
#
# smoke 集是按「4 来源 x 3 语言」各 200 行分层抽的,不是 head -N。probe 量的是峰值显存,而峰值
# 由它碰巧遇到的最长 batch 决定:av/ja 比 pornpics/zh 长约 35%,而洗牌后 head -N 有 49% 是
# pornpics —— 拿那种子集探出来的 bs 会偏大,正式训练跑到 av 长尾时 OOM。
#
# 注意 '"no"' 的引号。CLI 覆盖走 OmegaConf.from_cli,每个值都会被 YAML 解析一遍 —— 裸的 no
# 会变成布尔 False,transformers 随即报 "False is not a valid IntervalStrategy"。任何
# yes/no/on/off/y/n 形态的值都得带内层引号。
overrides() {   # $1 bs  $2 steps  $3 输出子目录  -> 填充 OVERRIDES 数组
  OVERRIDES=(
    dataset=mikomiko_desc_0730_smoke
    tokenized_path="${SAVE_DIR}/tokenized_smoke_0730"
    output_dir="${SAVE_DIR}/$3"
    # smoke/probe 的输出目录是一次性的。不设这个,重复探同一个 bs 会在第 0 步就被
    # "Output directory already exists and is not empty" 打回,看起来像跑失败了。
    overwrite_output_dir=true
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
#   bash train_desc_9b_0730.sh smoke              per_device 2,30 步
#   BS=8 STEPS=12 bash train_desc_9b_0730.sh smoke
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
# 2026-07-30 实测曲线(cutoff 2560,本数据,**内核缓存热态**):
#   bs=6 50.9% 2.18s/step 22.0samp/s | bs=8 61.9% 2.76s 23.2 | bs=12 94.8% 3.97s 24.2 | bs=16 OOM
# ⚠ 第一次跑任何一档都会明显偏慢(Triton 在 autotune)。要么先跑一档丢掉,要么只信重跑的数。
# 想复现:
#   bash train_desc_9b_0730.sh probe 6 8 12
#
#   bash train_desc_9b_0730.sh probe 6 8 10 12     Env: STEPS (15) | GPUS (0..7)
probe)
  [ $# -gt 0 ] || { echo "usage: bash train_desc_9b_0730.sh probe <bs> [bs ...]" >&2; exit 1; }
  export WANDB_DISABLED=true
  export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
  NGPU=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  mkdir -p logs
  set +e
  for BS in "$@"; do
    LOG="logs/probe_0730_bs${BS}_${NGPU}gpu.log"
    # 上一档(尤其是 OOM 那档)的进程要几十秒才把显存还给驱动。不等,采样器会把上一轮的
    # 残留当成本轮峰值 —— 实测见过 bs=8 报出 141,815 MiB 而 bs=12 只有 136,240 MiB 这种
    # 不可能的结果,和上一轮 bs=16 的 OOM 峰值 141,818 只差 3 MiB。
    for _ in $(seq 60); do
      USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      [ "${USED:-99999}" -lt 3000 ] && break
      sleep 5
    done
    echo "=== probing per_device=${BS} on ${NGPU} GPUs (起始显存 ${USED} MiB) -> ${LOG} ==="
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
# 正式训练:1 epoch,8x H200,ZeRO-3,9B 全参。超参全在 yaml 里,这里只钉卡、查盘、重定向日志。
#
# 3,312,103 行 / eff_batch 96 = 34,501 步。实测 24.2 samples/s -> 约 38 小时。
#
# 第 1 步之前还有一段一次性的 tokenize。实测 100,000 行一趟 3:08(563 ex/s;2,400 行那次的
# 28.5 ex/s 全是 64 个 worker 的启动开销,别拿它外推),日志里会跑 2 趟,3.31M 行合计约 1.8h。
# 之后重启复用 saves/.../tokenized_desc_0730_cut2560。改了 dataset / template / cutoff_len
# **必须删掉那个目录** —— 缓存按原文本键、不校验 dataset 名与 cutoff,命中旧 token 是静默的。
# (路径里带了 cut2560 就是为了让改 cutoff 时忘记删也不会撞车。)
#
# 跑 ~38h,关终端会 SIGTERM 掉它,所以要 detach:
#   nohup bash train_desc_9b_0730.sh full &        Env: GPUS (0..7) | LOG | SKIP_DISK_CHECK
full)
  export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
  # bs=12 峰值吃到 94.8%(136,240/143,771 MiB),只剩 7.5GB 要扛 34,501 步。expandable_segments
  # 让分配器用可增长的段而不是固定块,显著降低长跑中的碎片化 OOM —— 这是本档唯一的真实风险。
  # 换回 bs=6 x accum 2(同样 eff 96,显存 50.9%)时可以去掉。
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  mkdir -p logs "${SAVE_DIR}"

  # 9B 全参 ckpt ~100GB/个 x save_total_limit 2,加上 3.31M 行的 tokenized 缓存。盘满是在
  # 第 2000 步存档时炸,那时已经跑了两小时 —— 宁可现在就拦住。
  if [ -z "${SKIP_DISK_CHECK:-}" ]; then
    FREE_GB=$(df -BG --output=avail "${SAVE_DIR}" | tail -1 | tr -dc '0-9')
    echo "[train-desc] ${SAVE_DIR} 可用 ${FREE_GB}G"
    if [ "${FREE_GB:-0}" -lt 300 ]; then
      echo "[train-desc] 中止:少于 300G。2 个 ckpt(~200G)+ tokenized 缓存放不下。" >&2
      echo "             清盘,或 yaml 里设 save_only_model: true(省 ~72G/ckpt,代价是不可续训)," >&2
      echo "             或 SKIP_DISK_CHECK=1 强行开跑。" >&2
      exit 1
    fi
  fi

  LOG="${LOG:-logs/train_9b_grok_desc_0730_v1.log}"
  echo "[train-desc] GPUS=${CUDA_VISIBLE_DEVICES} -> ${LOG}"
  echo "[train-desc] 首次启动会先 tokenize 3.31M 行(实测外推 ~1.8h)才到第 1 步,别以为它卡住了。"
  exec llamafactory-cli train "${YAML}" "$@" >> "${LOG}" 2>&1
  ;;
esac
