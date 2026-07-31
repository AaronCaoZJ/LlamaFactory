#!/usr/bin/env bash
# DIAG 第二轮 —— 判定第一轮里那个「先跑的慢、后跑的快」到底是不是预热效应。
#
# 第一轮按执行顺序拿到的精确速率:
#   1st bs6_dw16  5.45 s/it   8.8 samp/s
#   2nd bs8_dw16  4.69 s/it  13.6
#   3rd bs12_dw16 13.98 s/it  6.9
#   4th bs6_dw2   2.15 s/it  22.3
#   5th bs6_dw32  2.24 s/it  21.4
# dw=2 和 dw=32 一样快、夹中间的 dw=16 最慢,这在物理上讲不通 —— 按顺序排却很整齐。
# 图片 I/O 已排除(冷读 3,509 img/s / 572 MiB/s,训练只要 ~22 img/s)。
# 所以本轮把三档 dw=16 原样重跑:若现在都变快,就与 worker 数无关,是缓存/JIT 预热;
# 若 bs12 依旧最慢,那它是显存逼近 95% 时分配器抖动,与预热无关。
set -uo pipefail
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"
cd "${LF_ROOT}"; source "${LF_VENV}/bin/activate"
export DISABLE_VERSION_CHECK=1 WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
_S="${LF_ROOT}/.cc-shim"
if [ -x "$_S/g++" ] && echo 'int main(){return 0;}' | "$_S/g++" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="$_S:$PATH" CC="$_S/gcc" CXX="$_S/g++" CUDAHOSTCXX="$_S/g++"
fi
OUT="${LF_ROOT}/logs/diag_speed_0730"; SAVE_DIR="${LF_ROOT}/saves/qwen3.5-9b/mikomiko"
mkdir -p "$OUT"

for bs in 6 8 12 6; do
  tag="rerun2_bs${bs}_$(date +%H%M%S)"
  for _ in $(seq 60); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${U:-99999}" -lt 3000 ] && break; sleep 5
  done
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 4; done ) > "$OUT/$tag.mem" 2>/dev/null &
  MP=$!
  # 注意 '"no"' 的内层引号:裸的 no 会被 OmegaConf 解析成布尔 False,
  # transformers 随即报 "False is not a valid IntervalStrategy"。
  llamafactory-cli train examples/train_full/qwen3_5_9b_mikomiko_grok_desc_0730.yaml \
    dataset=mikomiko_desc_0730_smoke \
    tokenized_path="${SAVE_DIR}/tokenized_smoke_0730" \
    output_dir="${SAVE_DIR}/diag_${tag}" overwrite_output_dir=true \
    per_device_train_batch_size="$bs" gradient_accumulation_steps=1 \
    dataloader_num_workers=16 max_steps=15 \
    save_strategy='"no"' eval_strategy='"no"' logging_steps=1 \
    plot_loss=false report_to=none num_train_epochs=1.0 > "$OUT/$tag.log" 2>&1
  rc=$?
  kill $MP 2>/dev/null; wait $MP 2>/dev/null; rm -rf "${SAVE_DIR}/diag_${tag}"
  r=$(tr '\r' '\n' < "$OUT/$tag.log" | grep -oE "1[0-5]/15 \[[0-9:]+<[0-9:]+, +[0-9.]+s/it" \
      | grep -oE "[0-9.]+s/it$" | tr -d 's/it' | tail -5 | awk '{s+=$1;n++} END{if(n)printf "%.2f", s/n}')
  pk=$(sort -n "$OUT/$tag.mem" 2>/dev/null | tail -1)
  if [ -n "$r" ]; then
    echo "[$(date +%H:%M:%S)] bs=$bs  ${r} s/it  $(awk -v b=$bs -v r=$r 'BEGIN{printf "%.1f", b*8/r}') samples/s  峰值 ${pk} MiB"
  else
    grep -qi "out of memory" "$OUT/$tag.log" && echo "[$(date +%H:%M:%S)] bs=$bs  OOM  峰值 ${pk} MiB" \
      || echo "[$(date +%H:%M:%S)] bs=$bs  FAILED rc=$rc  见 $OUT/$tag.log"
  fi
done
echo RERUN2_DONE
