#!/usr/bin/env bash
# Find the largest per_device_train_batch_size that fits, on the SAME rank count the real
# run will use. Rank count matters: under ZeRO-3 the param/grad/optimizer shard is
# (model state / world_size), so a batch that fits on 4 ranks has *more* room on 8.
# Probing on 4 and then training on 8 would leave capacity on the table; probing on 8 and
# training on 8 is the only apples-to-apples read.
#
# Runs the 2,400-row smoke set for a few steps and reports peak reserved memory per rank.
# Peak lands on the batch holding the longest sequences, so it needs enough steps to see a
# few batches -- STEPS=15 is the floor, not a formality.
#
#   bash scripts/qwen3_5/mikomiko_desc/probe_batch.sh 8 16 24
set -uo pipefail

source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"
cd "$LF_ROOT"
source "${LF_VENV}/bin/activate"

STEPS="${STEPS:-15}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NGPU=$(awk -F',' '{print NF}' <<< "$GPUS")

export DISABLE_VERSION_CHECK=1
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES="$GPUS"
mkdir -p logs

for BS in "$@"; do
  LOG="logs/probe_bs${BS}_${NGPU}gpu.log"
  echo "=== probing per_device=$BS on $NGPU GPUs -> $LOG ==="
  # sample memory while it runs; nvidia-smi reports the allocator's reservation, which is
  # what actually has to fit, not just live tensors
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 4; done ) > "logs/.mem_bs${BS}" 2>/dev/null &
  MEMPID=$!

  llamafactory-cli train examples/train_full/qwen3_5_9b_mikomiko_grok_desc.yaml \
    dataset=mikomiko_desc_0721_smoke \
    tokenized_path="saves/qwen3.5-9b/mikomiko/tokenized_smoke_0721" \
    output_dir="saves/qwen3.5-9b/mikomiko/probe_bs${BS}" \
    per_device_train_batch_size="$BS" \
    gradient_accumulation_steps=1 \
    max_steps="$STEPS" \
    save_strategy='"no"' \
    eval_strategy='"no"' \
    logging_steps=1 \
    plot_loss=false \
    report_to=none \
    num_train_epochs=1.0 > "$LOG" 2>&1
  RC=$?
  kill $MEMPID 2>/dev/null; wait $MEMPID 2>/dev/null

  PEAK=$(sort -n "logs/.mem_bs${BS}" 2>/dev/null | tail -1)
  rm -f "logs/.mem_bs${BS}"
  if [ $RC -ne 0 ]; then
    if grep -qi "out of memory" "$LOG"; then
      echo "  bs=$BS -> OOM (peak seen ${PEAK:-?} MiB)"
    else
      echo "  bs=$BS -> FAILED rc=$RC (not OOM; see $LOG)"
    fi
  else
    SPS=$(grep -oE "'train_samples_per_second': '[0-9.]+'" "$LOG" | tail -1 | grep -oE "[0-9.]+")
    echo "  bs=$BS -> OK  peak=${PEAK:-?} MiB / 143771 MiB  samples/s(incl warmup)=${SPS:-?}"
  fi
done
