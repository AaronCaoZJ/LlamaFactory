#!/usr/bin/env bash
# 4-GPU smoke test for the 20260721 description dataset.
#
# Answers only "does this config train at all": data loads, images decode, the template
# applies, loss is finite and moving, and what one step costs. It writes NO checkpoint
# (save_strategy=no) and runs on a 2,400-row stratified subset with its OWN tokenized
# cache, so it never builds the 1.34M-row cache the real run needs.
#
# Memory read off 4 GPUs is a LOWER bound for 8: under ZeRO-3 more ranks means smaller
# per-rank param/optimizer shards, so anything that fits here fits there.
#
#   bash scripts/qwen3_5/mikomiko_desc/smoke_4gpu.sh          # per_device 2, 30 steps
#   BS=8 STEPS=12 bash scripts/qwen3_5/mikomiko_desc/smoke_4gpu.sh
set -euo pipefail

source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"
cd "$LF_ROOT"
source "${LF_VENV}/bin/activate"

BS="${BS:-2}"
STEPS="${STEPS:-30}"
GPUS="${GPUS:-0,1,2,3}"
TAG="bs${BS}"

export DISABLE_VERSION_CHECK=1
export WANDB_DISABLED=true          # a smoke run should not litter the wandb project
export CUDA_VISIBLE_DEVICES="$GPUS"

# NOTE the '"no"' quoting below. CLI overrides go through OmegaConf.from_cli, which
# YAML-parses each value -- bare `no` becomes the boolean False, and transformers then
# dies with "False is not a valid IntervalStrategy". Any yes/no/on/off/y/n value passed
# as an override needs the inner quotes.
llamafactory-cli train examples/train_full/qwen3_5_9b_mikomiko_grok_desc.yaml \
  dataset=mikomiko_desc_0721_smoke \
  tokenized_path="saves/qwen3.5-9b/mikomiko/tokenized_smoke_0721" \
  output_dir="saves/qwen3.5-9b/mikomiko/smoke_${TAG}" \
  per_device_train_batch_size="$BS" \
  gradient_accumulation_steps=1 \
  max_steps="$STEPS" \
  save_strategy='"no"' \
  eval_strategy='"no"' \
  logging_steps=1 \
  plot_loss=false \
  report_to=none \
  num_train_epochs=1.0
