#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SH="${SCRIPT_DIR}/train_zwz_2b.sh"

GPUS="${GPUS:-4,5}"

echo "GPUS=${GPUS}"
echo "TRAIN_SH=${TRAIN_SH}"

echo "================================================================================"
echo "[train] 2B 0802 exchange_token + zwz_0723 + zwz_0724 + zwz_0802_long_collect 40ep"
echo "================================================================================"
GPUS="${GPUS}" bash "${TRAIN_SH}" \
  "examples/train_lora/qwen3_5_2b/mix_22-06_fk-pp/qwen3_5_2b_02_exchange_token_plus_zwz_0723_0724_0802_long_collect_40ep.yaml"
