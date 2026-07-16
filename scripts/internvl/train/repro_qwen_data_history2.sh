#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG="${CONFIG:-examples/train_lora/internvl/internvl3_5_2b_mix_22-06_fk-pp_02_exchange_token_history2.yaml}"
export MODEL_PATH="${MODEL_PATH:-/workspace1/zechen/hf_download/InternVL3_5-2B-HF}"

exec bash "${SCRIPT_DIR}/repro_qwen_data.sh"
