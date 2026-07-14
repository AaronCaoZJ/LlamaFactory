#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export CONFIG="${CONFIG:-examples/train_lora/internvl/internvl3_5_1b_mix_22-06_fk-pp_02_exchange_token.yaml}"
export MODEL_PATH="${MODEL_PATH:-/workspace1/zechen/hf_download/InternVL3_5-1B-HF}"

exec bash "${SCRIPT_DIR}/repro_qwen_data.sh"
