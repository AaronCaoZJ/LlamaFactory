#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

VENV_PATH="${VENV_PATH:-/storage/wenzheng/showrobot/LlamaFactory/.venv}"

CONFIG="${1:-${CONFIG:-examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_03_just_mix_zwz_new_prompt_add_horizon_flip.yaml}}"
MODEL_PATH="${MODEL_PATH:-/storage/wenzheng/showrobot/hf_download/models/Qwen3.5-9B}"

GPUS="${GPUS:-6,7}"
cd "${LF_ROOT}"

export PYTHONPATH="${LF_ROOT}/src:${PYTHONPATH:-}"

_SHIM="/storage/wenzheng/showrobot/LlamaFactory/.cc-shim"
if [ -x "${_SHIM}/gcc" ] && echo 'int main(){return 0;}' | "${_SHIM}/gcc" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}"
fi

if [ -f "${VENV_PATH}/bin/activate" ]; then
  source "${VENV_PATH}/bin/activate"
else
  echo "WARN: venv not found at ${VENV_PATH}; using current shell environment." >&2
fi

if [ ! -e "${MODEL_PATH}" ]; then
  echo "WARN: MODEL_PATH=${MODEL_PATH} does not exist. Override with MODEL_PATH=/path/to/InternVL3_5-2B-HF if needed." >&2
fi

python -c "import llamafactory; print('llamafactory:', llamafactory.__file__)"
python - "${MODEL_PATH}" <<'PY'
import sys
from transformers import AutoProcessor

model_path = sys.argv[1]
print("model_path:", model_path)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print("processor:", processor.__class__.__name__)
print("image_processor:", getattr(processor, "image_processor", None).__class__.__name__)
print("crop_to_patches:", getattr(processor, "crop_to_patches", None))
PY

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES="${GPUS}" \
  llamafactory-cli train "${CONFIG}" model_name_or_path="${MODEL_PATH}"
