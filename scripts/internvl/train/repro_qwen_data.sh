#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

VENV_PATH="${VENV_PATH:-/workspace1/zhijun/LlamaFactory/.venv}"
CONFIG="${CONFIG:-examples/train_lora/internvl/internvl3_5_2b_mix_22-06_fk-pp_02_exchange_token.yaml}"
if [ -z "${MODEL_PATH:-}" ]; then
  for _candidate in \
    /workspace1/zechen/hf_download/InternVL3_5-2B-HF \
    /workspace1/zhijun/hf_download/models/InternVL3_5-2B-HF \
    /workspace1/zechen/hf_download/InternVL3_5-2B; do
    if [ -e "${_candidate}" ]; then
      MODEL_PATH="${_candidate}"
      break
    fi
  done
  MODEL_PATH="${MODEL_PATH:-/workspace1/zechen/hf_download/InternVL3_5-2B-HF}"
fi
GPUS="${GPUS:-4,6}"

cd "${LF_ROOT}"

export PYTHONPATH="${LF_ROOT}/src:${PYTHONPATH:-}"

_SHIM="/workspace1/zhijun/LlamaFactory/.cc-shim"
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
