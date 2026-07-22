#!/usr/bin/env bash
# End-to-end helper: extract clips, convert RoboVQA, register dataset_info, write mixed yaml.
set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TOOL_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RAW_DIR="${RAW_DIR:-/storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA}"
KIND="${KIND:-reasoning}"                 # reasoning | understanding | both
NAME_SUFFIX="${NAME_SUFFIX:-}"            # e.g. 20k
MAX_SAMPLES="${MAX_SAMPLES:-}"            # e.g. 20000
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"         # set 1 after clips are extracted
VALIDATE_MEDIA="${VALIDATE_MEDIA:-1}"     # set 0 for faster conversion
MAKE_YAML="${MAKE_YAML:-1}"

if [ -z "${NAME_SUFFIX}" ] && [ -n "${MAX_SAMPLES}" ]; then
  NAME_SUFFIX="${MAX_SAMPLES}"
fi

if [ "${SKIP_EXTRACT}" != "1" ]; then
  RAW_DIR="${RAW_DIR}" bash "${TOOL_DIR}/extract_clips.sh"
else
  echo "[prepare] skip extract because SKIP_EXTRACT=1"
fi

convert_args=(--raw-dir "${RAW_DIR}" --kind "${KIND}" --name-suffix "${NAME_SUFFIX}")
if [ -n "${MAX_SAMPLES}" ]; then
  convert_args+=(--max-samples "${MAX_SAMPLES}")
fi
if [ "${VALIDATE_MEDIA}" = "1" ]; then
  convert_args+=(--validate-media)
fi

python "${TOOL_DIR}/convert_to_llamafactory.py" "${convert_args[@]}"

register_args=(--only "${KIND}" --name-suffix "${NAME_SUFFIX}" --overwrite)
python "${TOOL_DIR}/register_dataset_info.py" "${register_args[@]}"

if [ "${MAKE_YAML}" = "1" ]; then
  robovqa_dataset="robovqa_reasoning_lf"
  if [ "${KIND}" = "understanding" ]; then
    robovqa_dataset="robovqa_understanding_lf"
  fi
  if [ -n "${NAME_SUFFIX}" ]; then
    robovqa_dataset="${robovqa_dataset}_${NAME_SUFFIX}"
  fi
  python "${TOOL_DIR}/make_mixed_train_yaml.py" \
    --robovqa-dataset "${robovqa_dataset}" \
    --media-dir "${RAW_DIR}"
fi
