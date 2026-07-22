#!/usr/bin/env bash
# Extract RoboVQA clip archives in-place so JSON paths like clips/<id>.mp4 resolve.
set -euo pipefail

RAW_DIR="${RAW_DIR:-/storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA}"
CLIPS_DIR="${CLIPS_DIR:-${RAW_DIR}/clips}"

if [ ! -d "${RAW_DIR}" ]; then
  echo "ERROR: RAW_DIR does not exist: ${RAW_DIR}" >&2
  exit 1
fi

if [ ! -d "${CLIPS_DIR}" ]; then
  echo "ERROR: CLIPS_DIR does not exist: ${CLIPS_DIR}" >&2
  exit 1
fi

mapfile -t archives < <(find "${CLIPS_DIR}" -maxdepth 1 -type f -name 'clips_part_*.tar.gz' | sort)
if [ "${#archives[@]}" -eq 0 ]; then
  echo "ERROR: no clips_part_*.tar.gz archives found under ${CLIPS_DIR}" >&2
  exit 1
fi

before_count="$(find "${CLIPS_DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
echo "[extract] raw dir    : ${RAW_DIR}"
echo "[extract] clips dir  : ${CLIPS_DIR}"
echo "[extract] archives   : ${#archives[@]}"
echo "[extract] mp4 before : ${before_count}"

i=0
for archive in "${archives[@]}"; do
  i=$((i + 1))
  printf '[extract] (%03d/%03d) %s\n' "${i}" "${#archives[@]}" "$(basename "${archive}")"
  tar --no-same-owner --no-same-permissions --skip-old-files -xzf "${archive}" -C "${CLIPS_DIR}"
done

after_count="$(find "${CLIPS_DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
echo "[extract] mp4 after  : ${after_count}"

sample="${CLIPS_DIR}/4857681842253668822.mp4"
if [ -f "${sample}" ]; then
  echo "[extract] sample OK  : ${sample}"
else
  echo "WARN: expected sample clip is still missing: ${sample}" >&2
fi
