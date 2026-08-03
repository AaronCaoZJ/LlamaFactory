#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS="${GPUS:-0,1}"

echo "GPUS=${GPUS}"

echo "================================================================================"
echo "[queue] 0802 1/3 full long_collect"
echo "================================================================================"
GPUS="${GPUS}" bash "${SCRIPT_DIR}/train_zwz_0802_long_collect.sh"

echo "================================================================================"
echo "[queue] 0802 2/3 no task_3"
echo "================================================================================"
GPUS="${GPUS}" bash "${SCRIPT_DIR}/train_zwz_0802_long_collect_no_task3.sh"

echo "================================================================================"
echo "[queue] 0802 3/3 no task_3 and no task_4_1"
echo "================================================================================"
GPUS="${GPUS}" bash "${SCRIPT_DIR}/train_zwz_0802_long_collect_no_task3_task4_1.sh"
