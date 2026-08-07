#!/usr/bin/env bash
set -euo pipefail
# RoboLab 12 任务混训 (v3 prompt) on Qwen3.5-9B LoRA, 30 epochs
#
# 数据集已由 AgentRobot/scripts/robolab/prepare_lf_dataset.sh 转换并注册,
# 这里不再重复生成 —— 重跑转换会覆盖 rollout_lite.json,而训练进行中被覆盖会
# 让 fingerprint 与实际数据不一致。要重建数据集就先停训练。
#
#   数据: data/agentrobot/MVTOKEN/robolab_0805_12task/rollout_lite.json
#         7559 samples / 130 集 / 12 任务 (主任务 OrangeBlockInBowl 20 集)
#   注册: dataset_info.json -> robolab_0805_12task

# ═══ GPU / runtime knobs (edit here) ═══
GPU="${GPU:-1,3}"

source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"
TRAIN_CONFIG="${LLAMA_FACTORY_ROOT}/examples/train_lora/qwen3_5_9b/qwen3_5_9b_robolab_0805_12task.yaml"
DATASET_JSON="${LLAMA_FACTORY_ROOT}/data/agentrobot/MVTOKEN/robolab_0805_12task/rollout_lite.json"

export DISABLE_VERSION_CHECK=1  # transformers 5.6.1 > LF 硬编码上限 5.6.0；Qwen3.5 需新版

# Qwen3.5 的 GDN 反向用 tilelang JIT,nvcc 需要 gcc 宿主编译器；系统 gcc-12 缺 cc1plus。
_SHIM="${LLAMA_FACTORY_ROOT}/.cc-shim"
if [ -x "${_SHIM}/gcc" ] && echo 'int main(){return 0;}' | "${_SHIM}/gcc" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}"
fi

# ── 前置检查 ─────────────────────────────────────────────────────────────────
[ -f "${VENV_PATH}/bin/activate" ] || { echo "ERROR: venv 不存在 ${VENV_PATH}，先跑 env_setup.sh" >&2; exit 1; }
[ -f "${TRAIN_CONFIG}" ]  || { echo "ERROR: 训练配置不存在 ${TRAIN_CONFIG}" >&2; exit 1; }
[ -f "${DATASET_JSON}" ]  || { echo "ERROR: 数据集不存在 ${DATASET_JSON}
  先跑: bash /workspace1/zhijun/AgentRobot/scripts/robolab/prepare_lf_dataset.sh" >&2; exit 1; }

source "${VENV_PATH}/bin/activate"

# 样本数与图片可达性 —— 图片是绝对路径,指向 AgentRobot/rollouts,
# 那边被移动或清理过的话训练会在中途才报错,不如现在就查。
python - <<PY
import json, os, sys
d = json.load(open("${DATASET_JSON}"))
missing = [p for s in d[:200] for p in s["images"] if not os.path.exists(p)]
print(f"数据集: {len(d)} samples")
if missing:
    print(f"ERROR: 抽查前 200 条,有 {len(missing)} 张图片不存在,例如 {missing[0]}", file=sys.stderr)
    sys.exit(1)
print("图片抽查通过 (前 200 条)")
PY

echo "GPU=${GPU}  config=${TRAIN_CONFIG##*/}"
cd "${LLAMA_FACTORY_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${GPU}" \
  llamafactory-cli train "${TRAIN_CONFIG}"
