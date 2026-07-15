#!/usr/bin/env bash
# 方案 B —— 两路相机走 video 槽位（<video> 的两"帧"）而不是两张 <image>。
# Qwen 的 patch embed 是 3D conv（temporal_patch_size=2），两帧被融合成一组视觉 token：
# 每样本 64 个而不是 128 个，代价是 agentview / wrist 在每个空间位置被混合。
#
# 与 image 版互为对照实验：数据集 mix_22_27_v3_lite_video 是 mix_22_27_v3_lite 的 re-slot
# 版本（to_video_slot.py），样本 / 顺序 / prompt 完全相同，唯一变量就是模态槽位。
# 对照组：examples/train_lora/qwen3_5_9b/qwen3_5_9b_mix_22_27_v3.yaml
#
#   MODE=overfit   单卡，58 样本，先跑通管线再上正式训练（HANDOFF §3.2）
#   MODE=train     4132 样本，yaml 的 eff_bs=32 按 2 卡算（batch 4 × acc 4 × 2）
#
# 用法：
#   MODE=overfit GPU=4 bash scripts/qwen3_5/train/video_slot_train.sh
#   MODE=train   GPU=4,6 bash scripts/qwen3_5/train/video_slot_train.sh
set -euo pipefail

# ═══ GPU / runtime knobs (edit here) ═══
MODE="${MODE:-train}"                 # overfit | train
GPU="${GPU:-4,6}"                         # MODE=train 时给两张卡，如 GPU=4,6
MASTER_PORT="${MASTER_PORT:-29531}"     # 并发多任务时手工错开，否则 rendezvous 撞车

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${LF_ROOT}}"
VENV_PATH="${LLAMA_FACTORY_VENV:-${LLAMA_FACTORY_ROOT}/.venv}"
CFG_DIR="${LLAMA_FACTORY_ROOT}/examples/train_lora/qwen3_5_9b"

case "${MODE}" in
  overfit) TRAIN_CONFIG="${CFG_DIR}/qwen3_5_9b_overfit_video.yaml" ;;
  train)   TRAIN_CONFIG="${CFG_DIR}/qwen3_5_9b_mix_22_27_v3_video.yaml" ;;
  *)       echo "ERROR: MODE must be 'overfit' or 'train', got '${MODE}'" >&2; exit 1 ;;
esac
[ -f "${TRAIN_CONFIG}" ] || { echo "ERROR: missing config ${TRAIN_CONFIG}" >&2; exit 1; }

export DISABLE_VERSION_CHECK=1  # transformers 5.6.1 > LF 硬编码上限 5.6.0；绕过版本闸
export FORCE_TORCHRUN=1         # 单卡 + deepspeed 时必需（LF 单 GPU 默认不走 torchrun）
export MASTER_PORT

# 不设的话 torch/OpenMP 按核数(384)开线程，每个 preprocessing worker 都拉 128~220 个：
# "Running tokenizer" 那步（多模态数据集其实是在解码+归一化每条样本的图像）会从 ~10s
# 退化到 5+ 分钟，整机 load 被推到 600+。这台机器是共享的，别把别人一起拖下水。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# gcc-11 垫片（Qwen3.5 GDN 反向 tilelang JIT 需 cc1plus；系统 gcc-12 缺）
_SHIM="${LLAMA_FACTORY_ROOT}/.cc-shim"
if [ -x "${_SHIM}/gcc" ] && echo 'int main(){return 0;}' | "${_SHIM}/gcc" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}"
fi

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv not found at ${VENV_PATH}." >&2; exit 1
fi
source "${VENV_PATH}/bin/activate"
cd "${LLAMA_FACTORY_ROOT}"

echo "[video-slot] MODE=${MODE}  GPU=${GPU}  port=${MASTER_PORT}"
echo "[video-slot] config=${TRAIN_CONFIG}"
exec env CUDA_VISIBLE_DEVICES="${GPU}" llamafactory-cli train "${TRAIN_CONFIG}"
