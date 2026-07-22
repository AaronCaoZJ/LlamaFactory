#!/usr/bin/env bash
# TRAIN — mikomiko 图 -> tag,Qwen3.5-2B 全参 SFT,4 卡 ZeRO-0。
#
#   bash train_tag_2b.sh                              # 默认 mix(124w + 0716),即 full_v1
#   DATASET=mikomiko_tag_train bash train_tag_2b.sh   # 只训 124w,即 full_v0 那一版
#   GPU=0,1,2,3 EPOCHS=1.0 bash train_tag_2b.sh
#   bash train_tag_2b.sh learning_rate=1e-5           # 任何 yaml key 都能当 key=value 追加
#
# 数据来自 data/mikomiko_tag/process_data.sh:
#   mikomiko_tag_train      = jsonl/cleaned/train.jsonl        (CSV 124w 交付)
#   mikomiko_tag_mix_train  = jsonl_0716/mix/mix_train.jsonl   (124w 与 0716 轮转交错)
#
# 1 epoch ~= 17.3k 步 ~= 8h(4x H100)。Ctrl-C 停;重跑会从 output_dir 续。
# 改了 dataset / template / cutoff_len 必须删掉 yaml 里的 tokenized_path 目录 —— 缓存按原文本
# 键、不校验 dataset 名,命中旧 token 是静默的。
#
# Env: GPU (4,5,6,7) | DATASET (mikomiko_tag_mix_train) | EPOCHS (2.0) | BASE_MODEL
set -euo pipefail

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

source "${LF_VENV}/bin/activate"
export DISABLE_VERSION_CHECK=1     # transformers 5.6.1 > LF 硬编码上限 5.6.0;Qwen3.5 需要新版

# Qwen3.5 的 GDN 反向内核在 Hopper 上走 tilelang(JIT),需要能用的 g++;env_setup.sh 会按机器
# 建 .cc-shim 垫片。只有垫片真能编译才前置,免得换机器后悬空的垫片挡住系统里好用的编译器。
_SHIM="${LF_ROOT}/.cc-shim"
if [ -x "${_SHIM}/g++" ] && echo 'int main(){return 0;}' | "${_SHIM}/g++" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}" CC="${_SHIM}/gcc" CXX="${_SHIM}/g++" CUDAHOSTCXX="${_SHIM}/g++"
fi

cd "${LF_ROOT}"

BASE_MODEL="${BASE_MODEL:-${MODELS_DIR}/Qwen3.5-2B}"
DATASET="${DATASET:-mikomiko_tag_mix_train}"
GPU="${GPU:-4,5,6,7}"

echo "[train-tag] GPU=${GPU}  dataset=${DATASET}  base=${BASE_MODEL}"
exec env CUDA_VISIBLE_DEVICES="${GPU}" \
  llamafactory-cli train examples/train_full/qwen3_5_2b_mikomiko_tag.yaml \
    model_name_or_path="${BASE_MODEL}" \
    dataset="${DATASET}" \
    num_train_epochs="${EPOCHS:-2.0}" \
    overwrite_output_dir=false "$@"
