#!/usr/bin/env bash
set -euo pipefail

# 2w 试水：抽 20000 条 -> 16000 train / 4000 test，再下载图片到 img/pornstar
# 注意：--limit 只在 --plan 生效，所以必须 plan+download 一起跑（单独 --download 读不到候选文件）。
# resolve machine paths: locate & source scripts/workspace_dir.sh (sets LF_ROOT, MODELS_DIR, LF_VENV, VLLM_VENV, AGENTROBOT_ROOT, HF_HOME)
_wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
source "$_wsd/scripts/workspace_dir.sh"
source "${LF_VENV}/bin/activate"
python "${LF_ROOT}/data/mikomiko_tag/dataset_builder.py" \
  --plan --download
