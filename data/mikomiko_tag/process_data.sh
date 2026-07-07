#!/usr/bin/env bash
set -euo pipefail

# 2w 试水：抽 20000 条 -> 16000 train / 4000 test，再下载图片到 img/pornstar
# 注意：--limit 只在 --plan 生效，所以必须 plan+download 一起跑（单独 --download 读不到候选文件）。
# machine-agnostic paths (see scripts/workspace_dir.sh)
_d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_d" != "/" ] && [ ! -f "$_d/scripts/workspace_dir.sh" ]; do _d="$(dirname "$_d")"; done
source "$_d/scripts/workspace_dir.sh"
source "${LF_VENV}/bin/activate"
python "${LF_ROOT}/data/mikomiko_tag/dataset_builder.py" \
  --plan --download
