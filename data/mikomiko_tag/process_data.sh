#!/usr/bin/env bash
set -euo pipefail

# 2w 试水：抽 20000 条 -> 16000 train / 4000 test，再下载图片到 img/pornstar
# 注意：--limit 只在 --plan 生效，所以必须 plan+download 一起跑（单独 --download 读不到候选文件）。
source /workspace1/zhijun/LlamaFactory/.venv/bin/activate
python /workspace1/zhijun/LlamaFactory/data/mikomiko_tag/mikomiko_dataset_builder.py \
  --plan --download 
