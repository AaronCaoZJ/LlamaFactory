#!/usr/bin/env bash
# 串行队列：等 2B 那个 tmux session 结束，再跑 0.8B。
# 放在仓库里而不是 /tmp，因为 tmux 会长时间执行它，脚本文件必须一直在。
set -uo pipefail
cd /workspace1/zhijun/LlamaFactory

echo "[queue] 等待 qwen2b_40ep 结束…"
while tmux has-session -t qwen2b_40ep 2>/dev/null; do sleep 60; done
echo "[queue] 2B 已结束，开始 0.8B  ($(date '+%F %T'))"

# 旧的 30ep 结果归档而不是覆盖 —— 被覆盖的目录会变成新旧 checkpoint 混合且看起来正常
SRC=saves/qwen3.5-0.8b/robot/mix_22-06_fk-pp/02_exchange_token
if [ -d "$SRC" ]; then
  rm -rf "${SRC}_30ep"; mv "$SRC" "${SRC}_30ep"
  echo "[queue] 已归档 0.8B 旧结果 -> $(basename ${SRC}_30ep)"
fi

source .venv/bin/activate
DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4 \
  llamafactory-cli train examples/train_lora/qwen3_5_0_8b/mix_22-06_fk-pp/qwen3_5_0_8b_02_exchange_token.yaml \
  2>&1 | tee /workspace1/zhijun/LlamaFactory/train_qwen0_8b_fkpp_40ep.log
echo "[queue] 0.8B 结束 ($(date '+%F %T'))，退出码 ${PIPESTATUS[0]}"
