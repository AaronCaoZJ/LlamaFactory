# 主 agent 独立复核：temperature=0 下的数值抖动与决策翻转

复核动机：Agent A 报 "greedy 不可复现，top1 logprob 抖动 ±0.13 nats、margin 极差 1.25"。
这是本次核心结论之一，独立验证。

方法：`verify_determinism.py`（定点重复）、`jitter_at_boundary.py`（先扫 50 条找最低 margin
样本，再在其上重复 25 次）。全部走**生产 server**（未关 prefix caching / CUDA graph），
串行请求，显式 `temperature: 0`，数据集 `ood_sample/v3/rollout_lite.json`。

margin 定义：一次生成里**最不确定的那个决策步**上 top1-top2 的 logprob 差（跳过 EOS）。

## 结果总表（每模型：50 条扫描 + 最低 margin 的 3 条各重复 25 次）

| 模型 | 端口 | margin 中位 | margin 最小 | margin<0.5 危险带 | 重复 25 次 margin std | 输出是否翻转 |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 8108 | 8.750 | **0.125** | 1/50 | 0.125 | **是** 24xMV_FWD + 1xMV_DOWN |
| Qwen3.5-2B | 8102 | 9.375 | 0.375 | 1/50 | 0.196 | **是** 24xMV_DOWN + 1xMV_FWD |
| Qwen3.5-9B | 8109 | 12.625 | 0.625 | 0/50 | 0.094 | 否 |
| gemma4-E4B | 8104 | 12.500 | 0.500 | 1/50 | 0.077 | 否 |
| InternVL3.5-1B | 8201 | 12.250 | 0.500 | 1/50 | 0.034 | 否 |
| InternVL3.5-2B | 8202 | 11.250 | 0.875 | 0/50 | 0.060 | 否 |

## 结论

1. **非确定性真实存在，但幅度强烈依赖置信度本身。**
   高置信样本（8109 idx24，margin≈13）重复 20 次：tok0 logprob 有 8 个不同值，极差仅
   `3e-6` nats —— 完全无害。
   模糊样本（8108 idx24 的方向 token，`_DOWN` .615 vs `_FWD` .291）重复 30 次：
   30/30 全不同，logprob 从 **-0.642 摆到 -0.385**，即 P(_DOWN) 在 **0.526~0.681** 间晃，
   绝对概率摆幅 **0.155**。
   → 抖动恰好在最需要稳定的地方最大。Agent A 报的 ±0.13 是模糊样本的量级，复核成立。

2. **抖动幅度按家族分层，差 4~6 倍**：Qwen3.5-0.8B/2B (std 0.13~0.22) >> Qwen3.5-9B ~
   gemma4-E4B (0.08~0.09) > InternVL3.5 (0.03~0.06)。InternVL 在多条样本上是**逐位可复现**
   （std=0.0000）。
   推测机制：Qwen3.5 是 GDN(Gated DeltaNet) 混合线性注意力，chunked scan kernel 规约顺序
   不固定；Agent A 实测 `VLLM_BATCH_INVARIANT=1` 对 Qwen3.5 直接报 `not supported for
   GDN_ATTN`，与此一致。InternVL3.5 是标准 softmax attention 稠密模型，抖动仅剩 bf16 ulp。

3. **只有两个小 Qwen 模型真的翻转了输出**，且只发生在全 50 条里 margin 最低的那 1 条上，
   频率 1/25。量化：每条 50 步 rollout 约 1 个危险步，该步约 4% 概率翻转
   → **约 4% 的 rollout 会出现 1 次动作翻转**。单步影响小，但闭环里一次错误动作可能级联。

4. **诚实的定量结论**：数值抖动是真的，但**不足以单独解释"每天性能都不一样"**。
   Agent A 测到的"不传 temperature -> 落到 temperature=1.0 纯采样"造成的
   **6.0pt 准确率摆动 / 7-of-50 条输出改变**要大一个量级。日间大幅波动更可能来自
   采样参数没固定 + 物理侧（光照/标定/物体位姿），而非 kernel 抖动。

## 复跑方式

`python3 <scratchpad>/jitter_at_boundary.py <port> <lora>`
