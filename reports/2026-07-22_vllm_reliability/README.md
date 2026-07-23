# vLLM 推理可靠性 + 六模型置信度分布 —— 完整报告

日期：2026-07-22 ｜ 数据集：`data/agentrobot/ood_sample`（rollout_052，真机 Franka，"pick up the
white cup and place it on the green coaster"，50 步，任务/物体均不在训练集内）

配套产物（本目录）：`data/*.jsonl` 原始逐样本概率、`data/summary.json`、`figs/*.png`、
`scripts/*.py` 可复跑脚本、以及四份分报告 `reportA/B/C/M_*.md`。

---

## 0. 结论速览

| # | 结论 | 量级 |
|---|---|---|
| 1 | 三个启动脚本的 `--override-generation-config` 被注释掉，且 Qwen3.5 无 `generation_config.json` → 不传 temperature 的客户端会掉到 **temperature=1.0 纯采样** | 准确率摆动 **6.0pt**，7/50 条输出改变 |
| 2 | 真机链路**确实**显式发了 `temperature: 0.0`，所以上面那条**不直接坑真机**，坑的是旁路客户端 | — |
| 3 | 即使 temperature=0，vLLM 也**不是逐位可复现**的，且抖动在模糊样本上最大 | Qwen3.5 小模型 P 摆幅 **0.155**，约 **4% 的 rollout** 出现 1 次动作翻转 |
| 4 | 抖动是 **Qwen3.5 架构特有**（GDN 线性注意力），InternVL3.5 近乎逐位可复现 | 家族间差 **4~6 倍** |
| 5 | 不确定性几乎**全在"往哪走"，不在"做不做事件"** | 方向熵是事件判别熵的 **4.8~11.5 倍** |
| 6 | 远离事件步时**没有**随机误触发；所有误触发都是**抢跑一步** | 远端 P(事件) 最大仅 **8.9e-05** |
| 7 | 高置信度**不代表正确**：P≥0.99 的预测里仍有 8~15% 是错的 | 见 §2.4 |
| 8 | 步 28 六个模型**一致高置信答错** → 大概率是标注问题，不是模型问题 | 0.9875~1.0000 |

---

## 1. 任务一：temperature 与推理可靠性

### 1.1 采样参数优先级（实测确认）

```
客户端请求 > --override-generation-config > generation_config.json > vLLM 内置默认
```

源码依据 `to_sampling_params()`：`if (temperature := self.temperature) is None:` 才去取
`default_sampling_params`。实测：给已有 override(temp=0) 的 :8109 发 `temperature=2.0`，
30 次出 **5 种不同输出** —— 客户端赢。

### 1.2 现状不一致

| 启动脚本 | override | base model 的 `generation_config.json` |
|---|---|---|
| `qwen3_5/eval/start_vllm_server_9.sh:112` | ✅ temp=0 | **不存在** |
| `gemma4/eval/start_vllm_server.sh:89` | ✅ temp=0 | do_sample=true, temp=1.0, top_k=64 |
| `internvl/eval/start_vllm_server.sh:143` | ✅ temp=0 | 有文件但无采样字段 |
| `qwen3_5/eval/start_vllm_server_0_8.sh:82` | ❌ **注释掉** | **不存在** |
| `qwen3_5/eval/start_vllm_server_2.sh:84` | ❌ **注释掉** | **不存在** |
| `qwen3_5/eval/start_vllm_server.sh:84`（27B） | ❌ **注释掉** | temp=0.6, top_k=20, top_p=0.95 |

"注释掉 override" + "无 generation_config.json" 的组合最危险：落到 vLLM 内置
`_DEFAULT_SAMPLING_PARAMS`（temperature=1.0 / top_p=1.0 / top_k=0）＝**纯随机采样**。

**实测**（50 条 × 5 轮，:8108 / :8102）：

| 条件 | 准确率范围 | 输出会变的样本 |
|---|---|---|
| 不传 temperature（落到内置默认） | **80.0 ~ 86.0%（6.0pt）** | 7/50 |
| 显式传 temperature=0 | 84.0 ~ 86.0%（2.0pt） | 1/50 |
| :8109 有 override（两种条件） | 78.0% 恒定 | 0/50 |

### 1.3 真机链路查证

`run_real.py:181` + `:508` 显式发 `temperature: 0.0`；三个 vllm 后端 `_is_hosted()` 均为 False，
payload 原样透传。**所以注释掉 override 不直接影响真机部署**，影响的是手工 curl、评测脚本、
旁路工具。两个例外：

- `trapi_chatgpt`（openai dialect）的 0.0 会被 `vlm/gemma_client.py:150-152` 丢弃
- `run.py:203` 读 `vlm_cfg["temperature"]` **无默认值**，与 `run_real.py:508` 的
  `.get("temperature", 0.0)` 行为不对称；4 个真机 yaml 的 `vlm:` 块最好显式写死

`top_p` / `top_k` / `seed` 全链路**从不传**。

### 1.4 temperature=0 下的数值抖动（我独立复核）

每模型：扫 50 条找 margin 最低的样本，再在其上重复 25 次。走**生产 server**、串行、显式 temp=0。
margin = 一次生成里最不确定的那个决策步上 top1−top2 的 logprob 差。

| 模型 | margin 中位 | 最小 margin | margin<0.5 | 重复 25 次 margin std | 输出是否翻转 |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 8.75 | **0.125** | 1/50 | 0.125 | **是** 24×MV_FWD + 1×MV_DOWN |
| Qwen3.5-2B | 9.38 | 0.375 | 1/50 | 0.196 | **是** 24×MV_DOWN + 1×MV_FWD |
| Qwen3.5-9B | 12.63 | 0.625 | 0/50 | 0.094 | 否 |
| gemma4-E4B | 12.50 | 0.500 | 1/50 | 0.077 | 否 |
| InternVL3.5-1B | 12.25 | 0.500 | 1/50 | 0.034 | 否 |
| InternVL3.5-2B | 11.25 | 0.875 | 0/50 | 0.060 | 否 |

三点：

1. **抖动幅度与置信度反向**。高置信样本（margin≈13）重复 20 次极差仅 `3e-6` nats；模糊样本
   （`_DOWN` .615 vs `_FWD` .291）重复 30 次，logprob 从 −0.642 摆到 −0.385，即
   **P 在 0.526~0.681 之间晃，绝对摆幅 0.155**。抖动恰好在最需要稳定的地方最大。
2. **这是 Qwen3.5 架构特有的**。Qwen3.5 用 GDN（Gated DeltaNet）混合线性注意力，chunked scan
   kernel 规约顺序不固定；实测 `VLLM_BATCH_INVARIANT=1` 对 Qwen3.5 直接报
   `not supported for GDN_ATTN`。InternVL3.5 是标准 softmax attention，多条样本上 std=0.0000。
   **prefix caching / CUDA graph / 并发 / seed 四个开关实测全部无效。**
3. **但要诚实**：翻转只发生在 margin 最低的那 1 条上、频率 1/25 → 约 **4% 的 rollout 出现 1 次
   动作翻转**。这不足以单独解释"每天性能不同"，比 §1.2 的 6pt 小一个数量级。日间大幅波动
   更可能来自采样参数没固定 + **物理侧（光照/标定/物体位姿）**。

### 1.5 顺带查出的问题

- `ShowRobot-VLM*.yaml` 的 `qwen27` 指向 `:8101`，**当前无任何进程监听**。若被他人占用会静默
  打到错模型（历史上踩过"端口不独占静默分流"）。
- 端口独占检查其余通过（5 个 server 各 1 个监听进程，无重复）。
- 全部 LoRA 目录 mtime 均早于对应 server 启动时间 → **当前无权重漂移**。

---

## 2. 任务二：六模型置信度分布

50 条 OOD 样本，逐 token top-20 前缀累乘得到完整 action 串概率（首 token 只能区分
MV / GRASP / RELEASE / DONE 四类，六个方向在第二个 token 上）。

### 2.1 总表

| 模型 | acc | 平均 top1 | 平均熵 | ECE | P≥0.99 的样本数 | 其中错误率 |
|---|---|---|---|---|---|---|
| **intern1b** | **0.86** | **0.9890** | **0.0257** | 0.146 | 46 | 8.7% |
| qwen2b | 0.84 | 0.9651 | 0.0958 | **0.125** | 37 | **8.1%** |
| qwen0.8b | 0.80 | 0.9710 | 0.0724 | 0.179 | 40 | 12.5% |
| qwen9b | 0.78 | 0.9799 | 0.0455 | 0.200 | 41 | 9.8% |
| intern2b | 0.78 | 0.9817 | 0.0575 | 0.202 | 41 | 14.6% |
| gemma-e4b | 0.76 | 0.9628 | 0.0814 | 0.203 | 36 | 11.1% |

⚠️ n=50 单条 rollout，acc 的 95% CI 约 **±11pt** —— **模型排序仅供参考**，不要据此下结论说
intern1b 比 qwen9b 好。

### 2.2 核心结论：方向 vs 事件（用户最关心的）

| 模型 | 首 token（4 类事件判别）maxP 均值 / 最小 / <0.9 例数 | 方向（6 类）maxP 均值 / 最小 / <0.9 例数 |
|---|---|---|
| qwen0.8b | 0.9957 / 0.7980 / 1 | 0.9737 / **0.4989** / 2 |
| qwen2b | 0.9936 / 0.7771 / 1 | 0.9699 / 0.5922 / 5 |
| qwen9b | 0.9979 / 0.9455 / 0 | 0.9795 / 0.6223 / 3 |
| intern1b | **0.9995** / 0.9739 / 0 | **0.9883** / 0.5749 / 1 |
| intern2b | 0.9976 / 0.8807 / 1 | 0.9827 / 0.7302 / 3 |
| gemma-e4b | 0.9965 / 0.8345 / 1 | 0.9626 / 0.5757 / **8** |

**方向熵是事件判别熵的 4.8~11.5 倍。** 300 个（样本×模型）对里，首 token <0.9 只有 **4 例**，
方向 <0.9 有 **22 例**。结论明确：**模型对"要不要抓/放/结束"几乎不含糊，含糊的全在"往哪走"。**

### 2.3 误触发风险

距真事件步 ≥2 的 41 条 MV 样本上，六模型的 P(GRASP+RELEASE+DONE) 最大值：

| 模型 | qwen0.8b | qwen9b | intern1b | intern2b | qwen2b | gemma-e4b |
|---|---|---|---|---|---|---|
| max P(事件) | 8.9e-05 | 3.5e-06 | 2.2e-06 | 2.4e-07 | 5.7e-07 | 1.1e-07 |

**没有随机误触发。** 所有 >1% 的事件概率**全部落在真事件的前一步**（i=23/42/48）——是**相位抢跑**：

| 模型 | i23（GRASP 前一步） | i42（RELEASE 前一步） | i48（DONE 前一步） |
|---|---|---|---|
| **qwen9b** | **GRASP (0.990)** | **RELEASE (0.962)** | **DONE (0.946)** |
| intern1b | MV_DOWN | RELEASE (1.000) | DONE (0.974) |
| gemma-e4b | MV_DOWN | RELEASE (1.000) | DONE (0.834) |
| intern2b | MV_DOWN | MV_DOWN (0.119) | DONE (0.999) |
| qwen0.8b | MV_DOWN | RELEASE (1.000) | MV_UP (0.202) |
| **qwen2b** | MV_DOWN | MV_DOWN (0.223) | MV_UP (0.085) |

**qwen9b 风险最高**（三个事件全部抢跑，且是唯一 GRASP 抢跑的——抓早了会撞/抓空）；
**qwen2b 最稳**（从未翻转 argmax）。

### 2.4 校准度：高置信度 ≠ 正确

**这是最该注意的一点。** P≥0.99 的预测里仍有 **8.1%~14.6%** 是错的（见 §2.1 末列）。
再看"答对时"与"答错时"的平均置信度差：

| 模型 | 答对时 | 答错时 | **差值（= 置信度的判别力）** |
|---|---|---|---|
| qwen2b | 0.9840 | 0.8662 | **0.1178** ← 最好 |
| gemma-e4b | 0.9889 | 0.8802 | 0.1087 |
| qwen0.8b | 0.9911 | 0.8907 | 0.1004 |
| intern2b | 0.9940 | 0.9379 | 0.0561 |
| qwen9b | 0.9916 | 0.9384 | 0.0532 |
| **intern1b** | 0.9895 | 0.9859 | **0.0036** ← 几乎无判别力 |

**重要修正**：intern1b 准确率最高，但它**答错时也是满信心**（0.9859），置信度差仅 0.0036。
如果你打算用置信度做兜底/人工介入，**intern1b 是最差的选择**；qwen2b 的置信度信息量最大
（差值 0.118 + 最低 ECE 0.125）。"选哪个模型"取决于你要不要用置信度做安全机制。

### 2.5 混淆结构

最大混淆轴是 **FWD ↔ DOWN**（互泄漏 0.13~0.35）——俯视 agentview 里深度歧义，符合直觉。
三条对立轴 FWD/BACK、LEFT/RIGHT、UP/DOWN 的泄漏**全为 0**，即历史上担心的
Franka/Piper 相机朝向倒置问题在这批模型上**没有出现**。

`mix_22-06_fk-pp_02_exchange_token` 系列的 exchange 风险已排除：`build_mix.py` 只交换 Piper 侧，
Franka 侧 verbatim 复用；`mix_swap_wrapper.py` 只在 Piper 部署时反交换。**Franka 数据直接评是
对的，不需要翻转 label**。实测佐证：三个 exchange 模型分给 MV_BACK 的概率同样 ≈0（最大 2.0e-03）。

### 2.6 一个数据标注问题

**步 28（真值 MV_UP）六个模型全部预测 MV_RIGHT**，置信度 0.9875~1.0000：

| gemma-e4b | intern1b | intern2b | qwen0.8b | qwen2b | qwen9b |
|---|---|---|---|---|---|
| 1.0000 | 0.9875 | 1.0000 | 0.9999 | 0.9997 | 1.0000 |

六个独立训练的模型不会同时犯同一个随机错误。**建议回看该帧标注**——若确为标注错误，
六个模型的真实准确率都要 +2pt。

---

## 3. 任务三：infer 代码统一

### 3.1 改动清单

| 文件 | 动作 | 行数 |
|---|---|---|
| `scripts/eval_common/mvtoken_client.py` | **新增**（共享核心：HTTP/编码/prompt 拼接/评测/logprobs/统计/CLI） | 706 |
| `scripts/eval_common/README.md` | **新增**（用法 + 字段表 + 映射表） | 212 |
| `scripts/qwen3_5/eval/infer.py` | 重写为薄入口 | 312 → 99 |
| `scripts/gemma4/eval/infer.py` | 重写为薄入口 | 224 → 101 |
| `scripts/internvl/eval/infer.py` | **新建** | 137 |
| `scripts/qwen3_5/eval/eval_mvtoken.py` | **删除**（备份在本目录同级 scratchpad） | −214 |
| `test_mvtoken.sh` / `run_eval.sh` | 修复失效路径 | — |

**`eval_mvtoken.py` vs `infer.py` 的差异与处置**：前者是历史遗留，带**两处训推失配**——
(1) 文本排在图片之前（正确顺序是图片在前）；(2) 用 `\n\n` 连接 instruction/input，而 LF 的
alpaca converter 用**单个** `\n`；外加 `VALID_TOKENS` 漏了 `DONE`。其独有的 `--no-stage` 消融
已搬进共享核心并加强。全仓 + AgentRobot 仅一处引用且指向早已不存在的路径 → 安全删除。

### 3.2 LoRA ↔ prompt 版本映射

21 个在线 LoRA 中 **11 个适用 ood_sample，全部配 v3**：

- `v3/rollout_lite.json`（10 个）：`mix_22_27_v3_9`、`mix_22_27_04_v3_9`、`mix_22-06_fk-pp_02`、
  `mix_22_27_v3_2`、`mix_22-06_fk-pp_02_2`、`zechen_repro`、`mix_22-06_fk-pp_02_08`、
  `gemma4_e4b_mix_22_27_v3`、`internvl3.5-2b`、`internvl3.5-1b`
- `v3/rollout_lite_video.json`（1 个）：`mix_22_27_v3_9_video`（mp4 fps **必须** = 2.0）
- **不适用（10 个）**：`piper_0705_v4_9`（v4-piper prompt + ego 视角，FWD/BACK 与 franka 相反）、
  `dual_cloth_*`×3（双臂 3 图 + STILL）、`*History2*`×3（6 图契约）、`ms0717_*`×3（ManiSkill 仿真）

### 3.3 用法

```bash
cd /workspace1/zhijun/LlamaFactory
.venv/bin/python scripts/qwen3_5/eval/infer.py eval \
  --api-url http://localhost:8109 --model mix_22_27_v3_9 \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs

.venv/bin/python scripts/internvl/eval/infer.py eval \
  --api-url http://localhost:8202 --model internvl3.5-2b -e ... --logprobs

.venv/bin/python scripts/gemma4/eval/infer.py  eval \
  --api-url http://localhost:8104 --model gemma4_e4b_mix_22_27_v3 -e ... --logprobs

# 其它
... single "描述图中场景" --image a.png --image b.png
... tokens --model <lora>      # 只打印 9 个 action 的 tokenizer 切分
... eval ... --raw / --no-stage
```

输出落到 `results/logprobs/<family>__<lora>__<evalset>__<ts>.jsonl` + `_summary.json`。

---

## 4. 行动清单（按优先级）

1. **恢复三个被注释的 override**：`start_vllm_server_0_8.sh:82`、`_2.sh:84`、`.sh:84`(27B)
   去掉行首 `# `。堵住 6.0pt 那条路。（**优先级最高、改动最小**）
2. **事件 token 用"连续 2 帧确认"而不是概率阈值**。误触发时置信度同样接近 1.0，阈值拦不住；
   但所有误触发都只早 1 步，等一帧即可消除，代价仅 1 步延迟。
3. **方向 token 用 0.99 阈值**（0.9 无效，只能拦 0~50% 的错误）。可拦下 43~67% 的错误，
   损失 8~28% 覆盖。**方向和事件不该共用阈值。**
4. **把 logprobs 常开并把每步 margin 记进 rollout 日志**。margin<0.5 的步做重采样投票或沿用
   上一步 —— 这是唯一能压住 Qwen3.5 数值抖动的办法（其余开关实测全失效）。阈值需留 0.15 裕度。
5. **修 `ShowRobot-VLM*.yaml` 里指向死端口 :8101 的 `qwen27`**。
6. **在 4 个真机 yaml 的 `vlm:` 块显式写死 `temperature: 0.0`**（`run.py:203` 无默认值）；
   并让 `gemma_client.py:150-152` 的 hosted-openai 分支保留 0.0。
7. **回看 ood_sample 步 28 的标注**。

---

## 5. 局限

- **n=50 单条 rollout**，acc 的 95% CI 约 ±11pt，模型排序不可靠；事件样本各 n=1；
  `MV_BACK` 无正样本，只能证明"无虚假泄漏"，**未验证模型能否正确输出 MV_BACK**。
- 抖动实验只测了**单步**；闭环会放大单步翻转，实际影响可能大于 4%。
- **物理侧（相机曝光/标定/物体位姿/光照）完全未覆盖**。若真机日间差异远大于 2~6pt，
  主因很可能在物理侧而不在推理栈。
- 27B / 12B 未测。
- 图内标签为英文（环境无 CJK 字体）。

## 6. 环境注意

本次有 subagent 误跑 `uv run tests/check_license.py`，触发 `uv sync` 重装了 `.venv`、
改写了 `uv.lock`。**已逐项验证环境完好**：torch 2.13.0+cu130 / transformers 5.6.0 /
fla 0.5.1 / tilelang 0.1.11 / tvm_ffi 0.1.11 / deepspeed 0.18.4 / llamafactory 0.9.6.dev0
全部可 import，CUDA 8 卡可见，Qwen3.5-2B config 正常加载。

**今后在本仓库请勿使用 `uv run` / `make license` / `make test`**（会触发 `uv sync` 重装
环境，而 fla-core / flash-linear-attention / tilelang / apache-tvm-ffi 是 `uv pip --no-deps`
手工装的）。改用 `.venv/bin/python` 或 `uvx`。
