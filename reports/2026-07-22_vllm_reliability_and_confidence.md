# vLLM 推理可靠性 & 六模型置信度分析报告

日期：2026-07-22　测试集：`data/agentrobot/ood_sample`（rollout_052，真机 Franka，50 帧 OOD）

---

## 0. 结论先行

1. **temperature 确实会影响模型行为，且当前配置不一致。** 采样参数优先级是
   `客户端请求 > --override-generation-config > generation_config.json > vLLM 内置默认`。
   三个 Qwen 启动脚本把 `--override-generation-config` 注释掉了，而 Qwen3.5 的模型目录里
   **没有 `generation_config.json`**，于是"客户端不传 temperature"时会掉到 vLLM 内置默认
   `temperature=1.0` —— **纯随机采样**。实测准确率在 80.0~86.0% 之间摆动（6.0pt）。
2. **真机链路暂时没被这个坑打到**：`run_real.py` 显式发 `temperature: 0.0`，vllm dialect 原样透传。
   受害的是手工 curl、评测脚本、以及 `trapi_chatgpt` 这类 hosted dialect（它会把 0.0 丢掉）。
3. **即使 temperature=0，vLLM 也不是逐位可复现的，而且这是 Qwen3.5 架构特有的。**
   抖动幅度按家族分层差 4~6 倍；两个小 Qwen 模型在最模糊的样本上**真的翻转了输出**。
   但折算下来只有约 4% 的 rollout 会出现 1 次动作翻转，**不足以单独解释"每天性能不同"**。
4. **置信度的核心结论：不确定性几乎全在"往哪走"，不在"做不做事件"。**
   首 token（判别 MV / GRASP / RELEASE / DONE 四类）平均置信度 0.9936~0.9995、最低 0.78；
   方向六选一平均 0.9626~0.9883、最低 **0.4989**。误触发抓取/结束的风险极低
   （远离事件步时 P(事件) 最大仅 8.9e-05）。
5. **但置信度阈值救不了大部分错误：30~55% 的错误是"高置信度错误"（P>0.99 但错）。**
6. **测试集本身有标注问题**：步 13 和步 28 上六个独立训练的模型**全部一致地**不同意标注，
   置信度 0.9875~1.0000。扣掉标注存疑与相位抢跑后，校正准确率 84~94%。

---

## 1. 环境与被测对象

GPU 0-3 被他人占用，GPU 4-6 跑既有 server，GPU 7 本次新起三个。

| 端口 | 模型 | 评测用 LoRA | override-generation-config |
|---|---|---|---|
| 8108 | Qwen3.5-0.8B | `mix_22-06_fk-pp_02_08` | ❌ 注释掉 |
| 8102 | Qwen3.5-2B | `mix_22_27_v3_2` | ❌ 注释掉 |
| 8109 | Qwen3.5-9B | `mix_22_27_v3_9` | ✅ temp=0 |
| 8104 | gemma4-E4B-it | `gemma4_e4b_mix_22_27_v3` | ✅ temp=0 |
| 8201 | InternVL3.5-1B | `internvl3.5-1b` | ✅ temp=0 |
| 8202 | InternVL3.5-2B | `internvl3.5-2b` | ✅ temp=0 |

> 启动 InternVL-1B 时踩到一次坑：flashinfer 的 JIT 需要 `.cc-shim`（gcc-11 垫片）的
> `CC/CXX/CUDAHOSTCXX/NVCC_PREPEND_FLAGS`，直接敲 `vllm serve` 会 ninja 编译失败。
> 各 `start_vllm_server*.sh` 里已有这段逻辑，绕过脚本时要记得带上。

---

## 2. 任务一：temperature 与推理可靠性

### 2.1 采样参数优先级（实测）

vLLM `to_sampling_params()` 的逻辑是 `if (temperature := self.temperature) is None:` 才去取
`default_sampling_params`。所以**客户端赢**：给有 override(temp=0) 的 8109 发 `temperature=2.0`，
同一请求 30 次出了 **5 种不同输出**。

完整优先级：`客户端请求 > --override-generation-config > generation_config.json > vLLM 内置默认`

各 base model 的 `generation_config.json` 现状：

| 模型 | generation_config.json | 无 override 时的落点 |
|---|---|---|
| Qwen3.5-0.8B / 2B / 9B | **不存在** | vLLM 内置 `temp=1.0, top_p=1.0, top_k=0` → **纯采样** |
| Qwen3.5-27B | `temp=0.6, top_k=20, top_p=0.95` | 采样 |
| gemma4-E4B-it | `do_sample=true, temp=1.0, top_k=64, top_p=0.95` | 采样 |
| InternVL3_5-2B-HF | 有文件但**不含采样字段** | vLLM 内置 → 纯采样 |

### 2.2 未固定 temperature 的代价

全量 50 条 × 5 轮，在无 override 的 8108/8102 上：

| 请求方式 | 准确率范围 | 输出会变的样本 |
|---|---|---|
| 不传 temperature | **80.0 ~ 86.0%（6.0pt）** | 7/50 |
| 显式传 temperature=0 | 84.0 ~ 86.0%（2.0pt） | 1/50 |
| 8109（有 override），两种条件 | 78.0% 恒定 | 0/50 |

### 2.3 temperature=0 下的数值抖动 —— 架构相关

受控实验：生产 server、串行请求、显式 `temperature: 0`；每模型先扫 50 条找出 margin
（一次生成里最不确定那步的 top1−top2 logprob 差）最低的样本，再在其上重复 25 次。

| 模型 | margin 中位 | 最小 margin | margin<0.5 | 重复 25 次 margin std | 输出是否翻转 |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 8.75 | **0.125** | 1/50 | 0.125 | **是** 24×MV_FWD + 1×MV_DOWN |
| Qwen3.5-2B | 9.38 | 0.375 | 1/50 | 0.196 | **是** 24×MV_DOWN + 1×MV_FWD |
| Qwen3.5-9B | 12.63 | 0.625 | 0/50 | 0.094 | 否 |
| gemma4-E4B | 12.50 | 0.500 | 1/50 | 0.077 | 否 |
| InternVL3.5-1B | 12.25 | 0.500 | 1/50 | 0.034 | 否 |
| InternVL3.5-2B | 11.25 | 0.875 | 0/50 | 0.060 | 否 |

三点结论：

1. **抖动幅度随置信度反向放大。** 高置信样本（margin≈13）重复 20 次，logprob 极差仅 `3e-6` nats；
   模糊样本上 0.8B 的方向 token 概率在 30 次相同请求里从 **0.526 摆到 0.681**（绝对摆幅 0.155）。
   抖动恰好在最需要稳定的地方最大。
2. **这是 Qwen3.5 架构特有的。** InternVL3.5 多条样本上完全逐位可复现（std=0.0000）。
   原因大概率是 Qwen3.5 的 GDN（Gated DeltaNet）线性注意力 chunked scan 规约顺序不固定
   —— `VLLM_BATCH_INVARIANT=1` 对 Qwen3.5 直接报 `not supported for GDN_ATTN`，对得上。
   **prefix caching / CUDA graph / 并发 / seed 四个开关实测全部无效。**
3. **量级要诚实**：翻转只发生在 margin 最低的 1/50 条上、频率 1/25，折算约 **4% 的 rollout
   会出现 1 次动作翻转**。比不上 2.2 节的 6pt，日间大幅波动更可能来自采样参数没固定 +
   物理侧（光照/标定/物体位姿）。

### 2.4 真机链路核查

- `run_real.py:181` 和 `:508` **确实显式发 `temperature: 0.0`**，三个 vllm 后端
  `_is_hosted()` 均为 False，payload 原样返回。→ 注释掉 override **不直接坑真机**。
- `top_p` / `top_k` / `seed` **从不传** —— 靠 server 默认，是个隐藏耦合。
- 例外：`trapi_chatgpt`（openai dialect）的 `temperature=0.0` 会被
  `vlm/gemma_client.py:150-152` 丢弃（它只在 ==1.0 时转发）。
- `run.py:203` 读 `vlm_cfg["temperature"]` **没有默认值**，与 `run_real.py:508` 的
  `.get("temperature", 0.0)` 行为不对称。

### 2.5 其他"日间漂移"因素（按可能性排序）

1. 采样参数未固定（见上）。
2. **配置 bug**：`ShowRobot-VLM*.yaml` 的 `qwen27` 指向 `:8101`，**当前无进程监听**
   —— 若被他人占用会静默打到错模型。
3. 数值抖动（~4%/rollout，见 2.3）。
4. 物理侧：光照、标定、物体初始位姿。**本次完全未覆盖**，若日间差异远大于 2~6pt，主因可能在这。
5. 已排除：端口独占检查通过（5 个 server 各 1 个监听进程，无重复）；
   LoRA 目录 mtime 全部早于对应 server 启动时间，当前无权重漂移。

### 2.6 修复清单

| 优先级 | 位置 | 改法 |
|---|---|---|
| P0 | `scripts/qwen3_5/eval/start_vllm_server_0_8.sh:82`<br>`start_vllm_server_2.sh:84`<br>`start_vllm_server.sh:84`(27B) | 去掉行首 `# `，恢复 `--override-generation-config` |
| P0 | 4 个真机 yaml 的 `vlm:` 块 | 显式写 `temperature: 0.0`（`run.py:203` 无默认值） |
| P1 | 客户端统一 | 把 `top_p: 1.0` / `top_k: -1` 也显式发出去，别靠 server 默认 |
| P1 | `ShowRobot-VLM*.yaml` | 修 `qwen27` 的端口，或补一个"启动时探活 + 校验 model id"的前置检查 |
| P2 | `vlm/gemma_client.py:150-152` | 让 hosted-openai 后端保留 `temperature=0.0`（先确认目标模型接受非 1.0） |
| P2 | 闭环 runner | 常开 `logprobs`，把每步 margin 记进 rollout 日志；margin<0.5 的步做重采样投票或沿用上一步 —— 这是唯一能压住 GDN 抖动的办法，其余开关全失效 |

---

## 3. 任务二：六模型置信度分布

### 3.1 方法

Qwen3.5 / InternVL 的 tokenizer 把 action 切成 2 个 sub-token（`MV_DOWN`→`MV`+`_DOWN`，
`GRASP`→`GR`+`ASP`；`RELEASE`/`DONE` 是单 token），gemma4 更碎（`MV_FWD`→`MV`/`_`/`FW`/`D`）。
**首 token 只能区分 MV / GRASP / RELEASE / DONE 四类，六个方向落在第二个 token 上。**
因此 action 概率一律走"逐 token top-20 前缀累乘"，不用首 token 概率冒充。

### 3.2 总体准确率与置信度

| 模型 | 准确率 | 平均 top1 | 平均熵 | ECE |
|---|---|---|---|---|
| InternVL3.5-1B | **0.86** | **0.9890** | **0.0257** | 0.146 |
| Qwen3.5-2B | 0.84 | 0.9651 | 0.0958 | **0.125** |
| Qwen3.5-0.8B | 0.80 | 0.9710 | 0.0724 | 0.179 |
| Qwen3.5-9B | 0.78 | 0.9799 | 0.0455 | 0.200 |
| InternVL3.5-2B | 0.78 | 0.9817 | 0.0575 | 0.202 |
| gemma4-E4B | 0.76 | 0.9628 | 0.0814 | 0.203 |

⚠️ n=50 单条 rollout，准确率的 95% CI 约 **±11pt**，**模型排序不具统计显著性**，只能看数量级。

### 3.3 【核心】方向 token vs 事件 token 的分布差异

| 模型 | 首 token(4类) 平均 maxP | 首 token 最低 | 方向(6选1) 平均 maxP | **方向最低** |
|---|---|---|---|---|
| gemma4-E4B | 0.9965 | 0.8345 | 0.9626 | 0.5757 |
| InternVL3.5-1B | 0.9995 | 0.9739 | 0.9883 | 0.5749 |
| InternVL3.5-2B | 0.9976 | 0.8807 | 0.9827 | 0.7302 |
| Qwen3.5-0.8B | 0.9957 | 0.7980 | 0.9737 | **0.4989** |
| Qwen3.5-2B | 0.9936 | 0.7771 | 0.9699 | 0.5922 |
| Qwen3.5-9B | 0.9979 | 0.9455 | 0.9795 | 0.6223 |

- 熵的差距更悬殊：首 token 0.0025~0.0175，方向 0.0285~0.0903 —— **方向的熵是事件判别的 4.8~11.5 倍**。
- 300 个（样本×模型）对里，首 token maxP<0.9 只有 **4 例**，方向 maxP<0.9 有 **22 例**。
- **结论：模型对"该不该抓/该不该结束"极其笃定，不确定性几乎全部集中在"往哪个方向走"。**

### 3.4 误触发风险（真机最关心的）

距离真实事件步 ≥2 的 41 条 MV 样本上，六模型 P(GRASP+RELEASE+DONE) 的最大值：

| 模型 | max P(事件) |
|---|---|
| gemma4-E4B | 1.09e-07 |
| InternVL3.5-2B | 2.38e-07 |
| Qwen3.5-2B | 5.72e-07 |
| InternVL3.5-1B | 2.23e-06 |
| Qwen3.5-9B | 3.47e-06 |
| Qwen3.5-0.8B | 8.85e-05 |

**无一例超过 0.01%。** 所有 13 例 P(事件)>1% 的情况**全部落在真实事件的前一步**
（i=23/42/48）—— 是**相位抢跑**，不是随机误触发。
- 风险最高：**Qwen3.5-9B**（三个事件全部抢跑，置信度 0.94~0.99，且是唯一 GRASP 抢跑的）
- 风险最低：**Qwen3.5-2B**（最大 0.223，从未翻转 argmax）

### 3.5 混淆结构

- 最大混淆轴是 **FWD ↔ DOWN**（互泄漏 0.13~0.35）—— 俯视 agentview 下的深度歧义。
- **三组对立轴 FWD/BACK、LEFT/RIGHT、UP/DOWN 的泄漏全为 0**，没有历史上那种相机朝向导致的倒置。
- MV_BACK 在本测试集无正样本，只能证明"无虚假泄漏"，**未验证模型能否正确输出 MV_BACK**。

### 3.6 校准度与"高置信度错误"

| 模型 | 高置信错误(P>0.99 但错) / 总错误 | 步号 |
|---|---|---|
| gemma4-E4B | 4/12 | 4, 18, 28, 42 |
| InternVL3.5-1B | 4/7 | 7, 13, 19, 42 |
| InternVL3.5-2B | 6/11 | 4, 12, 13, 18, 28, 48 |
| Qwen3.5-0.8B | 5/10 | 7, 13, 19, 28, 42 |
| Qwen3.5-2B | 3/8 | 7, 14, 28 |
| Qwen3.5-9B | 4/11 | 7, 11, 13, 28 |

**30~55% 的错误发生在 P>0.99 时** —— 置信度阈值只能拦住不到一半的错误。
方向 token 用 0.99 阈值可拦下 43~67% 的错误、损失 8~28% 覆盖率；0.9 阈值基本无效。

### 3.7 数据质量发现（意外收获）

按步统计"几个模型答错"：

| 步 | 标注 | 错的模型数 | 模型的一致预测 |
|---|---|---|---|
| 13 | MV_DOWN | **6/6** | MV_FWD ×6 |
| 28 | MV_UP | **6/6** | MV_RIGHT ×6（置信度 0.9875~1.0000） |
| 4 | MV_LEFT | 5/6 | MV_DOWN ×3, MV_FWD ×2 |
| 7 | MV_DOWN | 5/6 | MV_FWD ×5 |
| 19 | MV_RIGHT | 5/6 | MV_DOWN ×3, MV_FWD ×2 |
| 18 | MV_FWD | 4/6 | MV_DOWN ×4 |
| 42 | MV_DOWN | 4/6 | RELEASE ×4（抢跑 1 步） |
| 48 | MV_UP | 4/6 | DONE ×4（抢跑 1 步） |

六个**独立训练**的模型不会同时犯同一个随机错误。步 13、28 建议回看原始帧核对标注。

扣掉标注存疑（步 13/28）与相位抢跑（步 42/48）后的**校正准确率**：

| 模型 | 原始 acc | 标注存疑 | 相位抢跑 | 真实错误 | **校正 acc** |
|---|---|---|---|---|---|
| InternVL3.5-1B | 86.0% | 2 | 2 | 3 | **94.0%** |
| Qwen3.5-2B | 84.0% | 2 | 0 | 6 | 88.0% |
| Qwen3.5-0.8B | 80.0% | 2 | 1 | 7 | 86.0% |
| Qwen3.5-9B | 78.0% | 2 | 2 | 7 | 86.0% |
| gemma4-E4B | 76.0% | 2 | 2 | 8 | 84.0% |
| InternVL3.5-2B | 78.0% | 2 | 1 | 8 | 84.0% |

---

## 4. 任务三：infer 代码统一

### 4.1 `eval_mvtoken.py` vs `infer.py`

`scripts/qwen3_5/eval/eval_mvtoken.py` 是历史遗留版本，带**两处训推失配**：
1. 把**文本排在图片之前**（正确顺序是媒体在前，content 数组顺序 == 占位符顺序）
2. 用 `\n\n` 连接 instruction 和 input（LF alpaca converter 用的是**单个** `\n`）

外加 `VALID_TOKENS` 漏了 `DONE`、默认 DATASET/端口全过期。**已删除**（备份在 scratchpad），
其独有的 `--no-stage` 消融已搬进新核心并加强。

### 4.2 新结构

| 文件 | 行数 | 说明 |
|---|---|---|
| `scripts/eval_common/mvtoken_client.py` | 706 | 共享核心：HTTP / 媒体编码 / prompt 拼接 / 评测 / logprobs / 统计 / CLI |
| `scripts/eval_common/README.md` | 212 | 用法 + 字段表 + 映射表 |
| `scripts/qwen3_5/eval/infer.py` | 99（原 312） | 薄入口，保留 chat template / video fps 约束 |
| `scripts/gemma4/eval/infer.py` | 101（原 224） | 薄入口 |
| `scripts/internvl/eval/infer.py` | 137（新建） | 薄入口，含 untie/归一化自检 |

各家族的硬约束（chat template 对齐、content-format、mm-processor-kwargs、video fps）
原样保留在各自入口并带注释说明。

### 4.3 LoRA ↔ prompt 版本映射

**21 个在线 LoRA 中，11 个适用 ood_sample，全部配 v3。**

- `v3/rollout_lite.json`（10 个）：`mix_22_27_v3_9`、`mix_22_27_04_v3_9`、`mix_22-06_fk-pp_02`、
  `mix_22_27_v3_2`、`mix_22-06_fk-pp_02_2`、`zechen_repro`、`mix_22-06_fk-pp_02_08`、
  `gemma4_e4b_mix_22_27_v3`、`internvl3.5-2b`、`internvl3.5-1b`
- `v3/rollout_lite_video.json`（1 个）：`mix_22_27_v3_9_video`（mp4 fps 必须 = 2.0）
- **不适用（10 个）**：`piper_0705_v4_9`（v4-piper prompt + ego 视角）、`dual_cloth_*`×3
  （双臂 3 图 + STILL）、`internvl3.5-2b-History2*`×3（6 图契约）、`ms0717_*`×3（ManiSkill 仿真）

两项核查澄清：
- **exchange_token 无需翻转 label**：`build_mix.py` 只交换 Piper 侧，Franka 侧 verbatim 复用；
  `mix_swap_wrapper.py` 只在 Piper 部署时反交换。实测佐证：三个 exchange 模型分给 MV_BACK
  的概率同样 ≈0（最大 2.0e-03），无 FWD/BACK 倒置。
- `internvl3.5-1b/2b` 实际是用 `mix_22-06_fk-pp_02_exchange_token` 训的（非 mix_22_27），
  但三个训练集的 prompt 形状与 ood_sample **逐字节相同**（791 字符），用 v3 评测仍然正确。

### 4.4 用法

```bash
cd /workspace1/zhijun/LlamaFactory && source .venv/bin/activate

# Qwen3.5（:8109 / :8102 / :8108）
python scripts/qwen3_5/eval/infer.py eval \
  --api-url http://localhost:8109 --model mix_22_27_v3_9 \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs

# gemma4（:8104）
python scripts/gemma4/eval/infer.py eval \
  --api-url http://localhost:8104 --model gemma4_e4b_mix_22_27_v3 \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs

# InternVL3.5（:8202 / :8201）
python scripts/internvl/eval/infer.py eval \
  --api-url http://localhost:8202 --model internvl3.5-2b \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs

# 其它子命令
python scripts/<family>/eval/infer.py tokens --model <lora>   # 打印 9 个 action 的切分
python scripts/<family>/eval/infer.py single "..." --image a.png --image b.png
python scripts/<family>/eval/infer.py eval ... --raw          # 完整原始回复
python scripts/<family>/eval/infer.py eval ... --no-stage     # 消融（仅 v0 有 Stage 行）
```

logprobs 落到 `results/logprobs/<family>__<lora>__<evalset>__<ts>.jsonl` + `_summary.json`。
每行字段：`action_probs` / `prob_flags` / `legal_mass` / `first_token_probs` / `dir_probs` /
`entropy` / `margin` / `top1_prob` / `top2_prob` / `p_label` 等。

---

## 5. 落地建议（按优先级）

1. **恢复三个 Qwen 启动脚本的 `--override-generation-config`**，并在 4 个真机 yaml 里显式写
   `temperature: 0.0`。这是投入产出比最高的一条。
2. **事件 token 用"连续 2 帧确认"而不是概率阈值。** 误触发时置信度同样接近 1.0，阈值拦不住；
   但实测所有误触发都只早 1 步，等一帧即可消除，代价仅 1 步延迟。
3. **方向 token 用 0.99 阈值兜底**（0.9 无效）。方向和事件不该共用阈值 —— 两者的置信度分布
   差一个数量级。
4. **优先 InternVL3.5-1B**：准确率、置信度、熵、稳定性全面最好，且概率读数近乎逐位可复现。
   Qwen 小模型在模糊样本上概率摆幅达 0.155，任何阈值都要留 0.15 裕度。
5. **回看 ood_sample 步 13 / 28 的标注**，并检查训练集里是否有同类问题 —— 六模型一致反对
   标注是很强的信号。
6. **闭环 runner 常开 logprobs 并记录每步 margin**，既是抖动的唯一缓解手段，也是事后归因的
   唯一抓手。

---

## 6. 局限

- **n=50、单条 rollout**：准确率 95% CI 约 ±11pt，模型排序不具统计显著性。
- **事件样本各 n=1**：GRASP/RELEASE/DONE 的"正样本置信度"只能逐条看，不能求平均。
- **MV_BACK 无正样本**：只证明了"无虚假泄漏"，未验证模型能否正确输出它。
- **只测了单步**，闭环会放大单步翻转的影响，未覆盖。
- **物理侧完全未覆盖**（光照/标定/物体位姿）。若真机日间差异远大于 2~6pt，主因可能在这里。
- 27B / 部分 LoRA 的 margin 分布未测。

---

## 附录：产物清单

**仓库内**
- `scripts/eval_common/{mvtoken_client.py, README.md}`（新增）
- `scripts/{qwen3_5,gemma4,internvl}/eval/infer.py`（重写 / 新建）
- `scripts/qwen3_5/eval/eval_mvtoken.py`（**已删除**，备份见下）
- `results/logprobs/*.jsonl` + `*_summary.json`

**scratchpad**（`/tmp/claude-3014/.../scratchpad/`）
- `reportA_sampling.md` + `q1_*.json`~`q5_*.json` —— 采样可靠性实验原始数据
- `reportM_determinism_verify.md` + `jitter_at_boundary.py` —— 确定性复核（可复跑）
- `reportC_confidence.md` + `probs/*.jsonl` + `probs/{summary,thresholds}.json` + `figs/fig1-6.png`
- `reportB_infer.md` + `backup_orig/` —— 代码整理报告与被删文件备份

## 附：一处环境风险

整理代码期间误跑了一次 `uv run tests/check_license.py`，它触发 `uv sync` 重装了 `.venv`
并改写了 `uv.lock`（mtime 17:46）。**已逐项验证环境完好**：torch 2.13.0+cu130 / transformers 5.6.0 /
fla 0.5.1 / tilelang 0.1.11 / tvm_ffi 0.1.11 / deepspeed 0.18.4 / llamafactory 0.9.6.dev0
全部正常 import，CUDA 8 卡可见，Qwen3.5 config 可加载。

**今后在本仓库请勿使用 `uv run` / `make license` / `make test`**（它们会隐式 `uv sync`），
改用 `.venv/bin/python` 或 `uvx`。
