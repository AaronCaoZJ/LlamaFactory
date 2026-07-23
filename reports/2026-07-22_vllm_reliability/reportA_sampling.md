# 报告 A：vLLM 采样参数 / 确定性可靠性排查

**日期**：2026-07-22 ｜ **vLLM**：0.24.0（`/workspace1/zhijun/AgentRobot/.venv-vllm`）
**评测样本**：`/workspace1/zhijun/LlamaFactory/data/agentrobot/ood_sample/v3/rollout_lite.json`（50 条，`<image><image>` 布局）
**探针脚本**：`/tmp/claude-3014/-workspace1-zhijun/6a801ac6-4b1b-49b8-98d6-d62501f4249c/scratchpad/probe.py`
**原始数据**：同目录 `q1_*.json` / `q2_*.json` / `q3_*.json` / `q5_*.json`

---

## 0. 结论先行

1. **`--override-generation-config` 只是「兜底默认值」，客户端一旦显式传 `temperature`，客户端赢。**
   源码机制：`ChatCompletionRequest.to_sampling_params()` 里 `if (temperature := self.temperature) is None: temperature = default_sampling_params.get(...)`。
   实测：8109（server 侧 `temperature=0`）收到客户端 `temperature=2.0` 后，30 次请求出现 **5 种不同输出**。

2. **真机部署链路（vllm backend）实际发的是 `temperature: 0.0`，这条路是安全的。**
   `run_real.py:181` DEFAULTS + `run_real.py:508` `.get("temperature", 0.0)`；三个 vllm 后端 `_is_hosted()` 均为 **False**，`_finalize_payload` 原样返回。
   → **所以「启动脚本注释掉 override」这件事，对真机 vllm 链路不构成直接故障**；它只坑「不传 temperature 的客户端」（例如 `infer.py eval` 之外的临时脚本、curl、别人的评测代码）。

3. **真正的「每天不一样」有两个独立来源，且第二个连 `temperature=0` 都挡不住：**
   - **来源 A（可控）**：不传 temperature 的客户端 → 落到 vLLM 内置默认 **temperature=1.0 / top_p=1.0 / top_k=0** 的纯采样。8102 上同一条样本 30 次出现 2 种答案；全量 50 条重复 5 轮，准确率 **80.0 ~ 86.0%（波动 6.0pt）**，50 条里 **7 条**会变。
   - **来源 B（不可控，本次最重要的发现）**：**vLLM 的前向本身不是逐次可复现的**。串行、关掉 prefix caching、显式 `temperature=0` 的条件下，同一条请求重复 60 次，top-1 logprob 抖动可达 **±0.076 nats**，top1−top2 margin 在 **1.25 ~ 2.25** 之间摆（摆幅 1.0 nat）。margin 落在这个抖动带内的 token 会**真的翻转 argmax**：8102/8188 上 sample#5（margin 0.125）60 次里翻 11~13 次。全量 50 条重复 5 轮，即使 `temperature=0`，准确率仍有 **84.0 ~ 86.0%（波动 2.0pt）**、1/50 条不稳定。

4. **不是 prefix caching 的锅**：`--no-enable-prefix-caching` 的对照 server（8188）翻转率没有消失（11/60、6/60）。
5. **不是 CUDA graph 的锅**：再加 `--enforce-eager` 仍翻 10/60，logprob 抖动反而从 0.152 涨到 **0.260**。
6. **不是并发的锅**：串行 vs 并发 8 / 并发 32，翻转率同量级。
7. **`seed` 救不了**：8109 上 `temperature=1.0, seed=42` 重复 20 次仍出现 2 种结果——seed 只钉采样随机数，钉不住前向数值。
8. **`VLLM_BATCH_INVARIANT=1`（vLLM 0.24 自带的批不变模式）对 Qwen3.5 不可用**：`RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN`（Qwen3.5 用 Gated DeltaNet 线性注意力）。所以来源 B 目前**没有一键修法**，只能靠「提高 margin / 降低对单次 argmax 的依赖」来缓解。

---

## Q1 客户端显式传 `temperature` 时，server 的 `--override-generation-config` 还起作用吗？谁覆盖谁？

### 结论
**客户端覆盖 server。** `--override-generation-config` 写入的是 `default_sampling_params`，仅在客户端**没传**该字段时才生效。同理适用于 `top_p` / `top_k`。

### 机制（源码证据）
`/workspace1/zhijun/AgentRobot/.venv-vllm/lib/python3.11/site-packages/vllm/entrypoints/openai/chat_completion/protocol.py` → `ChatCompletionRequest.to_sampling_params()`：

```python
if (temperature := self.temperature) is None:
    temperature = default_sampling_params.get(
        "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])
if (top_p := self.top_p) is None:
    top_p = default_sampling_params.get("top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"])
if (top_k := self.top_k) is None:
    top_k = default_sampling_params.get("top_k", self._DEFAULT_SAMPLING_PARAMS["top_k"])
```

`ChatCompletionRequest` 字段默认值全是 `None`（`temperature=None, top_p=None, top_k=None, seed=None`），所以「没传」和「传了」是能区分的。
优先级链：**客户端显式值 > `--override-generation-config` > 模型目录 `generation_config.json` > vLLM 内置 `_DEFAULT_SAMPLING_PARAMS`**。

`_DEFAULT_SAMPLING_PARAMS = {'repetition_penalty': 1.0, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0}`

### 实验设计
同一 server（8109，启动带 `--override-generation-config {"temperature":0,...}`）、同一条样本（`rollout_lite.json` #0）、同一 LoRA（`mix_22_27_v3_9`）。
A 组显式 `temperature=0.0` ×20；B 组显式 `temperature=2.0` ×30。`max_tokens=4`，`logprobs=true, top_logprobs=20`。

### 实测数据
| 组 | 请求数 | 不同输出文本数 | 分布 |
|---|---|---|---|
| `temperature=0.0`（显式） | 20 | **1** | `MV_LEFT` ×20 |
| `temperature=2.0`（显式） | 30 | **5** | `MV_LEFT` ×26、`MV****…surround` ×1、`MV ценам:url not` ×1、`MVемым заставляетقا` ×1、`MV_LONGโฟนMAP` ×1 |

server 侧的 `temperature=0` 完全没能压住客户端的 2.0 → **客户端赢，结论明确**。

附带确认：返回的 logprobs 是 **raw logprobs（采样前）**——两组的 token0 logprob 完全相同（`-1.1920928244535389e-07`），不随 temperature 变。所以 logprobs 可以当作「前向数值」的探针，与采样解耦。

### 证据位置
- 原始数据：`scratchpad/q1_8109_raw.json`
- 8109 实际启动命令：`ps` 可见 `--override-generation-config {"temperature": 0, "top_p": 1.0, "top_k": -1}`（pid 238839）

---

## Q2 客户端不传 `temperature` 时，各 server 实际用什么？

### 结论
- **8109 / 8104 / 8202（有 override）**：落到 `temperature=0, top_p=1.0, top_k=-1` → **确定性 greedy**。
- **8108 / 8102（无 override 且模型目录无 `generation_config.json`）**：落到 vLLM 内置默认 **`temperature=1.0, top_p=1.0, top_k=0`（= 全词表纯采样）**。假设成立，已实测证实。
- **Qwen3.5-27B（:8101，无 override 但有 `generation_config.json`）**：会落到 `do_sample=true, temperature=0.6, top_k=20, top_p=0.95` —— 比 8102/8108 温和，但仍然是采样。
- **gemma4-E4B-it**：若没有 override 会落到 `temperature=1.0, top_k=64, top_p=0.95`；当前 8104 有 override，不受影响。

### 实验设计
payload 里**完全不含** `temperature` 字段（`probe.make_payload(temperature=None)`），同一条样本 #0，每个 server 各 30 次，`max_tokens=4`。

### 实测数据
| server | override | 不同输出数 | 分布 | token0 logprob 不同值数 / 抖动幅度 |
|---|---|---|---|---|
| 8109 Qwen3.5-9B | 有 | **1** | `MV_LEFT` ×30 | 1 / `0.0` |
| 8104 gemma4-E4B | 有 | **1** | `MV_DOWN` ×30 | 7 / `9.5e-07` |
| 8108 Qwen3.5-0.8B | **无** | 1 | `MV_LEFT` ×30 | 28 / `9.5e-05` |
| 8102 Qwen3.5-2B | **无** | **2** | `MV_LEFT` ×26、`MV_DOWN` ×4 | 13 / `2.5e-06` |

8102 直接发散（13% 的请求给出不同动作）。8108 这条样本没翻是因为模型太自信（top1 logprob ≈ −2e-4，p≈0.9998），**不代表它是 greedy**——见下面的对照实验。

### 关键对照（证明 8102 确实在采样，而不是别的原因）
同一 server（8102）、同一条样本 #0：

| 条件 | 40 次结果 |
|---|---|
| 显式 `temperature=0.0` | `MV_LEFT` ×40（**100% 一致**） |
| 不传 `temperature` | `MV_LEFT` ×38、`MV_DOWN` ×2 |

样本 #0 在 8102 上 token1 位置的 margin 是 **2.25 nats** → `p(MV_DOWN)/p(MV_LEFT) = e^-2.25 ≈ 0.105`，即 ~9.5% 翻转概率，实测 2/40 = 5%、上一轮 4/30 = 13%，与 **temperature=1.0 纯采样**的理论值一致。假设验证通过。

### 全量影响（最能说明「每天不一样」的数字）
50 条全量评测，重复多轮：

| server | 客户端条件 | 各轮准确率 | 极差 | 50 条里结果会变的条数 |
|---|---|---|---|---|
| 8102（无 override，2B） | `temperature=0.0` | 86.0 / 84.0 / 86.0 / 84.0 / 84.0 | **2.0pt** | **1 / 50** |
| 8102（无 override，2B） | **不传 temperature** | 82.0 / 80.0 / 84.0 / 84.0 / 86.0 | **6.0pt** | **7 / 50** |
| 8109（有 override，9B） | `temperature=0.0` | 78.0 / 78.0 / 78.0 | 0.0pt | 0 / 50 |
| 8109（有 override，9B） | **不传 temperature** | 78.0 / 78.0 / 78.0 | 0.0pt | 0 / 50 |

8109 两种条件都稳，是因为 override 把「不传」这条路也钉成了 temperature=0，**并且** 9B 的最小 margin（0.5 nats）高于数值抖动带。

### 证据位置
- `scratchpad/q2_no_temp_raw.json`、`scratchpad/q3_control_8102_s0.json`、`scratchpad/q5_eval_repeat.json`
- 内置默认值：`/workspace1/zhijun/AgentRobot/.venv-vllm/bin/python -c "from vllm import SamplingParams; p=SamplingParams(); print(p.temperature,p.top_p,p.top_k,p.seed)"` → `1.0 1.0 0 None`

---

## Q3 `temperature=0`（greedy）下 vLLM 是否严格可复现？

### 结论
**不是。** 三层结论：
1. **文本级**：绝大多数样本 100% 一致；但 **top1/top2 margin ≲ 0.5 nats 的 token 会真的翻转 argmax**。
2. **logprob 级**：**从来不是逐位一致**。同一请求重复 60 次，返回的 top-20 logprob 向量几乎每次都不同（60 次里 59~60 个不同签名）。
3. **抖动幅度足以翻转 argmax**：观测到 top1 logprob 抖动 **±0.076 nats**、top1−top2 margin 摆幅达 **1.0 nat**，远大于「浮点 epsilon」量级。

### 实验设计
1. **margin 普查**：50 条样本 × `max_tokens=8`，记录每个生成位置的 top1−top2 margin，找出最紧的样本。
2. **重复 60 次**：在最紧样本上，显式 `temperature=0`，串行 60 次。
3. **并发**：同一样本同时发 8 条 / 32 条。
4. **prefix caching**：(a) 逐请求变化的 `cache_salt`（同 prompt、不同缓存桶）；(b) 固定 `cache_salt`；(c) 另起 `--no-enable-prefix-caching` 的对照 server（:8188，GPU 7，`--gpu-memory-utilization 0.12`，实验后已 kill）。
5. **seed**：`temperature=1.0` + `seed=42` 重复 20 次，看 seed 能不能钉住采样。

### 实测数据

#### (a) margin 分布（50 条样本的所有生成 token 位置）
| server | 位置总数 | margin ≤0.13 | ≤0.6 | ≤1.1 | ≤2.1 | ≤3.1 | 中位数 |
|---|---|---|---|---|---|---|---|
| 8109 (9B) | 146 | 0 (0.0%) | 1 (0.7%) | 2 (1.4%) | 2 (1.4%) | 5 (3.4%) | 15.6 |
| 8102 (2B) | 149 | 1 (0.7%) | 1 (0.7%) | 1 (0.7%) | 5 (3.4%) | 8 (5.4%) | 12.0 |

→ **约 1~3% 的决策 token 处在「抖动可翻转」的危险带**，其余离得很远。注意 margin 全是 0.125 的整数倍（bf16 logit 量化）。

#### (b) 重复 60 次（显式 `temperature=0`）
| server | 样本 | token0 margin | 60 次文本分布 | top-20 logprob 不同签名数 | token0 logprob 抖动 |
|---|---|---|---|---|---|
| 8109 | #12（最紧位置 margin **0.5**） | 14.6 | `MV_FWD` ×60 ✅ | **59 / 60** | 2.4e-07 |
| 8102 | #5（最紧位置 margin **0.125**） | 12.3 | `MV_DOWN` ×47、`MV_FWD` ×**13** ❌ | **60 / 60** | 1.2e-06 |
| 8108 | #0 | 8.9 | `MV_LEFT` ×60 ✅ | **60 / 60** | 5.5e-05 |

**文本 100% 一致 ≠ 数值一致**：8109 的 60 次输出文本完全相同，但 top-20 logprob 向量有 59 种不同签名。

#### (c) 串行 vs 并发（`temperature=0`）
| server / 样本 | 串行 60 | 并发 8 | 并发 32 |
|---|---|---|---|
| 8109 #12 | 60/60 一致 | 8/8 一致 | 32/32 一致 |
| 8102 #5 | 47:13 | 7:1 | 28:4 |
| 8188 #5（无 prefix cache） | 54:6 | — | 31:1 |

翻转率 8%~22%，**并发不是主因**（串行同样翻）。

#### (d) prefix caching 的影响（8102，样本 #5，`temperature=0`）
| 条件 | 40~60 次分布 | 翻转率 |
|---|---|---|
| 默认（prefix caching 开，无 salt） | 47 : 13 | 22% |
| `cache_salt` 逐请求变化 | 32 : 8 | 20% |
| `cache_salt` 固定 | 37 : 3 | 7.5% |
| **`--no-enable-prefix-caching`（:8188 对照 server）** | 54 : 6 / 49 : 11 | **10~18%** |
| **`--no-enable-prefix-caching` + `--enforce-eager`（:8188）** | 50 : 10 | **17%** |

**关掉 prefix caching 翻转依然存在** → prefix caching 只是调制项，不是根因。
**再关掉 CUDA graph（`--enforce-eager`）也没用**，翻转率 17%，抖动反而更大（见下表）→ CUDA graph 捕获也不是根因。

#### (e) 抖动幅度（最关键的数字）
8188（无 prefix cache、串行、`temperature=0`）样本 #42，60 次：

| 指标 | 均值 | 极差 |
|---|---|---|
| token0 logprob（token `MV`） | −0.1760 | **0.1519** |
| top1−top2 margin（`MV` vs `RELEASE`） | 1.667（min 1.25 / max 2.25） | **1.0000** |
| 不同 margin 取值 | — | 35 种 |

同一样本在 **`--enforce-eager`**（再关掉 CUDA graph）下重测 60 次：

| 指标 | 均值 | 范围 | 极差 |
|---|---|---|---|
| token0 logprob | — | — | **0.2601**（比非 eager 的 0.1519 **更大**） |
| top1−top2 margin | 1.225 | 0.75 ~ 2.00 | **1.25** |

top1/top2 的 **token 身份 60 次全都是 `MV` / `RELEASE`**（没有换人），纯粹是数值在动：`p(MV)` 在 0.80~0.85 之间摆、`p(RELEASE)` 在 0.11~0.20 之间摆。
**推论**：任何 margin ≲ 0.5 nats 的 token 都可能在 greedy 下翻转，margin ≲ 1.0 的处在灰区。
**四个可调开关（prefix caching / CUDA graph / 并发度 / seed）全部关掉或钉死，抖动依然存在**，说明根因在 kernel 本身（归约顺序 / atomics），不是任何可配置的调度行为。

#### (f) seed 能不能救？
| server / 样本 | `temperature=1.0, seed=42` ×20 |
|---|---|
| 8109 #12 | `MV_FWD` ×16、`MV_DOWN` ×4 ❌ |
| 8102 #5 | `MV_DOWN` ×20 |
| 8108 #0 | `MV_LEFT` ×20 |

**`seed` 不能保证可复现**（8109 上带同一个 seed 的 20 次请求出现 2 种结果）——vLLM 的 per-request seed 只钉住采样用的随机数，钉不住前向数值。**不要用 seed 代替 temperature=0。**

#### (g) 批不变模式（`VLLM_BATCH_INVARIANT=1`）
vLLM 0.24 自带该开关（`vllm/envs.py:88, :574`），但对 Qwen3.5 **直接启动失败**：
```
RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN.
```
（Qwen3.5 用 Gated DeltaNet 线性注意力。日志：`scratchpad/ctrl_8189_batchinv.log`）

### 证据位置
`scratchpad/q3_margin_scan.json`、`q3_pertoken_margin.json`、`q3_determinism_summary.json`、`q3_determinism_raw.json`、`q3_cachesalt_8102_s5.json`、`q3_noprefix_8188.json`、`ctrl_8188_noprefix.log`、`ctrl_8189_batchinv.log`

---

## Q4 真机部署链路实际发出去的请求里到底有没有 `temperature`？

### 结论
**有，而且是 0.0。** 三个 vllm 后端 `_is_hosted()` 都是 **False**，`_finalize_payload` **原样返回**，temperature 不会被丢。
**但 `trapi_chatgpt`（hosted OpenAI dialect）后端的 `temperature=0.0` 会被丢掉** —— 这是 `_finalize_payload` 的已知行为，只影响 GPT 系后端。
**`top_p` / `top_k` / `seed` 在真机链路里从来没有传过。**

### 链路追踪
1. **温度来源**
   - `run_real.py:181`（`DEFAULTS["vlm"]`）：`"temperature": 0.0`
   - `run_real.py:508`：`temperature=float(vlm_cfg.get("temperature", 0.0))` —— 双重兜底
   - `configs/robot.yaml` / `robot_piper.yaml` / `ShowRobot-VLM.yaml` / `ShowRobot-VLM_piper.yaml` 的 `vlm:` 块**都没有 `temperature` 键**（已 grep 全仓确认），所以走 DEFAULTS 的 0.0。
   - 只有 `configs/robot_libero.yaml:44`、`robot_maniskill*.yaml:130/136` 显式写了 `temperature: 0.0`（sim 用）。

2. **dialect 判定**（`vlm/gemma_client.py:83-84, :123-127`）
   ```python
   self.provider   = str(provider or "vllm").lower()
   self.api_dialect = str(api_dialect or self.provider).lower()
   def _is_hosted(self): return self.provider in ("openai","gemini","trapi") \
                             or self.api_dialect in ("openai","gemini")
   ```
   vllm 后端 yaml 里只写了 `provider: vllm`、**没写 `api_dialect`** → `api_dialect = "vllm"` → `_is_hosted()` **False** → `_finalize_payload` 第一行 `if not self._is_hosted(): return payload` 直接原样返回。

3. **实测解析结果**（真实读 yaml + `core.config.resolve_vlm_config`，DEFAULTS 逐字节取自 `run_real.py:176-183`）：

| config | backend | cfg temperature | provider | dialect | hosted | **实际发出的 temperature** | base_url |
|---|---|---|---|---|---|---|---|
| ShowRobot-VLM_piper.yaml | qwen27 | 0.0 | vllm | vllm | False | **0.0** | :8101 |
| ShowRobot-VLM_piper.yaml | qwen9 | 0.0 | vllm | vllm | False | **0.0** | :8109 |
| ShowRobot-VLM_piper.yaml | qwen2 | 0.0 | vllm | vllm | False | **0.0** | :8102 |
| ShowRobot-VLM.yaml | qwen27/9/2 | 0.0 | vllm | vllm | False | **0.0** | :8101/:8109/:8102 |
| robot_piper.yaml | trapi_gemini | 0.0 | showrobot | gemini | True | **0.0**（gemini 分支转发） | :4140 |
| robot_piper.yaml | **trapi_chatgpt** | 0.0 | openai | openai | True | **被丢弃 ⚠** | trapi.research.microsoft.com |
| robot_piper.yaml | gemini | 0.0 | gemini | gemini | True | **0.0** | generativelanguage.googleapis.com |

4. **payload 构造点**（都传 `temperature`，都不传 top_p/top_k/seed）
   - `vlm/mvtoken_client.py:73-79`：`"temperature": c.temperature`
   - `vlm/dual_mvtoken_client.py:87-106`、`:150`：`"temperature": c.temperature`
   - `vlm/gemma_client.py:309-316`（`complete_token`，带 `guided_choice`）：`"temperature": self.temperature`
   - `vlm/gemma_client.py:340`（解析失败后的 retry）：硬编码 `"temperature": 0.0`
   - `vlm/dual_roles.py:347/405/439/498`、`vlm/roles.py:105/196`：显式 `temperature=0.0`
   - 全仓 grep `top_p` / `top_k` / `"seed"`：payload 里**一处都没有**（`run.py:105` 的 `seed` 是 sim 环境 reset 用的，不是采样 seed）

### 由此得到的重要修正
`start_vllm_server_0_8.sh:82` / `start_vllm_server_2.sh:84` / `start_vllm_server.sh:84`（27B）注释掉 override，**并不会**让真机 vllm 链路跑成采样——真机自己带了 0.0。
它坑的是**所有不传 temperature 的旁路客户端**：curl、临时评测脚本、别人的 harness。`scripts/qwen3_5/eval/infer.py` 自己是安全的（`chat()` 的 `temperature` 默认 0.0，且总会写进 payload）。

### 证据位置
- `run_real.py:176-183, :508-521`
- `vlm/gemma_client.py:83-86, :115-127, :128-165`
- `vlm/mvtoken_client.py:73-79`、`vlm/dual_mvtoken_client.py:87-106`
- `core/config.py:129-160`（`resolve_vlm_config`）

---

## Q5 除 temperature 外，导致「每天表现不同」的因素清单（按可能性排序）

### #1 vLLM 前向的运行间数值不确定性 → 近似平手 token 的 argmax 翻转 【已证实，无一键修法】
- **证据**：Q3(e)，margin 摆幅 1.0 nat；Q3(b)，8102/8188 sample#5 翻转率 8~22%；Q5 全量重复，`temperature=0` 下准确率仍波动 2.0pt、1/50 条不稳。
- **已排除的解释**：prefix caching（`--no-enable-prefix-caching` 照翻 10/60）、CUDA graph（`--enforce-eager` 照翻 10/60，抖动反而更大）、并发（串行照翻）、采样（显式 temperature=0 且宽 margin 对照样本 40/40 一致）、seed（`seed=42` 照翻）。
- **根因方向**：kernel 归约顺序 / atomics 的运行间不确定性。`VLLM_BATCH_INVARIANT=1` 是官方对策，但 **Qwen3.5(GDN_ATTN) 不支持**。
- **去哪验证**：`scratchpad/q3_noprefix_8188.json`；换非 GDN 模型（gemma4 / InternVL）复测 `VLLM_BATCH_INVARIANT=1` 是否可用。
- **缓解**：模型越大 margin 越宽（9B 最小 margin 0.5 vs 2B 的 0.125，9B 在本 evalset 上 0/50 不稳）；或在真机侧对动作做时间平滑 / 多数投票，不要让单次 argmax 直接驱动电机。

### #2 旁路客户端不传 temperature，落到 vLLM 默认纯采样 【已证实，一行可修】
- **证据**：Q2，8102 全量重复准确率 80.0~86.0（6.0pt），7/50 条会变；`_DEFAULT_SAMPLING_PARAMS.temperature = 1.0`。
- **触发条件**：任何直连 8108/8102/8101 且省略 `temperature` 的脚本。8109/8104/8202 因为有 override 免疫。
- **去哪验证**：`scratchpad/q2_no_temp_raw.json`、`q5_eval_repeat.json`。

### #3 server 启动参数漂移：同一端口不同时间起来的参数不一样 【结构性风险，今天没发作】
- **证据**：同一份 `start_vllm_server_*.sh` 里 override 有的注释了、有的没注释（`_0_8.sh:82` / `_2.sh:84` / `.sh:84` 注释掉，`_9.sh:112` / gemma4 `:89` / internvl `:143` 保留）。谁重启、用哪份脚本、有没有传 `TEMPERATURE=` 环境变量，决定了那天的 server 行为。
- **今天的实际状态（已核对 `ps` vs 脚本）**：8109 运行参数与 `start_vllm_server_9.sh` 当前内容一致（`GPU_UTIL=0.7`、LoRA 列表 8 个全对、override 在）。**当前无漂移。**
- **另一个漂移面**：`--gpu-memory-utilization` 不同（8109=0.7、8104=0.6、8202=0.7、8108=0.15、8102=0.18）→ KV cache block 数不同 → 调度/分块行为不同 → #1 的抖动表现不同。
- **去哪验证**：`ps -eo args | grep vllm` 对照 `git log -p scripts/*/eval/start_vllm_server*.sh`。

### #4 LoRA 名指向的目录被覆盖/重训 【结构性风险，今天没发作】
- **证据**：`--lora-modules NAME=PATH` 里 NAME 稳定但 PATH 内容可变。当前 mtime：
  - `saves/qwen3.5-9b/robot/mix_22_27_v3` → 2026-07-04 08:48
  - `saves/qwen3.5-2b/robot/mix_22_27_v3` → 2026-07-04 08:33
  - `saves/qwen3.5-27b/robot/mix_22_27_v3` → 2026-06-29 19:43
  - `saves/gemma4-e4b/robot/mix_22_27_v3` → 2026-07-12 12:22
  都早于 8109（1d4h 前）和 8104（1h23m 前）的启动时间 → **当前加载的权重与磁盘一致**。
- **风险**：同名重训后，**已经在跑的 server 不会重新加载**，磁盘和显存不一致；重启后又突然变了。这正是「昨天好好的今天不行」的典型形态。
- **去哪验证**：比较 adapter mtime 与 server 进程 `etime`。

### #5 配置默认后端指向一个没起来的端口 【已发现，会导致整段跑不了或悄悄换模型】
- **证据**：`ShowRobot-VLM.yaml` / `ShowRobot-VLM_piper.yaml` 的 `qwen27` 后端指向 `localhost:8101`，**当前 8101 上没有任何进程监听**（`ss -ltn` 无 8101）。`configs/robot.yaml:67-69` 的 `mvtoken` 后端同样指 8101。
- **影响**：选到该后端就直接连不上；若哪天别人在 8101 起了别的模型，请求会**静默打到错模型上**。
- **端口独占检查（对应你记忆里那个坑）**：本次 `ss -ltnp` 显示 8102/8104/8108/8109/8202 **各自只有一个 vllm 进程监听，无重复端口、无多进程抢占**。同机另一用户 husiyuan 在 8003~8006 跑 UI-TARS（GPU 0-3），端口不冲突。**今天没有分流问题。**

### #6 hosted OpenAI 后端把 temperature=0.0 丢掉 【已证实，只影响 GPT 系】
- **证据**：`vlm/gemma_client.py:150-152`，`temperature` 只在 `== 1.0` 时转发；`robot_piper.yaml` 的 `trapi_chatgpt` 落在这条路上。
- **影响**：该后端跑在服务方默认温度（通常 1.0）+ 服务方随时可能换模型版本 → 天级漂移最大。gemini/showrobot 后端不受影响（gemini 分支照常转发 0.0）。

### #7 图像编码 / 分辨率差异
- **现状（已核对，看起来是对的）**：`core/images.py:48-52` 真机图走 **PNG 无损**；video 槽位走 `frames_to_video_jpeg_data_url`，默认 `quality=100 + subsampling=0`（4:4:4），注释里记录了 q100/4:4:4 编码误差 0.267/255 vs q95 的 0.911。
- **风险点**：`quality` 是函数参数，调用方若传低值就悄悄退化；相机分辨率/裁剪若变，视觉 token 数变，等价于换了 prompt。
- **去哪验证**：grep `frames_to_video_jpeg_data_url(` 的所有调用点是否传了 `quality`；对比真机保存的 PNG 与训练集图像的 shape。

### #8 历史帧注入顺序 / prompt 前缀不一致
- **证据/契约**：`vlm/mvtoken_roles.py:63-93` 的 `_history_prefix` 注释明确写着「这是 TRAINING CONTRACT，必须与 `LlamaFactory/data/agentrobot/build_history_dataset.py` 的 `_history_prefix` 逐字节一致」，且 `history_frames=2` 的正路是 `prompts/v3/history2_mvtoken_generator_lite.txt`，用 `--history-prompt-prefix` 选。
- **风险**：走没走 `--history-prompt-prefix`、`image_layout` 选 interleaved 还是别的，会让 prompt 与训练分布错位——表现就是「换了个跑法就掉点」。
- **去哪验证**：把真机 `--debug` 打出的实际 prompt 与 `build_history_dataset.py` 产出的训练样本做 diff。

### #9 chat template / prompt-parity
- **现状**：所有 server 都挂了 LF 对齐模板（8109/8102/8108 挂 `chat_template_qwen3_5_lf.jinja`，8104 挂 gemma4n 版，8202 挂 internvl 版 + `--chat-template-content-format openai`）。`chat_template_qwen3_5_lf.jinja` 自 2026-07-13 后无改动，工作区无未提交改动。**当前正常。**
- **风险**：漏挂模板 → Qwen3.5 官方模板插空 `<think>` 块，与训练分布差 4 个 token，静默掉点。`infer.py:130-145` 的 `check_prompt_parity()` 会拦，但真机链路没有这个检查。

### #10 机械 / 相机侧变化
- 相机外参漂移、光照、抓取物位置、夹爪磨损、标定变化 → 观测分布变了，模型没变。
- **去哪验证**：把每天真机的首帧存档做像素级/直方图对比；或用固定复现场景跑一组 fixed-observation 回放，把「模型侧」和「物理侧」的方差分开。这是唯一能把 #1~#9 与物理变化解耦的实验。

---

## 修复建议（精确到文件:行号）

### R1【高，1 行 ×3】把注释掉的 `--override-generation-config` 恢复
- `/workspace1/zhijun/LlamaFactory/scripts/qwen3_5/eval/start_vllm_server_0_8.sh:82`
- `/workspace1/zhijun/LlamaFactory/scripts/qwen3_5/eval/start_vllm_server_2.sh:84`
- `/workspace1/zhijun/LlamaFactory/scripts/qwen3_5/eval/start_vllm_server.sh:84`（27B，:8101）

改法：去掉行首 `# `，与 `start_vllm_server_9.sh:112` 保持一致：
```bash
  --override-generation-config "{\"temperature\": ${TEMPERATURE}, \"top_p\": 1.0, \"top_k\": -1}"
```
（`TEMPERATURE` 三份脚本里都已有 `TEMPERATURE="${TEMPERATURE:-0}"` 默认，无需另加。）
**收益**：把「客户端忘了传 temperature」这条 6.0pt 波动的路堵死。27B 那份收益最大——它的 `generation_config.json` 是 `temperature=0.6/top_k=20/top_p=0.95`，不加 override 就是实打实的采样。
**注意**：这只改默认值，改完仍然挡不住显式传 `temperature>0` 的客户端（Q1）。

### R2【高，1 行】给真机 hosted-openai 后端保留 temperature
`/workspace1/zhijun/AgentRobot/vlm/gemma_client.py:150-152`
```python
            temperature = payload.get("temperature")
            if temperature is not None and float(temperature) == 1.0:
                out["temperature"] = 1.0
```
建议改成「只有确实是 reasoning 模型才丢弃」，或至少让它可配置：
```python
            temperature = payload.get("temperature")
            # 只有 reasoning 模型才拒绝自定义 temperature；其余照常转发（0.0 = 确定性）
            if temperature is not None and (self.reasoning_effort is None or float(temperature) == 1.0):
                out["temperature"] = float(temperature)
```
**收益**：`trapi_chatgpt` 后端不再跑在服务方默认温度上。
**风险**：某些 OpenAI reasoning 模型会对非 1.0 的 temperature 报 400；上面的写法用 `reasoning_effort` 是否配置来区分，`robot_piper.yaml` 的 `trapi_chatgpt` 配了 `reasoning_effort: medium`，所以行为不变——**要真正生效需要人工确认目标模型是否接受 temperature=0**，建议先手测一次再改。

### R3【高，配置层】把 `temperature` 显式写进真机 config，别依赖代码默认
`/workspace1/zhijun/AgentRobot/configs/ShowRobot-VLM.yaml`、`ShowRobot-VLM_piper.yaml`、`robot.yaml`、`robot_piper.yaml` 的 `vlm:` 块（分别在 `ShowRobot-VLM_piper.yaml:34-35`、`robot_piper.yaml:43-44` 附近）加一行：
```yaml
vlm:
  base_url: http://localhost:8000/v1
  temperature: 0.0        # 确定性；缺省时依赖 run_real.py DEFAULTS，改动易失守
```
**收益**：`run.py:203` 用的是 `float(vlm_cfg["temperature"])`（**没有默认值，缺键直接 KeyError**），而 `run_real.py:508` 有默认。两条入口行为不一致，写进 config 可消除这个不对称。

### R4【中，运维】固化「一个端口一份启动脚本」+ 启动自检
- 现象：`ShowRobot-VLM*.yaml` 的 `qwen27` 指向 :8101，但 8101 当前没进程。
- 建议：在 `run_real.py:321-324` 已有的 vllm 健康检查基础上，把 **实际命中的 server 启动参数**也打进 rollout 日志（`GET /v1/models` 拿 model root + 记录 `ps` 的 `--override-generation-config` / `--gpu-memory-utilization` / LoRA 路径 + adapter mtime）。
- **收益**：出现「今天不对」时，能直接从日志判定是 #3/#4/#5 里的哪一个，而不用回忆。

### R5【中，缓解 #1】不要让单次 argmax 直接驱动电机
数值抖动无法在 vLLM 0.24 + Qwen3.5 上根除：`VLLM_BATCH_INVARIANT=1` 因 GDN_ATTN 不可用，`--no-enable-prefix-caching` / `--enforce-eager` / 降并发 / 传 seed **四条全部实测无效**。可行缓解，按代价从低到高：
1. **优先用更大的 backbone**：9B 在本 evalset 上 0/50 不稳，2B 是 1/50、0.8B 更差。
2. **在 `vlm/mvtoken_client.py:73-79` 的 payload 加 `"logprobs": True, "top_logprobs": 5` 并常开**（现在只在 `debug=True` 时加，见 `:82-83`），把每步 margin 记进 rollout 日志。margin < 0.5 的步就是「今天可能不一样」的那些步，事后能直接定位。代价约几 KB/步。
3. **对 margin < 阈值的步做 3 次重采样投票**，或沿用上一步动作（真机侧动作平滑）。这是唯一能真正压住 #1 的办法。

### R6【低，验证向】在非 GDN 模型上确认 `VLLM_BATCH_INVARIANT=1` 可用
gemma4 / InternVL 不用 GDN 注意力，值得单独测一次：若可用，这两条链路可以拿到真正 bit 级可复现的推理，作为「模型侧方差 = 0」的基线来隔离物理侧变化（#10）。

---

## 剩余风险 / 本次没能覆盖的

1. **来源 #1 没有根治方案**。`VLLM_BATCH_INVARIANT=1` 对 Qwen3.5 直接报错；升级 vLLM 是否支持 GDN_ATTN 未验证。在此之前，Qwen3.5 链路上「greedy 完全可复现」做不到。
2. **抖动幅度只在 2B/0.8B/9B 上量化过**，27B（:8101，未运行）和 gemma4/InternVL 的 margin 分布没测。gemma4 8104 的 token0 logprob 抖动是 9.5e-07（比 8108 的 5.5e-05 小两个量级），但没做低 margin 样本的翻转测试。
3. **50 条 evalset 只覆盖单步 token 预测**，没有闭环。闭环里误差会累积——一步翻转会改变后续所有观测，天级差异会被放大到远超 2~6pt。真正的天级方差需要用固定初始状态的闭环回放来测。
4. **R2 的改法需要人工确认**目标 OpenAI reasoning 模型是否接受 `temperature=0`，我没有该后端的凭据，无法实测。
5. **#10（机械/相机）完全没有覆盖**——本次全部是离线固定图像。如果真机每天的差异远大于本文测出的 2~6pt，那主因很可能在物理侧，需要按 R4 的日志 + 固定场景回放来分离。
6. **`--gpu-memory-utilization` 对抖动的影响没有单独做 A/B**（8102 用 0.18、对照 8188 用 0.12，两者翻转率同量级但不是严格对照）。

---

## 附：实验用的临时 server（已清理）
- `:8188` Qwen3.5-2B + `mix_22_27_v3_2`，GPU 7，`--gpu-memory-utilization 0.12`，`--no-enable-prefix-caching`（第二轮再加 `--enforce-eager`）—— 用于 prefix-cache / CUDA-graph 隔离。数据：`q3_noprefix_8188.json`、`q3_eager_8188.json`。
- `:8189` 同上 + `VLLM_BATCH_INVARIANT=1` —— 启动失败（GDN_ATTN 不支持），失败日志保留在 `scratchpad/ctrl_8189_batchinv.log`。
- 未 kill 任何既有 server（8109/8104/8202/8108/8102 全程在跑）。
