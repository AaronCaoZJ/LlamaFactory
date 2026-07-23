# MVTOKEN 评测客户端（三家族共用）

一个共享核心 + 每家族一个薄入口。**任何"三家都一样"的逻辑只能改核心**，否则必然各自漂移。

| 文件 | 作用 |
|---|---|
| `scripts/eval_common/mvtoken_client.py` | 核心：HTTP / 媒体编码 / prompt 拼接 / 评测循环 / logprobs 分析 / 统计 / CLI |
| `scripts/qwen3_5/eval/infer.py` | Qwen3.5 入口（0.8B / 2B / 9B / 27B） |
| `scripts/gemma4/eval/infer.py` | Gemma-4 入口（E4B / 12B） |
| `scripts/internvl/eval/infer.py` | InternVL3.5 入口（1B / 2B / 4B / 8B） |

薄入口只声明一个 `FamilySpec`（默认端口 / 默认 LoRA / 要不要发 system / video 契约 /
preflight 自检），其余全部复用核心。

## 三家族通用铁律（错一处静默掉点）

1. **媒体排在文本之前** —— OpenAI `content` 数组顺序 = 占位符顺序。
2. **`instruction` + 单个 `\n` + `input`** —— LF alpaca converter 是 `"\n".join(...)`，不是 `\n\n`。
3. **剥掉 prompt 里字面量的 `<image>` / `<video>`** —— 训练时 LF 是就地替换的。

每个入口启动时都会跑一次 **preflight**（拿 `/tokenize` 看服务端真正渲染出来的 token），失配直接退出：

| 家族 | preflight 查什么 | 查不到、只能靠 server 脚本保证的 |
|---|---|---|
| qwen3_5 | 渲染结果里**不能**出现 `think` token（官方模板即使 `enable_thinking=false` 也插空 think 块，差 4 token） | — |
| gemma4 | 必须有 `<\|think\|>` system turn；generation prompt 必须以空 thought 段 `<channel\|>` 结尾 | — |
| internvl | 必须有 system turn（LF 注入的书生·万象那句 ~31 token）；`</img>` 之后**不能**是换行（官方 jinja / `content-format=string` 都会多插一个 `\n`） | `--hf-overrides` 强制 untie lm_head（只能靠 `finish_reason` 恒为 `length` 事后告警）、`--mm-processor-kwargs` 把归一化拉回 ImageNet（HTTP 完全查不到） |

> 服务端配置一律以 `scripts/<family>/eval/start_vllm_server*.sh` 为准，**不要改那些文件**。

## 用法

```bash
cd /workspace1/zhijun/LlamaFactory
```

### Qwen3.5（:8109 = 9B / :8102 = 2B / :8108 = 0.8B）

```bash
source .venv/bin/activate

# image 布局 + logprobs
python scripts/qwen3_5/eval/infer.py eval \
  --api-url http://localhost:8109 --model mix_22_27_v3_9 \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs

# video 槽位（只有 mix_22_27_v3_9_video 是这么训的；mp4 fps 必须 = 训练 yaml 的 video_fps=2.0）
python scripts/qwen3_5/eval/infer.py eval \
  --api-url http://localhost:8109 --model mix_22_27_v3_9_video \
  -e data/agentrobot/ood_sample/v3/rollout_lite_video.json -n 50 --logprobs

# 2B / 0.8B
python scripts/qwen3_5/eval/infer.py eval --api-url http://localhost:8102 --model mix_22_27_v3_2 \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs
python scripts/qwen3_5/eval/infer.py eval --api-url http://localhost:8108 --model mix_22-06_fk-pp_02_08 \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs
```

### Gemma-4（:8104 = E4B）

```bash
source .venv-gemma4/bin/activate       # 纯 HTTP 客户端，用 .venv 也行
export DISABLE_VERSION_CHECK=1

python scripts/gemma4/eval/infer.py eval \
  --api-url http://localhost:8104 --model gemma4_e4b_mix_22_27_v3 \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs
```

### InternVL3.5（:8202 = 2B / :8201 = 1B）

```bash
source .venv/bin/activate

python scripts/internvl/eval/infer.py eval \
  --api-url http://localhost:8202 --model internvl3.5-2b \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs

python scripts/internvl/eval/infer.py eval \
  --api-url http://localhost:8201 --model internvl3.5-1b \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs
```

### 其它子命令

```bash
python scripts/<family>/eval/infer.py single "描述图中场景" --image a.png --image b.png
python scripts/<family>/eval/infer.py tokens --model <lora>     # 只打印 9 个 action 的切分
python scripts/<family>/eval/infer.py eval ... --raw            # 看完整原始回复
python scripts/<family>/eval/infer.py eval ... --no-stage       # 消融：删掉 'Stage:' 行（只有 v0 有）
```

`--api-url` / `--model` / `--video-fps` 也可以用 `API_URL` / `MODEL_NAME` / `VIDEO_FPS` 环境变量给
（旧脚本兼容），子命令前后都能写。

## `--logprobs` 输出

请求每步 `logprobs=true, top_logprobs=20`，默认 `--max-tokens 8`（gemma4 的 `MV_FWD` 要 4 个 token，
别往下调）。输出两个文件，默认落在 `results/logprobs/`：

* `<family>__<lora>__<evalset>__<ts>.jsonl` —— 每条样本一行
* `<...>_summary.json` —— `meta` / `overall` / `per_label` / `confusion`

### 为什么不能用"首 token 概率"

9 个 action 都被切成多个 sub-token，**首 token 只能区分 4 类**（`MV` / `GR` / `RELEASE` / `DONE`），
六个方向全在后面的 token 上：

| action | Qwen3.5（0.8B/2B/9B） | InternVL3.5（1B/2B） | gemma4-E4B |
|---|---|---|---|
| `MV_FWD` | `MV` `_FWD` | `MV` `_FWD` | `MV` `_` `FW` `D` |
| `MV_BACK` / `_LEFT` / `_RIGHT` / `_UP` / `_DOWN` | `MV` `_XXX` | `MV` `_XXX` | `MV` `_` `XXX` |
| `GRASP` | `GR` `ASP` | `GR` `ASP` | `GR` `ASP` |
| `RELEASE` | `RELEASE` | `RELEASE` | `RELEASE` |
| `DONE` | `DONE` | `DONE` | `DONE` |

所以 `action_probs` 是**逐 token top-20 前缀累乘**出来的整串概率，不是首 token 概率。
每次跑 `--logprobs` 都会现场用 `/tokenize` 重查一遍切分并写进 `summary.meta.action_token_split`。

### JSONL 字段

| 字段 | 含义 |
|---|---|
| `sample_idx` / `sample_ord` | 在原评测集里的下标 / 本次运行的顺序 |
| `label` / `pred_token` / `pred_text` / `correct` | 真值 / 解析出的预测 / 原始回复 / 是否命中 |
| `finish_reason` | `stop` / `length`（全是 `length` = InternVL 的 untie-lm_head 没配对） |
| `pred_prob` | **P(整串预测 action)**，前缀累乘；不精确时为 `null` |
| `pred_prob_upper_bound` | 同上，但**永远有值**（不精确时是上界）—— 画图/统计用这个 |
| `pred_first_token_prob` | 第一个生成 token 的概率 |
| `top1_token` / `top1_prob` / `top2_token` / `top2_prob` / `margin` | **首 token 层面**的 top-2 与差值 |
| `entropy` | 首 token top-20 上的熵（nats，忽略 top-20 之外的质量） |
| `top20` | 首 token 的 `[[token, prob], …]`（20 条） |
| `gen_tokens` | 实际生成路径 `[[token, prob], …]`（含 EOS） |
| `action_probs` | `{9 个 action: prob \| null}`，只有 `exact` 时给数 |
| `action_probs_upper_bound` | `{9 个 action: prob}`，**永远有值**，是上界 |
| `action_prob_status` | `exact` / `branch_upper` / `topk_upper` / `len_upper`，见下 |
| `action_top1` / `action_top1_prob` / `action_top2` / `action_top2_prob` / `action_margin` | **action 层面**的排序（按上界） |
| `label_prob` / `label_prob_upper_bound` / `label_prob_exact` | P(真值 action) |
| `first_token_probs` / `first_token_probs_norm` | 首 token 四分类 `{MV, GR, RELEASE, DONE}` 的原始 / 归一化概率 —— "事件 vs 移动"判别信号 |
| `first_token_legal_mass` | 首 token 落在这 4 个合法前缀上的概率（**精确**） |
| `legal_mass` | 9 个 action 概率（exact 或上界）之和 —— 上界，用来看有多少质量跑到非法 token 上 |

`action_prob_status` 的四种情况（`action_probability()` 的 docstring 有完整推导）：

* `exact` —— 候选串一路沿实际生成前缀走到底（含"只在最后一个 token 上岔开"），值精确。
* `branch_upper` —— 中途岔开实际路径且串还没走完（如实际吐 `MV_*`、问 `GRASP`：知道 P(`GR`)，
  但 P(`ASP`|`GR`) 要重新前向才知道）。`action_probs` 给 `null`，上界 = 已知前缀概率。
  实测 `GR`→`ASP` ≈ 0.9999，这个上界很紧。
* `topk_upper` —— 某步需要的 token 掉出 top-20。上界 = 已知前缀概率 × 该步 top-20 最小值。
* `len_upper` —— 生成步数不够（`--max-tokens` 太小）。

> **建议**：做统计一律用 `*_upper_bound`（永远有值、单调），需要"严格精确"时再用
> `action_prob_status == "exact"` 过滤。

---

## LoRA ↔ 训练 prompt 版本 映射表 ★用错版本评出来的置信度全是废的

`ood_sample` 里的 prompt 版本：

| 版本 | 文件 | 特征 | 对应的训练 prompt |
|---|---|---|---|
| v0 | `v0/rollout_lite.json`（49 条） | `Image 1 (agentview)` 表述；`input` 里有 `Stage:` / `Gripper now:` / `Recent moves`；**无 DONE** | `mvtoken_0622_v0`（只在 27B 上训过，当前无 server） |
| v1 | `v1/rollout_affordance.json`（50） | `TARGET:` / `AFFORD:` 行 | 与 `mvtoken_0622_v1_affordance`（`Grasp target:` / `Grasp point:`）**并不逐字相同**，当前也没有对应 LoRA 在线 |
| v2 | `v2/rollout_lite.json`（50） | 极简 prompt（`BAsed on two camera views`） | `mvtoken_0622_v2_lite`（当前无 server） |
| **v3** | `v3/rollout_lite.json`（50） | `Agentview: overhead view` + `Recent moves` + DONE | **`mix_22_27_v3` / `mix_22_27_04_v3` / `mix_22-06_fk-pp_02*` / `ms_0717/*` 全都是这一版** |
| v3-video | `v3/rollout_lite_video.json`（50） | 同 v3，但走 `<video>` 槽位 | `mix_22_27_v3_lite_video` |

> ood_sample 的任务是 **"pick up the white cup and place it on the green coaster"**，
> 在上面任何训练集的任务列表里都没有出现过（也没有任何训练样本引用 `ood_sample/` 路径）——
> 是真 OOD。

### 表

| LoRA 名 | 端口 | base model | 训练数据集（dataset_info key） | 训练 prompt | 该配哪个 ood_sample | 适用？ |
|---|---|---|---|---|---|---|
| `mix_22_27_v3_9` | 8109 | Qwen3.5-9B | `mix_22_27_v3_lite`（4132，franka 0622+0627+0704） | v3 | `v3/rollout_lite.json` | ✅ |
| `mix_22_27_04_v3_9` | 8109 | Qwen3.5-9B | `mix_22_27_04_v3_lite`（5070，+ 0704 更多任务） | v3 | `v3/rollout_lite.json` | ✅ |
| `mix_22-06_fk-pp_02` | 8109 | Qwen3.5-9B | `mix_22-06_fk-pp_02_exchange_token`（7933，franka + piper，piper 的 FWD/BACK 已对调） | v3 | `v3/rollout_lite.json` | ✅（ood 是 franka 视角，直接评） |
| `mix_22_27_v3_9_video` | 8109 | Qwen3.5-9B | `mix_22_27_v3_lite_video`（4132，`<video>` 槽位，`video_fps: 2.0`） | v3 | `v3/rollout_lite_video.json` | ✅（**必须**用 video 那份，且 fps=2.0） |
| `piper_0705_v4_9` | 8109 | Qwen3.5-9B | `piper_0705_v4_lite`（2347，piper 单臂） | v4-piper（`Agentview: first-person view from the robot`） | — | ❌ 两处失配：prompt 的 Agentview 描述行不同；且 piper 是 ego 视角、FWD/BACK 与 franka 相反，拿 franka 的 ood_sample 评它监督是反的 |
| `dual_cloth_twice` / `once` / `chain` | 8109 | Qwen3.5-9B | `dual_cloth_v4_twice`(4364) / `once`(2182) / `chain`(2182, sharegpt) | dual-arm v4（3 图、多一个 `STILL` token、一次出两个 token） | — | ❌ 任务是双臂折衣、3 张图、契约完全不同 |
| `mix_22_27_v3_2` | 8102 | Qwen3.5-2B | `mix_22_27_v3_lite`（4132） | v3 | `v3/rollout_lite.json` | ✅ |
| `mix_22-06_fk-pp_02_2` | 8102 | Qwen3.5-2B | `mix_22-06_fk-pp_02_exchange_token`（7933） | v3 | `v3/rollout_lite.json` | ✅ |
| `zechen_repro` | 8102 | Qwen3.5-2B | **存疑**：`training_args.bin` 的 `output_dir` 写 `mix_22_27_v3_zechen_reproduce`，但 9920 步 ÷ 40 epoch ÷ (4×4×2) ⇒ ~7933 条，与 `mix_22-06_fk-pp_02_exchange_token` 吻合、与 `mix_22_27_v3` 的 4132 条对不上 | v3（两个候选都是 v3 body，**这一点不存疑**） | `v3/rollout_lite.json` | ✅ |
| `mix_22-06_fk-pp_02_08` | 8108 | Qwen3.5-0.8B | `mix_22-06_fk-pp_02_exchange_token`（7933） | v3 | `v3/rollout_lite.json` | ✅ |
| `gemma4_e4b_mix_22_27_v3` | 8104 | gemma4-E4B-it | `mix_22_27_v3_lite`（4132） | v3 | `v3/rollout_lite.json` | ✅ |
| `internvl3.5-2b` | 8202 | InternVL3_5-2B-HF | `mix_22-06_fk-pp_02_exchange_token`（7933） | v3 | `v3/rollout_lite.json` | ✅ |
| `internvl3.5-1b` | 8201 | InternVL3_5-1B-HF | `mix_22-06_fk-pp_02_exchange_token`（7933） | v3 | `v3/rollout_lite.json` | ✅ |
| `internvl3.5-2b-History2` | 8202 | InternVL3_5-2B-HF | `mix_22-06_fk-pp_02_exchange_token_history2` | v3 body + **history 前缀**（`AgentRobot/prompts/v3/history2_mvtoken_generator_lite.txt`），**6 张图**（3 时刻 × agentview/wrist，interleaved） | — | ❌ ood_sample 每条只有 2 张图、也没有 history 前缀；要评必须先造 history 版评测集 |
| `internvl3.5-2b-History2-PlainPrompt` | 8202 | InternVL3_5-2B-HF | `..._history2_plain` | plain v3 prompt + **6 张图**（interleaved） | — | ❌ 同上（图数不对） |
| `internvl3.5-2b-History2-VideoSlot` | 8202 | InternVL3_5-2B-HF | `..._history2_plain_video` | plain v3 prompt + **两个 `<video>` part**（agentview 3 帧 + wrist 3 帧） | — | ❌ 与 ood_sample 的"单 `<video>` 双帧"布局不是一回事 |
| `internvl3.5-2b-ms0717_blockpap` | 8202 | InternVL3_5-2B-HF | `ms0717_blockpap_oracle_wide`（5338） | v3 body，但任务是 *pick up the orange block and place it on the coaster* | — | ❌ ManiSkill 仿真数据、任务/画风都不同 |
| `internvl3.5-2b-ms0717_blockpap_follow` | 8202 | InternVL3_5-2B-HF | `ms0717_blockpap_follow`（5826） | 同上 | — | ❌ 同上 |
| `internvl3.5-2b-ms0717_stackcube_follow` | 8202 | InternVL3_5-2B-HF | `ms0717_stackcube_follow`（4185） | v3 body，任务是 *stack the red cube on top of the green cube* | — | ❌ 同上 |

**证据来源**：`data/dataset_info.json`、`examples/train_lora/*/*.yaml`、
`scripts/*/eval/start_vllm_server*.sh` 的 `--lora-modules`、
各 `saves/<...>/{trainer_state.json,training_args.bin,adapter_config.json}`
（zechen 的四个 InternVL + 一个 Qwen 在 `/workspace1/zechen/finetune/lora/*`）。
`global_step ÷ epochs × (per_device_bs × grad_accum × world_size)` 与各数据集条数逐一对得上，
可作为"这个 adapter 到底吃了哪份数据"的独立佐证。

**一句话结论**：当前在线的 21 个 LoRA 里，**能直接用 ood_sample 评的是 11 个**：

* 配 `v3/rollout_lite.json`（image，10 个）：`mix_22_27_v3_9`、`mix_22_27_04_v3_9`、
  `mix_22-06_fk-pp_02`、`mix_22_27_v3_2`、`mix_22-06_fk-pp_02_2`、`zechen_repro`、
  `mix_22-06_fk-pp_02_08`、`gemma4_e4b_mix_22_27_v3`、`internvl3.5-2b`、`internvl3.5-1b`
* 配 `v3/rollout_lite_video.json`（video，1 个）：`mix_22_27_v3_9_video`

其余 10 个（`piper_0705_v4_9` / `dual_cloth_*`×3 / `internvl3.5-2b-History2*`×3 /
`internvl3.5-2b-ms0717_*`×3）要么视角/任务不同、要么图数或槽位契约不同，**不适用**。
