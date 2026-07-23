# reportB — MVTOKEN 三家族推理评测代码统一 + logprobs

日期 2026-07-22 ｜ 仓库 `/workspace1/zhijun/LlamaFactory`

---

## 1. `eval_mvtoken.py` vs `infer.py` 差异分析与处置结论

### 1.1 差异（已逐条验证，与你的初步比对一致，并补充两条）

| # | 差异 | `scripts/qwen3_5/eval/infer.py`（新，312 行） | `scripts/qwen3_5/eval/eval_mvtoken.py`（旧，214 行） | 判定 |
|---|---|---|---|---|
| 1 | `<video>` 槽位 | 支持（2 图现场编无损 mp4 → `video_url`） | 不支持，只有 image | infer 胜 |
| 2 | 媒体 / 文本顺序 | 媒体在前（`content = [image…, text]`） | **文本在前**（`content = [text, image…]`） | eval_mvtoken **训推失配** |
| 3 | instruction/input 连接符 | `f"{instruction}\n{input}"`（单 `\n`，与 LF alpaca converter `"\n".join` 一致） | **`\n\n`** | eval_mvtoken **训推失配** |
| 4 | prompt-parity 自检 | 有（`/tokenize` 探 `<think>`，失配 `sys.exit`） | 无 | infer 胜 |
| 5 | `--no-stage` 消融 | 无 | 有 | 已搬进新核心 |
| 6 | `chat_template_kwargs.enable_thinking` | 无 | 有 | **应删**：挂了 LF 对齐模板后该参数无意义，留着反而诱导人以为可以不挂模板 |
| 7 | `VALID_TOKENS` | 9 个（含 `DONE`） | **8 个（缺 `DONE`）** | eval_mvtoken 会把所有 `DONE` 预测判成"解析失败" |
| 8 | 默认配置 | `:8109` / `mix_22_27_v3_9` / ood v3 video | `:8101` / `MVTOKEN` / `data/robot_rollout.json` | eval_mvtoken 全部过期（8101 无 server、该 json 不存在） |
| 9 | *(补)* dataset 常量 | 相对路径 | 绝对路径 `/workspace1/zhijun/...` | 不可移植 |
| 10 | *(补)* `--raw` 分支 | 统一 | `context = sample["input"]`（不走 `.get`），`input` 缺失即 KeyError | 对 ood v1/v2/v3（`input` 为 `""`）能跑但对无 `input` 键的数据集会崩 |

### 1.2 引用扫描（删除前的尽调）

```
grep -rn "eval_mvtoken" /workspace1/zhijun/LlamaFactory   # 排除 saves/
  .ai/espresso/ShowRobot-VLM_HANDOFF.md:251   （只是文档里吐槽它 DATASET 路径不存在）
  scripts/qwen3_5/eval/test_mvtoken.sh:37     python scripts/eval/eval_mvtoken.py …
grep -rn "eval_mvtoken" /workspace1/zhijun/AgentRobot     # 0 命中
```

* `scripts/eval/` 这个目录**根本不存在**（`ls: cannot access 'scripts/eval'`），所以
  `test_mvtoken.sh` 里那行命令**早就是坏的**；`check_prompt_parity.sh` 完全没有引用它。
* AgentRobot 侧只有注释性引用（`vlm/mvtoken_client.py`、`vlm/mvtoken_roles.py` 的 docstring 提到
  "mirrors LlamaFactory/scripts/eval/infer.py"），不是代码依赖。

### 1.3 结论

**你的判断成立，已执行**：删除 `scripts/qwen3_5/eval/eval_mvtoken.py`（原文件已备份到
`<scratchpad>/backup_orig/qwen3_5_eval_mvtoken.py`），`--no-stage` 搬进共享核心并**加强**
（原版只扫 `input`，新版对 instruction+input 合并后的整段扫，因为 ood v0 的 `Stage:` 在 `input`
里、别的数据集可能在 instruction 里）。`enable_thinking` 开关不保留。

---

## 2. 合并方案与文件清单

**一个共享核心 + 每家族一个薄入口。** 薄入口只声明一个 `FamilySpec`（默认端口 / 默认 LoRA /
要不要发 system / video 契约 / preflight 自检），别的全部复用核心，从结构上杜绝"同样的逻辑写三遍
各自漂移"。

| 文件 | 动作 | 行数 | 说明 |
|---|---|---|---|
| `scripts/eval_common/mvtoken_client.py` | **新增** | 683 | 核心：HTTP / 媒体编码 / prompt 拼接 / 评测循环 / **logprobs 分析** / 统计 / CLI |
| `scripts/eval_common/README.md` | **新增** | 213 | 用法 + logprobs 字段表 + **LoRA↔prompt 映射表** |
| `scripts/qwen3_5/eval/infer.py` | 重写 | 312 → 100 | 只留 Qwen3.5 硬约束 + preflight |
| `scripts/gemma4/eval/infer.py` | 重写 | 224 → 101 | 只留 gemma4 硬约束 + preflight |
| `scripts/internvl/eval/infer.py` | **新增** | 143 | 之前没有；InternVL 硬约束 + 两级 preflight |
| `scripts/qwen3_5/eval/eval_mvtoken.py` | **删除** | −214 | 见 §1（已备份） |
| `scripts/qwen3_5/eval/test_mvtoken.sh` | 改 | — | 原来指向不存在的 `scripts/eval/*.py`、`:8101`、`mvtoken_0622_v2`、`scripts/eval/ood_sample/`，全部过期；改成 `:8109` / `mix_22_27_v3_9` / ood v3 / `--logprobs` |
| `scripts/gemma4/eval/run_eval.sh` | 改 | — | `--evalset` → `-e`（eval 子命令现在必须显式给），默认命令改成 `--logprobs`，补 prompt 版本警告 |

**没动**：任何 `start_vllm_server*.sh`、`check_prompt_parity.sh`、`*.jinja`、任何 server 进程。

净减少约 350 行重复代码，同时新增了 logprobs + InternVL 入口。

### 各家族保留的硬约束（原样保留 + 注释）

| 家族 | 约束 | 落点 |
|---|---|---|
| Qwen3.5 | server 必须挂 `chat_template_qwen3_5_lf.jinja`（官方模板即使 `enable_thinking=false` 也插空 think 块，差 4 token） | `preflight()`：`/tokenize` 纯文本探针，出现 `think` 直接 `sys.exit` |
| Qwen3.5 | video 布局的 mp4 fps **必须等于**训练 yaml 的 `video_fps`（默认 2.0） | `SPEC.video_fps=2.0` + `encode_frames_as_video()` 注释；`--video-fps` / `VIDEO_FPS` 可覆盖 |
| gemma4 | server 必须挂 `chat_template_gemma4n_lf.jinja`；训练侧无条件注入 `default_system`，**vLLM 不会自动补**，客户端必须发 | `SPEC.system_prompt="You are a helpful assistant."` + `preflight()` 双检：必须有 `<\|think\|>`、generation prompt 必须以 `<channel\|>`（空 thought 段）结尾 |
| gemma4 | action 切分碎（`MV_FWD` = 4 token），`--max-tokens` 不能低于 8 | `SPEC.notes` + `--max-tokens` help |
| InternVL3.5 | server 必须 `--chat-template chat_template_internvl_lf.jinja` + `--chat-template-content-format openai` | `preflight()` (a) 首两个 token 必须是 `<\|im_start\|> system`（LF 注入的书生·万象 ~31 token）；(b) 2 图探针，`</img>` 之后**不能**是换行（官方 jinja / string content-format 都会多插 `\n`）——这一步同时盖住两种失配 |
| InternVL3.5 | `--hf-overrides` 强制 untie lm_head | **HTTP 查不到**。用行为特征兜底：跑完若 `finish_reason` 全是 `length` 就告警（该 bug 下模型永远吐不出 `<\|im_end\|>`） |
| InternVL3.5 | `--mm-processor-kwargs` 把归一化拉回 ImageNet | **HTTP 完全查不到**，只能靠"server 是 `start_vllm_server.sh` 起的"保证；已写进入口 docstring + `SPEC.notes` 每次运行打印 |
| 三家通用 | 媒体在文本之前 / `instruction` + 单 `\n` + `input` / 剥掉字面量 `<image>`·`<video>` | `MvTokenClient.build_messages()`、`_user_text()`，核心里只有一份 |

---

## 3. action token 切分证据（`/tokenize` 实测，2026-07-22）

命令：`POST /tokenize {"model": <lora>, "prompt": "<action>", "add_special_tokens": false, "return_token_strs": true}`

| action | Qwen3.5-9B(:8109) / 2B(:8102) / 0.8B(:8108) | InternVL3.5-2B(:8202) / 1B(:8201) | gemma4-E4B(:8104) |
|---|---|---|---|
| `MV_FWD` | `['MV','_FWD']` | `['MV','_FWD']` | **`['MV','_','FW','D']`** |
| `MV_BACK` | `['MV','_BACK']` | `['MV','_BACK']` | `['MV','_','BACK']` |
| `MV_LEFT` | `['MV','_LEFT']` | `['MV','_LEFT']` | `['MV','_','LEFT']` |
| `MV_RIGHT` | `['MV','_RIGHT']` | `['MV','_RIGHT']` | `['MV','_','RIGHT']` |
| `MV_UP` | `['MV','_UP']` | `['MV','_UP']` | `['MV','_','UP']` |
| `MV_DOWN` | `['MV','_DOWN']` | `['MV','_DOWN']` | `['MV','_','DOWN']` |
| `GRASP` | `['GR','ASP']` | `['GR','ASP']` | `['GR','ASP']` |
| `RELEASE` | `['RELEASE']` | `['RELEASE']` | `['RELEASE']` |
| `DONE` | `['DONE']` | `['DONE']` | `['DONE']` |

**结论：三家族的首 token 都只能区分 4 类 `{MV, GR, RELEASE, DONE}`，六个方向全在后面的 token 上；
gemma4 更糟 —— 它的第 2 个 token 恒为 `_`（零信息），方向在第 3 个 token 上，`MV_FWD` 还要第 4 个。
所以 `action_probs` 一律是「逐 token top-20 前缀累乘」出来的整串概率，绝不是首 token 概率。**

运行时实证（`gen_tokens` 字段，qwen3.5-9B 第 2 条）：`[["MV",1.0],["_FWD",0.99981],["<|im_end|>",1.0]]`；
gemma4 同一条：`[["MV",0.99998],["_",0.99999],["DOWN",0.88079],["<turn|>",1.0]]`。

每次 `--logprobs` 都会现场重查一遍切分并写进 `summary.meta.action_token_split`，不写死。

### 前缀累乘的精确性分级（`action_prob_status`）

一次前向只能拿到「在**实际生成前缀**条件下」每步的 top-20，所以：

| status | 含义 | `action_probs` | `action_probs_upper_bound` |
|---|---|---|---|
| `exact` | 候选串一路沿实际生成前缀走到底（含"只在最后一个 token 上岔开"，用的仍是同一条件分布） | 精确值 | = 精确值 |
| `branch_upper` | 中途岔开且串没走完（如实际吐 `MV_*`、问 `GRASP`：知道 P(`GR`)，但 P(`ASP`\|`GR`) 要重新前向）| `null` | 已知前缀概率（实测 `GR`→`ASP`≈0.9999，很紧） |
| `topk_upper` | 某步需要的 token 掉出 top-20 | `null` | 前缀概率 × 该步 top-20 最小值 |
| `len_upper` | 生成步数不够（`--max-tokens` 太小） | `null` | 已知前缀概率 |

**给下游的建议：统计一律用 `*_upper_bound`（永远有值、单调），要"严格精确"再按
`action_prob_status == "exact"` 过滤。** 实测同一 label 的 6 个方向几乎总是 `exact`
（它们共享首 token `MV`），跨类的（`GRASP`/`RELEASE`/`DONE`）多半是 `topk_upper`
且量级 ≤1e-8，对分析无影响。

---

## 4. LoRA ↔ 训练 prompt 版本 映射表 ★

（同一份表已写进 `scripts/eval_common/README.md`，那份是给人用的权威副本。）

### 4.1 ood_sample 各版本对应哪一版训练 prompt

| 版本 | 文件 | 特征 | 对应训练 prompt |
|---|---|---|---|
| v0 | `v0/rollout_lite.json`（49） | `Image 1 (agentview)`；`input` 里有 `Stage:`/`Gripper now:`/`Recent moves`；**无 DONE** | `mvtoken_0622_v0`（只在 27B 训过，**当前无 server**） |
| v1 | `v1/rollout_affordance.json`（50） | `TARGET:` / `AFFORD:` | 与 `mvtoken_0622_v1_affordance`（`Grasp target:`/`Grasp point:`）**并非逐字相同**；当前无对应 LoRA 在线 |
| v2 | `v2/rollout_lite.json`（50） | 极简（`BAsed on two camera views`） | `mvtoken_0622_v2_lite`（**当前无 server**） |
| **v3** | `v3/rollout_lite.json`（50） | `Agentview: overhead view` + `Recent moves` + DONE | **当前在线的所有 pick-place LoRA 都是这一版** |
| v3-video | `v3/rollout_lite_video.json`（50） | 同 v3 但 `<video>` 槽位 | `mix_22_27_v3_lite_video` |

ood_sample 的任务是 **"pick up the white cup and place it on the green coaster"**，
不在任何训练集的任务清单里，也没有任何训练样本引用 `ood_sample/` 路径（`grep -c ood_sample` = 0）——真 OOD。

### 4.2 主表

| LoRA 名 | 端口 | base model | 训练数据集 | 训练 prompt | 该配哪个 ood_sample | 适用？ |
|---|---|---|---|---|---|---|
| `mix_22_27_v3_9` | 8109 | Qwen3.5-9B | `mix_22_27_v3_lite`（4132，franka） | v3 | `v3/rollout_lite.json` | ✅ |
| `mix_22_27_04_v3_9` | 8109 | Qwen3.5-9B | `mix_22_27_04_v3_lite`（5070） | v3 | `v3/rollout_lite.json` | ✅ |
| `mix_22-06_fk-pp_02` | 8109 | Qwen3.5-9B | `mix_22-06_fk-pp_02_exchange_token`（7933，franka+piper，piper 的 FWD/BACK 已对调） | v3 | `v3/rollout_lite.json` | ✅（ood 是 franka 视角，直接评） |
| `mix_22_27_v3_9_video` | 8109 | Qwen3.5-9B | `mix_22_27_v3_lite_video`（4132，`video_fps: 2.0`） | v3 | **`v3/rollout_lite_video.json`** | ✅（必须用 video 那份，fps=2.0） |
| `piper_0705_v4_9` | 8109 | Qwen3.5-9B | `piper_0705_v4_lite`（2347，piper 单臂） | **v4-piper**（`Agentview: first-person view from the robot`） | — | ❌ 双重失配：prompt 的 Agentview 描述行不同；piper 是 ego 视角、FWD/BACK 与 franka 相反，拿 franka 的 ood_sample 评它监督是反的 |
| `dual_cloth_twice` / `once` / `chain` | 8109 | Qwen3.5-9B | `dual_cloth_v4_twice`(4364) / `once`(2182) / `chain`(2182, sharegpt) | dual-arm v4（3 图、多一个 `STILL`、一次出两个 token） | — | ❌ 双臂折衣，图数/token 集/契约都不同 |
| `mix_22_27_v3_2` | 8102 | Qwen3.5-2B | `mix_22_27_v3_lite`（4132） | v3 | `v3/rollout_lite.json` | ✅ |
| `mix_22-06_fk-pp_02_2` | 8102 | Qwen3.5-2B | `mix_22-06_fk-pp_02_exchange_token`（7933） | v3 | `v3/rollout_lite.json` | ✅ |
| `zechen_repro` | 8102 | Qwen3.5-2B | **存疑**（见 §4.4） | v3（**这一点不存疑**） | `v3/rollout_lite.json` | ✅ |
| `mix_22-06_fk-pp_02_08` | 8108 | Qwen3.5-0.8B | `mix_22-06_fk-pp_02_exchange_token`（7933） | v3 | `v3/rollout_lite.json` | ✅ |
| `gemma4_e4b_mix_22_27_v3` | 8104 | gemma4-E4B-it | `mix_22_27_v3_lite`（4132） | v3 | `v3/rollout_lite.json` | ✅ |
| `internvl3.5-2b` | 8202 | InternVL3_5-2B-HF | `mix_22-06_fk-pp_02_exchange_token`（7933） | v3 | `v3/rollout_lite.json` | ✅ |
| `internvl3.5-1b` | 8201 | InternVL3_5-1B-HF | `mix_22-06_fk-pp_02_exchange_token`（7933） | v3 | `v3/rollout_lite.json` | ✅ |
| `internvl3.5-2b-History2` | 8202 | InternVL3_5-2B-HF | `mix_22-06_fk-pp_02_exchange_token_history2` | v3 body + **history 前缀**（`AgentRobot/prompts/v3/history2_mvtoken_generator_lite.txt`）+ **6 张图**（3 时刻×agentview/wrist，interleaved） | — | ❌ ood_sample 每条只有 2 图、也没有前缀；要评必须先造 history 版评测集 |
| `internvl3.5-2b-History2-PlainPrompt` | 8202 | InternVL3_5-2B-HF | `..._history2_plain` | plain v3 + **6 张图**（interleaved） | — | ❌ 图数不对（prompt 对得上） |
| `internvl3.5-2b-History2-VideoSlot` | 8202 | InternVL3_5-2B-HF | `..._history2_plain_video` | plain v3 + **两个 `<video>` part**（agentview 3 帧 + wrist 3 帧，`data:video/jpeg`） | — | ❌ 与 ood_sample 的"单 `<video>` 双帧"不是一回事 |
| `internvl3.5-2b-ms0717_blockpap` | 8202 | InternVL3_5-2B-HF | `ms0717_blockpap_oracle_wide`（5338） | v3 body，任务 *pick up the orange block and place it on the coaster* | — | ❌ ManiSkill 仿真 |
| `internvl3.5-2b-ms0717_blockpap_follow` | 8202 | InternVL3_5-2B-HF | `ms0717_blockpap_follow`（5826） | 同上 | — | ❌ ManiSkill 仿真 |
| `internvl3.5-2b-ms0717_stackcube_follow` | 8202 | InternVL3_5-2B-HF | `ms0717_stackcube_follow`（4185） | v3 body，任务 *stack the red cube on top of the green cube* | — | ❌ ManiSkill 仿真 |

**合计 21 个在线 LoRA → 11 个适用**（10 个配 `v3/rollout_lite.json`，1 个配 `v3/rollout_lite_video.json`），
10 个不适用。

### 4.3 证据链

1. `scripts/*/eval/start_vllm_server*.sh` 的 `--lora-modules` → LoRA 名 ↔ 磁盘目录。
2. `examples/train_lora/*/*.yaml` 的 `output_dir` + `dataset` → 目录 ↔ dataset key。
3. `data/dataset_info.json` → dataset key ↔ json 文件；直接读文件第一条拿到 prompt 文本，与
   ood_sample 各版本逐段比对。
4. `saves/<...>/training_args.bin`（`output_dir`/`per_device_train_batch_size`/`gradient_accumulation_steps`/`world_size`）
   + `trainer_state.json`（`global_step`/`num_train_epochs`）+ `adapter_config.json`（`base_model_name_or_path`）。
   zechen 的 5 个 adapter（`/workspace1/zechen/finetune/lora/*`）没有 yaml，但 `training_args.bin` 里
   有 `output_dir` / `run_name` 字符串，直接指明了数据集。
5. **独立佐证**：`global_step ÷ epochs × (per_device_bs × grad_accum × world_size)` ≈ 数据集条数。
   逐一对得上（eff_bs 一律 32 或 64）：

   | LoRA | steps/epochs | 推出的样本数 | 数据集实际条数 |
   |---|---|---|---|
   | `mix_22_27_v3_9` | 3900/30 | 130×32 = 4160 | 4132 ✓ |
   | `mix_22_27_04_v3_9` | 4770/30 | 159×32 = 5088 | 5070 ✓ |
   | `piper_0705_v4_9` | 1480/20 | 74×32 = 2368 | 2347 ✓ |
   | `mix_22-06_fk-pp_02` | 4960/20 | 248×32 = 7936 | 7933 ✓ |
   | `mix_22_27_v3_9_video` | 3900/30 | 4160 | 4132 ✓ |
   | `dual_cloth_twice` | 2740/20 | 137×32 = 4384 | 4364 ✓ |
   | `dual_cloth_once`/`chain` | 1380/20 | 69×32 = 2208 | 2182 ✓ |
   | `mix_22_27_v3_2` | 2600/40 | 65×64 = 4160 | 4132 ✓ |
   | `mix_22-06_fk-pp_02_2`/`_08` | 7440/30 | 248×32 = 7936 | 7933 ✓ |
   | `gemma4_e4b_mix_22_27_v3` | 3900/30 | 4160 | 4132 ✓ |
   | `internvl3.5-1b`/`-2b`/`-History2*` | 9920/40 | 248×32 = 7936 | 7933 ✓ |
   | `ms0717_blockpap` | 5010/30 | 167×32 = 5344 | 5338 ✓ |
   | `ms0717_blockpap_follow` | 3660/20 | 183×32 = 5856 | 5826 ✓ |
   | `ms0717_stackcube_follow` | 2620/20 | 131×32 = 4192 | 4185 ✓ |
6. InternVL History2 三兄弟各自的 (图序, prompt) 契约，来自
   `AgentRobot/run_real_mvtoken.sh` 的注释与 `AgentRobot/vlm/mvtoken_roles.py`：
   `History2` = interleaved + `history2_mvtoken_generator_lite.txt`；
   `History2-PlainPrompt` = interleaved + plain v3；`History2-VideoSlot` = video + plain v3。

### 4.4 唯一一处"存疑"：`zechen_repro`

* `training_args.bin` 的 `output_dir` = `saves/qwen3.5-2b/robot/mix_22_27_v3_zechen_reproduce`
  → 指向 `mix_22_27_v3`（4132 条）。
* 但 `global_step=9920`、`num_train_epochs=40`、`per_device_bs=4`、`grad_accum=4`、`world_size=2`
  ⇒ 248 步/epoch × eff_bs 32 = **7936 条**，与 `mix_22-06_fk-pp_02_exchange_token`（7933）吻合，
  与 `mix_22_27_v3` 的 4132（应为 130 步/epoch → 5200 步）**对不上**。
* **推断**：要么 zechen 侧有一份同名但内容是 fk-pp mix 的数据，要么 `output_dir` 是从模板抄的没改。
  没有更强证据，标"存疑"。
* **但对评测没有影响**：两个候选数据集的 prompt body **都是 v3**（`Agentview: overhead view` +
  `Recent moves` + DONE），所以 `zechen_repro` 配 `v3/rollout_lite.json` 是确定的。

---

## 5. 三家族用法命令（可直接复制粘贴）

```bash
cd /workspace1/zhijun/LlamaFactory
```

**Qwen3.5**（:8109=9B / :8102=2B / :8108=0.8B）
```bash
source .venv/bin/activate
python scripts/qwen3_5/eval/infer.py eval \
  --api-url http://localhost:8109 --model mix_22_27_v3_9 \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs

# video 槽位（只有 mix_22_27_v3_9_video）
python scripts/qwen3_5/eval/infer.py eval \
  --api-url http://localhost:8109 --model mix_22_27_v3_9_video \
  -e data/agentrobot/ood_sample/v3/rollout_lite_video.json -n 50 --logprobs
```

**Gemma-4**（:8104=E4B）
```bash
source .venv-gemma4/bin/activate     # 纯 HTTP 客户端，用 .venv 也行
export DISABLE_VERSION_CHECK=1
python scripts/gemma4/eval/infer.py eval \
  --api-url http://localhost:8104 --model gemma4_e4b_mix_22_27_v3 \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs
```

**InternVL3.5**（:8202=2B / :8201=1B）
```bash
source .venv/bin/activate
python scripts/internvl/eval/infer.py eval \
  --api-url http://localhost:8202 --model internvl3.5-2b \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs
python scripts/internvl/eval/infer.py eval \
  --api-url http://localhost:8201 --model internvl3.5-1b \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 50 --logprobs
```

**其它子命令**
```bash
python scripts/<family>/eval/infer.py single "描述图中场景" --image a.png --image b.png
python scripts/<family>/eval/infer.py tokens --model <lora>       # 只打印 9 个 action 的切分
python scripts/<family>/eval/infer.py eval ... --raw              # 完整原始回复
python scripts/<family>/eval/infer.py eval ... --no-stage         # 消融（只有 v0 有 Stage 行）
python scripts/<family>/eval/infer.py eval ... --logprobs-out X.jsonl   # 自定义输出
```

`--api-url` / `--model` / `--video-fps` 也可用 `API_URL` / `MODEL_NAME` / `VIDEO_FPS` 环境变量（旧脚本兼容），
子命令前后都能写。

### `--logprobs` 输出

默认落在 `results/logprobs/<family>__<lora>__<evalset>__<ts>.jsonl` 与同名 `_summary.json`。
请求 `logprobs=true, top_logprobs=20`，`--max-tokens` 默认 8。

JSONL 每行（qwen3.5-9B / `mix_22_27_v3_9` 第 2 条，真实输出，top20 截断展示）：

```json
{"sample_idx":7,"sample_ord":1,"label":"MV_DOWN","pred_token":"MV_FWD","pred_text":"MV_FWD",
 "correct":false,"finish_reason":"stop",
 "pred_prob":0.9998134722644053,"pred_prob_upper_bound":0.9998134722644053,"pred_first_token_prob":1.0,
 "top1_token":"MV","top1_prob":1.0,"top2_token":" MV","top2_prob":4.14e-08,"margin":0.99999996,
 "entropy":1.1122716681672224e-06,
 "top20":[["MV",1.0],[" MV",4.14e-08],["mv",5.60e-09],["MF",4.94e-09], "…共20条"],
 "gen_tokens":[["MV",1.0],["_FWD",0.9998134722644053],["<|im_end|>",1.0]],
 "action_probs":{"MV_FWD":0.9998134722644053,"MV_BACK":null,"MV_LEFT":null,"MV_RIGHT":null,
                 "MV_UP":2.2155348104951453e-08,"MV_DOWN":0.00017952664513906985,
                 "GRASP":null,"RELEASE":null,"DONE":null},
 "action_probs_upper_bound":{"MV_FWD":0.99981347,"MV_BACK":1.43e-08,"MV_LEFT":1.43e-08,
                 "MV_RIGHT":1.43e-08,"MV_UP":2.22e-08,"MV_DOWN":0.00017953,
                 "GRASP":1.03e-10,"RELEASE":1.03e-10,"DONE":1.03e-10},
 "action_prob_status":{"MV_FWD":"exact","MV_BACK":"topk_upper","MV_LEFT":"topk_upper",
                 "MV_RIGHT":"topk_upper","MV_UP":"exact","MV_DOWN":"exact",
                 "GRASP":"topk_upper","RELEASE":"topk_upper","DONE":"topk_upper"},
 "action_top1":"MV_FWD","action_top1_prob":0.99981347,
 "action_top2":"MV_DOWN","action_top2_prob":0.00017953,"action_margin":0.99963395,
 "label_prob":0.00017952664513906985,"label_prob_upper_bound":0.00017952664513906985,
 "label_prob_exact":true,
 "first_token_probs":{"DONE":0.0,"GR":0.0,"MV":1.0,"RELEASE":0.0},
 "first_token_probs_norm":{"DONE":0.0,"GR":0.0,"MV":1.0,"RELEASE":0.0},
 "first_token_legal_mass":1.0,"legal_mass":0.9999930642864524}
```

字段完整解释见 `scripts/eval_common/README.md`。要点：

* `action_probs` = **整串** action 的概率（前缀累乘），不精确时为 `null`；
  `action_probs_upper_bound` **永远有值**（上界），做统计用它。
* `first_token_probs` 是 `{MV, GR, RELEASE, DONE}` 四类的**精确**概率（用 `/tokenize` 查出的真首
  token 分组，不是"任意前缀匹配"——后者会把 `M` 和 `MV` 重复计数导致 `legal_mass > 1`）。
* `legal_mass` = 9 个 action 上界之和（上界）；`first_token_legal_mass` 是精确的合法首 token 质量。

summary JSON（gemma4 那次的真实输出，节选）：

```json
{"meta":{"family":"gemma4","api_url":"http://localhost:8104","model":"gemma4_e4b_mix_22_27_v3",
  "evalset":"data/agentrobot/ood_sample/v3/rollout_lite.json","n_samples":10,"n_total":50,"seed":42,
  "layout":"image","video_fps":null,"max_tokens":8,"no_stage":false,
  "timestamp":"2026-07-22 18:38:31",
  "action_token_split":{"MV_FWD":["MV","_","FW","D"],"MV_BACK":["MV","_","BACK"], "…":"…"}},
 "overall":{"n":10,"correct":9,"accuracy":0.9,"mean_pred_prob":0.9594441415781938,
  "mean_label_prob":0.8832866709851579,"mean_entropy":3.9413271176744346e-05,
  "mean_margin":0.9999959600655071,"mean_action_margin":0.9304842729620747,
  "mean_legal_mass":0.9884151475318428,"mean_first_token_legal_mass":0.9999965668173105},
 "per_label":{"MV_DOWN":{"n":3,"correct":3,"accuracy":1.0,"mean_pred_prob":0.9106496648264614,
   "mean_label_prob":0.9106496648264614,"mean_entropy":2.65e-06}, "…":"…"},
 "confusion":{"MV_FWD":{"MV_FWD":4,"MV_DOWN":1}, "…":"…"}}
```

---

## 6. 实跑验证记录（全部真跑，2026-07-22 18:2x–18:4x）

服务器一个都没 kill；6 个 server 全程在线。种子固定 42，取同一批 10 条（`v3/rollout_lite.json`，50 条里抽 10）。

### 6.1 三家族主验证（10 条 + logprobs）

**qwen3.5-9B / `mix_22_27_v3_9` / :8109 —— 9/10 = 90.0%**
```
[qwen3_5] prompt parity OK（mix_22_27_v3_9 @ http://localhost:8109，10 tok，无 think token）
Tokens  : MV_FWD n=2 ['MV','_FWD'] … 首 token 只能区分 4 类 ['DONE','GR','MV','RELEASE']
  #       Label        Pred   ok  P(pred)   P(label)      H
  1     MV_DOWN     MV_DOWN    ✓   0.9997    0.9997  0.000
  2     MV_DOWN      MV_FWD    ✗   0.9998    0.0002  0.000
  3     MV_LEFT     MV_LEFT    ✓   1.0000    1.0000  0.000
  4      MV_FWD      MV_FWD    ✓   0.9999    0.9999  0.000
  5      MV_FWD      MV_FWD    ✓   1.0000    1.0000  0.000
  6      MV_FWD      MV_FWD    ✓   0.9999    0.9999  0.000
  7     MV_DOWN     MV_DOWN    ✓   1.0000    1.0000  0.000
  8      MV_FWD      MV_FWD    ✓   1.0000    1.0000  0.000
  9    MV_RIGHT    MV_RIGHT    ✓   1.0000    1.0000  0.000
 10      MV_FWD      MV_FWD    ✓   0.9993    0.9993  0.000
Per-label: MV_DOWN 2/3=66.7% (meanP(pred)=0.9999) / MV_FWD 5/5=100% / MV_LEFT 1/1 / MV_RIGHT 1/1
Overall : 9/10 = 90.0%   meanP(pred)=0.9999  meanH=0.000  legal_mass=1.0000
```

**gemma4-E4B / `gemma4_e4b_mix_22_27_v3` / :8104 —— 9/10 = 90.0%**
```
[gemma4] prompt parity OK（… 27 tok，system+空 thought 齐全）
Tokens  : MV_FWD n=4 ['MV','_','FW','D'] / MV_BACK n=3 / GRASP n=2 / RELEASE n=1 / DONE n=1
  1     MV_DOWN     MV_DOWN    ✓   0.8512    0.8512  0.000
  2     MV_DOWN     MV_DOWN    ✓   0.8808    0.8808  0.000
  3     MV_LEFT     MV_LEFT    ✓   0.9972    0.9972  0.000
  4      MV_FWD      MV_FWD    ✓   0.9859    0.9859  0.000
  5      MV_FWD      MV_FWD    ✓   1.0000    1.0000  0.000
  6      MV_FWD     MV_DOWN    ✗   0.8808    0.1192  0.000   ← label 走 branch_upper（gen: MV/_/DOWN）
  7     MV_DOWN     MV_DOWN    ✓   1.0000    1.0000  0.000
  8      MV_FWD      MV_FWD    ✓   1.0000    1.0000  0.000
  9    MV_RIGHT    MV_RIGHT    ✓   1.0000    1.0000  0.000
 10      MV_FWD      MV_FWD    ✓   0.9986    0.9986  0.000
Overall : 9/10 = 90.0%   meanP(pred)=0.9594  mean legal_mass=0.9884
```

**InternVL3.5-2B / `internvl3.5-2b` / :8202 —— 8/10 = 80.0%**（prompt 版本 = v3，见 §4）
```
[internvl] prompt parity OK（internvl3.5-2b @ http://localhost:8202，system turn 在、图像块后无多余换行）
  1     MV_DOWN     MV_DOWN    ✓   1.0000    1.0000  0.000
  2     MV_DOWN      MV_FWD    ✗   0.8933    0.1067  0.000
  3     MV_LEFT     MV_LEFT    ✓   0.9999    0.9999  0.000
  4-9   （MV_FWD ×3、MV_DOWN、MV_FWD、MV_RIGHT 全对，P≈1.0000）
 10      MV_FWD     MV_DOWN    ✗   0.8670    0.1330  0.000
Overall : 8/10 = 80.0%   meanP(pred)=0.9760  legal_mass=1.0000  finish_reason 全为 stop（untie 正常）
```

### 6.2 其余 8 个"适用"LoRA 也全部跑通（同一批 10 条）

| LoRA | 端口 | acc | meanP(pred) |
|---|---|---|---|
| `mix_22_27_04_v3_9` | 8109 | 9/10 = 90.0% | 0.9986 |
| `mix_22-06_fk-pp_02` | 8109 | 9/10 = 90.0% | 0.9548 |
| `mix_22_27_v3_9_video`（v3-video，n=5） | 8109 | 4/5 = 80.0% | 0.9933 |
| `mix_22_27_v3_2` | 8102 | 7/10 = 70.0% | 0.9675 |
| `mix_22-06_fk-pp_02_2` | 8102 | 9/10 = 90.0% | 0.9844 |
| `zechen_repro` | 8102 | 8/10 = 80.0% | 0.9992 |
| `mix_22-06_fk-pp_02_08` | 8108 | 8/10 = 80.0% | 0.9429 |
| `internvl3.5-1b` | 8201 | 9/10 = 90.0% | 0.9982 |

### 6.3 其它路径验证

* `--no-stage` 消融（ood v0，qwen 9B，n=5）：正常跑通，4/5。
* InternVL + video 评测集 → 按设计**明确报错**而不是悄悄退化：
  `[fatal] internvl 没有可用的 <video> 契约，但样本带了 videos 字段。`
* `tokens` 子命令在 6 个 server 上都跑过（§3 的表就是它的输出）。
* 所有 logprobs 产物在 `results/logprobs/` 与 `<scratchpad>/runs/`。

### 6.4 代码质量

```
uvx ruff check  scripts/eval_common/mvtoken_client.py scripts/{qwen3_5,gemma4,internvl}/eval/infer.py
  -> All checks passed!
uvx ruff format scripts/eval_common/mvtoken_client.py scripts/{qwen3_5,gemma4,internvl}/eval/infer.py
  -> 2 files reformatted, 2 files left unchanged（已应用，之后重跑三家族验证仍全部通过）
python tests/check_license.py scripts/eval_common scripts/internvl scripts/gemma4/eval scripts/qwen3_5/eval
  -> exit 0（4 个文件的 Apache 2.0 头都在）
```

> 全仓 `make quality` / `make license` **本来就是红的**（`scripts/` 下有 57 个历史 ruff 错误、
> `scripts/hf_upload.py` 缺 license 头），与本次改动无关；我只保证自己新增/修改的 4 个 py 文件干净。

---

## 7. 风险与遗留

1. **⚠️ 我误跑了一次 `uv run python3 tests/check_license.py`，它触发了 `uv sync`，
   把 `.venv` 重装了一遍**（"Uninstalled 35 packages / Installed 53 packages"）。
   已逐项复查：`torch 2.13.0+cu130`（cuda 可用）、`transformers 5.6.0`、`numpy 2.5.1`、
   `fla-core 0.5.1`、`flash-linear-attention 0.5.1`、`tilelang 0.1.11`、`apache-tvm-ffi 0.1.11`、
   `deepspeed 0.18.4`、`av`、`vllm 0.11.0`、`llamafactory 0.9.6.dev0` **全部还在且可 import**。
   但 **`uv run` / `make license` / `make test` 会 sync 这个仓库的 `.venv`**，
   在这台机器上请一律用 `.venv/bin/python …` 或 `uvx`（uvx 是隔离的，不动项目 venv）。
2. **InternVL 的两条 server 约束（untie lm_head / ImageNet 归一化）HTTP 查不到。** 前者有事后
   告警（`finish_reason` 全 `length`），后者**完全无法自检** —— 只能靠"server 一定用
   `scripts/internvl/eval/start_vllm_server.sh` 启动"这个约定。
3. **`piper_0705_v4_9` / `dual_cloth_*` / `History2*` / `ms0717_*` 目前没有配套评测集。**
   要评它们需要分别造：piper 视角的 OOD rollout、双臂折衣 OOD、history2（6 图 + 前缀）版
   ood_sample、ManiSkill OOD。History2 的评测集最容易补：拿现成的
   `AgentRobot/prompts/v3/history2_mvtoken_generator_lite.txt` + `data/agentrobot/build_history_dataset.py`
   对 ood_sample 重跑一遍即可（图序契约是"旧→新、每帧 agentview→wrist、当前帧在最后"）。
4. **`ood_sample` 只有 49–50 条、且全是同一条 rollout_052。** 10 条抽样的 per-label 统计
   样本量极小（表里多处 1/1），做置信度分析建议 `-n 50` 跑满，且注意标签分布极不均衡
   （MV_FWD/MV_DOWN 占大头，`GRASP`/`RELEASE`/`DONE` 各只有 1–2 条）。
5. `action_probs` 对"跨首 token 的候选"（`GRASP` vs `MV_*`）只有上界。若下游需要 9 个 action 的
   **精确**全概率，唯一办法是对每个候选各发一次带 `echo`/强制续写的请求（9× 开销），
   当前实现没做——因为实测这些跨类候选的上界都 ≤1e-8，对分析无影响。
