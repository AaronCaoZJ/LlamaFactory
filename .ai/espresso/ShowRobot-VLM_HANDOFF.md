# ShowRobot-VLM HANDOFF — 在 LlamaFactory 里训 MVTOKEN 的一套约定

> 给接手的人：这条 pipeline 把机器人 rollout（双路相机图 + 逐步动作）训成「看两张图 → 输出一个动作 token」的 VLM 策略。
> 本文只写**约定和习惯**（为什么这么摆、新东西该往哪放），踩坑细节见 [`scripts/gemma4/GEMMA4_DEBUG.md`](../../scripts/gemma4/GEMMA4_DEBUG.md)。
> 内容基于 2026-07-12 的实读代码。**先读**：`data/agentrobot/data_list.md`、`scripts/workspace_dir.sh`。

```
AgentRobot 采集 rollout ──► rollout_to_llamafactory.py ──► rollout_lite.json (alpaca 多模态)
                                                              │  注册进 data/dataset_info.json
                                                              ▼
                                        examples/train_lora/<家族>/<模型>_<数据集>.yaml
                                                              │  scripts/<家族>/train/train.sh
                                                              ▼
                                              saves/<模型>/robot/<数据集>/  (LoRA adapter)
                                                              │  scripts/<家族>/eval/start_vllm_server.sh
                                                              ▼
                                     OpenAI 兼容 server ◄──► infer.py eval  /  AgentRobot 真机闭环
```

动作 token 集合：`MV_FWD MV_BACK MV_LEFT MV_RIGHT MV_UP MV_DOWN GRASP RELEASE DONE`
（`DONE` 录制数据里没有，由转换脚本在每条 episode 末尾**合成**一条）。

---

## 1. 命名：四套写法，别搞混

| 场景 | 写法 | 例子 |
|---|---|---|
| 数据集目录 / dataset_info key | 下划线 + 日期缩写 | `mix_22_27_v3_lite` |
| yaml / scripts 目录 | **下划线** | `qwen3_5_9b`、`gemma4/` |
| `saves/` 目录 | **小写 + 点** | `qwen3.5-9b`、`gemma4-e4b` |
| wandb `run_name` | `output_dir` 去掉 `saves/` 再把 `/` 换成 `-` | `gemma4-e4b-robot-mix_22_27_v3` |

数据集名的构成：`<采集批次>_<prompt版本>_<模式>`
- `mix_22_27` = 0622 + 0627 两批数据；`mix_22_27_04` = 再加 0704。**22/27/04 就是采集日期的日**。
- `_v3` / `_v4` = **prompt 模板版本**（对应 `AgentRobot/prompts/<vN>/`），不是数据版本。v3 = Franka 统一 prompt（主力）；v4 = 按本体分开渲染（Franka/Piper）。
- `_lite` = **stage-free**（prompt 里只有 task / gripper_state / recent_moves，不依赖任何 VLM 预处理）。
  另有 `_subgoal` / `_affordance` 两条**早期探索路线，已废弃**——在训的全是 lite。

---

## 2. 数据侧

### 2.1 目录四层

```
data/agentrobot/MVTOKEN/<采集批次>/<任务>/<rollout>/     # 0622 / 0627_cleaned / 0705_piper …
                        └── v3/rollout_lite.json         # 转换产物落在批次目录下的 prompt 版本子目录
```
- 批次按**采集日期**命名；`_cleaned` = 过了 `clean_grasp_release.py`；`_piper` = Piper 机械臂（其余默认 Franka）。
- 任务前缀：`pap_` = pick-and-place，`stack_` = 堆叠，`rearrange_` = 摆字。
- rollout 叶子目录：`agentview/*.png`（俯视 / Piper 是第一人称）、`wrist/*.png`、`actions.jsonl`、`metadata.json`。
- `mix_*/` 只有 json 没有图片——因为**样本里的图片是绝对路径**，merge 就是纯拼接。
  代价：**数据集不能跨机器直接搬运**。

### 2.2 样本 schema（三条硬约定，评测侧必须逐字复刻）

```json
{"instruction": "<image><image>" + prompt, "input": "", "output": "MV_DOWN",
 "images": ["/abs/.../agentview/0000.png", "/abs/.../wrist/0000.png"]}
```
1. **`<image><image>` 必须在最前面**——两张图排在文本之前。
2. **`input` 恒为空**，一切塞进 `instruction`（所以 `"\n".join` 的拼接细节在评测侧也要一致）。
3. `recent_moves` = 最近 5 个 `MV_*`，**newest first**，`GRASP/RELEASE` 不入历史，空时填 `"none"`。

### 2.3 `process_data.sh` 是「活的实验日志」，不是幂等构建脚本

顶部集中定义 `DATA_DIR_*` 和 `TASK_MAP_*`（`"pap_banana=pick up the banana and place it on the blue plate"`），
**所有历史命令保留但注释掉**，用 `: <<'EOF' … EOF` heredoc 当小标题分节，只解注释当前要跑的那条。
新增一批数据 = 在末尾追加一个新块，不要删旧的。

### 2.4 注册进 `data/dataset_info.json`

所有 MVTOKEN 条目共用同一份 columns（无 `formatting` 字段，默认 alpaca）：
```json
{"file_name": "agentrobot/MVTOKEN/mix_22_27/v3/rollout_lite.json",
 "columns": {"prompt": "instruction", "query": "input", "response": "output", "images": "images"}}
```
`file_name` 是**相对 `data/` 的路径**。数据集与模型正交——**加新模型不需要动这个文件**。

### 2.5 video 槽位（方案 B）：两路相机当成两"帧"

一条**在跑的对照实验**，不是默认路线。两个视角塞进 `<video>` 而不是两个 `<image>`：

```json
{"instruction": "<video>" + prompt, "input": "", "output": "MV_DOWN",
 "videos": [["/abs/.../agentview/0000.png", "/abs/.../wrist/0000.png"]]}
```
`videos` 是**双层 list**（外层一个元素对应一个 `<video>`），路径必须绝对——`_find_medias` 只给扁平字符串列表拼 `media_dir`，nested 的不拼。dataset_info 里 columns 写 `"videos": "videos"`。

**为什么 token 会减半**：Qwen 的 patch embed 是 3D conv，`temporal_patch_size=2`，**每个视觉 token 必须吃满 2 帧**（`pixel_values` 的 1536 = 3×**2**×16×16）。
- `<image><image>`：每张图沿时间轴**自我复制**填满槽位 → 各自 64 token，共 **128**。
- `<video>`=[agentview, wrist]：两个不同视角一人一个时间槽，`grid_t = 2÷2 = 1` → 只有 **64** token。
  **省下的正是"第二个视角的独立表示"**：两个视角在每个空间位置被 3D conv 加权求和。
  相邻视频帧这么融合是合理的（空间对齐），但 agentview / wrist 视角完全不同，同位置像素毫无对应关系。
- 想在 video 槽位里保持两个视角独立，只能喂 4 帧 `[A,A,W,W]`（`grid_t=2`）——而这在数值上**与 `<image><image>` 完全相同**
  （`pixel_values` allclose），白白多一个时间戳，**没有任何收益**。所以能选的只有「两图」和「拼接成一张」。

产物与开关：
| 东西 | 说明 |
|---|---|
| `to_video_slot.py` | 把已有 image 版数据集 re-slot 成 video 版。**样本 / 顺序 / prompt 逐字不变**，唯一变量是模态槽位——这才是干净的对照实验 |
| `rollout_to_llamafactory.py --video-slot` | 正统入口：新采集的数据直接出 video 版 |
| `qwen3_5_9b_mix_22_27_v3_video.yaml` | 训练配置；对照组是同目录的 `qwen3_5_9b_mix_22_27_v3.yaml` |
| `scripts/qwen3_5/train/video_slot_train.sh` | `MODE=overfit\|train` |

⚠️ **`video_fps` 是个隐藏的 prompt 参数**：它决定渲染出的时间戳文本（`video_fps: 2.0` → `<0.2 seconds>`）。
评测侧把两张图编码成 mp4 时**必须用同一个 fps**，否则时间戳漂移、prompt 失配。`infer.py` 用 `VIDEO_FPS` 环境变量同步。

---

## 3. 训练侧

### 3.1 yaml：模板化程度极高，抄就完了

位置 `examples/train_lora/<家族>/<模型>_<数据集>.yaml`（多路线实验再下沉一层，如 `qwen3_5_9b/mix_22-06_fk-pp/`）。
固定 7 个 `###` 分节：`model / method / LoRA / dataset / output / train / distributed`。

新建一份时**只需要改 5 个字段**，其余整段复制：
`model_name_or_path`、`template`、`dataset`、`output_dir`、`run_name`（+ 按模型大小选 deepspeed stage）。

不成文的硬约定：

| 约定 | 说明 |
|---|---|
| **`eff_bs` 恒为 32** | 换卡数只调 `gradient_accumulation_steps` 补回来（单卡就 8），别动 batch size |
| **LoRA 超参全线一致** | `rank 64 / alpha 128 / dropout 0.05 / target all`，几乎没人动 |
| **`image_max_pixels: 65536`** | = 256×256，与 rollout 图实际分辨率对齐。**推理 yaml 必须写同一个值** |
| **`overwrite_cache: true`** | 16 份 yaml 全部默认打开。datasets 的缓存指纹只 hash 图片**路径字符串**，图片被原地覆盖它察觉不到；全量重算 4132 条仅 ~17s，不值得赌（详见 §6） |
| **`### train` 前必有一段手算注释** | `samples / GPU数 / eff_bs / steps-per-epoch / 总步数`，方便 review 时核对 |
| deepspeed 按大小选 | 0.8B → z0；2B/9B/12B → z2；27B → z3。**注释里写清理由** |
| `freeze_vision_tower: true` | 只训语言侧 |

template 选择：Qwen3.5 全系 = `qwen3_5_nothink`；Gemma-4 E4B/12B = `gemma4n` + `enable_thinking: false`（26B/31B 才用 `gemma4`）。

### 3.2 先跑 overfit，再跑正式训练

`data/agentrobot/overfit_test/rollout_000`（单条 episode）→ 注册为 dataset `overfit` → 用 `*_overfit.yaml`
（`lora_dropout: 0.0`、`lr_scheduler_type: constant`、`cutoff_len: 512`、batch=1）**故意过拟合**。
跑通 = 数据管线 / 模板渲染 / 多模态输入 / LoRA 挂载全链路 OK。**上新模型系列时的第一个动作。**

### 3.3 脚本骨架

`scripts/<家族>/train/train.sh` 统一长这样：
```bash
set -euo pipefail
# ═══ GPU / runtime knobs (edit here) ═══     # 这条分隔线 = "可编辑区"标记，保留它
GPU="${GPU:-4,5}"
MODEL="${MODEL:-e4b}"     # 12b | e4b        # 多模型用 case 分支选 yaml
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"
...
export DISABLE_VERSION_CHECK=1               # 必需，理由写进注释
exec env CUDA_VISIBLE_DEVICES="${GPU}" llamafactory-cli train "${TRAIN_CONFIG}"
```
那条 `source` 的一行式是**深度无关**的（向上找 `scripts/workspace_dir.sh`）——**新脚本原样抄，别改**。

关键环境变量：

| 变量 | 何时需要 |
|---|---|
| `DISABLE_VERSION_CHECK=1` | **总是**。LF 硬编码 transformers ≤5.6.0，而 gemma4 要 ≥5.10 |
| `FORCE_TORCHRUN=1` | **单卡 + deepspeed** 时必需（LF 单 GPU 默认不走 torchrun） |
| `MASTER_PORT` | 并发多任务时手工错开（29511/29512/…），否则 rendezvous 撞车 |

### 3.4 `workspace_dir.sh` + `.env.paths`：路径不进 git

同一份代码要在多台机器跑，所以路径全部外置：`.env.paths`（**gitignored**，每台机器一份）导出
`MODELS_DIR / LF_VENV / VLLM_VENV / AGENTROBOT_ROOT / HF_HOME`；`.env.paths.example` 是模板（tracked）。
`workspace_dir.sh` 负责推导 `LF_ROOT`、source 它、并**在这里就把错配置 fail 掉**（而不是等模型加载 200 行后才炸）。
首次上机：`cp .env.paths.example .env.paths` 再改。调试：`WORKSPACE_DIR_DEBUG=1`。

### 3.5 `saves/` 布局

```
saves/<模型>/robot/<数据集>[/<路线>]/     # robot/ 这层是任务域（另有 mikomiko/）
```
叶子目录里，最终 adapter（`adapter_config.json` + `adapter_model.safetensors`）与 `checkpoint-*/` **并列**。
⚠️ **训练被中途 kill 的话顶层没有 adapter**，只有 checkpoint —— 此时 `--lora-modules` 必须指向具体 `checkpoint-N`。
正常跑完就用顶层目录，**别再指 checkpoint-N**。

---

## 4. 评测侧

### 4.1 脚本职责

| 文件 | 干什么 |
|---|---|
| `start_vllm_server.sh` | **推荐**。一次挂多个 LoRA：`--lora-modules key=path …`，`--max-lora-rank 64` |
| `start_hf_server.sh` | `llamafactory-cli api <examples/inference/*.yaml>`，备选后端 |
| `infer.py` | 推理客户端，两种模式：`single`（VQA 探针）/ `eval`（批量 token 准确率） |
| `run_eval.sh` | 封装：设 `API_URL`/`MODEL_NAME`/`EVALSET`，选一条 `EVAL_CMD`（其余注释掉），带时间戳写日志 |

端口约定（隐含但一致）：`8101` Qwen-27B、`8104` Gemma-4、`8108` Qwen-0.8B、`8109` Qwen-9B、`8114` HF backend。
**`MODEL_NAME` 必须等于 `--lora-modules` 的 key**——vLLM 会校验（不对就 404），HF backend 不校验。

注释符号约定：`#!` = 可编辑的旋钮，`#*` = 导出/可覆盖的变量。

### 4.2 ⚠️ prompt 必须逐 token 对齐训练（最大的坑）

推理侧的 content 拼装错一处，模型就退化（复读 / 吞掉 `MV_` 前缀 / 乱答）。三条铁律：

1. **图片排在文本之前**（OpenAI `content` 数组顺序 = 占位符顺序，两个后端都如此）。
2. **必须有 system turn**（LF 训练时注入 `default_system`）——HF backend 自动补，**vLLM 不会**。
3. **instruction 与 input 用单个 `\n` 连接**（converter 是 `"\n".join`），不是 `\n\n`。

**两个家族的官方 `chat_template.jinja` 都拼不出训练分布，vLLM 必须挂自己那份**：

| 家族 | `--chat-template` | 官方模板错在哪 |
|---|---|---|
| Gemma-4 | `scripts/gemma4/eval/chat_template_gemma4n_lf.jinja` | 缺 system turn + 空 thought 段。详见 GEMMA4_DEBUG.md §3 |
| Qwen3.5 | `scripts/qwen3_5/eval/chat_template_qwen3_5_lf.jinja` | **即使 `enable_thinking=false` 也会插一个空 think 块** `<think>\n\n</think>\n\n`（不是"不插"！），比 LF 的 `qwen3_5_nothink` 多 4 个 token |

Qwen3.5 那条是 2026-07-13 才发现的：`start_vllm_server_9.sh` 之前没挂 `--chat-template`，
所以**在那之前的所有 qwen3.5 评测数字都是在多 4 个 token 的失配 prompt 下测的**。已修，image 布局和 video 槽位都逐 token 验证过。

**验证方法**（换模型/改模板后务必跑）：用 vLLM `/tokenize`（`return_token_strs=true`）打印 prompt token，
和 LF `template.encode_oneturn()` 的结果**逐 token 比对**，必须完全相同。

### 4.3 评测集与日志

- `data/agentrobot/ood_sample/v3/rollout_lite.json`：一条**训练集里没有的任务**，跨模型共享的泛化评测集，
  所有 `run_eval.sh` 默认指向它。`id_sample/` 是对应的 in-distribution 对照组。
- 日志：`results/<模型家族>/<YYYYmmdd_HHMMSS>.txt`，脚本写固定日志头再 `tee -a` 追加 stdout。
- `infer.py eval` 常用参数：`-n 100 --raw`（看完整回复）/ 不加 `--raw`（只取 token，快）/ `--seed 42`（固定抽样）。

---

## 5. 加一个新模型系列：checklist

1. `.env.paths` 里确认 `MODELS_DIR` 下有 base 权重；需要新 venv 就建 `.venv-<model>`。
2. `examples/train_lora/<家族>/<模型>_overfit.yaml` → **先把 overfit 跑通**（§3.2）。
3. `examples/train_lora/<家族>/<模型>_<数据集>.yaml`（改那 5 个字段）。
4. `examples/inference/<模型>_lora.yaml`（HF backend 用；`image_max_pixels` **必须与训练一致**）。
5. `scripts/<家族>/train/{train.sh, overfit_test_train.sh}`（抄 gemma4 版，换 venv / `case MODEL` / 版本闸注释）。
6. `scripts/<家族>/eval/{start_vllm_server.sh, start_hf_server.sh, infer.py, run_eval.sh}`（抄 gemma4 版，分配新端口）。
7. **验证 prompt 渲染逐 token 对齐**（§4.2）；不齐就写一份 `chat_template_<model>_lf.jinja`。
8. 踩到坑就写一份 `scripts/<家族>/<MODEL>_DEBUG.md`（gemma4 那份是范例）。
9. **不需要动** `data/dataset_info.json`。

---

## 6. 已知的坏路径 / 陷阱

`examples/train_lora/` 做过一次「平铺 → 按模型建子目录」的重构，**但部分老脚本的路径没跟着改**，现在是坏的：

- `scripts/qwen3_5/train/train.sh` → 指向 `examples/train_lora/qwen3_5_9b_piper_0705_v4.yaml`（实际在 `qwen3_5_9b/` 下），
  且它引用的 `TASK_MAP` 数组已被注释掉，跑起来会 unbound variable 挂掉。真正在用的是 `mix_fk-pp_train.sh`。
- `scripts/qwen3_5/train/overfit_test_train.sh`、`scripts/gemma4/train/overfit_test_train.sh` 同类问题。
- `scripts/qwen3_5/eval/test_mvtoken.sh` 里的路径也是旧的（`scripts/eval/…`、`ood_sample/v2/…`）。
- `eval_mvtoken.py` 的 `DATASET` 默认值指向一个**不存在的**绝对路径 → 必须传 `--evalset`。

其他：

- ⚠️ `hf_download/models/Qwen3.5-0.5B` **目录名是错的**，里面实际是 0.8B 权重。
- ⚠️ `Image features and image tokens do not match` = 序列里 `<|image_pad|>`（gemma 是 soft token）的个数与 ViT 实际吐出的特征数对不上。
  **改 yaml 参数不会触发它**——`dataset.map()` 的 fingerprint 会 hash 整个 processor 对象，
  实测改 `image_max_pixels`（65536 → 131072）指纹就变了，缓存自动 miss 并重算，`cutoff_len` / `template` 同理。
  真正的元凶是**图片被原地覆盖而路径没变**（重新采集、`clean_grasp_release.py` 改图、换分辨率导出同名 png）：
  fingerprint 只 hash json 里的**路径字符串**，看不到磁盘上 png 的内容，于是复用了一份按旧图算出的占位符数量——而且是**静默**的。
  所以所有训练 yaml 已默认 `overwrite_cache: true`（全量重算 ~17s，多卡下只有 rank 0 重算）。要手动清缓存就删 `~/.cache/huggingface/datasets/json`。
  数据涨到几十万条、重算变成分钟级之后，别退回 `false`，改用 `tokenized_path` 显式缓存（路径自己命名，和数据版本绑定，比隐式指纹可控）。
- ⚠️ **vLLM 的端口不独占**：同端口起第二个 vLLM 不会报错，两个都能起来，请求被内核轮流分发，
  评测结果会变成两个 server 的混合。测之前先 `ss -lptn "sport = :8104"` 确认监听者只有一个。
- ⚠️ yaml 注释里的样本数（如「3738 samples」）可能已过期（`mix_22_27_v3_lite` 实际 4132 条）——**别盲信注释**。
