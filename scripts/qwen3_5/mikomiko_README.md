# mikomiko

三批数据、两个训练任务。数据处理一个入口,每个任务各一个训练脚本、一个推理+可视化脚本。

## 数据 → 训练 的对应

```
第一批 CSV 124w   -> jsonl/cleaned/train.jsonl    ─┐
                                                   ├─ mix ─> mix_train.jsonl ─> 2B tag 训练
第二批 0716 tag   -> jsonl_0716/train.jsonl       ─┘

第三批 0721 描述  -> jsonl_desc_0721/train.jsonl  ──────────────────────────> 9B desc 训练
```

前两批是**混着训**的(轮转交错成一份 `mix_train.jsonl`),所以三批数据对应两个训练。
`full_v0` = 只训第一批;`full_v1` = 训 mix。

## 五个脚本

| 干什么 | 脚本 |
|---|---|
| 数据处理(三批) | `data/mikomiko_tag/process_data.sh` |
| tag 训练 | `scripts/qwen3_5/mikomiko_tagger/train_tag_2b.sh` |
| tag 推理 + 可视化 | `scripts/qwen3_5/mikomiko_tagger/infer_tag_2b.sh` |
| desc 训练 | `scripts/qwen3_5/mikomiko_grok_desc/train_desc_9b.sh` |
| desc 推理 + 可视化 | `scripts/qwen3_5/mikomiko_grok_desc/infer_desc_9b.sh` |

每个脚本都**自包含**:头部一行 `source scripts/workspace_dir.sh` 拿机器路径(和仓库里其它脚本一样),
其余从上往下读完就是全部逻辑,没有额外的公共库要跳。不带子命令跑任何一个,都会打印它自己的命令表。

### 数据 · `process_data.sh`

```bash
bash process_data.sh csv                    # 第一批,自带下载 -> jsonl/
bash process_data.sh download               # 取图 -> img_0716/(~250 GB,后两批共用)
bash process_data.sh tag0716  --build       # 第二批 -> jsonl_0716/
bash process_data.sh clean    --apply       # 去掉整条复读的行
bash process_data.sh mix                    # 前两批交错 -> mix_train.jsonl
bash process_data.sh desc0721 --build && bash process_data.sh verify   # 第三批 + 门禁
```

builder 默认 `--plan`(空跑,打印丢弃漏斗),要写盘必须显式 `--build`。**先看漏斗是设计,不是形式。**

### tag 任务(Qwen3.5-2B 全参,4 卡 ZeRO-0)

```bash
bash train_tag_2b.sh                              # 默认 mix
DATASET=mikomiko_tag_train bash train_tag_2b.sh   # 只训第一批

bash infer_tag_2b.sh serve                  # vLLM :8110
bash infer_tag_2b.sh eval  [STEP] [GPU]     # 固定 400 张 -> evalmini_history.tsv
bash infer_tag_2b.sh seen  [STEP] [GPU]     # 200 张训练集图 -> seen_history.tsv
bash infer_tag_2b.sh eval-vllm [STEP]       # 同样 400 张,打已起的服务,快很多
FORCE=1 N=200 GPU=4 bash infer_tag_2b.sh viz
FORCE=1 GPU=0 bash infer_tag_2b.sh viz-onlyfans
```

`eval` 和 `seen` 要对着读:seen F1 **远高于** unseen → 拟合-泛化 gap 真实存在,加数据多样性有空间;两者**接近** → 标签自相矛盾,堆步数没用。

### desc 任务(Qwen3.5-9B 全参,8×H200 ZeRO-3)

```bash
bash train_desc_9b.sh smoke              # 能不能训起来,不写 ckpt
bash train_desc_9b.sh probe 8 12 16      # 找最大 per_device batch
nohup bash train_desc_9b.sh full &       # ~15h,必须 detach

bash infer_desc_9b.sh serve              # vLLM :8121
FORCE=1 bash infer_desc_9b.sh viz        # 微调 vs 未微调基座并排
```

`probe` 必须在**真实卡数**上跑:ZeRO-3 的分片是 (模型状态 / world_size),4 卡塞得下的 batch 在 8 卡上还有余量,拿 4 卡的结论去训 8 卡是浪费显存。

desc 侧**没有 eval 段**,是有意的:描述是自由散文,措辞不同也可以完全正确,BLEU/ROUGE 主要测用词运气。能机械判定的只有"形状"(语种、4 段齐不齐、字数、重复度、有没有撞 token 上限),这些判据在 `visualization/metrics_desc.py`,由 `viz` 直接画进页面。

## 四个必须对齐的地方(错一个就白跑)

1. **图在前、文本在后** — 数据构造时 `instruction = "<image>" + prompt`。
2. **`image_max_pixels=262144`** — 训练、推理、起服务三处必须一致,由两个 infer 脚本的 `serve_vllm` 和 `infer_mikomiko.preprocess_image()` 保证。
3. **不能有空 think 块** — checkpoint 自带的 jinja 会在 `assistant\n` 后插 `<think>\n\n</think>\n\n`,训练模板 `qwen3_5_nothink` 从不发这个块。这 4 个 token 值 **1.2pt microF1**。所以 `serve_vllm` 永远发 `chat_template_qwen3_5_lf.jinja`,不发 ckpt 自带的。
4. **tokenized 缓存按原文本键、不校验 dataset 名** — 改了 dataset / template / cutoff_len **必须删** `tokenized_path` 目录,否则静默命中旧 token。

## 每个脚本头部那几行是什么

所有路径来自 `.env.paths`,由 `scripts/workspace_dir.sh` 加载(仓库统一约定,不是 mikomiko 特有的):

```bash
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] ...)"
source "${LF_VENV}/bin/activate"
export DISABLE_VERSION_CHECK=1     # transformers 5.6.1 > LF 硬编码上限 5.6.0
```

训练脚本还多一段 `.cc-shim` 编译器垫片前置 —— Qwen3.5 的 GDN 反向内核在 Hopper 上走 tilelang JIT,
需要能用的 g++。**只在垫片真能编译时才前置**,免得换机器后悬空的垫片挡住系统里好用的编译器。

## 代码结构

```
scripts/qwen3_5/
  mikomiko_README.md            # 本文件
  mikomiko_tagger/
    train_tag_2b.sh  infer_tag_2b.sh
    infer_mikomiko.py           # 唯一的推理实现(hf + vllm 双后端)
    metrics_mikomiko.py         # 唯一的打分实现
    chat_template_qwen3_5_lf.jinja
    visualization/              # 抽样 + 出页,README 讲判据
  mikomiko_grok_desc/
    train_desc_9b.sh  infer_desc_9b.sh
    visualization/              # infer_desc.py 复用 infer_mikomiko 的预处理与 parity 检查
```

推理和打分各只有一份实现,hf / vllm / 审阅页三条路都走它们 —— 这样三边的数字才可比。

两个 `visualization/README.md` 讲的是**判据和坑**(为什么描述任务不做 BLEU、语种怎么判、bf16 下为什么非批不变),不是启动说明,值得单独读。
