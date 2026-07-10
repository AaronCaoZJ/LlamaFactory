# seen/unseen 抽样审阅页生成器

对发布版 tagger checkpoint(`${MODELS_DIR}/Mikomiko_pornpic_tagger/checkpoint-<STEP>`)抽样
seen(train)/unseen(test_unseen_mini)各 N 张图,推理后生成**自包含 HTML 审阅页**
(图片 base64 内嵌,可直接发送):一行四卡,每卡列出 post tag / 分类标签 / gemini gold / pred,
并给出每图 **tag 级 + 词级 P/R/F1**。

**推理和打分都不在这个目录里实现**,统一走上级的两个模块,保证与 `test_mikomiko.sh` 口径一致:

| 职责 | 模块 |
|---|---|
| 生成(hf / vllm 双后端、prompt 与图像预处理) | `../infer_mikomiko.py` |
| 打分(`per_image()` / `aggregate()`) | `../metrics_mikomiko.py` |
| 抽样 + 反查 catalog | `sample_data.py` |
| 出页 | `build_html.py` |

## 一条命令

```bash
cd scripts/qwen3_5/mikomiko_tagger/visualization
export HF_HOME=/workspace1/zhijun/hf_download

FORCE=1 N=200 GPU=4 bash build_html.sh     # 全量 400 张(200 seen + 200 unseen)
bash build_html.sh                          # 复用已有预测,只重建 HTML(不占 GPU)
```

默认后端 **vLLM + bf16**,脚本自己起服务、等就绪、跑完自动关(日志在 `WORK_DIR/vllm_server.log`)。
400 张耗时:抽样 3s | 服务启动 ~90s | 推理 ~40s | 缩略图+出页 ~70s。
中间产物在 `saves/qwen3.5-2b/mikomiko/viz_review/`(samples.json → samples_pred.json → *.html)。
常用 env:`N`、`SEED`、`GPU`、`PORT`、`BACKEND`(vllm/hf)、`DTYPE`(bf16/fp32,仅 hf)、
`CKPT_STEP`、`WORK_DIR`、`OUT`。**不带 `FORCE=1` 会直接复用旧预测**。

服务已经在跑就复用它(不重复起、跑完也不关):

```bash
API=http://localhost:8111 FORCE=1 N=200 bash build_html.sh
```

不想起服务就走 transformers;要逐位可复现的产物再加 `DTYPE=fp32`:

```bash
FORCE=1 N=200 GPU=4 BACKEND=hf bash build_html.sh
```

## 三个必须对齐的地方(错一个就白跑)

1. **图在前、文本在后** —— 数据构造时 instruction = `"<image>" + prompt`。
2. **image_max_pixels=262144** —— `infer_mikomiko.preprocess_image()` 逐行复刻 LF 的 `mm_plugin`。
3. **不能有空 think 块** —— checkpoint 自带 jinja 在 `enable_thinking` 为假时会在 `assistant\n`
   后插入 `<think>\n\n</think>\n\n`,而训练模板 `qwen3_5_nothink` 从不发这个块。这 4 个 token
   值 **1.2pt microF1 / 1.7pt microP**,还会把每图复合 tag 从 5.6 抬到 6.1(400 张实测,395 张
   预测改变)。hf 后端自动剥掉;vllm 后端靠 `chat_template_train_parity.jinja` +
   `check_prompt_parity()` 启动自检兜底。

## 注意

- gold = gemini 逐图 tag(与训练标签同源);post tag / category 只是 post 级参考,
  gold 本身不一致,FP/FN 需对照图片人工判断(这正是此页的用途)。
- **dtype**:bf16 下此模型**非批不变**(48 张实测 bs=8 与 bs=1 仅 26/48 逐字一致,零 padding 的
  同图 batch 则完全一致 → 是 bf16 舍入而非 padding);fp32 下 bs=1/4/8/16 全部逐字一致,且
  fp32 bs=8 比 bf16 bs=1 更快。默认 fp32,只有要复现 vLLM/官方数字时才用 bf16(差约 0.3pt)。
- 每组 24 张时抽样噪声 ±10pt 量级,定量结论以 400 张评测为准。
