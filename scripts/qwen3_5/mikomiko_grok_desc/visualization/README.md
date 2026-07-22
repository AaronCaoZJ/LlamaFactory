# grok_desc 抽样审阅页生成器

对 20260721 描述模型（`saves/qwen3.5-9b/mikomiko/grok_desc_v0`）抽样 seen/unseen × en/ja/zh 各 N 张，
推理后生成**自包含 HTML 审阅页**（图片 base64 内嵌，可直接发送）：每张图把
**gold / 微调后 / 未微调基座**三栏并排，支持按语言、数据划分、异常类型筛选和分页。

这是 tag 任务审阅页（`../../mikomiko_tagger/visualization/`）的姊妹流程，但**不是同一套判据**：
tag 任务打 tag 级 P/R/F1，描述任务是自由散文，那套指标在这里没有意义。

## 一条命令

```bash
cd scripts/qwen3_5/mikomiko_grok_desc

FORCE=1 bash infer_desc_9b.sh viz          # 全量 120 张 + 基座对照（2 张空闲 H200，约 6 分钟）
bash infer_desc_9b.sh viz                  # 复用已有预测，只重建 HTML（不占 GPU，4 秒）
FORCE=1 WITH_BASE=0 bash infer_desc_9b.sh viz   # 只跑微调模型，省一半 GPU 时间
```

中间产物在 `saves/qwen3.5-9b/mikomiko/viz_desc_0721/`（samples.json → samples_pred.json → *.html）。
常用 env：`N`（每语言每划分的张数，默认 20）、`SEED`、`WITH_BASE`、`GPU_SFT`/`GPU_BASE`、
`PORT_SFT`/`PORT_BASE`、`CKPT`、`WORK_DIR`、`OUT`。

| 职责 | 模块 |
|---|---|
| 分层抽样（每语言等量）+ 反查图片 | `sample_data.py` |
| 生成（vLLM，保留 finish_reason / token 数） | `infer_desc.py`（图像预处理与 prompt 对齐复用 `../../mikomiko_tagger/infer_mikomiko.py`） |
| 结构体检判据 | `metrics_desc.py` |
| 起服务（微调 / 基座各一次） | `../infer_desc_9b.sh` 的 `serve_vllm` |
| 出页 | `build_html.py`（由 `../infer_desc_9b.sh viz` 串起来） |

## 三个必须对齐的地方（错一个就白跑）

1. **图在前、文本在后** —— 数据构造时 instruction = `"<image>" + prompt`。
2. **image_max_pixels=262144** —— 与训练一致，由 `infer_mikomiko.preprocess_image()` 保证。
3. **两个模型用同一份 chat 模板** —— 基座与 ckpt 的 `chat_template.jinja` 逐字节相同，都用
   `chat_template_qwen3_5_lf.jinja`（去掉空 think 块）serve，prompt token 完全一致，
   差异只来自权重。给基座用它自带模板会多注入一个空 `<think></think>`，那就不是对照了。

## 为什么每种语言等量抽样

数据集 80/10/10 en/zh/ja，且**每张图只有一种语言**（不是平行语料），
prompt 里的语言块是模型判断该用哪种语言回答的**唯一信号**。
随机抽 60 条 train 会落到约 48 en / 6 zh / 6 ja，对 ja 什么都说不了。

## 判据说明（`metrics_desc.py`）

**刻意没有做与 gold 的相似度打分。** 描述是自由文本，措辞完全不同也可以完全正确，
BLEU/ROUGE 主要测的是用词运气，还容易被当成准确率来读。能机械判定的只有"形状对不对"：

| 指标 | 含义 |
|---|---|
| 语言 | 正文语种是否等于 prompt 要求的语种 —— 这份数据的命门 |
| 4段 | 4 个必需小标题是否齐全、是否按序 |
| 字数 / 比 | 中位字符数，以及同一张图 预测/gold 的字数比（跨语言字数不可比：zh gold 中位 787 字、en 2635） |
| 重复度 | 与 dataset builder 同一口径的 40-gram 覆盖率，≥0.3 是训练时被丢弃的阈值 |
| 异常 | 撞 token 上限 / 重复循环 / 无正文 |
| think | 输出里自带 `<think>` 块的比例 |

**语言判两次**，两者可以不一致：`header_lang`（抄对了小标题）与 `body_lang`（正文真的换了语言）。
只抄对标题、正文却是另一种语言，是"学会了模板没学会开关"，只有分开判才看得出来。

`body_lang` 先比 CJK 字符 vs 拉丁词、再用假名区分中日。反过来做（"有假名就是日语"）
会在真实数据上翻车：描述会逐字引用图上水印，有一条英文 gold 引用了日文片假名水印
「カリビアンコム」，7 个假名对 367 个英文词，假名优先的规则会判成日语。

## 注意

- **think 块单独剥离**：两个模型喂的是同一份无-think 提示词，基座 **120/120** 都会自己开一个
  `<think>`（微调后 0/120）。若把它算进正文，字数会虚高、思考内容多为英文还会把中日样本的
  语种判定带偏。页面折叠展示、不计入字数。
- **结构指标区分不出两个模型的内容质量**：基座靠 prompt 里的格式示例就能把 4 段撑到 90%。
  真正的差异在内容层（是否直呼具体行为、是否提取画面文字、是否编造场景），
  那是人判断的事 —— 这正是这一页要放图片的原因。
- gold 只是众多合法答案之一，措辞不同不等于说错；FP/FN 请对照图片人工判断。
