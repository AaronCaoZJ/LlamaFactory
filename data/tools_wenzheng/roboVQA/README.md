# RoboVQA -> LLaMA-Factory 工具说明

这套工具把本地 RoboVQA 数据：

```text
/storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA
```

整理成当前 LLaMA-Factory 可以直接训练的 video 数据集。默认推荐先用
`robovqa_reasoning_*.json`，它的输出保留 `<think>...</think><answer>...</answer>`。

## 文件作用

- `extract_clips.sh`
  解压 `clips/clips_part_*.tar.gz` 到 `clips/`。JSON 里视频路径是 `clips/<id>.mp4`，
  所以必须解到这个目录本身。
- `convert_to_llamafactory.py`
  流式读取原始大 JSON，输出 Alpaca 风格 JSONL：
  `instruction/input/output/videos`。不会一次性把 1GB+ JSON 读进内存。
- `register_dataset_info.py`
  把转换后的 JSONL 注册进 `data/dataset_info.json`。
- `make_mixed_train_yaml.py`
  基于当前 `qwen3_5_9b_03_just_mix_zwz_new_prompt_add_horizon_flip.yaml`
  生成一个 `plus_robovqa` 混合训练 yaml。
- `prepare_all.sh`
  顺序调用上面几个脚本，适合第一次快速跑通。

## 推荐运行顺序

在 repo 根目录运行：

```bash
cd /storage/wenzheng/showrobot/LlamaFactory
```

### 1. 解压视频

```bash
bash data/tools_wenzheng/roboVQA/extract_clips.sh
```

检查：

```bash
find /storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA/clips -maxdepth 1 -name '*.mp4' | wc -l
test -f /storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA/clips/4857681842253668822.mp4 && echo OK
```

### 2. 先做一个小样本 smoke test

```bash
python data/tools_wenzheng/roboVQA/convert_to_llamafactory.py \
  --kind reasoning \
  --max-samples 20000 \
  --name-suffix 20k \
  --validate-media
```

输出：

```text
data/robovqa/robovqa_reasoning_lf_20k.jsonl
```

注册到 `data/dataset_info.json`：

```bash
python data/tools_wenzheng/roboVQA/register_dataset_info.py \
  --only reasoning \
  --name-suffix 20k \
  --overwrite
```

生成混合训练 yaml：

```bash
python data/tools_wenzheng/roboVQA/make_mixed_train_yaml.py \
  --robovqa-dataset robovqa_reasoning_lf_20k \
  --media-dir /storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA
```

生成的 yaml 默认在：

```text
examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/qwen3_5_9b_03_just_mix_zwz_new_prompt_add_horizon_flip_plus_robovqa.yaml
```

### 3. 一键跑 smoke test

如果想一次跑完上面的流程：

```bash
MAX_SAMPLES=20000 NAME_SUFFIX=20k bash data/tools_wenzheng/roboVQA/prepare_all.sh
```

如果视频已经解压过：

```bash
SKIP_EXTRACT=1 MAX_SAMPLES=20000 NAME_SUFFIX=20k bash data/tools_wenzheng/roboVQA/prepare_all.sh
```

### 4. 转全量 reasoning

```bash
python data/tools_wenzheng/roboVQA/convert_to_llamafactory.py \
  --kind reasoning \
  --validate-media

python data/tools_wenzheng/roboVQA/register_dataset_info.py \
  --only reasoning \
  --overwrite

python data/tools_wenzheng/roboVQA/make_mixed_train_yaml.py \
  --robovqa-dataset robovqa_reasoning_lf \
  --media-dir /storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA
```

## 训练建议

RoboVQA reasoning 全量约 92 万条，你当前 robot action 数据是一两万条量级。直接全量 concat
会让 RoboVQA 主导训练。建议先用 `20k` 或 `50k` 小样本混合试验；效果稳定后再扩大。

混合 yaml 的关键字段是：

```yaml
dataset: mix_22-06_fk-pp_03_just_mix_zwz_new_prompt_add_horizon_flip,robovqa_reasoning_lf_20k
media_dir: /storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA
video_max_pixels: 65536
video_min_pixels: 1024
video_fps: 2.0
video_maxlen: 64
cutoff_len: 4096
```

`media_dir` 只负责把 RoboVQA 的相对视频路径 `clips/<id>.mp4` 解析到真实文件。
你现有 robot 样本大多是绝对图片路径，不会被它破坏。
