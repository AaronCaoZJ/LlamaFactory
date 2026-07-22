# 空间增强：添加双视角水平翻转样本

这个工具用于给 MVTOKEN 双视角机器人数据添加水平镜像增强。它会在原始数据基础上为每条样本 add 一条翻转样本，使训练样本数量变成 2 倍。

默认输入数据：

```bash
data/agentrobot/MVTOKEN/mix_22-06_fk-pp/03_just_mix/rollout_lite_zwz_new_prompt.json
```

默认输出数据：

```bash
data/agentrobot/MVTOKEN/mix_22-06_fk-pp/03_just_mix/rollout_lite_zwz_new_prompt_add_horizon_flip.json
```

也就是说，不传 `--output` 时，脚本会在同一目录下生成一个新的 JSON，不会覆盖输入 JSON。

## 做了什么

对每个被训练 JSON 引用到的 rollout，脚本会在原目录旁边创建一个 `_horizon_flip` 后缀目录，例如：

```bash
data/agentrobot/MVTOKEN/0704_cleaned/pap_pink_cube/rollout_000
data/agentrobot/MVTOKEN/0704_cleaned/pap_pink_cube/rollout_000_horizon_flip
```

在新的 rollout 目录里：

- `agentview` 图像做水平左右翻转。
- `wrist` 图像做水平左右翻转。
- `actions.jsonl` 里的 `MV_LEFT` 和 `MV_RIGHT` 互换。
- `metadata.json` 里的动作 token 序列也互换 `MV_LEFT` 和 `MV_RIGHT`。
- `visualization.mp4` 使用 ffmpeg 重新编码为可播放的 H.264 MP4：`libx264 + yuv420p + faststart`。

在新的训练 JSON 里：

- 每条原始样本后面追加一条对应的翻转样本。
- 翻转样本的 `images` 指向 `_horizon_flip` rollout。
- 翻转样本的模型预测目标 `output` 会互换 `MV_LEFT` 和 `MV_RIGHT`。
- prompt 里的 recent move history 行会互换 `MV_LEFT` 和 `MV_RIGHT`。
- prompt 里的 action 定义说明不改，因为动作语义定义本身仍然是同一套。

脚本是幂等的：如果输入 JSON 里已经包含 `_horizon_flip` 样本，脚本会先只取非翻转的 base 样本重新生成，不会把数据继续扩大成 4 倍、8 倍。

## 使用方法

只预览计划，不写文件：

```bash
python data/agentrobot/zwz_process_code/spatial_augmentation/add_horizon_flip_augment.py --dry-run
```

生成默认的新 JSON：

```bash
python data/agentrobot/zwz_process_code/spatial_augmentation/add_horizon_flip_augment.py
```

指定输出 JSON：

```bash
python data/agentrobot/zwz_process_code/spatial_augmentation/add_horizon_flip_augment.py \
  --output data/agentrobot/MVTOKEN/mix_22-06_fk-pp/03_just_mix/rollout_lite_zwz_new_prompt_add_horizon_flip.json
```

如果已经有 `_horizon_flip` rollout，想重新生成图像、`actions.jsonl`、`metadata.json` 和可播放的 H.264 `visualization.mp4`：

```bash
python data/agentrobot/zwz_process_code/spatial_augmentation/add_horizon_flip_augment.py --overwrite-rollouts
```

如果确实想原地覆盖输入 JSON，可以显式指定同一个输出路径：

```bash
python data/agentrobot/zwz_process_code/spatial_augmentation/add_horizon_flip_augment.py \
  --output data/agentrobot/MVTOKEN/mix_22-06_fk-pp/03_just_mix/rollout_lite_zwz_new_prompt.json
```

## 检查方式

检查样本数量：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("data/agentrobot/MVTOKEN/mix_22-06_fk-pp/03_just_mix/rollout_lite_zwz_new_prompt_add_horizon_flip.json")
samples = json.loads(path.read_text())
flipped = [s for s in samples if any("_horizon_flip" in p for p in s["images"])]
print("total:", len(samples))
print("base:", len(samples) - len(flipped))
print("flipped:", len(flipped))
PY
```

检查一个翻转视频是否是可播放的 H.264：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,codec_tag_string,pix_fmt,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 \
  data/agentrobot/MVTOKEN/0704_cleaned/pap_pink_cube/rollout_000_horizon_flip/visualization.mp4
```

期望输出里包含：

```text
codec_name=h264
codec_tag_string=avc1
pix_fmt=yuv420p
```

## 添加 Franka 的 FWD/BACK 规则反转样本

第二个增强是在 `add_horizon_flip` 数据基础上，只对 Franka 样本再 add 一份规则反转样本。Piper 样本不变。

默认输入数据：

```bash
data/agentrobot/MVTOKEN/mix_22-06_fk-pp/03_just_mix/rollout_lite_zwz_new_prompt_add_horizon_flip.json
```

默认输出数据：

```bash
data/agentrobot/MVTOKEN/mix_22-06_fk-pp/03_just_mix/rollout_lite_zwz_new_prompt_add_horizon_flip_add_augment_rules.json
```

这个增强只改训练 JSON，不复制或改动图像目录：

- 只处理 Franka 样本，按 `images` 路径里不含 `piper` 判断。
- prompt 里 `MV_FWD` 和 `MV_BACK` 的详细定义互换。
- 模型预测目标 `output` 里的 `MV_FWD` 和 `MV_BACK` 互换。
- prompt 的 recent move history 行里的 `MV_FWD` 和 `MV_BACK` 互换。
- `MV_LEFT/MV_RIGHT`、`MV_UP/MV_DOWN`、`GRASP/RELEASE/DONE` 不变。

因为 `add_horizon_flip` 后的数据包含 `7933` 条原始样本和 `7933` 条水平翻转样本，其中 Franka 共 `10140` 条，所以这个增强会再 add `10140` 条 Franka 规则反转样本。最终总数应为：

```text
15866 + 10140 = 26006
```

预览计划：

```bash
python data/agentrobot/zwz_process_code/spatial_augmentation/add_franka_fwd_back_augment_rules.py --dry-run
```

生成默认的新 JSON：

```bash
python data/agentrobot/zwz_process_code/spatial_augmentation/add_franka_fwd_back_augment_rules.py
```

检查样本数量：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("data/agentrobot/MVTOKEN/mix_22-06_fk-pp/03_just_mix/rollout_lite_zwz_new_prompt_add_horizon_flip_add_augment_rules.json")
samples = json.loads(path.read_text())
piper = [s for s in samples if "piper" in " ".join(s["images"])]
franka = [s for s in samples if "piper" not in " ".join(s["images"])]
rule_aug = [
    s for s in franka
    if "- MV_FWD: move the end effector farther from the AgentView camera, toward the robot body." in s["instruction"]
]
print("total:", len(samples))
print("piper:", len(piper))
print("franka:", len(franka))
print("franka_rule_aug:", len(rule_aug))
PY
```
