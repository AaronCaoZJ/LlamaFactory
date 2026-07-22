# Task Template Augmentation

这个目录用于把 `02_exchange_token/rollout_lite.json` 里单一的 `Task:` 描述替换成固定的多模板描述。

默认输入：

```text
data/agentrobot/MVTOKEN/mix_22-06_fk-pp/02_exchange_token/rollout_lite.json
```

默认输出：

```text
data/agentrobot/MVTOKEN/mix_22-06_fk-pp/02_exchange_token/rollout_lite_task_template_aug.json
```

## 文件

- `task_templates.json`
  17 个 unique task；普通 pick/place/stack task 各 6 条语义等价模板，`rearrange the letters to spell "SHOW"` 单独按 rollout 分成 S/H/O/W 修正组。模板只改任务描述，不改 action token 列表、recent moves、图像路径或输出标签。
  模板刻意覆盖直接命令、目标状态、夹爪执行、从当前位置转移、完成条件等不同句式；同时保留原始物体名、目标承载物和 `on/in/on top of` 空间关系，避免引入歧义。
  SHOW 模板会根据样本 image path 中的 rollout 选择对应模板组，例如 `rollout_006` 是把 S 放到 H 左边，`rollout_007/009` 是把 H 放到 S 和 O 之间，`rollout_010/011` 是把 O 放到 H 和 W 之间，`rollout_008/018/019` 是把 W 放到最右边。每条模板仍然保留最终拼成 `SHOW` 的约束。
- `apply_task_template_aug.py`
  应用模板并生成新的 LLaMA-Factory JSON。
- `make_task_contact_sheet.py`
  从每个 unique task 抽一条样本，生成 agentview/wrist 首帧拼图，用来人工检查模板是否 grounded。

## 运行顺序

先生成一张检查图：

```bash
python data/agentrobot/zwz_process_code/task_template_aug/make_task_contact_sheet.py \
  --output /tmp/task_template_aug_contact_sheet.jpg
```

预览将要替换的统计，不写文件：

```bash
python data/agentrobot/zwz_process_code/task_template_aug/apply_task_template_aug.py --dry-run
```

生成增强后的 JSON：

```bash
python data/agentrobot/zwz_process_code/task_template_aug/apply_task_template_aug.py \
  --stats-output data/agentrobot/MVTOKEN/mix_22-06_fk-pp/02_exchange_token/rollout_lite_task_template_aug.stats.json
```

默认 `--assignment sample`，同一个普通 task 的样本会在 6 个模板里轮换；SHOW 样本会先按具体 rollout 选中对应的 S/H/O/W 修正组，再在该组的 6 个模板里轮换，尽量提高 instruction wording 的覆盖度。

注意：SHOW 的 `left/right` 模板是基于原始未水平翻转图像人工检查得到的。如果要对水平翻转后的 SHOW 数据做同类增强，需要重新检查图像并同步调整左右关系。

如果想让同一个 rollout 的所有 timestep 使用同一个 task wording，运行：

```bash
python data/agentrobot/zwz_process_code/task_template_aug/apply_task_template_aug.py \
  --assignment rollout
```

## 接到 dataset_info

如果要训练这份数据，在 `data/dataset_info.json` 里加一个条目即可：

```json
"mix_22-06_fk-pp_02_exchange_token_task_template_aug": {
  "file_name": "agentrobot/MVTOKEN/mix_22-06_fk-pp/02_exchange_token/rollout_lite_task_template_aug.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "images": "images"
  }
}
```

然后 yaml 里使用：

```yaml
dataset: mix_22-06_fk-pp_02_exchange_token_task_template_aug
```
