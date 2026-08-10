# Visual prompt 实验 — 样本清单

由 `build_visual_prompt_dataset.py --write` 自动生成，**不要手工编辑**。

**12011 条样本 / 233 个 rollout**，两个 prompt 变体共用同一批样本：

| 变体 | 数据集文件 | prompt | `Task:` 那行 |
|---|---|---|---|
| `task` | `rollout_lite_visual_prompt.json` | `prompt_v3_visual_prompt.txt` | 每条不同，是真实任务描述 |
| `notask` | `rollout_lite_visual_prompt_notask.json` | `prompt_v3_visual_prompt_notask.txt` | 每条相同，只指向绿框和红叉 |

`notask` 是真正检验 visual prompt 的那个设置：文本在所有样本里完全一样，「抓哪个、放哪」只能从第三张图里读。`task` 变体保留下来做对照。
来源是 `rollout_lite_plus_zwz_0723_0724_0802_long_collect.json`（22584 条 / 333 个 rollout），该基础集本身就不含 `_horizon_flip` 数据。

每条样本带三张图 —— `[agentview, wrist, visual_prompt]`，prompt 用 `prompt_v3_visual_prompt.txt`。
`visual_prompt.png` 是首帧 agentview，绿框标出要抓的物体，红十字标出松手的位置。

## 用了的样本

「质检结论」来自 `MVTOKEN/visual_prompt_qa.json` 的 `category_verdict` 字段，记录的是这条标注**当初是怎么来的、人工看过之后的结论**（含义见文末对照表）；
这里只统计实际进了数据集的那些 rollout。

| 类别 | rollout | 样本 | 质检结论 |
|---|---:|---:|---|
| `0622/pap_banana` | 8 | 434 | 自动生成后抽查通过 8 条 |
| `0622/pap_mango` | 14 | 815 | 自动生成后抽查通过 14 条 |
| `0622/pap_yellow_cup` | 18 | 928 | 自动生成后抽查通过 18 条 |
| `0622/stack_white_bowl` | 29 | 1629 | 语义对但框偏松（padding/偏移约 10px） 29 条 |
| `0704_cleaned/pap_pink_cube` | 6 | 333 | 人工逐条改过框 6 条 |
| `0705_piper/pap_banana` | 9 | 469 | 整类重标过（2026-08-09，annotation_version 3） 9 条 |
| `0705_piper/pap_blue_block` | 6 | 272 | 整类重标过（2026-08-09，annotation_version 3） 6 条 |
| `0705_piper/pap_mango` | 5 | 255 | 整类重标过（2026-08-09，annotation_version 3） 5 条 |
| `0705_piper/pap_orange_block` | 7 | 295 | 整类重标过（2026-08-09，annotation_version 3） 7 条 |
| `0705_piper/pap_tennis` | 8 | 366 | 整类重标过（2026-08-09，annotation_version 3） 8 条 |
| `0705_piper/stack_gray_cup` | 3 | 212 | 整类重标过（2026-08-09，annotation_version 3） 3 条 |
| `0706_piper/pap_white_big_cube` | 5 | 247 | 人工逐条改过框 5 条 |
| `0706_piper/pap_white_cube` | 7 | 269 | 人工逐条改过框 7 条 |
| `zwz_0723/zwz_task_1` | 9 | 480 | 自动生成后抽查通过 9 条 |
| `zwz_0723/zwz_task_2` | 8 | 406 | 自动生成后抽查通过 8 条 |
| `zwz_0723/zwz_task_3` | 9 | 471 | 自动生成后抽查通过 9 条 |
| `zwz_0723/zwz_task_4` | 5 | 255 | 自动生成后抽查通过 5 条 |
| `zwz_0723/zwz_task_5` | 9 | 477 | 人工逐条改过框 9 条 |
| `zwz_0723/zwz_task_6` | 7 | 363 | 自动生成后抽查通过 7 条 |
| `zwz_0724/dif_wrist_and_gripper/task_1` | 6 | 314 | 人工逐条改过框 6 条 |
| `zwz_0724/dif_wrist_and_gripper/task_2` | 3 | 142 | 人工逐条改过框 3 条 |
| `zwz_0724/dif_wrist_and_gripper/task_3` | 4 | 179 | 人工逐条改过框 4 条 |
| `zwz_0724/dif_wrist_and_gripper/task_4` | 3 | 167 | 自动生成后抽查通过 3 条 |
| `zwz_0724/easy_spatial_left_right/task_10` | 7 | 356 | 自动生成后抽查通过 7 条 |
| `zwz_0724/easy_spatial_left_right/task_11` | 5 | 216 | 自动生成后抽查通过 5 条 |
| `zwz_0724/easy_spatial_left_right/task_12` | 6 | 271 | 自动生成后抽查通过 4 条、人工逐条改过框 2 条 |
| `zwz_0724/easy_spatial_left_right/task_5` | 4 | 159 | 自动生成后抽查通过 4 条 |
| `zwz_0724/easy_spatial_left_right/task_6` | 5 | 249 | 人工逐条改过框 5 条 |
| `zwz_0724/easy_spatial_left_right/task_7` | 4 | 204 | 自动生成后抽查通过 2 条、人工逐条改过框 2 条 |
| `zwz_0724/easy_spatial_left_right/task_8` | 8 | 464 | 人工逐条改过框 8 条 |
| `zwz_0724/easy_spatial_left_right/task_9` | 6 | 314 | 人工逐条改过框 6 条 |
| **合计** | **233** | **12011** | |

## 没用的样本

| 类别 | rollout | 样本 | 原因 |
|---|---:|---:|---|
| `0622/stack_white_bowl` | 1 | 2 | 已排除：整条轨迹没有 GRASP 也没有 RELEASE，不是抓放任务 |
| `0627_cleaned/pap_banana` | 1 | 24 | 已排除：有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标 |
| `0627_cleaned/pap_gray_mug` | 2 | 23 | 已排除：有 RELEASE 没有 GRASP，是放置/退回片段，物体从没被抓起 |
| `0627_cleaned/pap_gray_mug` | 2 | 65 | 已排除：有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标 |
| `0627_cleaned/pap_mango` | 1 | 14 | 已排除：有 RELEASE 没有 GRASP，是放置/退回片段，物体从没被抓起 |
| `0627_cleaned/pap_mango` | 3 | 99 | 已排除：有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标 |
| `0627_cleaned/stack_pink_cup` | 2 | 55 | 已排除：有 RELEASE 没有 GRASP，是放置/退回片段，物体从没被抓起 |
| `0627_cleaned/stack_pink_cup` | 2 | 44 | 已排除：有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标 |
| `0704_cleaned/pap_mango` | 2 | 124 | 已排除：有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标 |
| `0704_cleaned/pap_orange_block` | 4 | 165 | 已排除：有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标 |
| `0704_cleaned/rearrange_show` | 6 | 316 | 已排除：任务没有单一可见的放置落点，一个框加一个十字表达不了 |
| `0705_piper/pap_blue_block` | 2 | 7 | 已排除：有 RELEASE 没有 GRASP，是放置/退回片段，物体从没被抓起 |
| `0705_piper/pap_blue_block` | 2 | 95 | 已排除：有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标 |
| `0705_piper/pap_orange_block` | 1 | 51 | 已排除：有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标 |
| `0705_piper/pap_tennis` | 1 | 44 | 已排除：有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标 |
| `0705_piper/pap_wrench` | 4 | 122 | 已排除：任务没有单一可见的放置落点，一个框加一个十字表达不了 |
| `0705_piper/rearrange_show` | 2 | 97 | 已排除：任务没有单一可见的放置落点，一个框加一个十字表达不了 |
| `0705_piper/stack_gray_cup` | 1 | 62 | 已排除：有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标 |
| `zwz_0723/zwz_task_2` | 1 | 19 | 已排除：有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标 |
| `zwz_0724/easy_spatial_left_right/task_7` | 1 | 31 | 已排除：整条轨迹没有 GRASP 也没有 RELEASE，不是抓放任务 |
| `zwz_0802_long_collect/task_2` | 11 | 502 | 0802 long collect 整批不做：long-horizon 多步收集，中间有多个 subgoal |
| `zwz_0802_long_collect/task_3` | 7 | 1246 | 0802 long collect 整批不做：long-horizon 多步收集，中间有多个 subgoal |
| `zwz_0802_long_collect/task_4_1` | 12 | 2074 | 0802 long collect 整批不做：long-horizon 多步收集，中间有多个 subgoal |
| `zwz_0802_long_collect/task_4_2` | 17 | 3532 | 0802 long collect 整批不做：long-horizon 多步收集，中间有多个 subgoal |
| `zwz_0802_long_collect/task_5` | 12 | 1760 | 0802 long collect 整批不做：long-horizon 多步收集，中间有多个 subgoal |
| **合计** | **100** | **10573** | |

## 筛选规则

规则读 `MVTOKEN/visual_prompt_qa.json` 和 `MVTOKEN/visual_prompt_exclusions.json`，没有硬编码名单：

- 保留 `category_verdict` ∈ {fixed, loose, ok, reannotated}；
- 丢掉 `visual_prompt_exclusions.json` 里登记的（这些 rollout 根本没有标注文件 —— 轨迹缺 GRASP、缺 RELEASE，或者任务本身表达不了）；
- 丢掉整个 `zwz_0802_long_collect` root —— long-horizon 多步收集，中间有多个 subgoal，一个源框加一个目标十字表达不了（2026-08-10 决定）。

**`loose` 保留了**：这些框（全是 `0622/stack_white_bowl`）偏移最多约 10px，但仍然落在正确的物体上，而绿框在这里要传达的就是「哪个物体」—— 模型输出的是动作 token，不是框。
想要严格集就把构建脚本里 `KEEP_VERDICTS` 的 `loose` 去掉重跑。

`_horizon_flip` 不会出现，因为基础集本来就没有；训练也不使用翻转数据。

## 质检结论对照表

| verdict | 含义 |
|---|---|
| `ok` | 自动生成后抽查通过 |
| `reannotated` | 整类重标过（2026-08-09，annotation_version 3） |
| `fixed` | 人工逐条改过框 |
| `loose` | 语义对但框偏松（padding/偏移约 10px） |
| `mostly-ok` / `out_of_scope` | 大体可用，target 有时圈住一组物体 / 判定不做视觉提示标注；只出现在 0802 long collect，本实验没用到 |
