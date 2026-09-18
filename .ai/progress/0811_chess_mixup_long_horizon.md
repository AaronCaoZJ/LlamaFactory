# chess mixup long-horizon 混合训练（Qwen3.5-9B LoRA）

目标：用 `chess_mixup_long_horizon` + `chess_mixup_single_task` 的**混合 loss** 训一版 long-horizon 策略。
base = `/workspace1/zhijun/hf_download/models/Qwen3.5-9B_mix-fk-pp_merged`（stock 9B + `mix_22-06_fk-pp/02_exchange_token` LoRA 合并的 MVTOKEN 策略，非原始 VLM）。

## TODO

- [x] 1. 数据分析：样本量 / 动作分布 / 帧重叠 / 标签冲突
- [x] 2. 数据落地：前缀重写 + 逐张 stat 校验
- [x] 3. 注册 `data/dataset_info.json`
- [x] 4. 新增训练 yaml
- [x] 5. 更新训练脚本
- [x] 6. 静态校验（不占 GPU）
- [ ] 7. 启动训练 + 跟 loss ← **待用户自行启动**
- [ ] 8. 评测（重点看子任务边界帧的 DONE）

## 1. 数据分析结论

源：`/workspace1/zechen/DATA/chess/{mixup_long_horizon,mixup_single_task}/rollout_lite.json`

| | long_horizon | single_task |
|---|---|---|
| 样本数 | 1690 | 1723 |
| rollout 数 | 11（000~010） | 11（同一批） |
| 唯一帧（图片对） | 1679 | 1679 |
| DONE | 11 | 44 |
| 其余动作分布 | 完全相同 | 完全相同 |
| instruction 长度 | 1319~1363 字符 | 921~991 字符 |

**两者是同一批 rollout 的两种 prompt 标注**，不是两批数据：

- 图片帧集合**完全一致**（交集 1679，两边各自都没有独占帧）。
- `single_task` 的 prompt 只给**当前子任务**，每个子任务结束记一次 `DONE` → 4 子任务 × 11 rollout = 44。
- `long_horizon` 的 prompt 给**完整任务列表**（4 个子任务按序），只有整条 rollout 结束才 `DONE` → 11。

**冲突监督（重要，非数据错误）**：33 帧在两个数据集里标签不同 ——
`long` 标 `MV_UP`(23) / `MV_RIGHT`(10)（继续做下一个子任务），`single` 标 `DONE`（当前子任务已完成）。
33 = 44 − 11，正好是前 3 个子任务的边界帧。

这意味着混合训练时，**`DONE` 的判据必须由 instruction 的任务范围来区分**。这是混合 loss 的价值所在（模型学会"完成的是哪一层任务"），也是最大风险点：如果区分度不够，long-horizon 闭环会在子任务边界**提前吐 DONE 而终止**。评测时要专门看这 33 帧。

## 2. 数据落地

`data/agentrobot/MVTOKEN/chess/relocate_images.py` 从单数据集扩成多数据集（保留 `mixup_filtered` 原行为）：

```bash
python3 data/agentrobot/MVTOKEN/chess/relocate_images.py                    # 全部 3 个
python3 data/agentrobot/MVTOKEN/chess/relocate_images.py mixup_long_horizon # 单个
```

结果（前缀 `/users/zechen/...` → 本仓，逐张 stat）：

- `mixup_filtered`: 1690 样本 / 3380 图，全部可读
- `mixup_long_horizon`: 1690 样本 / 3380 图，全部可读
- `mixup_single_task`: 1723 样本 / 3446 图，全部可读

三者共用同一份图片树 `chess/mixup`（软链到 `/workspace1/zechen/DATA/chess/mixup`），只有 prompt 不同，无需重复拷贝像素。

## 3. 数据集注册

`data/dataset_info.json` 新增 `chess_mixup_long_horizon` / `chess_mixup_single_task`，列映射与 `chess_mixup_filtered` 一致（instruction/input/output/images）。JSON 已验证合法（30 条）。

## 4. 训练 yaml

`examples/train_lora/qwen3_5_9b/chess/qwen3_5_9b_chess_mixup_long_horizon.yaml`

| 参数 | 值 | 说明 |
|---|---|---|
| model_name_or_path | `../hf_download/models/Qwen3.5-9B_mix-fk-pp_merged` | 已 merge 的 MVTOKEN 策略 |
| template | `qwen3_5_nothink` | 与该 merged 模型 export 时一致 |
| dataset / mix_strategy | 两集 / `concat` | 3413 样本，1690:1723 ≈ 1:1 等权 |
| lora | r=64, α=128, dropout=0.05, target=all | 同师兄 |
| freeze_vision_tower / projector | true / false | 同师兄 |
| image_max_pixels / cutoff_len | 65536 / 2048 | 实测 seq 357~474，余量 1574 |
| bs × accum × cards | 4 × 8 × 1 = **32**（单卡） | 双卡时 accum 自动变 4 |
| lr / sched / warmup / clip | 1.0e-4 / cosine / 0.1 / 1.0 | 同师兄 |
| num_train_epochs | **20** | 107 步/epoch → **2140 步** |
| deepspeed | ds_z2 | 同师兄 |

epoch=20 的依据：本仓已验证的 `chess_mixup_filtered`（1690 样本 × 40 epoch ≈ 2110 步）收敛良好；这里样本翻倍，20 epoch 给到同一量级的优化步数（2140 步），且每帧在**每种 prompt scope 下**仍各被看 20 次。同时与师兄 InternVL 版的 107 steps/epoch × 20 = 2133 步完全对齐，便于横向比较。

**加权变体**（未启用，写在 yaml 注释里）：若闭环出现子任务边界提前 DONE，改成
`mix_strategy: interleave_over` + `interleave_probs: 0.65,0.35` 偏向 long-horizon。
注意 `interleave_probs` 在 `concat` 下会被直接拒绝。

## 5. 训练脚本

`scripts/qwen3_5/train/chess_mixup_train.sh`（在原脚本上扩展，未新建重复脚本）：

- `VARIANT=long_horizon`（默认）/ `filtered`，各自对应 config、数据校验、log 名，旧实验仍可复现。
- **按卡数自动推导 `gradient_accumulation_steps`**，把 eff_bs 恒定钉在 32（1 卡→8，2 卡→4，4 卡→2），通过命令行 `gradient_accumulation_steps=N` 覆盖 yaml。
  这是防止手改 yaml 时 eff_bs 静默漂移、把 lr 曲线改掉的关键改动。
- 新增 `DRY_RUN=1` 只打印将要执行的命令。
- 启动前逐个校验所需 json 与图片软链，缺失时直接报错并给出重建命令。

## 6. 静态校验结果（未占用 GPU）

走 LlamaFactory 真实数据管线（parser + tokenizer + template + mm plugin），CPU 上跑 8+8 样本：

- yaml 解析通过，`dataset` 两集、`mix_strategy=concat` 生效，concat 后样本数 = 两集之和
- 每样本 **128 个 image pad token**（2 视图 × 64），符合 `image_max_pixels=65536`
- 监督只落在动作 token 上（`MV_FWD` 被切成 3 个 sub-token + `<|im_end|>` = 4 个受监督位，prompt 全 -100）
- seq len min/mean/max = **357 / 417 / 474**，`cutoff_len=2048` 余量充足
- 脚本 `bash -n` 通过；单卡/双卡/filtered/非法 variant 四种 DRY_RUN 行为均正确

## 7. 启动方式

```bash
cd /workspace1/zhijun/LlamaFactory

# 先确认卡：写脚本时（08-11）只有 GPU2 空闲，其余 7 张均被占
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

bash scripts/qwen3_5/train/chess_mixup_train.sh              # 单卡 GPU2，accum 自动=8
GPUS=2,3 bash scripts/qwen3_5/train/chess_mixup_train.sh     # 双卡，accum 自动=4
FOREGROUND=1 bash scripts/qwen3_5/train/chess_mixup_train.sh # 前台
DRY_RUN=1 bash scripts/qwen3_5/train/chess_mixup_train.sh    # 只看命令

tail -f saves/qwen3.5-9b/robot/chess/logs/long_horizon.log
```

- 产出：`saves/qwen3.5-9b/robot/chess/mixup_long_horizon`（loss 曲线 `training_loss.png`）
- 耗时估算：历史双卡 2120 步 = 2h52m（4.7 s/it）。单卡 accum 翻倍 → 2140 步约 **5.5~6 小时**；双卡约 3 小时。
- 显存：9B LoRA + ZeRO-2 + 冻结视觉塔，seq ~0.5k，约 30~45GB/卡。

## 8. 后续注意

- **评测重点**：那 33 个子任务边界帧。要分别用两种 prompt scope 测同一帧，确认 long-horizon prompt 下不会提前 DONE。
- 训练产出的 `adapter_model.safetensors` 默认 0600，同事读不了，需要 `chmod 664`。
- 闭环部署时 prompt（含 `Recent moves, newest first:` 历史行）必须与训练分布逐字节一致。
