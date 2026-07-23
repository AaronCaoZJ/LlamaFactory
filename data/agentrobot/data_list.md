# File path
- RAW Data: HF `YanzheChen/real-data`
- Processed data: HF `aaroncaozj/MVTOKEN_datasets`
- Data processing scripts: @**LlamaFactory**/data/agentrobot/
- Datasets config: @**LlamaFactory**/data/dataset_info.json
- Prompt txt files: @**AgentRobot**/prompts/

# Datasets
### mvtoken_0622/v(n)
- @/workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/0622
- 4 * tasks, ~80 rollouts (3k+ data pairs)
- Franka data only
- v0, v1, v2 (not good), v3 correspond to 4 versions of prompts @/workspace1/zhijun/AgentRobot/prompts

### mix_22_27/v3
- @/workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/mix_22_27
- Mix of ~80 rollouts from 0622 and *~12 from 0627*
- The 0627 dataset mainly focuses on solving 抓取物体时*前后*判断不准. It contains two categories: *grasp/* and *release/*. Use `clean_grasp_release.py` to remove `release` from grasp folder and `grasp` from release folder, so as to construct augmented data for grasp and release respectively.
- Franka data only
- v3 prompt only

### mix_22_27_04/v3
- @/workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/mix_22_27_04
- Mix of ~80 rollouts from 0622, ~12 from 0627 and *~18 from 0704*
- The 0627 dataset mainly focuses on solving 抓取物体时*左右*判断不准，或不懂的*抬起 gripper 重新抓*. Similarly, using `clean_grasp_release.py` to remove `release` from grasp folder and `grasp` from release folder.
- Franka data only
- v3 prompt only

### piper_0705/v3
- @/workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/piper_0705
- *~50 from Piper*
- Piper data only
- v3 prompt only

### mix_22-06_fk-pp
- /workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/mix_22-06_fk-pp
- Mix of data till 0706
- ~120 from Franka and ~60 from Piper
- v3 for just_mix train, and v4 for hardware aware prompt, LlamaFactory/scripts/qwen3_5/train/mix_fk-pp_train.sh for more details

### ms_0717 (ManiSkill 仿真, 2 cm 原子, 0717)
- @/workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/ms_0717 (README 在目录内,生成脚本 @AgentRobot/scripts/maniskill/)
- **300 rollouts / 15349 pairs, 3 个数据集各 100 集**;每 token ≙ 2 cm,**每帧后严格沿单轴移动**(斜向步 0%)
- **三者相机完全统一**(agentview = BlockPAP 标定前视 RealSense 渲染 640×480,**2026-07-23 起存盘为 256×256 letterbox**(resize_with_pad,与真机数据格式一致,原图备份在 `_backup_agentview_640x480/`);wrist = 居中腕相机 256² 已按部署 transform 翻转),可直接混训
  - `ms0717_blockpap_oracle_wide` — BlockPAP 取放,**方案 A**(特权 oracle 直接规划),5338 样本
  - `ms0717_blockpap_follow` — BlockPAP 取放,**方案 D**(连续 demo → 2cm 原子重执行 + 成功校验),5826 样本
  - `ms0717_stackcube_follow` — **官方 StackCube-v1** 堆叠(相机换成 BlockPAP 机位,数值验证逐位一致),方案 D,4185 样本
- 两个 blockpap 子集同场景同任务同相机,仅 tokenize 方式不同 → 干净的 **A/D 对照**
- 训练:`scripts/internvl/train/ms0717_{oracle_wide,blockpap_follow,stackcube_follow}_2b.sh`(InternVL3.5-2B LoRA,超参一致)
- 已知偏斜:`MV_BACK` 仅 3-5%(机械臂静止位在所有物体前方,取物恒为 +X);详见目录 README
- v3 prompt;可视化 dataset_browser.html + 各子集 preview.mp4 在数据目录内
