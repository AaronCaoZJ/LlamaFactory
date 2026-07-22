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
- @/workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/ms_0717
- **238 rollouts / 12738 pairs, 5 子集 = 2 种闭环方案**(两个主力各 100 集完全随机布局:`ms0717_blockstack_follow`=BlockStack 堆叠/方案D,`ms0717_blockpap_oracle_wide`=BlockPAP 取放/方案A)(生成脚本 @AgentRobot/scripts/maniskill/,README 同目录)
- 相机对齐 BlockPAP real2sim(标定前视 external_cam 640×480 + 居中腕相机 256²,wrist 已按部署 transform 翻转);每 token ≙ 2 cm,**每帧后严格沿单轴移动**
- `ms0717_blockpap_oracle_wide`(方案 A,**100 集**,block+coaster 各在 ~22×43cm 独立随机,间距≥12cm,100/100 成功)· `ms0717_blockpap_oracle`(方案 A,24 集,BlockPAP 原生 random——**coaster 仅 4×2cm 抖动**,多样性差)· `ms0717_{pick,stack}cube_follow`(方案 D,各 7 集);步长 19.6-19.7±0.3-0.8 mm,斜向步 0%,每集过成功校验
- 已知偏斜:`MV_BACK` 仅 3.4%(机械臂静止位 x≈0.34 在所有物体前方,取物恒为 +X);详见目录 README
- v3 prompt;可视化 dataset_browser.html + 各子集 preview.mp4 在数据目录内