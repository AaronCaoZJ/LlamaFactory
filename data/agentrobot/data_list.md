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