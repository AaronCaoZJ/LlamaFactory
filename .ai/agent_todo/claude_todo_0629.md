 现在又有两组数据，分别是 @/workspace1/zhijun/hf_download/datasets/MVTOKEN_RAW/0627/grasp 和 @/workspace1/zhijun/hf_download/datasets/MVTOKEN_RAW/0627/release 我希望清洗这两波数据，将 release 文件夹的 grasp 全部去除，将 grasp 文件夹的 release 全部去除。

 处理完后得到一个文件夹叫做 0627_cleaned 包含所有 rollout_xxx 文件夹，将这个文件夹移动到 LlamaFactory/data/agentrobot/MVTOKEN 下

 过程中用到的 py 脚本写下来存放在 LlamaFactory/data/agentrobot，命令行命令写进 process_data.sh