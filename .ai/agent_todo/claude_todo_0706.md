<!-- 之前我们使用 0705 的 piper 数据直接训了一个 qwen 35 9b，现在我又增加了 0706 数据，两个任务名称分别是 pap_white_cube=Pick up the white cube and place it on the pink cube，pap_white_big_cube=Pick up the white block and place it on the orange block。

然后你要帮我混合 piper 数据和 mix_22_27_04 数据。混合的时候需要考虑几种路线：

路线一是上下反转所有 piper 的输入图像，然后和 mix_22_27_04 的 franka 数据使用统一的 prompt v3。然后直接混合训练。

路线二是不反转图像，但是将 piper 数据中的 back 和 fwd 直接调换，在 piper 部署的时候，输出 back 实际往前走，这就和 franka 视角下的往画面上走是 back 的语意对齐了。

路线三是仅使用 v4 的两种 hardware aware 构造 prompt，对图像和 token 不做任何更改。

请你分别实现。构造出来的数据集分别放在 @/workspace1/zhijun/LlamaFactory/data/agentrobot/MVTOKEN/mix_22-06_fk-pp/01_flip_img 02_exchange_token 03_just_mix

相应的在训练 config 中 @/workspace1/zhijun/LlamaFactory/examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp 构造相应的 yaml 文件，对应三条路线，使用 9b 为模版。在 dataset_info 中构造相应的数据集。在 @/workspace1/zhijun/LlamaFactory/scripts/qwen3_5/train 中写一个 mix_fk-pp_train.sh 同时启动三个训练，分在034567六张卡上训练，目前5卡正在跑 vllm server，当你准备好所有训练所需的东西后，请你帮我关闭并开始训练 -->