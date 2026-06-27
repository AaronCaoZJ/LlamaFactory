我现在需要基于 llamma factory 进行 vlm 模型 qwen 3.5 27b 的训练，你需要协助我完成下面几件事，请规划并按顺序解决：

1. git clone https://github.com/AaronCaoZJ/LlamaFactory.git

2. 参考官方文档，尤其是查找 qwen 3.5 27b 最佳实践 https://llamafactory.readthedocs.io/zh-cn/latest/，给我一套 lora 微调的具体方案和流程，写入 LlamaFactory/lora_ft.md，请注意也在其中增加一段，告诉我 lora 的原理，各种 lora 的

3. 参考官方文档，尤其是其中的对于数据的构建、预处理、格式等的描述，我现在想要构建一些图文对数据，大概是两个个相机视角图片，指令是上下左右前后的 token，然后能够让 vlm 更适应这种形式的问答，参考这个自定义数据集文档给我完整的数据集构建流程，并写入 LlamaFactory/data.md

以上两个文件和 git clone 操作允许直接操作，直接写入。