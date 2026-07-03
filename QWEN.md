# Qwen 家族速览（VL / 数字系列 / Omni 的关系与差异）

> 整理日期：2026-06-28。本文用于在 LlamaFactory 中微调 Qwen 系列模型前快速理解各分支的定位、架构与命名逻辑。
> 注：部分细节（尤其 Qwen3.5）综合自官方博客、HF model card、技术报告与第三方解读，可能存在版本差异，落地前请以对应模型的官方 model card 为准。

---

## 1. 一句话结论

Qwen（通义千问，阿里巴巴 Qwen 团队）是一个**按"代际数字 + 能力分支"组织的模型矩阵**：

- **数字系列**（Qwen / Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5）= 纯文本（LLM）主干，是整个家族的"底座"和代际节奏。
- **能力分支**在每一代主干基础上派生，命名为 `Qwen{N}-{能力}`：
  - `-VL`：视觉-语言（图像/视频理解）。
  - `-Omni`：全模态（文本+图像+音频+视频输入，文本+语音输出）。
  - 其他：`-Coder`（代码）、`-Math`（数学）、`-Audio`（音频理解）、`-Embedding`/`-Reranker`（检索）。
- **趋势**：从 Qwen3 → Qwen3.5，主干本身正在**走向原生多模态**（文本-视觉从预训练阶段就融合），分支与主干的界限在变模糊。

---

## 2. 命名与分支逻辑

| 维度 | 含义 | 例子 |
|------|------|------|
| 数字代际 | 第几代，决定架构底座与训练数据规模 | Qwen2.5、Qwen3、Qwen3.5 |
| 能力后缀 | 该代主干的某个专用分支 | `-VL` / `-Omni` / `-Coder` / `-Math` |
| 规模标记 | 参数量；MoE 用 `总量-A激活量` | `32B`（稠密）、`235B-A22B`（MoE，总 235B 激活 22B） |
| 训练后类型 | 推理/对齐形态 | `-Instruct`（直接回答）、`-Thinking`（带思维链）、`-Base`（基座） |
| 服务规格 | API/托管档位（多为闭源） | `-Max`、`-Plus`、`-Flash`、`-Light` |

阅读技巧：`Qwen3-VL-235B-A22B-Thinking` = 第3代 · 视觉语言分支 · MoE(总235B/激活22B) · 思考版。

---

## 3. 家族时间线（主干 + 分支）

| 主干（LLM） | 时间 | 规模 | 关键多模态/专用分支 |
|------|------|------|------|
| Qwen（1代） | 2023-09 | 1.8B / 7B / 14B / 72B | Qwen-VL、Qwen-Audio（2023） |
| Qwen1.5 | 2024-02 | 0.5B–72B + MoE | — |
| Qwen2 | 2024-06 | 0.5B–72B、57B-A14B | Qwen2-VL、Qwen2-Audio、Qwen2-Math |
| Qwen2.5 | 2024-09 | 0.5B–72B | Qwen2.5-VL、Qwen2.5-Coder、Qwen2.5-Omni、Qwen2.5-1M（长上下文） |
| Qwen3 | 2025-04 | 稠密 0.6B–32B；MoE 30B-A3B / 235B-A22B | Qwen3-VL、Qwen3-Omni、Qwen3-Coder、Qwen3-Embedding |
| Qwen3.5 | 2026-02 | 397B-A17B（旗舰，基于 Qwen3-Next） | 原生多模态融入主干；Qwen3.5-Omni、Qwen3-Coder-Next |

> 注：Qwen3-Max（>1T 参数）、Qwen3.5-Plus 等为闭源 API；开源权重通常走 Apache 2.0，发布于 Hugging Face / ModelScope。早期 1 代 72B 与部分大模型曾为研究/专有许可。

---

## 4. 数字系列（纯文本 LLM 主干）演进

- **架构基线**：早期基于类 LLaMA Transformer，标配 RoPE、GQA、SwiGLU、Pre-Norm RMSNorm。
- **Qwen2/2.5**：激进扩数据（up to 18T tokens）、强化多语种、DPO/GRPO 对齐、长上下文（Qwen2.5-1M 用 Dual Chunk Attention + YaRN 到 1M）。
- **Qwen3**：36T tokens、119 种语言；**统一稠密 + MoE 两种形态**；引入"思考/非思考"双模式（同一模型可切换是否输出思维链）。
- **Qwen3.5**：见下一节，转向 **Qwen3-Next 架构 + 原生多模态**。

---

## 5. 视觉分支 Qwen-VL 系列（重点）

定位：图像/视频 + 文本的多模态理解（OCR、定位 grounding、文档、视频）。

| 版本 | 时间 | 要点 |
|------|------|------|
| Qwen-VL | 2023-08 | ViT 视觉编码器 + 位置感知 adapter 接到 LLM，开启图文融合、OCR、grounding |
| Qwen2-VL | 2024 | 动态分辨率、原生多分辨率处理 |
| Qwen2.5-VL | 2025-01 | 更强文档/视频理解 |
| **Qwen3-VL** | 2025-09 | 当前主力开源多模态，技术报告 arXiv:2511.21631 |

**Qwen3-VL 关键升级：**
- 规模齐全：稠密 **2B / 4B / 8B / 32B**，MoE **30B-A3B / 235B-A22B**；各有 `-Instruct` 与 `-Thinking`。
- 上下文 **256K**（YaRN 扩展 RoPE），支持文本/图像/视频交错输入。
- **Interleaved-MRoPE**：时间/高/宽多轴旋转位置编码。
- **DeepStack**：把 ViT 多层级特征注入 LLM 浅层，做层次化视觉-语言融合。
- 视频**显式时间戳对齐**，提升时序理解。
- 衍生 `Qwen3-VL-Embedding` 等检索模型。

---

## 6. 全模态分支 Qwen-Omni 系列

定位：**一个模型统一感知文本/图像/音频/视频，并实时输出文本与自然语音**。

- **核心架构：Thinker–Talker**
  - **Thinker**：负责理解与推理，产出文本/语义。
  - **Talker**：负责把语义合成为流式语音。
  - 两者解耦的好处：可在"想"和"说"之间插入 RAG、安全过滤、函数调用。
  - Qwen3/3.5-Omni 中 Thinker 与 Talker 都采用 **Hybrid-Attention MoE** 设计。
- **模态**：输入 = 文本+图像+音频+视频（单次推理）；输出 = **文本 + 语音**。
  - ⚠️ 明确边界：Omni **不生成图像/视频**，只理解它们；只生成文字和语音。
- **演进**：Qwen2.5-Omni（2025-03）→ Qwen3-Omni（技术报告 arXiv:2509.17765）→ Qwen3.5-Omni。
- **语言覆盖（以 3.5-Omni 量级）**：语音识别 ~113 种语言/方言，语音生成 ~36 种；上下文 256K（≈10+ 小时音频）；Flash 档流式语音延迟可低至 ~234ms。
- **服务档位**：Plus（高质量、含音色克隆）/ Flash（生产默认）/ Light（边缘/低成本）。

---

## 7. Qwen3.5 模型细节（2026-02 旗舰）

**首发：Qwen3.5-397B-A17B**（总 397B / 每 token 激活 17B 的 MoE）。

**底座：Qwen3-Next 架构**，关键点：
- **混合注意力**：线性注意力 **Gated DeltaNet（Gated Delta Networks）** + **Gated Attention** 混合，兼顾长序列效率与表达力。
- **高稀疏度 MoE**：397B 总参，仅激活 17B。
- **Multi-Token Prediction（MTP）**：多 token 预测，提升训练信号与解码效率。
- **训练稳定性优化** + **FP8 训练管线**（激活显存约降 50%，>10% 提速）。

**原生多模态**：文本-视觉**从预训练起就融合**（不再是后接视觉模块），本身即视觉-语言模型。视觉基准示例：MathVision 88.6 / OmniDocBench 90.8 / OCRBench 93.1。

**规格**：
- 上下文：原生 **262,144（256K）**，可扩展到 **~1M**。
- 词表：**250K**（较 Qwen3 的 150K 扩大，多数语言编解码效率提升 10–60%）。
- 语言：**201 种**语言/方言（Qwen3 为 119）。
- 三种推理模式：自适应思考+工具调用（默认）/ 深度思维链 / 即时无思考。
- 吞吐：32K/256K 上下文下解码速度约为 Qwen3-Max 的 **8.6× / 19×**，质量相当。
- 许可：开源权重 **Apache 2.0**（HF / ModelScope）；**Qwen3.5-Plus** 为闭源托管（提供 1M 上下文）。
- 后续：Qwen3.6（如 Qwen3.6-35B-A3B，2026-04，Apache 2.0）、Qwen3-Coder-Next 等。

---

## 7.5 深入：Qwen3.5 的注意力、Mamba 与 Gated Attention

Qwen3.5 / Qwen3-Next 不是用单一一种注意力，而是**两种层交替堆叠，比例约 3:1**：
**3 份 Gated DeltaNet + 1 份 Gated Attention**。下面分别说清楚，并回答"Mamba 是不是它的组件"。

### (1) Qwen3.5 里有 Mamba 这个结构吗？—— 没有，但有"亲戚"

**先给结论：模型里没有一个叫 Mamba 的层，但有一种和 Mamba 同源的层，叫 Gated DeltaNet。**

打个比方理解它们的关系：

```
普通注意力（softmax）：每生成一个词都要回看前面所有词
        → KV cache 越存越大，序列越长越慢、越费显存

Gated DeltaNet（和 Mamba 同一类）：只维护一个"固定大小的记忆"
        → 不管序列多长，记忆大小不变 → 省显存、长文本快
```

- **Mamba** 是这类"固定记忆"模型的代表作。
- **Gated DeltaNet** 是它的"近亲"：核心记忆更新方式来自另一条线（DeltaNet），但**借用了 Mamba2 的门控机制**——论文名字就直说了《Gated Delta Networks: **Improving Mamba2** with Delta Rule》。
- 所以准确说法是：**Qwen3.5 用的是"基于 Mamba2 改进的 Gated DeltaNet"，不是 Mamba 本身。**

> 之前文档里写的"Mamba 血统"只是比喻，意思就是这层和 Mamba 同源、并复用了 Mamba 那套底层工程代码（见下一节）——这正是训练时会冒出 mamba 字样的原因。

### (2) 训练报 "mamba" 相关错误 → 多半是缺内核包

因为和 Mamba 同源，Gated DeltaNet 复用了 Mamba 的底层 CUDA 内核。这些内核要单独装，没装好就报错。常见三件套：

| 依赖包 | 作用 | 典型报错 |
|------|------|----------|
| `causal-conv1d` | 因果一维卷积内核 | ImportError / 未编译 / 和 CUDA 版本对不上 |
| `mamba-ssm` | Mamba 选择性扫描内核 | 找不到模块、编译失败、缺算子 |
| `flash-linear-attention`（`fla`） | Gated DeltaNet 的高效实现 | 版本太旧、缺对应算子 |

**排查顺序（结合 LlamaFactory）：**
1. 先确认 **transformers 版本**够新、支持 Qwen3-Next/Qwen3.5（旧版本根本没有对应的模型代码，会直接报结构错）。
2. 按官方 model card 装 `causal-conv1d` / `mamba-ssm` / `flash-linear-attention`，版本要和你的 **CUDA / PyTorch** 对齐；**优先用预编译 wheel**，少用源码编译，省得踩 CUDA 工具链的坑。
3. 看完整报错：是 **import 失败**（缺包/版本不对）还是 **运行时算子失败**（CUDA 架构/编译不匹配）？两者解法不同，别凭感觉猜。
4. 这类模型通常是 **MoE + 线性注意力**，显存和并行配置和普通稠密模型不一样，照着官方推荐配。

### (3) Gated Attention 是什么？

就是**普通注意力 + 两个小改动**，让训练更稳：

- **输出门控**：注意力算完结果 `Y` 后，再乘一个 0~1 之间的"开关" `σ(X·Wθ)`（σ 是 sigmoid），逐通道决定"这部分输出留多少"。
  公式：`Y' = Y ⊙ σ(X·Wθ)`。好处：消除 **Attention Sink / Massive Activation** 这类异常，数值更稳，能撑到 ~100 万 token 的超长上下文。
- **QK-Norm**：对 Q、K 先做一次归一化（RMSNorm），稳定训练。

> 出自 Qwen 团队 NeurIPS 2025 论文《Gated Attention for LLMs》。它保留了普通注意力"精确回看全文"的能力，正好补上 Gated DeltaNet 的短板——所以两者按 3:1 搭配：**大部分用省显存的 DeltaNet，少量用精确的 Gated Attention。**

---

## 8. 关系与差异总表

| 分支 | 输入 | 输出 | 架构特点 | 典型用途 |
|------|------|------|----------|----------|
| 数字系列（LLM） | 文本 | 文本 | Dense/MoE，思考双模式 | 通用对话、推理、Agent |
| **-VL** | 文本+图像+视频 | 文本 | ViT+LLM；MRoPE/DeepStack | 多模态理解、OCR、文档、视频 |
| **-Omni** | 文本+图/音/视频 | 文本+语音 | Thinker–Talker MoE | 实时语音助手、全模态交互 |
| -Coder | 文本/代码 | 代码 | LLM 主干代码特化 | 代码生成、补全、agentic coding |
| -Math | 文本 | 文本 | LLM 主干数学特化 | 数学推理 |

**核心区别记忆点：**
- VL **只读图、不说话**；Omni **能读图/音/视频、还能说话**；数字系列**纯文本**。
- 从 Qwen3.5 起，"主干"自己就具备原生多模态，VL 更像是其多模态能力的专门强化与小尺寸释放。

---

## 9. 与 LlamaFactory 的关系（落地提示）

- LlamaFactory 支持 Qwen 系列的 SFT / LoRA / QLoRA / DPO 等微调；选模型时注意：
  - **纯文本任务** → 选数字系列（Qwen3 / Qwen3.5 对应规模）。
  - **图文/视频理解任务** → 选 `Qwen*-VL`，需确认安装多模态依赖与正确的 `template`（如 `qwen2_vl` / `qwen3_vl` 类模板）及处理器（processor）。
  - **语音/全模态** → `Qwen*-Omni`，注意音频/视频数据管线与额外依赖，确认 LlamaFactory 版本是否已支持该 template。
- MoE 版本（如 `30B-A3B` / `235B-A22B`）显存与并行配置不同于稠密版，微调前先核对官方 model card 的 `chat_template`、上下文长度与 transformers 版本要求。
- 选 `-Instruct` 还是 `-Thinking` 取决于是否需要思维链；继续预训练/对齐基座则选 `-Base`。

---

## 10. 参考来源

- Qwen 官方博客：https://qwen.ai/blog?id=qwen3.5 ；Qwen3 博客：https://qwenlm.github.io/blog/qwen3/
- Qwen Wikipedia：https://en.wikipedia.org/wiki/Qwen
- GitHub：https://github.com/QwenLM/Qwen3 ，https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL 技术报告：https://arxiv.org/abs/2511.21631
- Qwen3-Omni 技术报告：https://arxiv.org/html/2509.17765v1
- Qwen-series 综述：https://www.emergentmind.com/topics/qwen-series-models
- Qwen3.5 解读：https://lushbinary.com/blog/qwen-3-5-developer-guide-benchmarks-architecture-integration-2026/ ；Qwen3.5-Omni：https://wavespeed.ai/blog/posts/what-is-qwen3-5-omni/
- Gated DeltaNet / Mamba2 关系：Sebastian Raschka https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/ ；论文《Gated Delta Networks: Improving Mamba2 with Delta Rule》
- Gated Attention：Qwen 团队 NeurIPS 2025《Gated Attention for LLMs》 https://openreview.net/forum?id=1b7whO4SfY ；https://sebastianraschka.com/llm-architecture-gallery/gated-attention/
- Qwen3.5 注意力混合解读：https://huggingface.co/blog/mlabonne/qwen35
