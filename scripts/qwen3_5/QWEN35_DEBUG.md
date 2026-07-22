# Qwen3.5 训练 / 推理 chat template 失配

> **结论**：训练侧模板统一为 `qwen3_5_nothink`；vLLM 服务必须挂
> `--chat-template scripts/qwen3_5/eval/chat_template_qwen3_5_lf.jinja`。
> 否则训推不匹配，会静默掉点。

## 1. 失配长什么样

| 渲染方 | 来源 | `assistant\n` 之后 |
|---|---|---|
| **LF `qwen3_5_nothink`**（训练真值） | `data/template.py` 的 Python slots。**训练不走 jinja**，两套实现互不校验 | *（空）* |
| 官方 jinja，默认 / `enable_thinking=true` | Qwen3.5 出厂 `chat_template`。LF 不覆盖它（见下），`saves/` 里存的也是这份 | `<think>\n` |
| 官方 jinja，`enable_thinking=false` | 同上，同一份 jinja 的另一个分支 | `<think>\n\n</think>\n\n` |
| 本仓库 `chat_template_qwen3_5_lf.jinja` | 手写：官方 jinja 只改 generation prompt 分支（§3） | *（空），训推一致* |

双方都叫 "no think"，实现却不是一回事：**LF 是一个 token 都不加，官方 jinja 是加一个闭合的空 think 块**。
所以 `enable_thinking=false` 救不了——它只是把「开着的 think」换成「闭合的空 think」。官方 jinja 只有这两种模式，
传什么参数都拼不出「什么都不加」。

**LF 为什么不覆盖 Qwen 的 jinja**：`Template.fix_jinja_template`（`template.py:273`）只在
`tokenizer.chat_template is None` **或** `replace_jinja_template=True` 时才写 jinja。`qwen3_5` / `qwen3_5_nothink`
都没开这个开关（全仓库只有 `custom` / `chatml` / `default` 等 7 个模板开了），而 Qwen3.5 出厂自带 chat_template
→ LF 原样放过，`save_pretrained` 再把这份出厂版抄进 checkpoint。

## 2. 后果：静默掉点，而非崩溃

失配是**末尾追加**型：训练 prompt 是推理 prompt 的完整前缀，没有信息丢失；多出来的 `<think>` 又是基座
极熟悉的 token，任务先验压得住 → **输出看起来完全正常**。代价却是实打实的：mikomiko tagger 实测
**microF1 掉 1.2pt**。

> 对比 Gemma-4（`scripts/gemma4/GEMMA4_DEBUG.md`）：那边丢的是整个 system turn，公共前缀仅 2 token，
> 直接复读 / 乱答。**Gemma 会崩，Qwen 只是静默变差——所以 Qwen 的更难发现。**

## 3. 解决方法

### 现方案（已实施，零重训）

服务端挂自定义 jinja。它相对官方 jinja 只改一处（generation prompt 分支），并保留 `enable_thinking=true` 后门：

```jinja
{#- 官方：总是发 think -#}                    {#- 本仓库：仅在显式要求时才发 -#}
{%- if enable_thinking ... is false %}        {%- if enable_thinking ... is true %}
    {{- '<think>\n\n</think>\n\n' }}              {{- '<think>\n' }}
{%- else %}                                   {%- endif %}
    {{- '<think>\n' }}
{%- endif %}
```

六个 server 脚本已全部挂载：`eval/start_vllm_server{,_9,_2,_0_8}.sh`（27B/9B/2B/0.8B）、
`mikomiko_tagger/infer_tag_2b.sh` 与 `mikomiko_grok_desc/infer_desc_9b.sh` 里的 `serve_vllm`。
已验证 **4 个尺寸 × 3 个场景（单轮 / 多轮 / 带 system），token 级全部一致**。

> `eval/` 与 `mikomiko_tagger/` 下两份 jinja **完全相同**，改一份要同步另一份。

### 备选（不改 jinja，但要重训所有模型）

让训练侧去迁就官方 jinja。实测逐字节一致，好处是摆脱自定义 jinja 的维护负担，代价是**已训模型全部作废**。

```yaml
template: qwen3_5          # 不是 _nothink
enable_thinking: false     # prompt 尾部拼空 think 块，不计 loss
```
```bash
vllm serve ... --default-chat-template-kwargs '{"enable_thinking": false}'   # 不挂 --chat-template
```

> 两条路**不可混用**：同一个 server 只能选一种 serve 配置，混着必错一边。

# 校验工具：`check_prompt_parity.sh`

改模板、换 server、加新 LoRA 之后必跑：

```bash
bash scripts/qwen3_5/eval/check_prompt_parity.sh                       # 默认 :8109 / piper_0705_v4_9
PORT=8109 LORA=dual_cloth_once \
  DATA=data/agentrobot/MVTOKEN/dual_cloth/v4/rollout_dual_once.json \
  bash scripts/qwen3_5/eval/check_prompt_parity.sh                     # 换被测对象
```

脚本用 `POST /tokenize`（不做推理，只按服务端当前 chat template 渲染 prompt 再切 token——这是唯一能看到
**vLLM 真正喂给模型那串 token** 的手段），做两级检查：

1. **尾部探针**：纯文本 PROBE，看 `assistant\n` 之后有没有混进 `<think>`。漏挂 `--chat-template` 在这步就会挂。
2. **逐 token 对拍**：拿数据集第一条真样本（带图），比对训练侧与服务端的完整 token 序列。
   只有这步能验证图像展开成多少个 `<|image_pad|>`——那是运行时算的，读 jinja 看不出来。

alpaca 与 sharegpt 两种样本格式都支持（后者如 dual 的 `--chain`）。退出码：0 一致 / 1 失配 / 2 server 不可达。

实测通过（9B + `piper_0705_v4`，2 张 256×256 图）：训练侧 337 tok / 服务端 337 tok，128 个
`<|image_pad|>` 完全对齐，逐 token 一致；端到端推理 5/5 命中。

> 尾部探针那一级已在两个推理入口做成**启动断言**（`check_prompt_parity()`）：
> `eval/infer.py` 与 `mikomiko_tagger/infer_mikomiko.py` 每次访问 server 前先验一次，
> 渲染出 `<think>` 就直接 fatal 退出，不会让人跑完一整轮才发现失配。
