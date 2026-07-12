# Gemma-4 训练 / 评测踩坑记录

机器 showlab15（共享 8×H200）。训练 venv `.venv-gemma4`（transformers 5.12.1），
服务 venv `/workspace1/zhijun/AgentRobot/.venv-vllm`（vLLM 0.24.0）。

---

## 0. 模型家族差异（先搞清楚这个，很多坑源于此）

| 模型 | `model_type` | LF `template` | 结构特点 |
|---|---|---|---|
| E2B / E4B / **12B** | `gemma4`(nano) / **`gemma4_unified`**(12B) | **`gemma4n`** | E4B 有 per-layer-embedding(PLE) + KV 共享层；12B 没有 |
| 26B-A4B / 31B | `gemma4` | `gemma4` | thinking 模型 |

- **E4B 和 12B 虽然架构不同，但共用 `template: gemma4n`**（见 `extras/constants.py` 的 register_model_group）。
- E4B：`vision_tower` + `audio_tower` + `embed_vision/embed_audio`；12B：只有 `vision_embedder`。
  两者在 `model/model_utils/visual.py` 里是**两条**独立的 composite 注册。
- E4B 的 42 层中，第 24 层起 KV 共享（`num_kv_shared_layers=18`），所以 adapter 里
  24 层以后**没有 k_proj/v_proj** —— 这是正常的，不是训练出错。

## 1. 环境

- **`export DISABLE_VERSION_CHECK=1`**：gemma4 需 transformers≥5.10，但 LF 硬编码上限 5.6.0，不绕过会直接拒绝启动。
- **vLLM 必须 ≥0.24.0**：0.22 会因架构未注册 + o_proj 异构 head_dim 崩溃（PR #44429 才加了原生支持）。
- **flashinfer 要 JIT 编译**：必须 `source .venv-vllm/bin/activate`（`ninja` 装在 venv/bin），并
  `export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11`（节点 gcc-12 缺 cc1plus）。
  直接调 `$VENV/bin/vllm` 不激活 → `FileNotFoundError: 'ninja'`。
- 不需要 `--enforce-eager`，0.24.0 下 torch.compile 正常。

## 2. 训练

- **`Image features and image tokens do not match`** → 十有八九是**预处理缓存过期**（改了 plugin/processor 后，
  旧缓存的图像占位符数量对不上，如 488 vs 512）。修：`overwrite_cache: true` 或删 `~/.cache/huggingface/datasets/json`。
  注意 256×256 的图不受 `image_max_pixels` 影响（低于上限不缩放）。
- **单卡 + deepspeed 报 "Please use FORCE_TORCHRUN=1"**：单 GPU 时 LF 默认不走 torchrun，配了 deepspeed 就必须
  `FORCE_TORCHRUN=1`；或干脆去掉 deepspeed（单张 H200 跑 12B LoRA 只占 ~33GB）。
- `freeze_vision_tower: true` + `lora_target: all` → adapter 只含语言层，正确。
- ⚠️ **E4B 上 `lora_target: all` 会把 PLE 通路也训进去**（`per_layer_input_gate` / `per_layer_projection` /
  `per_layer_model_projection`，占 adapter 686 个权重里的 170 个）。vLLM 能正常加载，**不是 bug**；
  但若要规避，显式写 `lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。

## 3. 评测 / 服务：**prompt 必须逐 token 对齐训练**（最大的坑）

训练侧（LF `gemma4n` 模板 + alpaca converter）的真实渲染是：

```
<bos><|turn>system\n<|think|>You are a helpful assistant.<turn|>\n<|turn>user\n<|image|><|image|>TEXT<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

推理侧任何一处对不上都会让模型退化（复读、吞掉 `MV_` 前缀、乱答）。三条铁律：

1. **图片排在文本之前**（训练数据 `<image><image>` 在 instruction 开头）。
   OpenAI `content` 数组的顺序 = 占位符顺序，两个后端都如此。
2. **必须有 system**：LF 注入 `default_system` 且 `format_system` 硬编码 `<|think|>`。
   HF backend 会自动补；**vLLM 不会**。
3. **instruction 与 input 用单个 `\n` 连接**（converter 是 `"\n".join`），不是 `\n\n`。

> **gemma4 官方 `chat_template.jinja` 永远拼不出训练分布**：它不渲染 system 轮，
> 且 `add_generation_prompt` 只给 `<|turn>model\n`，缺结尾那段空 thought。
> → vLLM 必须 `--chat-template scripts/gemma4/eval/chat_template_gemma4n_lf.jinja`（本仓库已提供，复刻训练渲染）。
> → HF backend（`llamafactory-cli api`）复用 LF 模板，天然对齐，不需要它。

**对齐之后，vLLM 与 HF backend 结果逐条一致**（E4B 20 条 OOD 均为 17/20）。优先用 vLLM（吞吐 + 支持 `guided_choice`）。

验证方法（改完模板务必跑）：用 vLLM 的 `/tokenize`（`return_token_strs=true`）打印 prompt token，
和 LF `template.encode_oneturn()` 的结果逐 token 比对，必须完全相同。

### 3.1 「我不想 thinking，为什么 prompt 里还有 `<|think|>` 和 thought 段？」

**你确实没有在 thinking** —— 输出就是干净的单个动作 token，没有 CoT。那两个标记不是"开启思考"：

- **结尾的 `<|channel>thought\n<channel|>` 是一个"空的、已闭合的思考块"，它恰恰是关闭 thinking 的手段。**
  见 `ReasoningTemplate.encode_oneturn`（template.py:423-434）：`enable_thinking=false` 时先删掉 response 里的
  CoT，再把这个空 thought 段拼到 **prompt 侧且不计 loss**（`enable_thinking=true` 才拼到 response 侧计 loss）。
  语义是"思考段到此为止，直接给答案"——思考位被预先填空封上，模型没得思考。
- **system 里的 `<|think|>` 是 LF 的粗糙实现**：`gemma4n` 的 `format_system` 把它**无条件**写死，
  不管 `enable_thinking` 是什么；而 gemma4 官方 jinja 里 `<|think|>` 只在 `enable_thinking=true` 时才发
  （官方语义 = 开思考）。所以这是个语义矛盾的冗余标记。净效果：prompt 说"可以思考"但思考块已闭合 → 直接出答案。

**结论：合不合理已不重要，模型是在这个分布上训完的，推理侧必须原样复刻**，少发任何一个都会失配退化
（今天那批缺 `MV_` 前缀的输出就是例子）。硬编码它们 = 复现训练，不是"想要 think"。

想要真正干净的 prompt 只能**重新训练**：gemma4 **没有** `_nothink` 变体（LF 只给 ernie / qwen3 系注册了），
需自己注册 `gemma4n_nothink`——`format_system` 去掉 `<|think|>`，并用普通 `Template` 而非 `ReasoningTemplate`
（这样不会追加空 thought 段）。收益仅是省几个 token，模型行为不变（现在本来就不思考）。
**不建议为此重训**，除非将来要切到复刻该模板成本很高的推理框架，那就在下一轮训练时顺手改掉。

## 4. 其他

- LF 的 API server（`llamafactory-cli api`）只接受 `model/messages/tools/do_sample/temperature/top_p/n/
  presence_penalty/max_tokens/stop/stream`。多余字段（`guided_choice` / `chat_template_kwargs` / `logprobs`）
  被 pydantic **静默忽略**，不报错也不生效 —— 依赖约束解码的客户端要注意。
  另外它**忽略 `model` 字段**，永远用启动 yaml 里加载的模型；vLLM 则会校验，名字不对直接 404。
- 训练完成后顶层 `saves/.../` 目录就是最终 adapter，别再指向中间的 `checkpoint-N`。
- 量化变体 `gemma-4-12B-it-qat-w4a16-ct` 仍有未解决的 bug（num_soft_tokens / expert_weights），bf16 不受影响。
- 节点 GPU 常被别人占，跑前 `nvidia-smi` 挑空卡；曾遇 GPU 出 uncorrectable ECC 后卡在 `cudaErrorDevicesUnavailable`。

## 5. 快速排查清单

| 症状 | 先查 |
|---|---|
| 输出复读 / 缺 token 前缀 / 乱答 | prompt 对齐（§3）；同端口是否有第二个 server（§4） |
| `Image features and image tokens do not match` | 预处理缓存（§2） |
| vLLM 请求 404 | model 名 ≠ `--lora-modules` 的 key |
| 启动报 transformers 版本 | `DISABLE_VERSION_CHECK=1` |
| `FileNotFoundError: 'ninja'` | 没 activate vllm venv（§1） |
| prompt 里有 `<|think|>` / thought 段，是在思考吗？ | 不是，那是**关闭**思考的写法（§3.1） |
