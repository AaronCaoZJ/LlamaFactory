# Gemma-4 训练 / 评测踩坑记录

> **机器** showlab15（共享 8×H200）
> **训练 venv** `.venv-gemma4`（transformers 5.12.1）
> **服务 venv** `/workspace1/zhijun/AgentRobot/.venv-vllm`（vLLM 0.24.0）

---

## 速查：症状 → 病因

| 症状 | 先查 |
|---|---|
| **输出复读 / 吞 `MV_` 前缀 / 乱答** | prompt 没对齐训练分布（§4，最大的坑） |
| `Image features and image tokens do not match` | 预处理缓存过期（§3） |
| 启动报 transformers 版本不兼容 | 忘了 `DISABLE_VERSION_CHECK=1`（§1） |
| 单卡报 `Please use FORCE_TORCHRUN=1` | 配了 deepspeed（§3） |
| adapter 里 24 层以后没有 k_proj/v_proj | **正常**，E4B 的 KV 共享层（§2） |
| prompt 里有 `<\|think\|>` 和 thought 段，是在思考吗？ | **不是**，那是**关闭**思考的写法（§4.2） |
| 想删掉 `"You are a helpful assistant."` | 已训模型别删；`default_system: ""` 是陷阱（§4.3） |

## 开跑前的硬规矩（照抄即可）

```bash
# 训练
export DISABLE_VERSION_CHECK=1        # 不设直接拒绝启动

# 服务（必须 activate，不能直接调 $VENV/bin/vllm）
source /workspace1/zhijun/AgentRobot/.venv-vllm/bin/activate
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
ss -lptn "sport = :$PORT"             # 确认端口没被旧 server 占（§5.1）
vllm serve ... --chat-template scripts/gemma4/eval/chat_template_gemma4n_lf.jinja
```

1. **vLLM 必须挂 `--chat-template`**，官方模板拼不出训练分布 → §4
2. **改过 plugin/processor/数据 就要 `overwrite_cache: true`** → §3
3. **评测前确认端口独占**，vLLM 不会报端口冲突 → §5.1

---

## 1. 环境

| 坑 | 原因 | 修 |
|---|---|---|
| 启动被版本检查拒绝 | gemma4 需 transformers≥5.10，LF 硬编码上限 5.6.0 | `export DISABLE_VERSION_CHECK=1` |
| vLLM 崩溃 | 0.22 架构未注册 + o_proj 异构 head_dim（PR #44429 才加原生支持） | **vLLM ≥ 0.24.0** |
| JIT 编译失败 | 节点 gcc-12 缺 cc1plus | `export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11` |

不需要 `--enforce-eager`，0.24.0 下 torch.compile 正常。

## 2. 模型家族差异（很多坑的根源）

| 模型 | `model_type` | LF `template` | 结构特点 |
|---|---|---|---|
| E2B / E4B | `gemma4`（nano） | `gemma4n` | per-layer-embedding(PLE) + KV 共享层 |
| **12B** | **`gemma4_unified`** | **`gemma4n`** | 无 PLE、无 KV 共享 |
| 26B-A4B / 31B | `gemma4` | `gemma4` | thinking 模型 |

- **E4B 和 12B 架构不同，但共用 `template: gemma4n`**（`extras/constants.py` 的 `register_model_group`）。
- 视觉塔注册是**两条独立的** composite（`model/model_utils/visual.py`）：
  E4B = `vision_tower` + `audio_tower` + `embed_vision/embed_audio`；12B = 只有 `vision_embedder`。
- **E4B 的 42 层中，第 24 层起 KV 共享**（`num_kv_shared_layers=18`）→ adapter 里 24 层以后
  **没有 k_proj/v_proj 是正常的**，不是训练出错。

## 3. 训练

**`Image features and image tokens do not match`**
- **原因**：十有八九是**预处理缓存过期** —— 改了 plugin/processor 后，旧缓存的图像占位符数量对不上（如 488 vs 512）。
- **修**：`overwrite_cache: true`，或删 `~/.cache/huggingface/datasets/json`。
- 注：256×256 的图不受 `image_max_pixels` 影响（低于上限不缩放），别往这个方向查。

**单卡 + deepspeed 报 `Please use FORCE_TORCHRUN=1`**
- **原因**：单 GPU 时 LF 默认不走 torchrun，配了 deepspeed 就必须走。
- **修**：`FORCE_TORCHRUN=1`；或干脆去掉 deepspeed（单张 H200 跑 12B LoRA 只占 ~33GB）。

**LoRA target 的两个注意点**
- `freeze_vision_tower: true` + `lora_target: all` → adapter 只含语言层，**正确**。
- ⚠️ **E4B 上 `lora_target: all` 会把 PLE 通路也训进去**（`per_layer_input_gate` / `per_layer_projection` /
  `per_layer_model_projection`，占 adapter 686 个权重里的 170 个）。vLLM 能正常加载，**不是 bug**；
  要规避就显式写 `lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。

## 4. 评测 / 服务：prompt 必须逐 token 对齐训练 ★最大的坑

训练侧（LF `gemma4n` + alpaca converter）的**真实渲染**：

```
<bos><|turn>system\n<|think|>You are a helpful assistant.<turn|>\n<|turn>user\n<|image|><|image|>TEXT<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

**官方 `chat_template.jinja` 永远拼不出它**，差距是结构性的：

| | 官方 jinja | 训练侧 |
|---|---|---|
| system turn | 只在 `enable_thinking=true` / 有 tools / 数据显式带 system 时才发（jinja:179）——**本项目三条都不满足**（dataset 无 system 列） | **无条件发** |
| 生成前缀 | `add_generation_prompt` 只给 `<\|turn>model\n` | 还要拼**结尾空 thought 段** |

> 实测 E4B 单轮：训练 26 tok vs 官方 10 tok，**公共前缀仅 2 tok，整个 system turn 丢掉**。
> **失配后果是复读 / 吞 `MV_` 前缀 / 乱答，不是掉几个点。**

**怎么修**
- **vLLM 必须挂** `--chat-template scripts/gemma4/eval/chat_template_gemma4n_lf.jinja`（已提供，复刻训练渲染）。
- HF backend（`llamafactory-cli api`）复用 LF 模板，**天然对齐**，不需要它。

**另三条铁律**（任一处对不上，同样退化）
1. **图片排在文本之前** —— OpenAI `content` 数组顺序 = 占位符顺序。
2. **必须有 system** —— HF backend 自动补，**vLLM 不会**。
3. **instruction 与 input 用单个 `\n` 连接** —— converter 是 `"\n".join`，不是 `\n\n`。

**结论 & 验证**
- 对齐后 vLLM 与 HF 结果**逐条一致**（E4B 20 条 OOD 均 17/20，连错的 3 条都一样）→ 优先用 vLLM（吞吐 + `guided_choice`）。
- **改模板必跑**：vLLM `/tokenize`（`return_token_strs=true`）vs LF `template.encode_oneturn()`，**逐 token 必须相同**。

### 4.1 `<|think|>` 和 thought 段 = **关闭**思考，不是开启

- 结尾 `<|channel>thought\n<channel|>` 是**空的、已闭合的**思考块 → 思考位被预先封死，模型直接出答案。
  （`ReasoningTemplate.encode_oneturn`，`template.py:423-434`：`enable_thinking=false` 时删掉 response 的 CoT，
  把空段拼到 **prompt 侧、不计 loss**）
- system 里的 `<|think|>` 是 LF **无条件**写死的；官方 jinja 里它只在 `enable_thinking=true` 时发（**语义相反**）。

模型就是在这个分布上训完的，**推理侧原样复刻即可**，别自作聪明改。

### 4.2 想去掉 `"You are a helpful assistant."`？

**已训模型：别动。** dataset 无 system 列 → 这句进了**每一条样本**，删掉 = 重演 §4 的失配。

**`default_system: ""` 是陷阱**（`<|think|>` 硬绑在 `format_system` slot 里）：

| 配置 | system turn | `<\|think\|>` |
|---|---|---|
| 不设（默认） | 有 | ✅ |
| `default_system: ""` | **无** | **❌ 一起丢** |
| `default_system: "自定义"` | 有 | ✅ |

要改只能**重训**：换成有用的 `default_system: "<任务指令>"`（`<|think|>` 会保留），推理侧同步改
`chat_template_gemma4n_lf.jinja` 里的 `namespace(system=...)`。
彻底删掉 system turn 须改 LF 源码（把 `<|think|>` 从 `format_system` 解耦 + 注册 `gemma4n_nothink` 用普通 `Template`）——**不值得**。

## 5. 服务 / 运维

### 5.1 LF API server（`llamafactory-cli api`）的两个坑

- **只接受**这些字段：`model` / `messages` / `tools` / `do_sample` / `temperature` / `top_p` / `n` /
  `presence_penalty` / `max_tokens` / `stop` / `stream`。
  多余字段（`guided_choice` / `chat_template_kwargs` / `logprobs`）被 pydantic **静默忽略**，
  **不报错也不生效** —— 依赖约束解码的客户端要特别注意。
- **它忽略 `model` 字段**，永远用启动 yaml 里加载的那个模型；vLLM 则会校验，名字对不上直接 **404**。

## 6. 验证：`/tokenize` 对拍（改模板 / 换 server 必跑）

`POST /tokenize` **不做推理**，只按服务端当前 chat template 把 messages 渲染成最终 prompt 再切 token。
它是唯一能看见 **vLLM 真正喂给模型那串 token** 的办法（图像展开成多少 token 是运行时算的，读 jinja 看不出来）。

### 6.1 快速版：一眼看 prompt 头尾

```bash
curl -s http://127.0.0.1:8104/tokenize -H 'Content-Type: application/json' -d '{
  "model": "gemma4_e4b_mix_22_27_v3",
  "messages": [{"role":"user","content":"PROBE"}],
  "add_generation_prompt": true, "return_token_strs": true
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('前6:',d['token_strs'][:6]); print('末4:',d['token_strs'][-4:])"
```

**期望**：

```
前6: ['<bos>', '<|turn>', 'system', '\n', '<|think|>', 'You']      ← system turn + <|think|> 都在
末4: ['<|channel>', 'thought', '\n', '<channel|>']                  ← 结尾空 thought 段
```

少了 system turn 或结尾 thought 段 = **server 漏挂 `--chat-template`**，立刻停下来修（§4）。

### 6.2 完整版：带图逐 token 对拍训练侧

```bash
cd $LF_ROOT && DISABLE_VERSION_CHECK=1 .venv/bin/python - <<'PY'
import json, base64, urllib.request, logging; logging.disable(logging.WARNING)
from transformers import AutoTokenizer, AutoProcessor
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

PORT, LORA  = 8104, "gemma4_e4b_mix_22_27_v3"                          # ← 改这里
BASE        = "/workspace1/zhijun/hf_download/models/gemma4-E4B-it"
DATA        = "data/agentrobot/MVTOKEN/mix_22_27/v3/rollout_lite.json"
TPL, MAXPIX = dict(template="gemma4n", enable_thinking=False), 65536    # 与训练 yaml 一致

tok  = AutoTokenizer.from_pretrained(BASE)
proc = AutoProcessor.from_pretrained(BASE); setattr(proc, "image_max_pixels", MAXPIX)
tpl  = get_template_and_fix_tokenizer(tok, DataArguments(**TPL))
s    = json.load(open(DATA))[0]

# 训练侧真值（alpaca converter: instruction + input 用单个 \n 连接）
content = "\n".join(x for x in [s["instruction"], s["input"]] if x)
msgs = tpl.mm_plugin.process_messages(
    [{"role":"user","content":content}, {"role":"assistant","content":s["output"]}],
    s["images"], [], [], proc)
train, _ = tpl.encode_oneturn(tok, msgs)

# 服务端：<image> 占位符顺序 = content 数组顺序（图在前，文本在后）
durl = lambda p: "data:image/png;base64," + base64.b64encode(open(p,"rb").read()).decode()
c = [{"type":"image_url","image_url":{"url":durl(p)}} for p in s["images"]]
c.append({"type":"text","text": s["instruction"].replace("<image>","",len(s["images"]))})
req = {"model":LORA, "messages":[{"role":"user","content":c}],
       "add_generation_prompt":True, "return_token_strs":True}
srv = json.load(urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{PORT}/tokenize", data=json.dumps(req).encode(),
    headers={"Content-Type":"application/json"}), timeout=90))["tokens"]

print(f"训练侧 {len(train)} tok / 服务端 {len(srv)} tok")
if srv == train:
    print("🎉 逐 token 完全一致")
else:
    lcp = next((i for i,(a,b) in enumerate(zip(train,srv)) if a!=b), min(len(train),len(srv)))
    print(f"❌ 失配！公共前缀 {lcp}")
    print(f"   训练侧后续: {tok.decode(train[lcp:lcp+12])!r}")
    print(f"   服务端后续: {tok.decode(srv[lcp:lcp+12])!r}")
PY
```

**实测通过**（E4B + `mix_22_27_v3`，2 张 256×256 图）：训练侧 745 tok / 服务端 745 tok，**逐 token 一致**。
