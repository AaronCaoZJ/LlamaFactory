# Qwen3.5 训练 / 推理 template 失配

> 全系训练模板 = **`qwen3_5_nothink`**（18 个 yaml 无一例外）。
> 服务必须挂 `--chat-template scripts/qwen3_5/eval/chat_template_qwen3_5_lf.jinja`。

---

## 症结（一句话）

**训练侧 prompt 结尾不加任何 think token；而官方 jinja 无论传什么参数都会加。**

官方 jinja 的输出空间只有两个元素——`<think>\n`（开着的思考）和 `<think>\n\n</think>\n\n`
（**空 think 块，不是"不加"**）。训练要的"什么都不加"**不在这个集合里**。

→ 这不是配置没调对，是**官方模板压根没有 nothink 这个模式**，调任何参数都拼不出来。

## 1. 训练侧 token 序列（LF `qwen3_5_nothink`）

```
prompt   : <|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n
response : A<|im_end|>\n                                            ← 算 loss
```

**零 think token。** assistant 头之后直接就是答案。多轮 / 带 system 同理。

## 2. 推理侧：三方渲染对比

| 谁 | `assistant\n` 之后 | |
|---|---|---|
| **LF `qwen3_5_nothink`**（训练用） | *（什么都没有）* | ← 训练真值 |
| 官方 jinja，默认 / `enable_thinking=true` | `<think>\n` | ❌ 多 2 tok |
| 官方 jinja，`enable_thinking=false` | `<think>\n\n</think>\n\n` | ❌ 多 4 tok（**空块 ≠ 不加**） |
| 本仓库 `chat_template_qwen3_5_lf.jinja` | *（什么都没有）* | ✅ 复刻训练 |

自定义 jinja = 官方 jinja **只改一处**（generation prompt）：

```jinja
{#- 官方：总是发 think -#}                    {#- 本仓库：只在显式要求时才发 -#}
{%- if enable_thinking ... is false %}        {%- if enable_thinking ... is true %}
    {{- '<think>\n\n</think>\n\n' }}              {{- '<think>\n' }}
{%- else %}                                   {%- endif %}
    {{- '<think>\n' }}
{%- endif %}
```

保留了 `enable_thinking=true` 后门（想开思考模式仍可开）。
`eval/` 与 `mikomiko_tagger/` 下两份**字节完全相同**，改一份要同步另一份。

## 3. 失配从哪来

1. **LF 训练不走 jinja**，走 `template.py` 的 Python slots。两套独立实现，**没有任何一致性校验**。
2. LF 的 `fix_jinja_template`（`template.py:275`）只在 `tokenizer.chat_template is None`
   **或** `replace_jinja_template=True` 时才覆盖 jinja。`qwen3_5` / `qwen3_5_nothink` **都没设**这个开关，
   而 Qwen3.5 自带 chat_template → **LF 原样放过，不覆盖**。
3. 存 checkpoint 时 `save_pretrained` 把这份**官方原版** jinja 抄进去。

> ⚠️ **`saves/.../chat_template.jinja` 是 Qwen 出厂的，不是训练用的。** 别拿它当训练分布的依据。
> （只有 `custom` / `chatml` / `default` / `fewshot` / `vicuna` 等 7 个模板设了 `replace_jinja_template=True`，
> LF 才会生成 jinja；所有主流模型家族都不在其列——qwen 是常态，不是例外。）

## 4. 后果：**静默掉点**，不是崩

Qwen 的失配是**末尾追加**——训练 prompt 是推理 prompt 的**完整前缀**，零信息丢失。
多出来的 `<think>` 又是基座极熟悉的 token，任务先验压得住 → **输出看着完全正常**。

但代价实打实：**mikomiko tagger 实测掉 1.2pt microF1**。

> 对比 Gemma-4（见 `scripts/gemma4/GEMMA4_DEBUG.md`）：那边丢掉整个 system turn，
> 公共前缀仅 2 tok → 直接复读 / 乱答。**Gemma 会崩，Qwen 只是静默变差——所以 Qwen 的更难发现。**

## 5. 怎么修

**现方案（已实施，零重训）**：五个 server 脚本全部挂 `--chat-template`：

| 脚本 | 模型 |
|---|---|
| `eval/start_vllm_server.sh` | 27B |
| `eval/start_vllm_server_9.sh` | 9B |
| `eval/start_vllm_server_2.sh` | 2B |
| `eval/start_vllm_server_0_8.sh` | 0.8B |
| `mikomiko_tagger/start_vllm_server_mikomiko.sh` | 2B |

已验证：**4 个尺寸 × 3 个场景（单轮 / 多轮 / 带 system），token 级全部一致**。

**备选（不改 jinja，但要重训所有模型）**：让训练侧去迁就官方 jinja——

```yaml
template: qwen3_5          # 不是 _nothink
enable_thinking: false     # → prompt 尾部拼空 think 块，不计 loss
```
```bash
vllm serve ... --default-chat-template-kwargs '{"enable_thinking": false}'   # 不挂 --chat-template
```

实测**逐字节一致**。好处是摆脱自定义 jinja 的维护负担；代价是**已训模型全部作废**。
两条路**不能混用**（同一个 server 只能选一种 serve 配置，混着必错一个）。

## 6. 验证：`/tokenize` 对拍（改模板 / 换 server 必跑）

`POST /tokenize` **不做推理**，只按服务端当前 chat template 把 messages 渲染成最终 prompt 再切 token。
它是唯一能看见 **vLLM 真正喂给模型那串 token** 的办法（图像展开成多少 `<|image_pad|>` 是运行时算的，读 jinja 看不出来）。

### 6.1 快速版：一眼看 prompt 尾巴

```bash
curl -s http://127.0.0.1:8109/tokenize -H 'Content-Type: application/json' -d '{
  "model": "piper_0705_v4_9",
  "messages": [{"role":"user","content":"PROBE"}],
  "add_generation_prompt": true, "return_token_strs": true
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['count'], d['token_strs'][-4:])"
```

**期望**（`Ċ` = `\n`）：

```
10 ['Ċ', '<|im_start|>', 'assistant', 'Ċ']   ← 结尾干净，无 <think>
```

出现 `'<think>'` = **server 漏挂 `--chat-template`**，立刻停下来修。

### 6.2 完整版：带图逐 token 对拍训练侧

```bash
cd $LF_ROOT && DISABLE_VERSION_CHECK=1 .venv/bin/python - <<'PY'
import json, base64, urllib.request, logging; logging.disable(logging.WARNING)
from transformers import AutoTokenizer, AutoProcessor
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

PORT, LORA  = 8109, "piper_0705_v4_9"                                  # ← 改这里
BASE        = "/workspace1/zhijun/hf_download/models/Qwen3.5-9B"
DATA        = "data/agentrobot/MVTOKEN/0705_piper/v4/rollout_lite.json"
TPL, MAXPIX = dict(template="qwen3_5_nothink"), 65536                  # 与训练 yaml 一致

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

**实测通过**（9B + `piper_0705_v4`，2 张 256×256 图）：训练侧 337 tok / 服务端 337 tok，
128 个 `<|image_pad|>` 完全对上，**逐 token 一致**；端到端推理 5/5 命中。

> `mikomiko_tagger/infer_mikomiko.py` 的 `check_prompt_parity()` 已把这个检查做成**启动断言**。
> `eval/infer.py` **还没有** —— 目前只靠人记得挂 flag。
