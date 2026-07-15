#!/usr/bin/env bash
# 训推 prompt 一致性校验：把 vLLM 真正喂给模型的那串 token，与 LF 训练侧逐 token 对拍。
#
# 改 chat template / 换 server / 加新 LoRA 之后必跑（背景见 ../QWEN35_DEBUG.md）。
# 失配是静默的：训练 prompt 是推理 prompt 的完整前缀，多出来的 <think> 又是基座极熟悉的
# token，输出看着完全正常，只是悄悄掉点。除了对拍，没有别的办法发现。
#
# 两级检查：
#   1) 尾部探针 —— 纯文本 PROBE，看 assistant\n 之后有没有混进 <think>。
#      漏挂 --chat-template 的话这一步就会挂，不用等第 2 步。
#   2) 逐 token 对拍 —— 拿数据集第一条真样本（带图），比对训练侧与服务端的完整 token 序列。
#      只有这一步能验证图像展开成多少个 <|image_pad|>（运行时算的，读 jinja 看不出来）。
#
# 用法（默认查 :8109 的 piper_0705_v4_9）：
#   bash scripts/qwen3_5/eval/check_prompt_parity.sh
#   PORT=8109 LORA=dual_cloth_once   DATA=data/agentrobot/MVTOKEN/dual_cloth/v4/rollout_dual_once.json \
#     bash scripts/qwen3_5/eval/check_prompt_parity.sh
#   PORT=8109 LORA=dual_cloth_chain  DATA=data/agentrobot/MVTOKEN/dual_cloth/v4/rollout_dual_chain.json \
#     bash scripts/qwen3_5/eval/check_prompt_parity.sh          # sharegpt 样本也支持
#   PORT=8101 LORA=mix_22_27_v3 BASE=${MODELS_DIR}/Qwen3.5-27B ... （27B server）
#
# 退出码：0 = 逐 token 一致；1 = 失配或探针发现 <think>；2 = server 不可达 / 配置错。
set -uo pipefail

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"
cd "${LF_ROOT}"

# ── 被测对象（按需覆盖）────────────────────────────────────────────────────────
PORT="${PORT:-8109}"
LORA="${LORA:-piper_0705_v4_9}"
BASE="${BASE:-${MODELS_DIR}/Qwen3.5-9B}"
DATA="${DATA:-data/agentrobot/MVTOKEN/0705_piper/v4/rollout_lite.json}"
TEMPLATE="${TEMPLATE:-qwen3_5_nothink}"   # 必须与训练 yaml 的 template 一致
MAXPIX="${MAXPIX:-65536}"                 # 必须与训练 yaml 的 image_max_pixels 一致
URL="http://127.0.0.1:${PORT}"

# DUMP=1    并排打印两侧完整 token 序列（连续的 <|image_pad|> 折叠成 ×N，否则 192 个会刷屏）
# DUMP=full 一个都不折叠，逐 token 全打
DUMP="${DUMP:-0}"

echo "════════════════════════════════════════════════════════════════"
echo "  server   : ${URL}"
echo "  LoRA     : ${LORA}"
echo "  base     : ${BASE}"
echo "  数据     : ${DATA}"
echo "  template : ${TEMPLATE}   (image_max_pixels=${MAXPIX})"
echo "════════════════════════════════════════════════════════════════"

[ -f "${DATA}" ] || { echo "ERROR: 数据集不存在: ${DATA}" >&2; exit 2; }
curl -sf -m 5 "${URL}/v1/models" >/dev/null || {
  echo "ERROR: server 不可达 (${URL})。先起 server：" >&2
  echo "       bash scripts/qwen3_5/eval/start_vllm_server_9.sh" >&2
  exit 2
}

source .venv/bin/activate
export DISABLE_VERSION_CHECK=1

# ── 1) 尾部探针 ───────────────────────────────────────────────────────────────
echo
echo "── 1) 尾部探针（纯文本，看有没有混进 <think>）"
PORT="${PORT}" LORA="${LORA}" python3 - <<'PY' || exit 1
import json, os, sys, urllib.request

url = f"http://127.0.0.1:{os.environ['PORT']}/tokenize"
req = {"model": os.environ["LORA"],
       "messages": [{"role": "user", "content": "PROBE"}],
       "add_generation_prompt": True, "return_token_strs": True}
d = json.load(urllib.request.urlopen(urllib.request.Request(
    url, data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=30))

tail = d["token_strs"][-4:]
print(f"   {d['count']} tok, 尾部 = {tail}")
if any("think" in t for t in d["token_strs"]):
    print("   ✗ 尾部出现 <think> —— server 漏挂了 --chat-template", file=sys.stderr)
    sys.exit(1)
print("   ✓ 尾部干净（assistant\\n 之后无 think token）")
PY

# ── 2) 逐 token 对拍 ──────────────────────────────────────────────────────────
echo
echo "── 2) 逐 token 对拍（带图，训练侧 vs 服务端）"
PORT="${PORT}" LORA="${LORA}" BASE="${BASE}" DATA="${DATA}" \
TEMPLATE="${TEMPLATE}" MAXPIX="${MAXPIX}" DUMP="${DUMP}" python3 - <<'PY' || exit 1
import base64, json, logging, os, sys, urllib.request
logging.disable(logging.WARNING)
from transformers import AutoTokenizer, AutoProcessor
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

PORT, LORA = os.environ["PORT"], os.environ["LORA"]
BASE, DATA = os.environ["BASE"], os.environ["DATA"]
TEMPLATE, MAXPIX = os.environ["TEMPLATE"], int(os.environ["MAXPIX"])

tok = AutoTokenizer.from_pretrained(BASE)
proc = AutoProcessor.from_pretrained(BASE)
setattr(proc, "image_max_pixels", MAXPIX)
tpl = get_template_and_fix_tokenizer(tok, DataArguments(template=TEMPLATE))

s = json.load(open(DATA))[0]
images = s.get("images") or []

# 样本格式两种：sharegpt(messages，如 dual --chain) / alpaca(instruction+input)。
# 对拍只需要「第一个 user turn + generation prompt」这段前缀 —— 它就是 server 在
# add_generation_prompt=True 时渲染出的东西，后续 assistant 轮不影响这段。
if "messages" in s:
    user_text = next(m["content"] for m in s["messages"] if m["role"] == "user")
    asst_text = next(m["content"] for m in s["messages"] if m["role"] == "assistant")
else:
    user_text = "\n".join(x for x in [s["instruction"], s.get("input", "")] if x)
    asst_text = s["output"]

msgs = tpl.mm_plugin.process_messages(
    [{"role": "user", "content": user_text}, {"role": "assistant", "content": asst_text}],
    images, [], [], proc)
train, _ = tpl.encode_oneturn(tok, msgs)

# 服务端：content 数组顺序 = <image> 占位符顺序（图在前、文本在后，与 converter 一致）
durl = lambda p: "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
content = [{"type": "image_url", "image_url": {"url": durl(p)}} for p in images]
content.append({"type": "text", "text": user_text.replace("<image>", "", len(images))})
req = {"model": LORA, "messages": [{"role": "user", "content": content}],
       "add_generation_prompt": True, "return_token_strs": True}
srv = json.load(urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{PORT}/tokenize", data=json.dumps(req).encode(),
    headers={"Content-Type": "application/json"}), timeout=90))["tokens"]

pad = tok.convert_tokens_to_ids("<|image_pad|>")
print(f"   训练侧 {len(train)} tok / 服务端 {len(srv)} tok"
      f"   ({len(images)} 图 -> {sum(1 for t in train if t == pad)} 个 <|image_pad|>)")

lcp = next((i for i, (a, b) in enumerate(zip(train, srv)) if a != b), min(len(train), len(srv)))
same = srv == train


def dump(collapse: bool) -> None:
    """并排打印两侧完整 token 序列，标出第一处失配。

    collapse=True 时把连续同一 token 折叠成 '×N'（192 个 <|image_pad|> 否则会把差异淹掉）。
    折叠只在两侧该段完全相同时进行，失配处永远逐 token 展开。
    """
    n = max(len(train), len(srv))
    print(f"\n   {'idx':>5}  {'训练侧':<28} {'服务端':<28}")
    print(f"   {'-' * 5}  {'-' * 28} {'-' * 28}")
    i = 0
    while i < n:
        a = train[i] if i < len(train) else None
        b = srv[i] if i < len(srv) else None
        run = 1
        if collapse and a is not None and a == b:
            while (i + run < n
                   and (train[i + run] if i + run < len(train) else None) == a
                   and (srv[i + run] if i + run < len(srv) else None) == a):
                run += 1
        s_a = tok.convert_ids_to_tokens([a])[0] if a is not None else "—"
        s_b = tok.convert_ids_to_tokens([b])[0] if b is not None else "—"
        if run > 1:
            print(f"   {i:>5}  {s_a + f' ×{run}':<28} {s_b + f' ×{run}':<28}")
        else:
            mark = "" if a == b else "   ← 失配"
            print(f"   {i:>5}  {s_a:<28} {s_b:<28}{mark}")
        i += run


if os.environ.get("DUMP", "0") != "0":
    dump(collapse=os.environ["DUMP"] != "full")

if same:
    print("\n   ✓ 逐 token 完全一致")
    sys.exit(0)

print(f"\n   ✗ 失配：公共前缀 {lcp} token", file=sys.stderr)
print(f"     训练侧后续: {tok.decode(train[lcp:lcp + 12])!r}", file=sys.stderr)
print(f"     服务端后续: {tok.decode(srv[lcp:lcp + 12])!r}", file=sys.stderr)
if os.environ.get("DUMP", "0") == "0":
    print("     完整序列对比：加 DUMP=1 重跑", file=sys.stderr)
sys.exit(1)
PY

echo
echo "✓ 训推 prompt 一致"
