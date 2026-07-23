"""独立复核：同一请求串行重复 N 次，看 logprob 是否逐位一致。"""
import base64, json, math, statistics, sys, urllib.request

port, model, idx, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
s = json.load(open("data/agentrobot/ood_sample/v3/rollout_lite.json"))[idx]
enc = lambda p: "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
txt = s["instruction"].replace("<image>", "").strip()
if s.get("input"):
    txt = f"{txt}\n{s['input']}"
content = [{"type": "image_url", "image_url": {"url": enc(p)}} for p in s["images"]]
content.append({"type": "text", "text": txt})
payload = {"model": model, "messages": [{"role": "user", "content": content}],
           "max_tokens": 8, "temperature": 0, "logprobs": True, "top_logprobs": 5}
data = json.dumps(payload).encode()

texts, t0_lp, t1_lp, margins = [], [], [], []
for _ in range(n):
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    ch = json.loads(urllib.request.urlopen(req, timeout=180).read())["choices"][0]
    texts.append(ch["message"]["content"])
    toks = ch["logprobs"]["content"]
    t0_lp.append(toks[0]["logprob"])
    tl = toks[0]["top_logprobs"]
    margins.append(tl[0]["logprob"] - tl[1]["logprob"])
    if len(toks) > 1:
        t1_lp.append(toks[1]["logprob"])

def rep(name, v):
    if not v:
        return
    print(f"  {name:14} uniq={len(set(v)):2d}/{len(v)}  min={min(v):.6f}  max={max(v):.6f}  "
          f"极差={max(v)-min(v):.6f}  std={statistics.pstdev(v):.2e}")

print(f"port={port} model={model} idx={idx} label={s['output']} n={n}")
print(f"  输出文本         uniq={len(set(texts))} -> {dict((t, texts.count(t)) for t in set(texts))}")
rep("tok0 logprob", t0_lp)
rep("tok1 logprob", t1_lp)
rep("tok0 margin", margins)
