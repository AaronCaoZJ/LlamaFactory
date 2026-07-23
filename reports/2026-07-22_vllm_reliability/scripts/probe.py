#!/usr/bin/env python3
"""sampling / determinism probe -- stdlib only.
Request construction mirrors training distribution:
  1. images before text in the OpenAI content array (order == <image> placeholder order)
  2. user text = instruction minus <image> placeholders + (single "\n" + input) when input non-empty
Reference impl: /workspace1/zhijun/LlamaFactory/scripts/qwen3_5/eval/infer.py (encode_image / chat)
"""
import base64, json, sys, threading, time, urllib.error, urllib.request
from collections import Counter
from pathlib import Path

DATA = "/workspace1/zhijun/LlamaFactory/data/agentrobot/ood_sample/v3/rollout_lite.json"
SCRATCH = "/tmp/claude-3014/-workspace1-zhijun/6a801ac6-4b1b-49b8-98d6-d62501f4249c/scratchpad/"
_IMAGE_TOKEN = "<image>"
_IMG_CACHE = {}

def encode_image(path):
    if path in _IMG_CACHE:
        return _IMG_CACHE[path]
    suffix = Path(path).suffix.lstrip(".").lower()
    mime = {"jpg":"jpeg","jpeg":"jpeg","png":"png","gif":"gif","webp":"webp"}.get(suffix,"jpeg")
    with open(path,"rb") as f:
        data = base64.b64encode(f.read()).decode()
    uri = "data:image/%s;base64,%s" % (mime, data)
    _IMG_CACHE[path] = uri
    return uri

def load_samples():
    with open(DATA) as f:
        return json.load(f)

def build_messages(sample, prefix_salt=""):
    instruction = sample["instruction"]
    sample_input = sample.get("input") or ""
    user_text = (instruction + "\n" + sample_input) if sample_input else instruction
    clean_text = user_text.replace(_IMAGE_TOKEN, "").strip()
    if prefix_salt:
        clean_text = prefix_salt + clean_text
    content = [{"type":"image_url","image_url":{"url":encode_image(p)}} for p in sample["images"]]
    content.append({"type":"text","text":clean_text})
    return [{"role":"user","content":content}]

def post(port, payload, timeout=300):
    req = urllib.request.Request("http://localhost:%d/v1/chat/completions" % port,
        data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"__http_error__": e.code, "__body__": e.read().decode(errors="replace")[:800]}
    except Exception as e:
        return {"__error__": repr(e)}

def make_payload(model, sample, temperature=None, top_p=None, top_k=None, seed=None,
                 max_tokens=4, logprobs=True, top_logprobs=20, prefix_salt="", extra=None):
    p = {"model":model, "messages":build_messages(sample, prefix_salt), "max_tokens":max_tokens}
    if temperature is not None: p["temperature"] = temperature
    if top_p is not None: p["top_p"] = top_p
    if top_k is not None: p["top_k"] = top_k
    if seed is not None: p["seed"] = seed
    if logprobs:
        p["logprobs"] = True
        p["top_logprobs"] = top_logprobs
    if extra: p.update(extra)
    return p

def extract(resp):
    if "__error__" in resp or "__http_error__" in resp:
        return {"ok": False, "raw": resp}
    ch = resp["choices"][0]
    out = {"ok":True, "text":ch["message"]["content"], "finish":ch.get("finish_reason")}
    lp = ch.get("logprobs")
    if lp and lp.get("content"):
        first = lp["content"][0]
        out["tok0"] = first["token"]
        out["tok0_logprob"] = first["logprob"]
        out["top"] = [(t["token"], t["logprob"]) for t in (first.get("top_logprobs") or [])]
    return out

def run_serial(port, model, sample, n, **kw):
    return [extract(post(port, make_payload(model, sample, **kw))) for _ in range(n)]

def run_concurrent(port, model, sample, n, **kw):
    out = [None]*n
    payloads = [make_payload(model, sample, **kw) for _ in range(n)]
    def work(i):
        out[i] = extract(post(port, payloads[i]))
    ths = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    return out, time.time()-t0

def summarize(res):
    ok = [r for r in res if r.get("ok")]
    tc = Counter(r["text"] for r in ok)
    lps = [r["tok0_logprob"] for r in ok if "tok0_logprob" in r]
    uniq = sorted(set(repr(x) for x in lps))
    return {"n":len(res), "n_ok":len(ok), "n_unique_text":len(tc),
            "text_counts":dict(tc.most_common()),
            "n_unique_tok0_logprob":len(uniq), "tok0_logprob_values":uniq[:8],
            "tok0_logprob_spread": (max(lps)-min(lps)) if lps else None,
            "errors":[r["raw"] for r in res if not r.get("ok")][:3]}

def dump(obj, name):
    p = SCRATCH + name
    with open(p, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print("[dump] " + p, file=sys.stderr)
