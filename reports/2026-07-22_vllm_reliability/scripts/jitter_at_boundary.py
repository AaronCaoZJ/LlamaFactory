"""受控实验：每个模型先扫 50 条找最低 margin 的样本，再在这些样本上重复 R 次测抖动。

抖动定义在"最不确定的那个生成步"上 —— 那才是决策可能翻转的位置。
"""
import base64, json, statistics, sys, urllib.request

DS = json.load(open("data/agentrobot/ood_sample/v3/rollout_lite.json"))
_cache = {}


def enc(p):
    if p not in _cache:
        _cache[p] = "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
    return _cache[p]


def ask(port, model, s):
    txt = s["instruction"].replace("<image>", "").strip()
    if s.get("input"):
        txt = f"{txt}\n{s['input']}"
    content = [{"type": "image_url", "image_url": {"url": enc(p)}} for p in s["images"]]
    content.append({"type": "text", "text": txt})
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "max_tokens": 8, "temperature": 0,
                       "logprobs": True, "top_logprobs": 5}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    ch = json.loads(urllib.request.urlopen(req, timeout=180).read())["choices"][0]
    toks = ch["logprobs"]["content"]
    # 只看真正在做选择的步：跳过 EOS/结束符
    steps = [t for t in toks if not t["token"].startswith("<")]
    margins = [t["top_logprobs"][0]["logprob"] - t["top_logprobs"][1]["logprob"] for t in steps]
    worst = min(range(len(margins)), key=lambda i: margins[i]) if margins else 0
    return ch["message"]["content"].strip(), margins[worst] if margins else 99.0, worst


def main(port, model):
    scan = []
    for i, s in enumerate(DS):
        try:
            text, m, w = ask(port, model, s)
            scan.append((m, i, text, w))
        except Exception as e:
            print(f"  [warn] idx={i} {e!r}")
    scan.sort()
    print(f"\n=== {model} @:{port}  最低 margin 的 4 条: "
          + ", ".join(f"idx{i}(m={m:.3f},{t})" for m, i, t, _ in scan[:4]))
    print(f"    全 50 条 margin: min={scan[0][0]:.3f} p25={scan[len(scan)//4][0]:.3f} "
          f"中位={scan[len(scan)//2][0]:.3f}")
    danger = sum(1 for m, *_ in scan if m < 0.5)
    print(f"    margin<0.5 的样本数（危险带）: {danger}/{len(scan)}")

    R = 25
    for m0, idx, _t, wpos in scan[:3]:
        outs, ms = [], []
        for _ in range(R):
            text, m, _w = ask(port, model, DS[idx])
            outs.append(text)
            ms.append(m)
        flip = len(set(outs))
        print(f"    idx{idx:<3} 首测margin={m0:.3f} | {R} 次重复: 输出种类={flip} "
              f"{dict((o, outs.count(o)) for o in set(outs)) if flip > 1 else ''} "
              f"| margin极差={max(ms)-min(ms):.4f} std={statistics.pstdev(ms):.4f}")


main(sys.argv[1], sys.argv[2])
