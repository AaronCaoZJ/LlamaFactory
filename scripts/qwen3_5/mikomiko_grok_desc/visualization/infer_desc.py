#!/usr/bin/env python3
"""infer_desc.py — step 2/3: generate descriptions for the review page.

Thin wrapper around the tagger's inference entry point. Image preprocessing, the "<image> first,
prompt second" ordering and the empty-think-block parity guard are IMPORTED from
../../mikomiko_tagger/infer_mikomiko.py rather than reimplemented -- those three are the things
that silently cost accuracy when they drift, and there should be exactly one copy of each.

What this file adds, and the only reason it exists instead of calling infer_mikomiko directly:
it keeps `finish_reason` and `completion_tokens` per row. For a ~30-token tag list nobody cares;
for a 500-950 token description "did the model stop, or did it hit the cap" is the difference
between a complete answer and a truncated one, and the review page has to be able to say which.

Writes samples.json back out with three fields added per row:
    pred_<tag>, finish_<tag>, ntok_<tag>          (tag = --tag, e.g. "sft" / "base")
so the SFT run and the base run can accumulate into the same file.

Usage:
    python infer_desc.py --input WORK/samples.json --output WORK/samples_pred.json \
        --api http://localhost:8121 --model desc_sft --tag sft --max-new-tokens 1536
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "mikomiko_tagger"))
from infer_mikomiko import check_prompt_parity, encode_image_b64, prompt_of  # noqa: E402


def chat(api, model, text, image_path, max_tokens, retries=3):
    """One completion. Returns (content, finish_reason, completion_tokens)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_image_b64(image_path)}},  # image FIRST
            {"type": "text", "text": text},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    last = ""
    for _ in range(retries):
        try:
            req = urllib.request.Request(f"{api}/v1/chat/completions", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=600) as resp:
                d = json.loads(resp.read().decode())
            choice = d["choices"][0]
            return (choice["message"]["content"] or "",
                    choice.get("finish_reason", "?"),
                    (d.get("usage") or {}).get("completion_tokens", 0))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
        except Exception as e:
            last = repr(e)
        time.sleep(2.0)
    print(f"[warn] request failed after {retries} tries: {last}", file=sys.stderr)
    return "", "error", 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--api", default="http://localhost:8121")
    ap.add_argument("--model", default="desc")
    ap.add_argument("--tag", required=True, help="field suffix: pred_<tag> (e.g. sft, base)")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    rows = json.load(open(args.input, encoding="utf-8"))
    print(f"[infer:{args.tag}] {len(rows)} rows | {args.model} @ {args.api}", flush=True)
    check_prompt_parity(args.api, args.model)

    out = [None] * len(rows)
    done, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(chat, args.api, args.model, prompt_of(r["instruction"]),
                          r["image"], args.max_new_tokens): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(rows):
                print(f"  [infer:{args.tag}] {done}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    for r, (text, finish, ntok) in zip(rows, out):
        r[f"pred_{args.tag}"] = (text or "").strip()
        r[f"finish_{args.tag}"] = finish
        r[f"ntok_{args.tag}"] = ntok

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump(rows, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    empty = sum(1 for r in rows if not r[f"pred_{args.tag}"])
    capped = sum(1 for r in rows if r[f"finish_{args.tag}"] == "length")
    ntoks = sorted(r[f"ntok_{args.tag}"] for r in rows)
    print(f"[infer:{args.tag}] {len(rows)} preds in {time.time()-t0:.0f}s | empty={empty} | "
          f"hit token cap={capped} | median tokens={ntoks[len(ntoks)//2]} | max={ntoks[-1]}")
    print(f"[infer:{args.tag}] -> {args.output}")


if __name__ == "__main__":
    main()
