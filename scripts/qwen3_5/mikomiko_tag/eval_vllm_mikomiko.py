#!/usr/bin/env python3
"""
eval_vllm_mikomiko.py — fast mikomiko image->tag eval against a vLLM OpenAI server.

Sends each eval image to the server with the SAME prompt shape as training (image FIRST, then the
tagging prompt — the mikomiko builder emits instruction = "<image>" + prompt), greedy / no-think,
collects predictions, and scores them with metrics_mikomiko (identical metrics to the hf path).

Prereq: start the server first ->
    bash scripts/qwen3_5/mikomiko_tag/start_vllm_server_mikomiko.sh [STEP]

Usage:
    python scripts/qwen3_5/mikomiko_tag/eval_vllm_mikomiko.py --step 11530
    python scripts/qwen3_5/mikomiko_tag/eval_vllm_mikomiko.py --evalset data/mikomiko_tag/jsonl/eval_mini.jsonl \
        --api http://localhost:8110 --model mikomiko --concurrency 32 -n 400
"""
import argparse, base64, json, os, sys, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import metrics_mikomiko  # same directory

IMAGE_TOKEN = "<image>"


def encode_image(path):
    suffix = Path(path).suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "jpeg")
    with open(path, "rb") as f:
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()


def chat(api, model, text, image_path, max_tokens, retries=3):
    # image FIRST, then text -> matches training ("<image>" + prompt). This ordering is what the
    # gemma4 eval-mismatch bug was about; keep image before text for parity.
    content = [
        {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
        {"type": "text", "text": text},
    ]
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    last = ""
    for _ in range(retries):
        try:
            req = urllib.request.Request(f"{api}/v1/chat/completions", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
        except Exception as e:
            last = repr(e)
        time.sleep(1.0)
    print(f"[warn] request failed after {retries} tries: {last}", file=sys.stderr)
    return ""   # empty prediction -> scored as a miss, keeps alignment


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[2]   # .../LlamaFactory
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", default=str(root / "data/mikomiko_tag/jsonl/eval_mini.jsonl"))
    ap.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8110"))
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "mikomiko"))
    ap.add_argument("--step", default="11530")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("-n", "--max-samples", type=int, default=None, help="cap #samples (debug)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.evalset, encoding="utf-8")]
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"[eval] {len(rows)} samples -> {args.api} (model={args.model}, concurrency={args.concurrency})")

    def one(row):
        text = row["instruction"].replace(IMAGE_TOKEN, "").strip()
        return chat(args.api, args.model, text, row["images"][0], args.max_tokens)

    t0 = time.time()
    preds = [None] * len(rows)
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(one, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            i = futs[fut]
            preds[i] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(rows):
                print(f"  [eval] {done}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0
    print(f"[eval] generated {len(rows)} predictions in {dt:.0f}s ({len(rows)/dt:.1f}/s)")

    out_dir = Path(args.out_dir) if args.out_dir else \
        root / f"saves/qwen3.5-2b/mikomiko/predict_sanity/runs/vllm_step_{args.step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for r, pd in zip(rows, preds):
            f.write(json.dumps({"label": r["output"], "predict": pd or ""}, ensure_ascii=False) + "\n")
    print(f"[eval] predictions -> {pred_path}")

    history = root / "saves/qwen3.5-2b/mikomiko/predict_sanity/evalmini_history.tsv"
    metrics_mikomiko.score(str(pred_path), args.evalset, f"{args.step}(vllm)",
                         str(history), str(out_dir / "metrics.json"))
    print(f"[eval] saved -> {out_dir}/")


if __name__ == "__main__":
    main()
