#!/usr/bin/env python3
"""eval_vllm_mikomiko.py — thin wrapper: eval the 400-image set against a running vLLM server.

Generation and scoring both live elsewhere now, so every path reports identical numbers:
  generation -> infer_mikomiko.py (backend=vllm)   scoring -> metrics_mikomiko.py
This file only wires the two together with the historical CLI + output layout.

Prereq: bash scripts/qwen3_5/mikomiko_tagger/start_vllm_server_mikomiko.sh [STEP]
That script serves chat_template_qwen3_5_lf.jinja. Pointed at a server running the checkpoint's
stock template, infer_mikomiko.check_prompt_parity() aborts: the stock template appends an empty
"<think>\n\n</think>\n\n" after "assistant\n", 4 tokens absent from the SFT prompt, worth -1.2pt
microF1.

Usage:
    python scripts/qwen3_5/mikomiko_tagger/eval_vllm_mikomiko.py --step 11530
    python scripts/qwen3_5/mikomiko_tagger/eval_vllm_mikomiko.py --evalset data/mikomiko_tag/jsonl/eval_mini.jsonl \
        --api http://localhost:8110 --model mikomiko --concurrency 32 -n 400
"""
import argparse, os
from pathlib import Path

import infer_mikomiko  # same directory
import metrics_mikomiko


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

    rows, raw, kind = infer_mikomiko.load_rows(args.evalset)
    if args.max_samples:
        rows, raw = rows[: args.max_samples], raw[: args.max_samples]
    print(f"[eval] {len(rows)} samples -> {args.api} (model={args.model}, concurrency={args.concurrency})")

    preds = infer_mikomiko.generate_vllm(rows, api=args.api, model=args.model,
                                         concurrency=args.concurrency, max_new_tokens=args.max_tokens)

    out_dir = Path(args.out_dir) if args.out_dir else \
        root / f"saves/qwen3.5-2b/mikomiko/predict_sanity/runs/vllm_step_{args.step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    infer_mikomiko.save_preds(pred_path, preds, raw, kind)
    print(f"[eval] predictions -> {pred_path}")

    history = root / "saves/qwen3.5-2b/mikomiko/predict_sanity/evalmini_history.tsv"
    metrics_mikomiko.score(str(pred_path), args.evalset, f"{args.step}(vllm)",
                           str(history), str(out_dir / "metrics.json"))
    print(f"[eval] saved -> {out_dir}/")


if __name__ == "__main__":
    main()
