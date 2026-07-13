#!/usr/bin/env python3
"""infer_mikomiko.py — the single inference entry point for the mikomiko image->tag tagger.

One implementation of the prompt, the image preprocessing and the decode loop, behind two
backends. Everything else (test_mikomiko.sh's scoring, the review page, ad-hoc evals) should call
this rather than re-implement generation.

TRAINING PARITY — the three things that silently cost accuracy if you get them wrong:
  1. image FIRST, then the tagging prompt (the dataset builder emits "<image>" + prompt);
  2. image pre-resized to image_max_pixels=262144, mirroring LlamaFactory's mm_plugin;
  3. NO empty think block. The checkpoint's jinja appends "<think>\\n\\n</think>\\n\\n" after
     "assistant\\n" whenever enable_thinking is falsy, but the SFT template (`qwen3_5_nothink`)
     never emitted it. Those 4 tokens cost 1.2pt microF1 / 1.7pt microP and inflate composite
     over-generation (5.6 -> 6.1 per image); 395/400 predictions changed when measured on
     eval_mini. The hf backend strips the block; the vllm backend requires the server to be
     started with --chat-template chat_template_qwen3_5_lf.jinja (asserted at startup).

BACKEND — vllm by default: it is the engine that serves this model in production, and bf16 is what
training and every archived eval used. hf drives transformers directly (no server).

DTYPE (hf backend only; the vllm server is bf16) — bf16 weights are cast to fp32 losslessly, so
fp32 does not change the model, only the arithmetic precision of the forward pass. Consequence:
in bf16 greedy output is NOT batch-invariant (48-image probe: bs=8 matched bs=1 on only 26/48; a
padding-free batch of identical images matched exactly, so the cause is bf16 rounding under ragged
batch shapes, not padding). In fp32 output is bitwise identical for bs=1/4/8/16. Aggregate metrics
differ by ~0.3pt microF1 — inside the ±1.5pt sampling noise of a 400-image set. Use fp32 when you
need a reproducible artifact, bf16 to match the served model.

Inputs (auto-detected):
  *.jsonl  alpaca eval rows  {instruction, output, images:[path], _src?}
  *.json   review samples    [{instruction, image, gemini, split}]
Outputs mirror the input shape: predictions.jsonl {label, predict} for the former, the samples list
with a `pred` field for the latter.

Usage:
    # score the 400-image eval set through the HF backend
    python infer_mikomiko.py --input data/mikomiko_tag/jsonl/eval_mini.jsonl \\
        --output saves/.../predictions.jsonl --score --step 17296

    # predictions for the review page (called by visualization/build_html.sh)
    CUDA_VISIBLE_DEVICES=6 python infer_mikomiko.py --input WORK/samples.json --output WORK/samples_pred.json

    # against a running vLLM server (start_vllm_server_mikomiko.sh)
    python infer_mikomiko.py --backend vllm --api http://localhost:8110 --input ... --output ...
"""
import argparse, base64, json, math, os, sys, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                   # .../LlamaFactory
DEFAULT_CKPT = Path(os.environ.get("MODELS_DIR", "/workspace1/zhijun/hf_download/models")) / \
    "Mikomiko_pornpic_tagger/checkpoint-17296"

IMAGE_TOKEN = "<image>"
IMAGE_MAX_PIXELS = 262144        # must match training image_max_pixels
IMAGE_MIN_PIXELS = 32 * 32
THINK_BLOCK = "<think>\n\n</think>\n\n"
PROMPT_TAIL = "<|im_start|>assistant\n"
THINK_OPEN_ID = 248068           # '<think>' in the qwen3.5 vocab


# ── shared: image + prompt ───────────────────────────────────────────────────────────────────
def preprocess_image(path):
    """Mirror LlamaFactory mm_plugin._preprocess_image (base + Qwen2VL override)."""
    image = Image.open(path)
    if image.width * image.height > IMAGE_MAX_PIXELS:
        f = math.sqrt(IMAGE_MAX_PIXELS / (image.width * image.height))
        image = image.resize((int(image.width * f), int(image.height * f)))
    if image.width * image.height < IMAGE_MIN_PIXELS:
        f = math.sqrt(IMAGE_MIN_PIXELS / (image.width * image.height))
        image = image.resize((int(image.width * f), int(image.height * f)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    if min(image.width, image.height) < 28:
        image = image.resize((max(image.width, 28), max(image.height, 28)))
    if image.width / image.height > 200:
        image = image.resize((image.height * 180, image.height))
    if image.height / image.width > 200:
        image = image.resize((image.width, image.width * 180))
    return image


def prompt_of(instruction):
    """The tagging prompt with the "<image>" placeholder removed (image is passed separately)."""
    return instruction.replace(IMAGE_TOKEN, "").strip()


def render_prompt(processor, instruction):
    """Chat-template render at training parity: strip the template's empty think block so the
    prompt ends right after `assistant\\n`, token-for-token identical to `qwen3_5_nothink`."""
    text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},                                   # image first, then text
            {"type": "text", "text": prompt_of(instruction)},
        ]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    text = text.replace(THINK_BLOCK, "")
    if not text.endswith(PROMPT_TAIL):                           # template drifted -> fail loud
        raise RuntimeError(f"prompt does not end with {PROMPT_TAIL!r}: ...{text[-60:]!r}")
    return text


# ── row schema: {instruction, image, gold, src} ──────────────────────────────────────────────
def load_rows(path):
    """Read an alpaca eval jsonl or a review samples.json into the canonical row schema."""
    if str(path).endswith(".jsonl"):
        raw = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [dict(instruction=r["instruction"], image=r["images"][0],
                     gold=r.get("output", ""), src=r.get("_src", "?")) for r in raw]
        return rows, raw, "jsonl"
    raw = json.load(open(path, encoding="utf-8"))
    rows = [dict(instruction=r["instruction"], image=r["image"],
                 gold=r.get("gemini", ""), src=r.get("split", "?")) for r in raw]
    return rows, raw, "json"


def save_preds(path, preds, raw, kind):
    """Write predictions back in the shape the caller's input implied."""
    if kind == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for r, p in zip(raw, preds):
                f.write(json.dumps({"label": r.get("output", ""), "predict": p}, ensure_ascii=False) + "\n")
    else:
        for r, p in zip(raw, preds):
            r["pred"] = p
        json.dump(raw, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ── backend: transformers ────────────────────────────────────────────────────────────────────
def generate_hf(rows, ckpt, dtype="fp32", batch_size=8, max_new_tokens=128):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = torch.float32 if dtype == "fp32" else torch.bfloat16
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    # Loading + upcasting 5.4GB of weights is the long silent phase; announce it or the caller
    # thinks the script hung.
    print(f"[hf] loading {dtype} weights onto cuda:{dev} (30-60s) ...", flush=True)
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"                    # batched generation
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt, dtype=torch_dtype, device_map="cuda", trust_remote_code=True,
    ).eval()
    print(f"[hf] model ready in {time.time()-t_load:.0f}s "
          f"({torch.cuda.memory_allocated()/2**30:.1f} GiB); generating bs={batch_size}", flush=True)

    preds, t0 = [], time.time()
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        images = [preprocess_image(r["image"]) for r in batch]
        texts = [render_prompt(processor, r["instruction"]) for r in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
        if torch_dtype == torch.float32 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].float()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        for ids in out[:, inputs["input_ids"].shape[1]:]:
            preds.append(processor.tokenizer.decode(ids, skip_special_tokens=True).strip())
        done = min(i + batch_size, len(rows))
        eta = (time.time() - t0) / done * (len(rows) - done)
        print(f"  [hf] {done}/{len(rows)} ({time.time()-t0:.0f}s, eta {eta:.0f}s)", flush=True)
    print(f"[hf] {len(rows)} preds in {time.time()-t0:.0f}s ({len(rows)/(time.time()-t0):.1f}/s)", flush=True)
    return preds


# ── backend: vLLM OpenAI server ──────────────────────────────────────────────────────────────
def encode_image_b64(path):
    suffix = Path(path).suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "jpeg")
    with open(path, "rb") as f:
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()


def check_prompt_parity(api, model):
    """Abort if the server's chat template still injects the empty think block."""
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": "x"}],
                          "add_generation_prompt": True,
                          "chat_template_kwargs": {"enable_thinking": False}}).encode()
    try:
        req = urllib.request.Request(f"{api}/tokenize", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            tokens = json.loads(resp.read().decode()).get("tokens") or []
    except Exception as e:                                       # endpoint absent -> skip the guard
        print(f"[warn] prompt-parity check skipped ({e!r})", file=sys.stderr)
        return
    if THINK_OPEN_ID in tokens:
        sys.exit("[fatal] server injects an empty <think></think> block (train/infer mismatch, "
                 "-1.2pt microF1). Restart it with start_vllm_server_mikomiko.sh, which passes "
                 "--chat-template chat_template_qwen3_5_lf.jinja")
    print("[vllm] prompt parity OK (no think block)")


def _chat(api, model, text, image_path, max_tokens, retries=3):
    content = [
        {"type": "image_url", "image_url": {"url": encode_image_b64(image_path)}},   # image first
        {"type": "text", "text": text},
    ]
    payload = json.dumps({
        "model": model, "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0.0,
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
    return ""                                                    # empty pred -> scored as a miss


def generate_vllm(rows, api="http://localhost:8110", model="mikomiko", concurrency=32, max_new_tokens=128):
    check_prompt_parity(api, model)
    preds, done, t0 = [None] * len(rows), 0, time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_chat, api, model, prompt_of(r["instruction"]), r["image"], max_new_tokens): i
                for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            preds[futs[fut]] = (fut.result() or "").strip()
            done += 1
            if done % 50 == 0 or done == len(rows):
                print(f"  [vllm] {done}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[vllm] {len(rows)} preds in {time.time()-t0:.0f}s ({len(rows)/(time.time()-t0):.1f}/s)")
    return preds


def generate(rows, backend="vllm", **kw):
    if backend == "hf":
        return generate_hf(rows, kw.get("ckpt", str(DEFAULT_CKPT)), kw.get("dtype", "bf16"),
                           kw.get("batch_size", 8), kw.get("max_new_tokens", 128))
    if backend == "vllm":
        return generate_vllm(rows, kw.get("api", "http://localhost:8110"), kw.get("model", "mikomiko"),
                             kw.get("concurrency", 32), kw.get("max_new_tokens", 128))
    raise ValueError(f"unknown backend {backend!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="eval *.jsonl (alpaca rows) or review *.json (samples)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    # hf backend
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="bf16 = served-model parity (default), drifts with batch; fp32 = batch-invariant greedy")
    ap.add_argument("--batch-size", type=int, default=8)
    # vllm backend
    ap.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8110"))
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "mikomiko"))
    ap.add_argument("--concurrency", type=int, default=32)
    # optional scoring (only meaningful when rows carry gold)
    ap.add_argument("--score", action="store_true", help="score with metrics_mikomiko after generating")
    ap.add_argument("--step", default="adhoc", help="label written to the history row")
    ap.add_argument("--history", default=None)
    ap.add_argument("--metrics-out", default=None)
    args = ap.parse_args()

    rows, raw, kind = load_rows(args.input)
    print(f"[infer] {len(rows)} rows | backend={args.backend} | {args.input}", flush=True)
    preds = generate(rows, backend=args.backend, ckpt=args.ckpt, dtype=args.dtype,
                     batch_size=args.batch_size, api=args.api, model=args.model,
                     concurrency=args.concurrency, max_new_tokens=args.max_new_tokens)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_preds(args.output, preds, raw, kind)
    empty = sum(1 for p in preds if not p)
    print(f"[infer] empty preds={empty} -> {args.output}")

    if args.score:
        if kind != "jsonl":
            sys.exit("[fatal] --score expects an alpaca *.jsonl input (it carries `output` + `_src`)")
        sys.path.insert(0, str(HERE))
        import metrics_mikomiko
        out_dir = os.path.dirname(os.path.abspath(args.output))
        metrics_mikomiko.score(args.output, args.input, args.step,
                               args.history or os.path.join(out_dir, "history.tsv"),
                               args.metrics_out or os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
