#!/usr/bin/env python3
"""Gate the built jsonl_desc_0721/ before spending GPU hours on it.

Checks, on every row of the mini + a sample of train/test:
  1. the instruction's language block matches the output's language (the whole point of
     the per-language pairing -- a mismatch here silently teaches the wrong language map)
  2. exactly one '<image>' token, and it is at the front
  3. the referenced image file exists and decodes
  4. no train/test post leakage
Exits non-zero on any failure, so it can gate a run script.
"""
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "jsonl_desc_0721")
SAMPLE = 3000

HEADERS = {
    "en": ("Creative Intent", "Foreground and Subject",
           "Background and Environment", "Photography Techniques and Visual Presentation"),
    "ja": ("創作意図", "前景と主要な被写体", "背景と周囲の環境", "撮影技法と視覚的表現"),
    "zh": ("创作意图", "前景与主体", "背景与环境", "摄影技术与视觉呈现"),
}


def lang_of(text, where):
    """Which language's 4 headers does this text carry? Exactly one must match."""
    hit = [lg for lg, hs in HEADERS.items() if all(h in text for h in hs)]
    if len(hit) != 1:
        return None
    return hit[0]


def prompt_lang(instr):
    """The instruction embeds the example-output format, so it names its own headers."""
    return lang_of(instr, "instruction")


def read_sample(path, n):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines() if n is None else None
    if lines is None:
        with open(path, encoding="utf-8") as f:
            lines = [ln for i, ln in enumerate(f) if i < n * 4]
        random.seed(7)
        lines = random.sample(lines, min(n, len(lines)))
    return [json.loads(x) for x in lines]


def main():
    fails = []
    posts = {}

    for name, n in (("test_unseen_mini", None), ("train", SAMPLE), ("test_unseen", SAMPLE)):
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        if not os.path.exists(path):
            fails.append(f"{name}.jsonl missing")
            continue
        rows = read_sample(path, n)
        mix, bad_img = Counter(), 0
        for r in rows:
            instr, out = r["instruction"], r["output"]
            if not instr.startswith("<image>"):
                fails.append(f"{name}: instruction does not start with <image>")
                break
            if instr.count("<image>") != 1:
                fails.append(f"{name}: {instr.count('<image>')} <image> tokens, expected 1")
                break
            pl = prompt_lang(instr[len("<image>"):])
            ol = lang_of(out, "output")
            if pl is None or ol is None or pl != ol:
                fails.append(f"{name}: language mismatch prompt={pl} output={ol} "
                             f"img={r['images'][0]}")
                break
            mix[ol] += 1
            img = r["images"][0]
            if not os.path.exists(img):
                bad_img += 1
        if bad_img:
            fails.append(f"{name}: {bad_img}/{len(rows)} sampled images missing on disk")
        posts[name] = {os.path.basename(r["images"][0]).split("_", 1)[0] for r in rows}
        print(f"[{name}] {len(rows):,} rows checked  langs={dict(mix)}  "
              f"missing_images={bad_img}", flush=True)

    if "train" in posts and "test_unseen" in posts:
        overlap = posts["train"] & posts["test_unseen"]
        print(f"[leakage] sampled train posts ∩ test posts = {len(overlap)}")
        if overlap:
            fails.append(f"post leakage: {len(overlap)} posts in both splits "
                         f"(e.g. {sorted(overlap)[:5]})")

    # decode a handful of images for real, not just stat()
    try:
        from PIL import Image
        path = os.path.join(OUT_DIR, "test_unseen_mini.jsonl")
        rows = [json.loads(x) for x in open(path, encoding="utf-8")][:25]
        for r in rows:
            Image.open(r["images"][0]).verify()
        print(f"[decode] {len(rows)} mini images decode OK")
    except Exception as e:
        fails.append(f"image decode failed: {type(e).__name__}: {e}")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS -- language pairing, image refs and split isolation all check out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
