#!/usr/bin/env python3
"""sample_data.py — step 1/3 of the grok_desc review page pipeline.

Samples N images per (language x split) from the 20260721 description dataset and writes the
rows the inference step consumes. seen = train.jsonl (the model saw these images AND their
target text, verbatim); unseen = test_unseen_mini.jsonl (whole posts held out of training).

WHY THE LANGUAGE STRATIFICATION IS NOT OPTIONAL
Each image carries exactly ONE language and the split is 80/10/10 en/zh/ja, so a naive random
sample of 60 train rows lands ~48 en / ~6 zh / ~6 ja and cannot say anything about ja at all.
The language block inside the instruction is the ONLY signal telling the model which language to
answer in (see data/mikomiko_tag/dataset_builder_desc_0721.py) -- whether that switch survived
training is the single most load-bearing question this page answers, so every language gets the
same n.

Language is read off the instruction's 4 section headers (the prompt embeds its own example
output format), matching data/mikomiko_tag/verify_desc_0721.py. Rows whose instruction matches
zero or more than one language are skipped rather than guessed at.

Output: WORK_DIR/samples.json
        [{split, lang, name, post_id, image, gold, instruction}]
Next:   ../../mikomiko_tagger/infer_mikomiko.py --input WORK/samples.json --output WORK/..._pred.json
        (or just run ../infer_desc_9b.sh viz, which drives all steps)

Usage:
    python sample_data.py [--n 20] [--seed 42] [--work-dir SAVES/viz_desc_0721]
"""
import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]                      # .../LlamaFactory
JSONL_DIR = ROOT / "data/mikomiko_tag/jsonl_desc_0721"
DEFAULT_WORK = ROOT / "saves/qwen3.5-9b/mikomiko/viz_desc_0721"

LANGS = ("en", "ja", "zh")
# Same 4 headers the builder and the verify gate use. Keep in sync with
# data/mikomiko_tag/dataset_builder_desc_0721.py:HEADERS -- a drift here silently mislabels
# every sample's language.
HEADERS = {
    "en": ("Creative Intent", "Foreground and Subject",
           "Background and Environment", "Photography Techniques and Visual Presentation"),
    "ja": ("創作意図", "前景と主要な被写体", "背景と周囲の環境", "撮影技法と視覚的表現"),
    "zh": ("创作意图", "前景与主体", "背景与环境", "摄影技术与视觉呈现"),
}


def lang_of(text):
    """Which language's 4 headers does this text carry? Exactly one must match, else None."""
    hit = [lg for lg, hs in HEADERS.items() if all(h in text for h in hs)]
    return hit[0] if len(hit) == 1 else None


def pick(rows, n, seed):
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def sample_unseen(path, n, seed):
    """test_unseen_mini is 1200 rows (400/lang) -- small enough to read whole and bucket."""
    by_lang = {lg: [] for lg in LANGS}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lg = lang_of(r["instruction"])
            if lg:
                by_lang[lg].append(r)
    print(f"[sample] unseen pool: " + " ".join(f"{lg}={len(v)}" for lg, v in by_lang.items()))
    return {lg: pick(v, n, seed) for lg, v in by_lang.items()}


def sample_seen(path, n, seed):
    """train.jsonl is 7.3 GB / 1.34M rows. One streaming pass, reservoir-sampling per language.

    Language is detected on the RAW line (the builder writes ensure_ascii=False, so the headers
    are literal text) to avoid json-parsing 1.34M rows; the ~60 survivors are parsed and their
    language re-confirmed on the parsed instruction before they are kept.
    """
    keep = n * 3                                    # oversample: some images may be missing on disk
    res = {lg: [] for lg in LANGS}                  # reservoir of raw lines
    seen_n = {lg: 0 for lg in LANGS}
    rng = random.Random(seed)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            lg = lang_of(line)
            if not lg:
                continue
            seen_n[lg] += 1
            if len(res[lg]) < keep:
                res[lg].append(line)
            else:                                   # classic reservoir: replace with prob keep/seen
                j = rng.randrange(seen_n[lg])
                if j < keep:
                    res[lg][j] = line
            if (i + 1) % 200_000 == 0:
                print(f"  [sample] scanned {i+1:,} train rows "
                      + " ".join(f"{k}={v:,}" for k, v in seen_n.items()), flush=True)
    print(f"[sample] train pool: " + " ".join(f"{lg}={v:,}" for lg, v in seen_n.items()))

    out = {}
    for lg in LANGS:
        rows = []
        for line in res[lg]:
            r = json.loads(line)
            if lang_of(r["instruction"]) == lg and os.path.exists(r["images"][0]):
                rows.append(r)
        out[lg] = pick(rows, n, seed)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="samples per language per split (default 20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--jsonl-dir", default=str(JSONL_DIR))
    args = ap.parse_args()

    jsonl_dir = Path(args.jsonl_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    out_path = os.path.join(args.work_dir, "samples.json")

    picked = {
        "unseen": sample_unseen(jsonl_dir / "test_unseen_mini.jsonl", args.n, args.seed),
        "seen": sample_seen(jsonl_dir / "train.jsonl", args.n, args.seed + 1),
    }

    samples = []
    for split in ("unseen", "seen"):
        for lg in LANGS:
            for r in picked[split][lg]:
                img = r["images"][0]
                name = os.path.basename(img)
                samples.append(dict(split=split, lang=lg, name=name,
                                    post_id=name.split("_", 1)[0], image=img,
                                    gold=r["output"], instruction=r["instruction"]))

    missing = [s["name"] for s in samples if not os.path.exists(s["image"])]
    if missing:
        raise SystemExit(f"[fatal] {len(missing)} sampled images not on disk, e.g. {missing[:3]}")

    json.dump(samples, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[sample] total={len(samples)} -> {out_path}")
    for split in ("seen", "unseen"):
        counts = {lg: sum(1 for s in samples if s["split"] == split and s["lang"] == lg)
                  for lg in LANGS}
        print(f"[sample] {split:<6} " + " ".join(f"{lg}={c}" for lg, c in counts.items()))


if __name__ == "__main__":
    main()
