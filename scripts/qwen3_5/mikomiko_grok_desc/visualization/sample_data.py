#!/usr/bin/env python3
"""sample_data.py — pick the review-page samples out of the 20260730 description dataset.

Samples N images per (source x language x split) and writes the rows the inference step
consumes. seen = jsonl_desc_0730/train.jsonl (the model saw these images AND their target text,
verbatim); unseen = each source's own test_unseen_mini.jsonl (whole posts/products/creators held
out of training).

Stratified by SOURCE as well as language, which the 0721 version did not need. The merged set is
49.1% pornpics / 24.3% av / 22.6% of / 4.0% oneione, so sampling on language alone would hand a
review page ~1 oneione image in 25 — and oneione is the only source that is majority Chinese and
the only one carrying cosplay/reading material. Per-source cells keep every source visible at a
fixed budget.

Language is read off the section headers, and BOTH header vocabularies count: av was annotated
under its own spec (制作意図 / 前景と主要被写体 / ...) while the other three use grokv8/grokv10
(創作意図 / 前景と主要な被写体 / ...). Recognising only the latter would silently drop every av
Japanese row — 80% of av — from the sample. The variants live in metrics_desc.HEADER_VARIANTS,
which the page's health table shares, so the two can never drift apart.

`system` is carried through: av rows have a non-empty one (its spec is system + a user prompt
with the work's metadata) and inference must replay it.

    python sample_data.py [--n 5] [--seed 42] [--work-dir SAVES/viz_desc_0730]
        (or just run ../infer_desc_9b.sh viz, which drives all steps)
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics_desc import HEADER_VARIANTS, LANGS  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]                      # .../LlamaFactory
TAG_DIR = ROOT / "data/mikomiko_tag"
MERGED_TRAIN = TAG_DIR / "jsonl_desc_0730/train.jsonl"
DEFAULT_WORK = ROOT / "saves/qwen3.5-9b/mikomiko/viz_desc_0730"

# image path marker -> source. Same mapping mix_desc_0730.py uses.
SOURCES = {"av": "/av/", "of": "/onlyfans/", "oneione": "/oneione/", "pornpics": "/pornpics/"}


def source_of(image_path):
    for src, mark in SOURCES.items():
        if mark in image_path:
            return src
    return None


def lang_of(text):
    """Which language's 4 headers does this text carry (any variant)? Exactly one, else None."""
    hit = [lg for lg, vs in HEADER_VARIANTS.items()
           if any(all(h in text for h in hs) for hs in vs)]
    return hit[0] if len(hit) == 1 else None


def pick(rows, n, seed):
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def sample_unseen(n, seed):
    """{(source, lang) -> rows}. Each source's mini is 200 rows, small enough to read whole."""
    out, pools = {}, {}
    for src in SOURCES:
        path = TAG_DIR / f"jsonl_{src}_0730/test_unseen_mini.jsonl"
        if not path.exists():
            print(f"[sample] WARN {path} missing, skipping {src} unseen")
            continue
        by_lang = {lg: [] for lg in LANGS}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                lg = lang_of(r["output"])
                if lg and os.path.exists(r["images"][0]):
                    by_lang[lg].append(r)
        for lg in LANGS:
            pools[(src, lg)] = len(by_lang[lg])
            out[(src, lg)] = pick(by_lang[lg], n, seed)
    print("[sample] unseen pool: " + " ".join(f"{s}/{lg}={v}" for (s, lg), v in pools.items()))
    return out


def sample_seen(path, n, seed):
    """{(source, lang) -> rows} from the 18.9 GB merged train.jsonl, one streaming pass.

    Reservoir-sample per cell on the RAW line: the builders write ensure_ascii=False, so both the
    headers and the image path are literal text and neither needs a json.loads. Only the ~2k
    survivors get parsed. Parsing 3.31M rows to sample 120 would dominate the whole viz run.
    """
    keep = n * 4                                     # oversample: some images may be gone from disk
    cells = [(s, lg) for s in SOURCES for lg in LANGS]
    res = {c: [] for c in cells}
    seen_n = {c: 0 for c in cells}
    rng = random.Random(seed)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            lg = lang_of(line)
            if not lg:
                continue
            src = source_of(line)
            if not src:
                continue
            c = (src, lg)
            seen_n[c] += 1
            if len(res[c]) < keep:
                res[c].append(line)
            else:                                    # classic reservoir: replace with prob keep/seen
                j = rng.randrange(seen_n[c])
                if j < keep:
                    res[c][j] = line
            if (i + 1) % 500_000 == 0:
                print(f"  [sample] scanned {i+1:,} train rows", flush=True)
    print("[sample] train pool: " + " ".join(f"{s}/{lg}={v:,}" for (s, lg), v in seen_n.items()))

    out = {}
    for c in cells:
        rows = []
        for line in res[c]:
            r = json.loads(line)
            if lang_of(r["output"]) == c[1] and os.path.exists(r["images"][0]):
                rows.append(r)
        out[c] = pick(rows, n, seed)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5,
                    help="samples per source per language per split (default 5 -> 4x3x2x5 = 120)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--train", default=str(MERGED_TRAIN))
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    out_path = os.path.join(args.work_dir, "samples.json")

    picked = {
        "unseen": sample_unseen(args.n, args.seed),
        "seen": sample_seen(Path(args.train), args.n, args.seed + 1),
    }

    samples = []
    for split in ("unseen", "seen"):
        for src in SOURCES:
            for lg in LANGS:
                for r in picked[split].get((src, lg), []):
                    img = r["images"][0]
                    name = os.path.basename(img)
                    samples.append(dict(split=split, lang=lg, source=src, name=name,
                                        post_id=Path(img).parent.name, image=img,
                                        gold=r["output"], instruction=r["instruction"],
                                        system=r.get("system", "")))

    missing = [s["name"] for s in samples if not os.path.exists(s["image"])]
    if missing:
        raise SystemExit(f"[fatal] {len(missing)} sampled images not on disk, e.g. {missing[:3]}")

    json.dump(samples, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[sample] total={len(samples)} -> {out_path}")
    for split in ("seen", "unseen"):
        row = {f"{s}/{lg}": sum(1 for x in samples
                                if x["split"] == split and x["source"] == s and x["lang"] == lg)
               for s in SOURCES for lg in LANGS}
        thin = {k: v for k, v in row.items() if v < args.n}
        print(f"[sample] {split}: {sum(row.values())} "
              + (f"(不足 {args.n} 的格: {thin})" if thin else "(每格都满)"))


if __name__ == "__main__":
    sys.exit(main())
