#!/usr/bin/env python3
"""
mix_desc_0730.py
================
Assemble the 20260730 description delivery from its four per-source packages.

  --merge        jsonl_{av,of,oneione,pornpics}_0730/train.jsonl
                   -> jsonl_desc_0730/train.jsonl        (dataset: mikomiko_desc_0730_train)
  --shrink-mini  rewrite each package's test_unseen_mini.jsonl down to MINI_ROWS rows
  --smoke        jsonl_desc_0730/smoke.jsonl             (dataset: mikomiko_desc_0730_smoke)
                   SMOKE_PER_CELL rows per source x language, for train_desc_9b_0730.sh
                   smoke/probe. Drawn from the merged TRAIN file, so it cannot contaminate
                   eval. Stratified rather than head -N because probe measures peak memory,
                   which is set by the longest batch it happens to see: av/ja is ~35% longer
                   than pornpics/zh, and a head -N of a shuffled file is 49% pornpics.

Merge is a real external shuffle, not a round-robin append. mix_train_jsonl.py interleaves
in blocks, which is fine for two files of similar size but wrong here: the four inputs span
1,625,718 down to 132,156 rows, so a round-robin exhausts oneione at 1/12 of the way through
and the last ~800k lines come out pure pornpics. Trainer's RandomSampler would hide that
today, but any later switch to `streaming: true` shuffles inside a fixed-size buffer and
would then see the physical order. Two passes: scatter lines to SHARDS temp files at random,
shuffle each shard in memory (~19.6 GB / 48 ≈ 410 MB), concatenate.

Shrinking is destructive but reversible: every builder seeds its mini sample (random.seed(42))
so `dataset_builder_desc_<src>_0730.py --build` regenerates the full 1,200-row file byte for
byte. Sampling is stratified, not head -200 -- the minis are built language-balanced (400 per
language) and pornpics is additionally balanced across its two prompt versions, so an
unstratified cut would silently undo that. At MINI_ROWS=200 each language cell holds ~67 rows:
enough for a stable per-dataset eval LOSS (each row contributes ~1k tokens, so the estimate
averages over ~200k tokens), not enough for per-language regression detection -- for that, use
test_unseen.jsonl, which is untouched.
"""
import argparse
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("av", "of", "oneione", "pornpics")
PKG_DIR = {s: os.path.join(HERE, f"jsonl_{s}_0730") for s in SOURCES}
OUT_DIR = os.path.join(HERE, "jsonl_desc_0730")
MERGED = os.path.join(OUT_DIR, "train.jsonl")

MINI_ROWS = 200
SMOKE_PER_CELL = 200            # x 4 sources x 3 langs = 2,400 rows
SHARDS = 48
SEED = 20260730
SMOKE = os.path.join(OUT_DIR, "smoke.jsonl")

# Section markers, per language. av's spec words its ja headers differently from the
# grokv8/grokv10 wording the other three use, so both spellings have to be recognised or
# every av row lands in the zh bucket.
LANG_MARK = {
    "en": ("**Creative Intent**",),
    "ja": ("**創作意図**", "**制作意図**"),
    "zh": ("**创作意图**",),
}
LANGS = ("en", "ja", "zh")


def lang_of(output):
    for lg in ("en", "ja"):
        if any(m in output for m in LANG_MARK[lg]):
            return lg
    return "zh"


def load_prompt_blocks(fname):
    path = os.path.join(HERE, "prompt", fname)
    blocks = [b.strip() for b in open(path, encoding="utf-8").read().split("\n\n\n")]
    return {b for b in blocks if b}


def cell_of(source, row):
    """Stratification cell. pornpics carries two prompt versions and its mini is balanced
    across them as well as across languages; collapsing that to language alone would let
    the 71/29 v8/v10 skew back in."""
    lg = lang_of(row["output"])
    if source != "pornpics":
        return lg
    body = row["instruction"][len("<image>"):]
    return (lg, "v8" if body in cell_of.V8 else "v10" if body in cell_of.V10 else "?")


cell_of.V8 = load_prompt_blocks("prompt_grokv8.txt")
cell_of.V10 = load_prompt_blocks("prompt_grokv10.txt")


def merge():
    inputs = [(s, os.path.join(PKG_DIR[s], "train.jsonl")) for s in SOURCES]
    for s, p in inputs:
        if not os.path.exists(p):
            raise SystemExit(f"[fatal] missing input: {p}")
    os.makedirs(OUT_DIR, exist_ok=True)

    rng = random.Random(SEED)
    tmpdir = tempfile.mkdtemp(prefix="mix_desc_0730_", dir=OUT_DIR)
    try:
        handles = [open(os.path.join(tmpdir, f"{i:03d}.jsonl"), "w", encoding="utf-8")
                   for i in range(SHARDS)]
        counts = Counter()
        for source, path in inputs:
            n = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    handles[rng.randrange(SHARDS)].write(line)
                    n += 1
            counts[source] = n
            print(f"[merge] scattered {source:9s} {n:>9,} rows", flush=True)
        for h in handles:
            h.close()

        total = 0
        with open(MERGED, "w", encoding="utf-8") as out:
            for i in range(SHARDS):
                sp = os.path.join(tmpdir, f"{i:03d}.jsonl")
                lines = open(sp, encoding="utf-8").readlines()
                rng.shuffle(lines)
                out.writelines(lines)
                total += len(lines)
                os.remove(sp)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"[merge] {MERGED}")
    for s in SOURCES:
        print(f"[merge]   {s:9s} {counts[s]:>9,}  ({counts[s] / total:.1%})")
    print(f"[merge]   {'TOTAL':9s} {total:>9,}  "
          f"({os.path.getsize(MERGED) / 2**30:.1f} GiB)", flush=True)


def shrink_mini():
    for source in SOURCES:
        path = os.path.join(PKG_DIR[source], "test_unseen_mini.jsonl")
        if not os.path.exists(path):
            print(f"[mini] {source}: missing {path}, skipped", flush=True)
            continue
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        if len(rows) <= MINI_ROWS:
            print(f"[mini] {source}: already {len(rows)} rows, left alone", flush=True)
            continue

        by_cell = defaultdict(list)
        for r in rows:
            by_cell[cell_of(source, r)].append(r)
        cells = sorted(by_cell, key=str)

        # Spread the remainder over the first cells rather than dropping it, so the total
        # is exactly MINI_ROWS.
        base, extra = divmod(MINI_ROWS, len(cells))
        rng = random.Random(SEED)
        kept = []
        for i, c in enumerate(cells):
            want = base + (1 if i < extra else 0)
            pool = by_cell[c]
            if len(pool) < want:
                print(f"[mini] {source}: WARN cell {c} has {len(pool)} < {want}", flush=True)
            kept.extend(rng.sample(pool, min(want, len(pool))))
        rng.shuffle(kept)

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        mix = Counter(lang_of(r["output"]) for r in kept)
        print(f"[mini] {source:9s} {len(rows):>5,} -> {len(kept):>3} rows  ("
              + " ".join(f"{lg}={mix[lg]}" for lg in LANGS)
              + (f" | {len(cells)} cells" if source == "pornpics" else "") + ")", flush=True)


SRC_MARK = {"av": "/av/", "of": "/onlyfans/", "oneione": "/oneione/", "pornpics": "/pornpics/"}


def source_of(image_path):
    for src, mark in SRC_MARK.items():
        if mark in image_path:
            return src
    return "?"


def smoke():
    if not os.path.exists(MERGED):
        raise SystemExit(f"[fatal] {MERGED} missing -- run --merge first.")
    # Reservoir sample per (source, language) cell in one streaming pass: the merged file
    # is ~19 GB, far past what fits in memory.
    rng = random.Random(SEED)
    res, seen = defaultdict(list), Counter()
    with open(MERGED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cell = (source_of(r["images"][0]), lang_of(r["output"]))
            seen[cell] += 1
            pool = res[cell]
            if len(pool) < SMOKE_PER_CELL:
                pool.append(line)
            else:
                j = rng.randrange(seen[cell])
                if j < SMOKE_PER_CELL:
                    pool[j] = line

    rows = [ln for cell in sorted(res) for ln in res[cell]]
    rng.shuffle(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SMOKE, "w", encoding="utf-8") as f:
        f.writelines(rows)
    print(f"[smoke] {SMOKE}: {len(rows):,} rows from {len(res)} cells", flush=True)
    for cell in sorted(res):
        print(f"[smoke]   {cell[0]:9s} {cell[1]}  {len(res[cell]):>4}  "
              f"(pool {seen[cell]:,})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merge", action="store_true", help="build jsonl_desc_0730/train.jsonl")
    ap.add_argument("--shrink-mini", action="store_true",
                    help=f"rewrite each test_unseen_mini.jsonl to {MINI_ROWS} rows")
    ap.add_argument("--smoke", action="store_true",
                    help=f"build jsonl_desc_0730/smoke.jsonl ({SMOKE_PER_CELL}/cell)")
    args = ap.parse_args()
    if not (args.merge or args.shrink_mini or args.smoke):
        ap.error("nothing to do: pass --merge, --shrink-mini and/or --smoke")
    if args.shrink_mini:
        shrink_mini()
    if args.merge:
        merge()
    if args.smoke:
        smoke()


if __name__ == "__main__":
    main()
