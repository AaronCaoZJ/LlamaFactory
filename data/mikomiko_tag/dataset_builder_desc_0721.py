#!/usr/bin/env python3
"""
dataset_builder_desc_0721.py
============================
Build the LlamaFactory alpaca jsonl for the 20260721 *description* delivery.
Standalone dataset -- the tag datasets in jsonl/ and jsonl_0716/ are NOT touched.

Source : RAW/effective_20260721_llm_training.parquet   (1,369,397 rows)
Prompt : prompt_grokv8.txt     -- ONE instruction in THREE languages, '\\n\\n\\n'-separated,
                                 in the order en, ja, zh.
Images : img_0716/{post_id}_{image_name}   (already 100% on disk; verified 1,369,397/1,369,397)
Output : jsonl_desc_0721/{train,test_unseen,test_unseen_mini}.jsonl

This is a different task from the tag datasets: the target is a 4-section prose
description (~500-950 output tokens), not a comma-separated tag list (~30).

  Language -- each image carries exactly ONE language (verified: 0 duplicate keys,
  1 language per image), split 80/10/10 en/zh/ja. So the three languages are NOT a
  parallel corpus, and the prompt block is the only signal telling the model which
  language to answer in. Pairing row.desc_lang with its own prompt block is therefore
  load-bearing, not cosmetic: feed every row the English block and the 274k ja/zh rows
  become noise. assert_prompt_blocks() fails loudly if the file's block order drifts.

  images -- img_0716/, never img/. The 1.23M files in img/ are the '/cut/' variant
  (watermark banner stripped) and 135,747 keys collide by name with different pixels.
  Pointing there trains on images that are not the ones the text describes, silently.

Filtering (~0.2% of rows; all are annotator failures that would teach broken format):
  - desc_lang not in {en, ja, zh}          (6 rows; desc_desc parsed to all-None)
  - fewer than 4 section headers present   (~1.7k; truncated or degenerate)
  - raw_output >= MAX_CHARS                (runaway generations, up to 27,714 chars)
  - repetition_score >= MAX_REP            ('enticing  enticing  enticing ...' loops)

Split: train + test_unseen, whole posts held out so no post straddles the boundary.
HOLDOUT_FRAC is 0.02 here, not the 0716 builder's 0.10: at 1.37M rows a 10% holdout
parks 137k samples (~200M training tokens at this task's length) against an eval that
never reads more than a few thousand. 2% still leaves 20x the largest mini.

    python dataset_builder_desc_0721.py --plan     # funnel + coverage, writes nothing
    python dataset_builder_desc_0721.py --build    # write jsonl_desc_0721/
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "RAW", "effective_20260721_llm_training.parquet")
PROMPT_FILE = os.path.join(HERE, "prompt_grokv8.txt")
IMG_DIR = os.path.join(HERE, "img_0716")
OUT_DIR = os.path.join(HERE, "jsonl_desc_0721")

LANGS = ("en", "ja", "zh")          # the order the blocks appear in prompt_grokv8.txt
SPLIT_SEED = 0
POST_HOLDOUT_FRAC = 0.02
MINI_PER_LANG = 400                 # mikomiko_HANDOFF.md: 400 is the smallest n with a usable CI
MAX_CHARS = 6000                    # p99 of a healthy row is ~3.5k chars; beyond this is runaway
MAX_REP = 0.3                       # fraction of the text covered by one repeated 40-gram
REP_N = 40

# The 4 section headers each language's prompt asks for. A row missing any of them did
# not follow the format, whatever else it contains.
HEADERS = {
    "en": ("Creative Intent", "Foreground and Subject",
           "Background and Environment", "Photography Techniques and Visual Presentation"),
    "ja": ("創作意図", "前景と主要な被写体", "背景と周囲の環境", "撮影技法と視覚的表現"),
    "zh": ("创作意图", "前景与主体", "背景与环境", "摄影技术与视觉呈现"),
}


def load_prompts():
    """{lang -> instruction text}, asserting the file's block order is still en, ja, zh."""
    if not os.path.exists(PROMPT_FILE):
        sys.exit(f"[fatal] {PROMPT_FILE} missing -- it defines the task framing and the "
                 f"language switch.")
    blocks = [b.strip() for b in open(PROMPT_FILE, encoding="utf-8").read().split("\n\n\n")]
    blocks = [b for b in blocks if b]
    if len(blocks) != 3:
        sys.exit(f"[fatal] {PROMPT_FILE} split into {len(blocks)} blocks, expected 3 "
                 f"(en/ja/zh separated by a blank line pair).")
    prompts = dict(zip(LANGS, blocks))
    # Each block must name its own language's section headers. If the file is ever
    # reordered or re-translated, every row gets the wrong-language instruction and the
    # only symptom is a model that answers in the wrong language -- catch it here.
    for lg, text in prompts.items():
        missing = [h for h in HEADERS[lg] if h not in text]
        if missing:
            sys.exit(f"[fatal] prompt block #{LANGS.index(lg) + 1} was taken as '{lg}' but "
                     f"does not contain its headers {missing}. Block order in "
                     f"{PROMPT_FILE} changed?")
    print(f"[prompt] 3 blocks OK -- " +
          ", ".join(f"{lg}={len(prompts[lg])}ch" for lg in LANGS), flush=True)
    return prompts


def repetition_score(s):
    """Fraction of the text covered by its most frequent non-overlapping REP_N-gram.

    A healthy paragraph scores <0.05; a generation stuck in a loop approaches 1.0.
    Cheap enough to run on every row (~62 grams per 2.5k-char description).
    """
    if len(s) < REP_N * 3:
        return 0.0
    grams = Counter(s[i:i + REP_N] for i in range(0, len(s) - REP_N, REP_N))
    return grams.most_common(1)[0][1] * REP_N / len(s)


def load_rows():
    """[(key, lang, raw_output)] straight from the parquet, unfiltered."""
    pf = pq.ParquetFile(PARQUET)
    rows = []
    for b in pf.iter_batches(batch_size=100_000,
                             columns=["post_id", "image_name", "desc_lang", "raw_output"]):
        d = b.to_pydict()
        rows.extend(zip((f"{p}_{i}" for p, i in zip(d["post_id"], d["image_name"])),
                        d["desc_lang"], d["raw_output"]))
    if len({k for k, _, _ in rows}) != len(rows):
        sys.exit("[fatal] duplicate '{post_id}_{image_name}' keys in the parquet -- the flat "
                 "image filename would collide.")
    return rows


def build(args):
    t0 = time.time()
    prompts = load_prompts()
    rows = load_rows()
    print(f"[plan] parquet rows={len(rows):,}  ({time.time() - t0:.0f}s)", flush=True)

    have = ({e.name for e in os.scandir(IMG_DIR) if e.is_file() and not e.name.endswith(".tmp")}
            if os.path.isdir(IMG_DIR) else set())
    print(f"[plan] {IMG_DIR}: {len(have):,} files on disk", flush=True)

    drop, kept = Counter(), []
    for key, lang, raw in rows:
        if lang not in HEADERS:
            drop["bad_lang"] += 1
        elif not raw or not raw.strip():
            drop["empty_output"] += 1
        elif any(h not in raw for h in HEADERS[lang]):
            drop["missing_section"] += 1
        elif len(raw) >= MAX_CHARS:
            drop["too_long"] += 1
        elif repetition_score(raw) >= MAX_REP:
            drop["repetition"] += 1
        elif key not in have:
            drop["no_image"] += 1
        else:
            kept.append((key, lang, raw.strip()))

    print(f"[plan] funnel: parquet={len(rows):,}"
          + "".join(f"  -{k}={v:,}" for k, v in drop.most_common())
          + f"  = usable {len(kept):,} ({len(kept) / len(rows):.2%} of delivery)", flush=True)
    if drop["no_image"]:
        print(f"[plan] NOTE {drop['no_image']:,} rows have no image in img_0716/. Expected 0 "
              f"-- the delivery was verified 100% covered. Investigate before training.",
              flush=True)
    if not kept:
        sys.exit("[fatal] no usable rows.")

    lang_n = Counter(l for _, l, _ in kept)
    print(f"[plan] language mix: "
          + "  ".join(f"{lg}={lang_n[lg]:,} ({lang_n[lg] / len(kept):.1%})" for lg in LANGS),
          flush=True)

    by_post = defaultdict(list)
    for r in kept:
        by_post[r[0].split("_", 1)[0]].append(r)
    posts = sorted(by_post)
    random.seed(SPLIT_SEED)
    random.shuffle(posts)
    n_hold = int(round(len(posts) * POST_HOLDOUT_FRAC))
    test = [r for p in posts[:n_hold] for r in by_post[p]]
    train = [r for p in posts[n_hold:] for r in by_post[p]]
    print(f"[plan] posts={len(posts):,}  held out={n_hold:,} ({POST_HOLDOUT_FRAC:.0%})"
          f"  -> train={len(train):,}  test_unseen={len(test):,}", flush=True)

    avg = sum(len(r[2]) for r in kept) / len(kept)
    print(f"[plan] avg output chars={avg:,.0f}", flush=True)

    if args.plan:
        k, lg, out = kept[0]
        print(f"[plan] sample -> {k}  lang={lg}\n         instruction[:80]="
              f"{prompts[lg][:80]!r}\n         output[:120]={out[:120]!r}")
        print("[plan] --plan, nothing written.", flush=True)
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    def dump(name, rowset):
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for key, lang, out in rowset:
                f.write(json.dumps({"instruction": "<image>" + prompts[lang],
                                    "input": "",
                                    "output": out,
                                    "images": [os.path.join(IMG_DIR, key)]},
                                   ensure_ascii=False) + "\n")
        mix = Counter(l for _, l, _ in rowset)
        print(f"[build] {name}.jsonl: {len(rowset):,} rows  ("
              + " ".join(f"{lg}={mix[lg]:,}" for lg in LANGS) + ")", flush=True)

    for name, rowset in (("train", train), ("test_unseen", test)):
        random.seed(1234)
        random.shuffle(rowset)
        dump(name, rowset)

    # Language-stratified, not a flat sample: at 80/10/10 a flat 1,200-row mini would
    # hand ja and zh ~120 rows each -- too few to tell a per-language regression from noise.
    random.seed(42)
    by_lang = defaultdict(list)
    for r in test:
        by_lang[r[1]].append(r)
    mini = []
    for lg in LANGS:
        pool = by_lang[lg]
        take = min(MINI_PER_LANG, len(pool))
        if take < MINI_PER_LANG:
            print(f"[build] WARN mini: only {take} {lg} rows in test_unseen "
                  f"(wanted {MINI_PER_LANG}); its CI will be wider than the others.", flush=True)
        mini.extend(random.sample(pool, take))
    random.shuffle(mini)
    dump("test_unseen_mini", mini)

    print(f"[build] DONE -> {OUT_DIR}  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build the 20260721 description dataset.")
    ap.add_argument("--plan", action="store_true", help="report funnel/coverage, write nothing")
    ap.add_argument("--build", action="store_true", help="write jsonl_desc_0721/")
    args = ap.parse_args()
    if not (args.plan or args.build):
        args.plan = True
    build(args)


if __name__ == "__main__":
    main()
