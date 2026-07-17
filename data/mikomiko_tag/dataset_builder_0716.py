#!/usr/bin/env python3
"""
dataset_builder_0716.py
=======================
Build the LlamaFactory alpaca jsonl for the 20260716 cover-hit delivery.
Standalone dataset -- the 124w set in jsonl/ is NOT merged and NOT touched.

Source : pornpics_tag_cover_dataset_20260716/..._first_batch_hits_20260716.parquet
Images : img_0716/{post_id}_{image_name}   (fetched by download_0716.py)
Output : jsonl_0716/{train,test_unseen,test_unseen_mini}.jsonl

Label = category + post_tag, comma-joined, Title Case.

  Ordering -- category first, then post_tag. Not a guess: on 200k rows of the old
  delivery, the labels are category-before-post_tag 99.7% of the time (the annotation
  run was handed the candidates in that order and returned them in it). The parquet
  stores each list alphabetically, so within-group order is already lost; the
  group order is what carries over.

  Casing -- the parquet is all-lowercase; the old dataset and tag_vocab.txt are
  Title Case with all-caps abbreviations (MILF, BBC, PAWG...). Rule:
    1. whole tag in tag_vocab.txt      -> reuse the vocab's exact surface form
    2. otherwise                       -> Title Case per word, except a word the
                                          vocab writes as an all-caps abbreviation,
                                          which keeps that form ('shemale milf' ->
                                          'Shemale MILF', not 'Shemale Milf')
    3. 'ai'                            -> 'AI'  (not in the old vocab; see AI_EXTRA)
  Rule 2 exists so one token never renders two ways across the corpus; without it
  'Amateur MILF' and 'Shemale Milf' would coexist. render_map() asserts that.

Split: train + test_unseen only (whole posts held out). No test_stratified -- the new
vocab is closed (3,391 tags, min DF 8, median 1,708) and has no rare tail, so the old
DF bands collapse: DF[1,10) holds 8 images against a 400/band target.

    python dataset_builder_0716.py --plan     # funnel + image coverage, writes nothing
    python dataset_builder_0716.py --build    # write jsonl_0716/
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
PARQUET = os.path.join(HERE, "pornpics_tag_cover_dataset_20260716",
                       "pornpics_category_tag_cover_first_batch_hits_20260716.parquet")
TAG_VOCAB = os.path.join(HERE, "tag_vocab.txt")
IMG_DIR = os.path.join(HERE, "img_0716")
OUT_DIR = os.path.join(HERE, "jsonl_0716")
PROMPT_FILE = os.path.join(HERE, "prompt.txt")

SPLIT_SEED = 0
POST_HOLDOUT_FRAC = 0.10
# 400, not the old builder's 200: mikomiko_HANDOFF.md concluded a 400-sample mini is the
# smallest that gives a usable CI (+-1.4); n=10 sanity checks misread quality.
MINI_EVAL_N = 400
AI_EXTRA = {"ai": "AI"}


def load_prompt():
    p = ""
    if os.path.exists(PROMPT_FILE):
        p = open(PROMPT_FILE, encoding="utf-8").read().strip()
    if not p:
        sys.exit(f"[fatal] {PROMPT_FILE} missing or empty -- it defines the task framing "
                 f"and must match the 124w dataset's.")
    return p


def render_map(tags):
    """{lowercase tag -> Title Case surface form} for every tag in the delivery."""
    vocab = {}
    for t in open(TAG_VOCAB, encoding="utf-8").read().split(","):
        t = " ".join(t.split()).strip()
        if t:
            vocab.setdefault(t.lower(), t)
    # words the old vocab writes as a bare all-caps abbreviation (BBC, MILF, PAWG, ...)
    caps = {k: v for k, v in vocab.items() if " " not in v and v.isupper()}
    caps.update(AI_EXTRA)

    out = {}
    for t in tags:
        out[t] = vocab[t] if t in vocab else " ".join(caps.get(w, w.title()) for w in t.split())

    # One token must never render two ways, or the model sees 'MILF' and 'Milf' as
    # different symbols for one concept. Loud failure beats silent inconsistency.
    seen = defaultdict(set)
    for low, surf in out.items():
        for lw, sw in zip(low.split(), surf.split()):
            seen[lw].add(sw)
    clash = {w: v for w, v in seen.items() if len(v) > 1}
    if clash:
        sys.exit(f"[fatal] token renders inconsistently: "
                 f"{ {w: sorted(v) for w, v in list(clash.items())[:8]} }")
    hit = sum(1 for t in tags if t in vocab)
    print(f"[case] {len(tags):,} tags: {hit:,} from tag_vocab.txt verbatim, "
          f"{len(tags) - hit:,} Title Cased (abbreviations preserved: "
          f"{', '.join(sorted(set(caps.values())))})", flush=True)
    return out


def load_rows():
    """[(key, [lowercase tags in output order])] -- category first, then post_tag."""
    f = pq.ParquetFile(PARQUET)
    rows, vocab = [], set()
    for b in f.iter_batches(batch_size=200_000,
                            columns=["post_id", "image_name", "category", "post_tag"]):
        d = b.to_pydict()
        for pid, img, cat, pt in zip(d["post_id"], d["image_name"], d["category"], d["post_tag"]):
            seen, tags = set(), []
            for t in list(cat) + list(pt):      # category before post_tag; dedupe keeps first
                if t not in seen:
                    seen.add(t)
                    tags.append(t)
            rows.append((f"{pid}_{img}", tags))
            vocab.update(tags)
    if len({k for k, _ in rows}) != len(rows):
        sys.exit("[fatal] duplicate '{post_id}_{image_name}' keys in the parquet.")
    return rows, vocab


def build(args):
    t0 = time.time()
    rows, vocab = load_rows()
    print(f"[plan] parquet rows={len(rows):,}  distinct tags={len(vocab):,} "
          f"({time.time() - t0:.0f}s)", flush=True)
    surface = render_map(vocab)

    have = ({e.name for e in os.scandir(IMG_DIR) if e.is_file() and not e.name.endswith(".tmp")}
            if os.path.isdir(IMG_DIR) else set())
    drop = Counter()
    kept = []
    for key, tags in rows:
        if key not in have:
            drop["no_image"] += 1          # not downloaded (yet) -> cannot train on it
            continue
        if not tags:
            drop["empty_tags"] += 1
            continue
        kept.append((key, ", ".join(surface[t] for t in tags)))
    print(f"[plan] funnel: parquet={len(rows):,}  -no_image={drop['no_image']:,}"
          f"  -empty_tags={drop['empty_tags']:,}  = usable {len(kept):,}"
          f"  ({len(kept) / len(rows):.1%} of delivery)", flush=True)
    if drop["no_image"]:
        print(f"[plan] NOTE {drop['no_image']:,} rows have no image on disk. If download_0716.py "
              f"is still running, rerun --build when it finishes.", flush=True)
    if not kept:
        sys.exit("[fatal] no usable rows -- is img_0716/ populated?")

    by_post = defaultdict(list)
    for key, out in kept:
        by_post[key.split("_", 1)[0]].append((key, out))
    posts = sorted(by_post)
    random.seed(SPLIT_SEED)
    random.shuffle(posts)
    n_hold = int(round(len(posts) * POST_HOLDOUT_FRAC))
    test = [r for p in posts[:n_hold] for r in by_post[p]]
    train = [r for p in posts[n_hold:] for r in by_post[p]]
    print(f"[plan] posts={len(posts):,}  held out={n_hold:,} ({POST_HOLDOUT_FRAC:.0%})"
          f"  -> train={len(train):,}  test_unseen={len(test):,}", flush=True)

    ntag = sum(o.count(",") + 1 for _, o in kept) / len(kept)
    print(f"[plan] avg tags/image={ntag:.2f}", flush=True)
    if args.plan:
        k, o = kept[0]
        print(f"[plan] sample -> {k}\n         {o}")
        print("[plan] --plan, nothing written.", flush=True)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    instr = "<image>" + load_prompt()
    for name, rowset in (("train", train), ("test_unseen", test)):
        random.seed(1234)
        random.shuffle(rowset)
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for key, out in rowset:
                f.write(json.dumps({"instruction": instr, "input": "", "output": out,
                                    "images": [os.path.join(IMG_DIR, key)]},
                                   ensure_ascii=False) + "\n")
        print(f"[build] {name}.jsonl: {len(rowset):,} rows", flush=True)

    random.seed(42)
    mini = random.sample(test, min(MINI_EVAL_N, len(test)))
    path = os.path.join(OUT_DIR, "test_unseen_mini.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for key, out in mini:
            f.write(json.dumps({"instruction": instr, "input": "", "output": out,
                                "images": [os.path.join(IMG_DIR, key)]}, ensure_ascii=False) + "\n")
    print(f"[build] test_unseen_mini.jsonl: {len(mini):,} rows", flush=True)
    print(f"[build] DONE -> {OUT_DIR}  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build the 20260716 cover-hit tagging dataset.")
    ap.add_argument("--plan", action="store_true", help="report funnel/coverage, write nothing")
    ap.add_argument("--build", action="store_true", help="write jsonl_0716/")
    args = ap.parse_args()
    if not (args.plan or args.build):
        args.plan = True
    build(args)


if __name__ == "__main__":
    main()
