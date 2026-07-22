#!/usr/bin/env python3
"""Analyse effective_20260721_llm_training.parquet:
  - language distribution (desc_lang) and how the 3 languages are assigned
  - is the row key (post_id, image_name) unique? does one image get >1 language?
  - source / dataset_id / described distribution
  - null / empty rates for the fields we would train on
  - local image coverage against img/ and img_0716/
"""
import os
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "RAW", "effective_20260721_llm_training.parquet")
IMG_DIRS = {"img": os.path.join(HERE, "img"), "img_0716": os.path.join(HERE, "img_0716")}

COLS = ["post_id", "image_name", "sample_id", "dataset_id", "source",
        "desc_lang", "described", "raw_output", "category", "post_tag"]


def main():
    pf = pq.ParquetFile(PARQUET)
    print(f"rows={pf.metadata.num_rows:,}  row_groups={pf.metadata.num_row_groups}", flush=True)

    lang = Counter()
    src = Counter()
    dsid = Counter()
    desc_flag = Counter()
    lang_by_post = defaultdict(set)     # post_id -> {langs}
    langs_per_key = defaultdict(set)    # "{pid}_{img}" -> {langs}
    keys = Counter()                    # duplicate detection
    empty_raw = 0
    raw_len = Counter()                 # bucketed char length
    n = 0

    for b in pf.iter_batches(batch_size=100_000, columns=COLS):
        d = b.to_pydict()
        for pid, img, sid, did, so, lg, dsc, raw in zip(
                d["post_id"], d["image_name"], d["sample_id"], d["dataset_id"],
                d["source"], d["desc_lang"], d["described"], d["raw_output"]):
            n += 1
            lang[lg] += 1
            src[so] += 1
            dsid[did] += 1
            desc_flag[dsc] += 1
            key = f"{pid}_{img}"
            keys[key] += 1
            langs_per_key[key].add(lg)
            lang_by_post[pid].add(lg)
            if not raw or not raw.strip():
                empty_raw += 1
            else:
                raw_len[min(len(raw) // 500, 20)] += 1
        if n % 500_000 == 0:
            print(f"  ...{n:,}", flush=True)

    print(f"\nscanned={n:,}")
    print(f"\n[desc_lang]  {dict(lang)}")
    print(f"  -> pct: { {k: f'{v/n:.1%}' for k, v in lang.items()} }")
    print(f"\n[source]     {dict(src.most_common(10))}")
    print(f"[dataset_id] {dict(dsid.most_common(10))}")
    print(f"[described]  {dict(desc_flag)}")
    print(f"[raw_output] empty={empty_raw:,}")

    dup = {k: c for k, c in keys.items() if c > 1}
    print(f"\n[key uniqueness] distinct '{{post_id}}_{{image_name}}' = {len(keys):,}, "
          f"duplicated keys = {len(dup):,}")
    if dup:
        ex = list(dup.items())[:5]
        print(f"  examples: {ex}")
    multi = Counter(len(v) for v in langs_per_key.values())
    print(f"[langs per image] {dict(multi)}   (1 => each image described in exactly one language)")

    post_multi = Counter(len(v) for v in lang_by_post.values())
    print(f"[langs per post]  {dict(post_multi)}  posts={len(lang_by_post):,}")

    print(f"\n[raw_output length] buckets of 500 chars "
          f"(bucket i = [{500}*i, 500*(i+1)) ):")
    for k in sorted(raw_len):
        label = f">={500*k}" if k == 20 else f"{500*k}-{500*(k+1)}"
        print(f"  {label:>12}: {raw_len[k]:,}")

    # ---- image coverage ----
    print("\n[image coverage] scanning local dirs ...", flush=True)
    for name, path in IMG_DIRS.items():
        if not os.path.isdir(path):
            print(f"  {name}: MISSING")
            continue
        have = {e.name for e in os.scandir(path) if e.is_file() and not e.name.endswith(".tmp")}
        hit = sum(1 for k in keys if k in have)
        print(f"  {name}: {len(have):,} files on disk; parquet keys present = "
              f"{hit:,}/{len(keys):,} ({hit/len(keys):.1%})", flush=True)
        IMG_DIRS[name] = have  # keep for union

    sets = [v for v in IMG_DIRS.values() if isinstance(v, set)]
    if len(sets) > 1:
        union = set().union(*sets)
        hit = sum(1 for k in keys if k in union)
        print(f"  UNION of dirs: parquet keys present = {hit:,}/{len(keys):,} "
              f"({hit/len(keys):.1%})  missing={len(keys)-hit:,}")


if __name__ == "__main__":
    sys.exit(main())
