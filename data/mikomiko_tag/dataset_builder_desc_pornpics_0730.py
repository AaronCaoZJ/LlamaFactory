#!/usr/bin/env python3
"""
dataset_builder_desc_pornpics_0730.py
=====================================
Build the LlamaFactory alpaca jsonl for the *pornpics* description delivery (20260730).
Standalone dataset; merged with the av / onlyfans / oneione halves only once all exist.

Source : /root/Kokoro2-EU-packed/caption/
         170w-pornpics_curated_full_watermark_flagged.parquet   (1,706,297 rows)
Prompt : TWO versions, picked per row -- see below.
Images : /root/Kokoro2-EU-packed/pornpics/{post_id}/{image_index}.webp
Output : jsonl_pornpics_0730/{train,test_unseen,test_unseen_mini}.jsonl

  Prompt version is per row, not per delivery. This is the only source so far annotated
  under two different prompts, and desc_desc.custom_id carries the version token:

      pornpics_split1_20260720__of_10535964_..._aba1.jpg__desc__v0.2.0__lang=en
                                                                ^^^^^^
      v0.2.0 -> prompt_grokv8.txt    1,213,690 rows (71.1%)
      v0.4.0 -> prompt_grokv10.txt     473,670 rows (27.8%)
      v0.3.0 -> dropped                 18,937 rows (1.1%)

  The mapping was established from the annotations themselves, not assumed. grokv10 asks
  for an age range and forbids absence statements ("no tattoos"); grokv8 does neither.
  Measured over raw_output:

      ver      lang       rows   mentions age range   writes absence statements
      v0.2.0   en      970,737                 0.0%                        6.7%
      v0.2.0   ja      121,540                 0.0%                       33.4%
      v0.2.0   zh      121,413                 0.0%                       27.5%
      v0.4.0   en      378,737                12.2%                        3.7%
      v0.4.0   zh       47,333                31.1%                        9.9%

  Zero age-range mentions across 1.2M v0.2.0 rows in all three languages is not a style
  difference, it is a prompt that never asked. Feeding those rows the grokv10 block would
  train the model to ignore an instruction it is being given -- the same failure mode as
  pairing a row with the wrong language block, and just as invisible in the loss.

  v0.3.0 is an intermediate revision matching neither file (its zh rows score high on
  BOTH age range and absence statements), so there is no prompt to pair it with. At 1.1%
  it is not worth reverse-engineering; VERSION_PROMPT below is the whole policy.

Filtering (~1.2% of rows):
  - custom_id version not in VERSION_PROMPT   (18,937 -- v0.3.0)
  - any of the 4 sections empty               (1,213 across the kept versions)
  - desc_lang not in {en, ja, zh}
  - len >= MAX_CHARS / repetition >= MAX_REP  (29 rows over 6000 chars)
  - image missing under IMG_ROOT

  LOCATOR_TAIL -- the local tree is flat, `pornpics/{post_id}/{image_index}.webp`, but the
  delivery mixes two crawls whose URLs do NOT share a prefix:

      pornpics_split1_20260720   1,059,932   .../cut/10535964/6.webp
      pornpics_tag_hits_20260716   615,905   pornstar-data.mikolab.bid/32483444/15.webp
                                    24,762   ...lesbian-shemale.mikolab.bid/shemale/32486570/11.webp
                                     4,673   .../lesbian/<post_id>/<index>.webp
                                       880   .../bondage/...      145   .../bukkake/...

  pornpics_url_locator.parquet certifies `url:/cut/<post_id>/<image_index>.webp` for all
  27,039,519 rows it covers, but it only covers the /cut/ crawl. Matching that rule
  literally silently drops the entire tag_hits batch -- 646,365 rows, 38% of the delivery
  and over half of everything annotated under grokv8 -- even though 618,958 of them (95.8%)
  are sitting on disk under the same flat layout. What is actually invariant across both
  crawls is the path TAIL: every one of the 1,706,297 URLs ends in
  `/{post_id}/{image_index}.webp`. That is what this builder checks; the leading segment
  (/cut/, nothing, /shemale/, /lesbian/, /bondage/, /bukkake/) is a CDN routing artefact
  and is not part of the local path. On-disk existence is then the real gate.

  Never address these files by image_name: in the split1 batch it is the ORIGINAL source
  filename ('10535964_007_aba1.jpg'), not the delivered basename ('6.webp').

Split: whole posts held out (701,132 galleries, median 2 images, max 20). Images inside
one pornpics gallery are the same model in the same set, minutes apart.

    python dataset_builder_desc_pornpics_0730.py --plan
    python dataset_builder_desc_pornpics_0730.py --build
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from urllib.parse import urlparse

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = ("/root/Kokoro2-EU-packed/caption/"
           "170w-pornpics_curated_full_watermark_flagged.parquet")
PROMPT_DIR = os.path.join(HERE, "prompt")
IMG_ROOT = "/root/Kokoro2-EU-packed/pornpics"
OUT_DIR = os.path.join(HERE, "jsonl_pornpics_0730")

# The whole version policy. A version absent from this dict is dropped by name, so a
# future delivery that introduces v0.5.0 fails loudly in the funnel instead of silently
# inheriting whichever prompt happened to be last.
VERSION_PROMPT = {
    "v0.2.0": "prompt_grokv8.txt",
    "v0.4.0": "prompt_grokv10.txt",
}

LANGS = ("en", "ja", "zh")          # the order the blocks appear in both prompt files
SPLIT_SEED = 0
SAMPLE_SEED = 7
POST_HOLDOUT_FRAC = 0.02
MINI_PER_LANG = 400
MAX_CHARS = 6000
MAX_REP = 0.3
REP_N = 40

# Both prompt versions ask for the same four sections under the same names; only the
# instruction bodies differ. load_prompts() asserts that per file.
HEADERS = {
    "en": ("Creative Intent", "Foreground and Subject",
           "Background and Environment", "Photography Techniques and Visual Presentation"),
    "ja": ("創作意図", "前景と主要な被写体", "背景と周囲の環境", "撮影技法と視覚的表現"),
    "zh": ("创作意图", "前景与主体", "背景与环境", "摄影技术与视觉呈现"),
}
SECTION_FIELDS = ("intent", "foreground_subject", "background_env", "photo_technique")

_VERSION = re.compile(r"__(v[\d.]+)__")


def load_prompts():
    """{version -> {lang -> instruction}}, asserting each file is still en, ja, zh."""
    out = {}
    for version, fname in VERSION_PROMPT.items():
        path = os.path.join(PROMPT_DIR, fname)
        if not os.path.exists(path):
            sys.exit(f"[fatal] {path} missing -- it is the task framing for {version}.")
        blocks = [b.strip() for b in open(path, encoding="utf-8").read().split("\n\n\n")]
        blocks = [b for b in blocks if b]
        if len(blocks) != 3:
            sys.exit(f"[fatal] {path} split into {len(blocks)} blocks, expected 3 "
                     f"(en/ja/zh separated by a blank line pair).")
        prompts = dict(zip(LANGS, blocks))
        for lg, text in prompts.items():
            missing = [h for h in HEADERS[lg] if h not in text]
            if missing:
                sys.exit(f"[fatal] {fname} block #{LANGS.index(lg) + 1} was taken as '{lg}' "
                         f"but does not contain its headers {missing}. Block order changed?")
        out[version] = prompts
        print(f"[prompt] {version} <- {fname}  "
              + ", ".join(f"{lg}={len(prompts[lg])}ch" for lg in LANGS), flush=True)
    return out


def repetition_score(s):
    """Fraction of the text covered by its most frequent non-overlapping REP_N-gram."""
    if len(s) < REP_N * 3:
        return 0.0
    grams = Counter(s[i:i + REP_N] for i in range(0, len(s) - REP_N, REP_N))
    return grams.most_common(1)[0][1] * REP_N / len(s)


def compose_output(lang, sections, markers):
    fmt = (lambda h: f"**{h}**") if markers == "bold" else (lambda h: h)
    return "\n\n".join(f"{fmt(h)}\n{s.strip()}" for h, s in zip(HEADERS[lang], sections))


def load_rows(args):
    """[(post_id, version, lang, output, image_path)] after content filtering."""
    if not os.path.exists(PARQUET):
        sys.exit(f"[fatal] {PARQUET} missing.")
    pf = pq.ParquetFile(PARQUET)
    names = set(pf.schema_arrow.names)
    need = {"post_id", "image_index", "url", "desc_desc", "desc_lang", "described"}
    if not need <= names:
        sys.exit(f"[fatal] {PARQUET} is missing {sorted(need - names)}. Found: "
                 f"{sorted(names)}")

    cols = ["post_id", "image_index", "url", "desc_desc", "desc_lang", "described"]
    drop, kept, seen = Counter(), [], set()
    n = 0
    for b in pf.iter_batches(batch_size=50_000, columns=cols):
        d = b.to_pydict()
        n += len(d["post_id"])
        for i in range(len(d["post_id"])):
            post_id, idx = d["post_id"][i], d["image_index"][i]
            key = (post_id, idx)
            if key in seen:
                drop["duplicate_key"] += 1
                continue
            seen.add(key)

            dd = d["desc_desc"][i]
            if not d["described"][i] or dd is None:
                drop["not_described"] += 1
                continue

            m = _VERSION.search(dd.get("custom_id") or "")
            version = m.group(1) if m else None
            if version not in VERSION_PROMPT:
                drop[f"version_{version or 'unparsed'}"] += 1
                continue

            lang = d["desc_lang"][i]
            if lang not in HEADERS:
                drop["bad_lang"] += 1
                continue

            # Address the file by the locator tail, never by image_name: in the split1
            # batch image_name is the ORIGINAL source filename ('10535964_007_aba1.jpg'),
            # not the delivered basename, so joining on it silently misses.
            # The leading segment is not part of the local layout and varies by batch --
            # see LOCATOR_TAIL in the module docstring.
            path_part = urlparse(d["url"][i]).path
            if not path_part.endswith(f"/{post_id}/{idx}.webp"):
                drop["url_off_formula"] += 1
                continue

            sections = [dd.get(f) or "" for f in SECTION_FIELDS]
            if not all(s.strip() for s in sections):
                drop["partial_sections"] += 1
                continue
            out = compose_output(lang, sections, args.markers)
            if len(out) >= MAX_CHARS:
                drop["too_long"] += 1
                continue
            if repetition_score(out) >= MAX_REP:
                drop["repetition"] += 1
                continue
            kept.append((post_id, version, lang, out,
                         os.path.join(IMG_ROOT, post_id, f"{idx}.webp")))
    return n, drop, kept


def build(args):
    t0 = time.time()
    prompts = load_prompts()
    n_rows, drop, kept = load_rows(args)
    print(f"[plan] parquet rows={n_rows:,}  ({time.time() - t0:.0f}s)", flush=True)

    missing = [r for r in kept if not os.path.exists(r[4])]
    if missing:
        gone = {r[4] for r in missing}
        kept = [r for r in kept if r[4] not in gone]
        drop["no_image"] = len(missing)

    print(f"[plan] funnel: parquet={n_rows:,}"
          + "".join(f"  -{k}={v:,}" for k, v in drop.most_common())
          + f"  = usable {len(kept):,} ({len(kept) / n_rows:.2%} of delivery)", flush=True)
    if drop["no_image"]:
        print(f"[plan] NOTE {drop['no_image']:,} rows have no file under {IMG_ROOT}.", flush=True)
        for r in missing[:3]:
            print(f"[plan]      e.g. {r[4]}", flush=True)
    if not kept:
        sys.exit("[fatal] no usable rows.")

    if args.target and len(kept) > args.target:
        random.seed(SAMPLE_SEED)
        kept = random.sample(kept, args.target)
        print(f"[plan] sampled down to the {args.target:,} ask", flush=True)

    ver_n = Counter(r[1] for r in kept)
    print("[plan] prompt version mix: "
          + "  ".join(f"{v}({VERSION_PROMPT[v]})={ver_n[v]:,} ({ver_n[v] / len(kept):.1%})"
                      for v in sorted(ver_n)), flush=True)
    lang_n = Counter(r[2] for r in kept)
    print("[plan] language mix: "
          + "  ".join(f"{lg}={lang_n[lg]:,} ({lang_n[lg] / len(kept):.1%})" for lg in LANGS),
          flush=True)
    for lg in LANGS:
        pool = [len(r[3]) for r in kept if r[2] == lg]
        if pool:
            print(f"[plan]   {lg}: avg output {sum(pool) / len(pool):,.0f} chars", flush=True)

    by_post = defaultdict(list)
    for r in kept:
        by_post[r[0]].append(r)
    posts = sorted(by_post)
    random.seed(SPLIT_SEED)
    random.shuffle(posts)
    n_hold = int(round(len(posts) * POST_HOLDOUT_FRAC))
    test = [r for p in posts[:n_hold] for r in by_post[p]]
    train = [r for p in posts[n_hold:] for r in by_post[p]]
    print(f"[plan] posts={len(posts):,}  held out={n_hold:,} ({POST_HOLDOUT_FRAC:.0%})"
          f"  -> train={len(train):,}  test_unseen={len(test):,}", flush=True)
    print("[plan] test_unseen language mix: "
          + "  ".join(f"{lg}={Counter(r[2] for r in test)[lg]:,}" for lg in LANGS), flush=True)

    if args.plan:
        post_id, version, lg, out, path = kept[0]
        print(f"[plan] sample -> post={post_id} version={version} lang={lg}\n"
              f"[plan]   instruction[:80]={('<image>' + prompts[version][lg])[:80]!r}\n"
              f"[plan]   output[:140]={out[:140]!r}\n"
              f"[plan]   image={path}")
        print("[plan] --plan, nothing written.", flush=True)
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    def dump(name, rowset):
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for _post, version, lang, out, img in rowset:
                f.write(json.dumps({"system": "",
                                    "instruction": "<image>" + prompts[version][lang],
                                    "input": "",
                                    "output": out,
                                    "images": [img]},
                                   ensure_ascii=False) + "\n")
        mix = Counter(r[2] for r in rowset)
        ver = Counter(r[1] for r in rowset)
        print(f"[build] {name}.jsonl: {len(rowset):,} rows  ("
              + " ".join(f"{lg}={mix[lg]:,}" for lg in LANGS) + " | "
              + " ".join(f"{v}={ver[v]:,}" for v in sorted(ver)) + ")", flush=True)

    for name, rowset in (("train", train), ("test_unseen", test)):
        random.seed(1234)
        random.shuffle(rowset)
        dump(name, rowset)

    # Stratified by language AND prompt version: a flat mini would be ~79% en and ~71%
    # grokv8, and a regression on the smaller cell would not be separable from noise.
    random.seed(42)
    by_cell = defaultdict(list)
    for r in test:
        by_cell[(r[2], r[1])].append(r)
    per_cell = max(1, MINI_PER_LANG // len(VERSION_PROMPT))
    mini = []
    for lg in LANGS:
        for version in sorted(VERSION_PROMPT):
            pool = by_cell[(lg, version)]
            take = min(per_cell, len(pool))
            if take < per_cell:
                print(f"[build] WARN mini: only {take} {lg}/{version} rows in test_unseen "
                      f"(wanted {per_cell}); its CI will be wider.", flush=True)
            mini.extend(random.sample(pool, take))
    random.shuffle(mini)
    dump("test_unseen_mini", mini)

    print(f"[build] DONE -> {OUT_DIR}  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build the 20260730 pornpics description dataset.")
    ap.add_argument("--plan", action="store_true", help="report funnel/coverage, write nothing")
    ap.add_argument("--build", action="store_true", help="write jsonl_pornpics_0730/")
    ap.add_argument("--target", type=int, default=0,
                    help="cap rows before splitting (default 0 = keep everything usable)")
    ap.add_argument("--markers", choices=("bold", "plain"), default="bold",
                    help="section marker style in the target text (default bold, '**X**')")
    args = ap.parse_args()
    if not (args.plan or args.build):
        args.plan = True
    build(args)


if __name__ == "__main__":
    main()
