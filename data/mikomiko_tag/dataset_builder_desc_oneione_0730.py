#!/usr/bin/env python3
"""
dataset_builder_desc_oneione_0730.py
====================================
Build the LlamaFactory alpaca jsonl for the *oneione* description delivery (20260730).
Standalone dataset; merged with the av / onlyfans / pornpics halves only once all exist.

Source : /root/Kokoro2-EU-packed/caption/oneione_t2i_unified_v1_20260730_quick/
         oneione_t2i_unified_v1.parquet   (147,265 rows, one row per image)
Prompt : prompt/prompt_grokv10.txt  -- ONE instruction in THREE languages,
         '\\n\\n\\n'-separated, in the order en, ja, zh.
Images : /root/Kokoro2-EU-packed + urlparse(url).path   (the URL path already starts
         with /oneione/, so the CDN layout and the local mirror are the same tree)
Output : jsonl_oneione_0730/{train,test_unseen,test_unseen_mini}.jsonl

Differences from the av builder, all forced by the source:

  Schema -- this delivery does NOT use the desc_av shape. There is no desc_desc struct,
  no desc_lang and no described flag. The four sections are four TOP-LEVEL
  struct<text,lang> columns, each carrying its OWN language tag. Row language is
  therefore derived, not read: we require all four tags to agree. 10,876 rows disagree
  or are blank (see below), which is exactly the population that would otherwise get a
  wrong-language instruction.

  Prompt -- grokv10 is a flat instruction with no metadata slots, so unlike av there is
  no system/user split and no title/tags injected. The row's extra.* metadata
  (character_name, series_name, reading article context, graphic author) is NOT fed to
  the model: the annotator did not see it either -- the reading captions describe the
  image and ignore the article text they are filed under. Rows still carry an empty
  "system" key so this file and jsonl_av_0730/ share one column mapping when merged.

  Markers -- grokv10 prints its four sections slash-joined with no asterisks, while
  grokv8 and the av specs ask for '**Creative Intent**' on its own line. The parquet
  stores the sections already split with the markers stripped, so the annotator's actual
  rendering is not recoverable from this file. Default is bold, matching av, grokv8 and
  the pornpics delivery -- a merged training set must not teach two formats keyed only
  by which source a row came from. --markers plain if that ever turns out wrong.

Filtering (7.6% of rows):
  - a section is blank                    (10,876; 10,865 of them are the
    foreground_subject paragraph alone, 9,195 of those Japanese. The other three
    sections on those rows are complete and normal, so this is a pipeline loss, not a
    refusal -- but a caption missing its main-subject paragraph is not a usable target,
    and keeping it would teach the model to skip that section. It costs 25% of the
    Japanese rows; --keep-partial exists to revisit that without editing the script.)
  - the four lang tags disagree, or are not one of en/ja/zh
  - len >= MAX_CHARS / repetition >= MAX_REP (safety valves; 0 rows today)
  - image missing under IMG_ROOT           (0 today -- the mirror is complete)

Split: train + test_unseen, whole posts held out (5,587 posts, median 15 images, max
249). HOLDOUT_FRAC is 0.03, not av's 0.02: at 136k usable rows 2% leaves ~2.7k in
test_unseen, and after the 53/27/20 zh/en/ja split the Japanese slice would sit right on
top of MINI_PER_LANG with nothing to spare once lumpy post sizes are accounted for.

    python dataset_builder_desc_oneione_0730.py --plan
    python dataset_builder_desc_oneione_0730.py --build
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from urllib.parse import urlparse

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = ("/root/Kokoro2-EU-packed/caption/oneione_t2i_unified_v1_20260730_quick/"
           "oneione_t2i_unified_v1.parquet")
PROMPT_FILE = os.path.join(HERE, "prompt", "prompt_grokv10.txt")
IMG_ROOT = "/root/Kokoro2-EU-packed"
URL_PREFIX = "/oneione/"
OUT_DIR = os.path.join(HERE, "jsonl_oneione_0730")

LANGS = ("en", "ja", "zh")          # the order the blocks appear in prompt_grokv10.txt
SPLIT_SEED = 0
SAMPLE_SEED = 7
POST_HOLDOUT_FRAC = 0.03
MINI_PER_LANG = 400
MAX_CHARS = 6000
MAX_REP = 0.3
REP_N = 40

# The 4 sections each language's block asks for, in order. grokv10 prints them
# slash-joined on its last line, which is what load_prompts() reads.
HEADERS = {
    "en": ("Creative Intent", "Foreground and Subject",
           "Background and Environment", "Photography Techniques and Visual Presentation"),
    "ja": ("創作意図", "前景と主要な被写体", "背景と周囲の環境", "撮影技法と視覚的表現"),
    "zh": ("创作意图", "前景与主体", "背景与环境", "摄影技术与视觉呈现"),
}
# The parquet column for each section, in the same order as HEADERS.
SECTION_COLUMNS = ("intent", "foreground_subject", "background_env", "photo_technique")


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
    # If the file is ever reordered or re-translated, every row gets the wrong-language
    # instruction and the only symptom is a model that answers in the wrong language.
    for lg, text in prompts.items():
        missing = [h for h in HEADERS[lg] if h not in text]
        if missing:
            sys.exit(f"[fatal] prompt block #{LANGS.index(lg) + 1} was taken as '{lg}' but "
                     f"does not contain its headers {missing}. Block order in "
                     f"{PROMPT_FILE} changed?")
    print("[prompt] 3 blocks OK -- "
          + ", ".join(f"{lg}={len(prompts[lg])}ch" for lg in LANGS), flush=True)
    return prompts


def repetition_score(s):
    """Fraction of the text covered by its most frequent non-overlapping REP_N-gram."""
    if len(s) < REP_N * 3:
        return 0.0
    grams = Counter(s[i:i + REP_N] for i in range(0, len(s) - REP_N, REP_N))
    return grams.most_common(1)[0][1] * REP_N / len(s)


def compose_output(lang, sections, markers):
    fmt = (lambda h: f"**{h}**") if markers == "bold" else (lambda h: h)
    return "\n\n".join(f"{fmt(h)}\n{s.strip()}"
                       for h, s in zip(HEADERS[lang], sections) if s.strip())


def local_path(url):
    """Local mirror path for a delivery URL, or None if it is not shaped as expected."""
    path = urlparse(url).path
    if not path.startswith(URL_PREFIX):
        return None
    return IMG_ROOT + path


def load_rows(prompts, keep_partial):
    """[(post_id, type, lang, output, image_path)] after content filtering."""
    if not os.path.exists(PARQUET):
        sys.exit(f"[fatal] {PARQUET} missing.")
    pf = pq.ParquetFile(PARQUET)
    names = set(pf.schema_arrow.names)
    need = {"post_id", "image_index", "url", "extra", *SECTION_COLUMNS}
    if not need <= names:
        sys.exit(f"[fatal] {PARQUET} is missing {sorted(need - names)}. This builder expects "
                 f"the t2i.unified.v1 profile (four top-level struct<text,lang> sections). "
                 f"Found: {sorted(names)}")

    cols = ["post_id", "image_index", "url", "extra", *SECTION_COLUMNS]
    drop, kept, seen = Counter(), [], set()
    n = 0
    for b in pf.iter_batches(batch_size=20_000, columns=cols):
        d = b.to_pydict()
        n += len(d["url"])
        for i in range(len(d["url"])):
            key = (d["post_id"][i], d["image_index"][i])
            if key in seen:
                drop["duplicate_key"] += 1
                continue
            seen.add(key)

            vals = [d[c][i] or {} for c in SECTION_COLUMNS]
            texts = [(v.get("text") or "").strip() for v in vals]
            tags = [(v.get("lang") or "") for v in vals]

            filled = [t for t in texts if t]
            if not filled:
                drop["all_sections_blank"] += 1
                continue
            if len(filled) < 4 and not keep_partial:
                drop["blank_section"] += 1
                continue

            # Language is derived from the section tags, not read from a row column.
            # Only tags on non-blank sections can vote; a blank section's tag is "".
            voted = {tg for tg, t in zip(tags, texts) if t}
            if len(voted) != 1:
                drop["lang_disagreement"] += 1
                continue
            lang = voted.pop()
            if lang not in HEADERS:
                drop["bad_lang"] += 1
                continue

            path = local_path(d["url"][i])
            if path is None:
                drop["bad_url"] += 1
                continue

            out = compose_output(lang, texts, ARGS.markers)
            if len(out) >= MAX_CHARS:
                drop["too_long"] += 1
                continue
            if repetition_score(out) >= MAX_REP:
                drop["repetition"] += 1
                continue
            kept.append((d["post_id"][i], d["extra"][i]["type"], lang, out, path))
    return n, drop, kept


def build(args):
    t0 = time.time()
    prompts = load_prompts()
    n_rows, drop, kept = load_rows(prompts, args.keep_partial)
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
    elif args.target and len(kept) < args.target:
        print(f"[plan] SHORT of the {args.target:,} ask by {args.target - len(kept):,} rows "
              f"-- taking all {len(kept):,}.", flush=True)

    lang_n = Counter(r[2] for r in kept)
    print("[plan] language mix: "
          + "  ".join(f"{lg}={lang_n[lg]:,} ({lang_n[lg] / len(kept):.1%})" for lg in LANGS),
          flush=True)
    type_n = Counter(r[1] for r in kept)
    print("[plan] source mix:   "
          + "  ".join(f"{t}={type_n[t]:,} ({type_n[t] / len(kept):.1%})"
                      for t in sorted(type_n)), flush=True)
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
    hold_mix = Counter(r[2] for r in test)
    print("[plan] test_unseen language mix: "
          + "  ".join(f"{lg}={hold_mix[lg]:,}" for lg in LANGS), flush=True)

    if args.plan:
        pid, typ, lg, out, path = kept[0]
        print(f"[plan] sample -> post={pid} type={typ} lang={lg}\n"
              f"[plan]   instruction[:80]={('<image>' + prompts[lg])[:80]!r}\n"
              f"[plan]   output[:160]={out[:160]!r}\n"
              f"[plan]   image={path}")
        print("[plan] --plan, nothing written.", flush=True)
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    def dump(name, rowset):
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for _pid, _typ, lang, out, img in rowset:
                f.write(json.dumps({"system": "",
                                    "instruction": "<image>" + prompts[lang],
                                    "input": "",
                                    "output": out,
                                    "images": [img]},
                                   ensure_ascii=False) + "\n")
        mix = Counter(r[2] for r in rowset)
        print(f"[build] {name}.jsonl: {len(rowset):,} rows  ("
              + " ".join(f"{lg}={mix[lg]:,}" for lg in LANGS) + ")", flush=True)

    for name, rowset in (("train", train), ("test_unseen", test)):
        random.seed(1234)
        random.shuffle(rowset)
        dump(name, rowset)

    # Language-stratified: a flat mini would hand ja ~20% of the rows and its per-language
    # regression signal would sit inside the noise.
    random.seed(42)
    by_lang = defaultdict(list)
    for r in test:
        by_lang[r[2]].append(r)
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
    global ARGS
    ap = argparse.ArgumentParser(description="Build the 20260730 oneione description dataset.")
    ap.add_argument("--plan", action="store_true", help="report funnel/coverage, write nothing")
    ap.add_argument("--build", action="store_true", help="write jsonl_oneione_0730/")
    ap.add_argument("--target", type=int, default=0,
                    help="cap rows before splitting (default 0 = keep everything usable)")
    ap.add_argument("--markers", choices=("bold", "plain"), default="bold",
                    help="section marker style in the target text (default bold, '**X**')")
    ap.add_argument("--keep-partial", action="store_true",
                    help="keep rows with a blank section instead of dropping them "
                         "(emits a 3-section target; off by default)")
    ARGS = ap.parse_args()
    if not (ARGS.plan or ARGS.build):
        ARGS.plan = True
    build(ARGS)


if __name__ == "__main__":
    main()
