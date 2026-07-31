#!/usr/bin/env python3
"""
dataset_builder_desc_av_0730.py
===============================
Build the LlamaFactory alpaca jsonl for the *av* description delivery (20260730).
Standalone dataset -- jsonl/, jsonl_0716/ and jsonl_desc_0721/ are NOT touched.
This is the "av" half of the two-dataset delivery; the onlyfans half is built by
dataset_builder_desc_of_0730.py and the two are merged only once both exist.

Source : /root/Kokoro2-EU-packed/caption/desc_av.parquet     (900,008 rows)
Prompt : prompt/AV_prompt/{en,jp,zh}.md   -- three *task specs*, not three plain
         prompt blocks. Each spec carries a system prompt, a mustache user-prompt
         template and the section markers the annotator was told to emit.
Images : /root/Kokoro2-EU-packed/av/{object_key}
Output : jsonl_av_0730/{train,test_unseen,test_unseen_mini}.jsonl

Differences from the 0721 desc builder, all forced by the source:

  Prompt shape -- 0721 fed one flat instruction per language. The av annotator was
  called with a system prompt AND a user prompt carrying the work's metadata
  (title / tags_content / actress / tags_format). Those descriptions name the
  actress and lean on the tags, so dropping the metadata would train the model to
  invent names it cannot see. The split is preserved: system prompt -> the "system"
  column, rendered user prompt -> "instruction". dataset_info.json must therefore
  map "system" for these entries; a plain prompt/query/response/images mapping
  silently drops the whole task framing.

  IMAGE POSITION -- <image> goes at the HEAD of the instruction, even though the
  annotator's template puts it last, after an "image:" marker. The faithful order was
  tried first and it crashes training:

      RuntimeError: shape mismatch: value tensor of shape [3, 2618] cannot be
      broadcast to indexing result of shape [3, 2560]

  av is the only source that can exceed cutoff_len (it alone carries a separate system
  prompt), and it was the only source with <image> at the tail. When a row is over
  length, infer_seqlen() truncates the SOURCE from the right -- straight through the
  expanded image-pad block -- while image_grid_thw still claims the full count, so
  get_rope_index emits more positions than the truncated sequence has. Rare enough to
  clear a bs=2 smoke run over the first 480 rows and still kill a 34k-step run hours in
  (1 row in 2,400 here; ~0.04% of the delivery).

  With <image> at the head, truncation eats the metadata tail instead -- at worst a few
  studio tags -- and the image block is untouchable. The other three sources already
  build this way, so it also makes the merged file uniform.

  Language -- desc_lang is one of en/ja/zh per row and the file for ja is jp.md
  (LANG_SPEC below). The mix is ~80% ja / 10% zh / 10% en, and it is NOT a parallel
  corpus. en descriptions are also ~2.5x longer than ja and ~3.7x longer than zh
  (mean 2,494 / 969 / 682 chars), so a per-language eval mini is load-bearing.

  Output text -- the parquet stores the four sections already split into struct
  fields, with the "**...**" markers stripped. We re-emit them in the spec's own
  order and wording so the model learns the format the spec asks for. Marker text
  is read out of each md's "输出规范" section, so a re-worded spec fails loudly
  here instead of quietly training a format nobody asked for.

Filtering (~7.7% of rows):
  - described == False / desc_desc null      (63,910; never annotated)
  - any of the 4 sections empty              (5,163) -- 4,637 of these are annotator
    refusals ("I must decline this request.") parked in desc_desc.description with
    every section left null. They must not reach training.
  - desc_lang not in {en, ja, zh}
  - len >= MAX_CHARS / repetition >= MAX_REP (safety valves; ~4 rows today)
  - image missing under IMG_ROOT             (~1% of the delivery, see --plan)

Split: train + test_unseen, whole *products* held out (product_code, ~69k works,
median 11 images each) so no work straddles the boundary. Splitting on rows would
leak: images from one work share an actress, a set and often a background.

    python dataset_builder_desc_av_0730.py --plan     # funnel + coverage, writes nothing
    python dataset_builder_desc_av_0730.py --build    # write jsonl_av_0730/
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = "/root/Kokoro2-EU-packed/caption/desc_av.parquet"
PROMPT_DIR = os.path.join(HERE, "prompt", "AV_prompt")
IMG_ROOT = "/root/Kokoro2-EU-packed/av"
OUT_DIR = os.path.join(HERE, "jsonl_av_0730")

# desc_lang value -> spec filename. 'ja' in the data, 'jp' in the filename.
LANG_SPEC = {"en": "en.md", "ja": "jp.md", "zh": "zh.md"}
LANGS = ("en", "ja", "zh")

TARGET = 900_000                   # the 90w ask; capped at what survives filtering
SPLIT_SEED = 0
SAMPLE_SEED = 7
PRODUCT_HOLDOUT_FRAC = 0.02        # 0721's reasoning: 10% parks far more than any eval reads
MINI_PER_LANG = 400
MAX_CHARS = 6000                   # p99 of a healthy en row is ~4.0k; beyond this is runaway
MAX_REP = 0.3
REP_N = 40

# parquet struct field -> position in the spec's section list. The spec lists its
# markers in this order; the parquet names them differently.
SECTION_FIELDS = ("intent", "foreground_subject", "background_env", "photo_technique")

# The annotator's user-prompt template names the actress slot 'actres'; the parquet
# column is 'actress'. Mapping it wrong renders an empty performer line on 850k rows.
VAR_COLUMNS = {"title": "title", "tags_content": "tags_content",
               "actres": "actress", "tags_format": "tags_format"}

_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.S | re.M)
_MARKERS = re.compile(r"期望章节（节标记）\s*[:：]\s*(.+)")
_SECTION_BLOCK = re.compile(r"\{\{#vars\.(\w+)\}\}(.*?)\{\{/vars\.\1\}\}", re.S)
_VAR = re.compile(r"\{\{vars\.(\w+)\}\}")


def _fenced_after(text, heading, path):
    """The first ``` block after `heading`. The specs are generated files -- if the
    generator ever changes the heading numbering this must fail, not guess."""
    i = text.find(heading)
    if i < 0:
        sys.exit(f"[fatal] {path}: heading {heading!r} not found. Spec format changed?")
    m = _FENCE.search(text, i)
    if not m:
        sys.exit(f"[fatal] {path}: no fenced block after {heading!r}.")
    return m.group(1).rstrip("\n")


def load_specs():
    """{lang -> (system_prompt, user_template, (marker, marker, marker, marker))}."""
    specs = {}
    for lang, fname in LANG_SPEC.items():
        path = os.path.join(PROMPT_DIR, fname)
        if not os.path.exists(path):
            sys.exit(f"[fatal] {path} missing -- it defines the task framing for lang={lang}.")
        text = open(path, encoding="utf-8").read()
        system = _fenced_after(text, "## 3. System Prompt", path)
        template = _fenced_after(text, "## 4. User Prompt 模板", path)

        m = _MARKERS.search(text)
        if not m:
            sys.exit(f"[fatal] {path}: '期望章节（节标记）' line missing -- cannot know which "
                     f"section markers to re-emit.")
        markers = re.findall(r"`([^`]+)`", m.group(1))
        if len(markers) != 4:
            sys.exit(f"[fatal] {path}: expected 4 section markers, got {len(markers)}: {markers}")
        # The system prompt shows the same four markers in its example-output block. If the
        # two disagree the spec was edited in one place only, and we would be training a
        # format the model was never asked for.
        missing = [k for k in markers if k not in system]
        if missing:
            sys.exit(f"[fatal] {path}: markers {missing} appear in 输出规范 but not in the "
                     f"system prompt. Spec is internally inconsistent.")
        # Every var the template references must be one we can fill from the parquet.
        unknown = {v for v in _VAR.findall(template)} - set(VAR_COLUMNS)
        unknown |= {v for v, _ in _SECTION_BLOCK.findall(template)} - set(VAR_COLUMNS)
        if unknown:
            sys.exit(f"[fatal] {path}: user template references unknown vars {sorted(unknown)}; "
                     f"VAR_COLUMNS knows {sorted(VAR_COLUMNS)}.")
        specs[lang] = (system, template, tuple(markers))
        print(f"[spec] {lang:2s} <- {fname}  system={len(system)}ch  template={len(template)}ch  "
              f"markers={markers}", flush=True)
    return specs


def render_user(template, values):
    """Fill the mustache template. Conditional blocks render only when their var is
    non-empty; the blank lines they leave behind are kept, because that is exactly
    what the annotator saw (see the reproduce example in zh.md)."""
    def block(m):
        var, body = m.group(1), m.group(2)
        v = values.get(VAR_COLUMNS[var])
        return body if v and str(v).strip() else ""

    out = _SECTION_BLOCK.sub(block, template)
    out = _VAR.sub(lambda m: str(values.get(VAR_COLUMNS[m.group(1)]) or ""), out)
    return out


def repetition_score(s):
    """Fraction of the text covered by its most frequent non-overlapping REP_N-gram."""
    if len(s) < REP_N * 3:
        return 0.0
    grams = Counter(s[i:i + REP_N] for i in range(0, len(s) - REP_N, REP_N))
    return grams.most_common(1)[0][1] * REP_N / len(s)


def compose_output(markers, sections):
    return "\n\n".join(f"{mk}\n{sec.strip()}" for mk, sec in zip(markers, sections))


def load_rows(specs):
    """[(object_key, product_code, lang, user_prompt, output)] after content filtering.

    Image existence is checked by the caller; everything here is content-only so the
    funnel stays readable when the download is still catching up.
    """
    pf = pq.ParquetFile(PARQUET)
    cols = ["object_key", "product_code", "desc_lang", "described", "desc_desc",
            "title", "tags_content", "actress", "tags_format"]
    drop, kept = Counter(), []
    n = 0
    for b in pf.iter_batches(batch_size=50_000, columns=cols):
        d = b.to_pydict()
        n += len(d["object_key"])
        for i in range(len(d["object_key"])):
            lang, dd = d["desc_lang"][i], d["desc_desc"][i]
            if not d["described"][i] or dd is None:
                drop["not_described"] += 1
                continue
            if lang not in LANG_SPEC:
                drop["bad_lang"] += 1
                continue
            sections = [dd.get(f) or "" for f in SECTION_FIELDS]
            if not all(s.strip() for s in sections):
                # 4,637 of these are refusals parked in dd["description"]; the rest are
                # truncations. Neither is a usable target.
                drop["refusal_or_partial"] += 1
                continue
            system, template, markers = specs[lang]
            out = compose_output(markers, sections)
            if len(out) >= MAX_CHARS:
                drop["too_long"] += 1
                continue
            if repetition_score(out) >= MAX_REP:
                drop["repetition"] += 1
                continue
            values = {c: d[c][i] for c in ("title", "tags_content", "actress", "tags_format")}
            kept.append((d["object_key"][i], d["product_code"][i], lang,
                         render_user(template, values), out))
    return n, drop, kept


def build(args):
    t0 = time.time()
    specs = load_specs()
    n_rows, drop, kept = load_rows(specs)
    print(f"[plan] parquet rows={n_rows:,}  ({time.time() - t0:.0f}s)", flush=True)

    missing = [r for r in kept if not os.path.exists(os.path.join(IMG_ROOT, r[0]))]
    if missing:
        have = {r[0] for r in kept} - {r[0] for r in missing}
        kept = [r for r in kept if r[0] in have]
        drop["no_image"] = len(missing)

    print(f"[plan] funnel: parquet={n_rows:,}"
          + "".join(f"  -{k}={v:,}" for k, v in drop.most_common())
          + f"  = usable {len(kept):,} ({len(kept) / n_rows:.2%} of delivery)", flush=True)
    if drop["no_image"]:
        print(f"[plan] NOTE {drop['no_image']:,} rows have no file under {IMG_ROOT}. The av "
              f"mirror is still filling; re-run when the download job reports complete.",
              flush=True)
        for k, _, _, _, _ in missing[:3]:
            print(f"[plan]      e.g. {k}", flush=True)
    if not kept:
        sys.exit("[fatal] no usable rows.")

    if len(kept) > args.target:
        random.seed(SAMPLE_SEED)
        kept = random.sample(kept, args.target)
        print(f"[plan] sampled down to the {args.target:,} ask", flush=True)
    elif len(kept) < args.target:
        print(f"[plan] SHORT of the {args.target:,} ask by {args.target - len(kept):,} rows "
              f"-- taking all {len(kept):,}. The delivery does not contain more.", flush=True)

    lang_n = Counter(r[2] for r in kept)
    print("[plan] language mix: "
          + "  ".join(f"{lg}={lang_n[lg]:,} ({lang_n[lg] / len(kept):.1%})" for lg in LANGS),
          flush=True)
    for lg in LANGS:
        pool = [len(r[4]) for r in kept if r[2] == lg]
        if pool:
            print(f"[plan]   {lg}: avg output {sum(pool) / len(pool):,.0f} chars", flush=True)

    by_product = defaultdict(list)
    for r in kept:
        by_product[r[1]].append(r)
    products = sorted(by_product)
    random.seed(SPLIT_SEED)
    random.shuffle(products)
    n_hold = int(round(len(products) * PRODUCT_HOLDOUT_FRAC))
    test = [r for p in products[:n_hold] for r in by_product[p]]
    train = [r for p in products[n_hold:] for r in by_product[p]]
    print(f"[plan] products={len(products):,}  held out={n_hold:,} "
          f"({PRODUCT_HOLDOUT_FRAC:.0%})  -> train={len(train):,}  test_unseen={len(test):,}",
          flush=True)

    if args.plan:
        key, pc, lg, user, out = kept[0]
        print(f"[plan] sample -> {key}  product={pc}  lang={lg}")
        print(f"[plan]   system[:80]={specs[lg][0][:80]!r}")
        print(f"[plan]   instruction={user!r}")
        print(f"[plan]   output[:160]={out[:160]!r}")
        print("[plan] --plan, nothing written.", flush=True)
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    def dump(name, rowset):
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for key, _pc, lang, user, out in rowset:
                f.write(json.dumps({"system": specs[lang][0],
                                    # <image> at the HEAD, not at the template's own
                                    # "image:" marker -- see IMAGE POSITION in the docstring.
                                    "instruction": "<image>" + user.rstrip("\n"),
                                    "input": "",
                                    "output": out,
                                    "images": [os.path.join(IMG_ROOT, key)]},
                                   ensure_ascii=False) + "\n")
        mix = Counter(r[2] for r in rowset)
        print(f"[build] {name}.jsonl: {len(rowset):,} rows  ("
              + " ".join(f"{lg}={mix[lg]:,}" for lg in LANGS) + ")", flush=True)

    for name, rowset in (("train", train), ("test_unseen", test)):
        random.seed(1234)
        random.shuffle(rowset)
        dump(name, rowset)

    # Language-stratified: at 80/10/10 a flat mini would hand en and zh too few rows to
    # separate a per-language regression from noise.
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
    ap = argparse.ArgumentParser(description="Build the 20260730 av description dataset.")
    ap.add_argument("--plan", action="store_true", help="report funnel/coverage, write nothing")
    ap.add_argument("--build", action="store_true", help="write jsonl_av_0730/")
    ap.add_argument("--target", type=int, default=TARGET,
                    help=f"rows to keep before splitting (default {TARGET:,})")
    args = ap.parse_args()
    if not (args.plan or args.build):
        args.plan = True
    build(args)


if __name__ == "__main__":
    main()
