#!/usr/bin/env python3
"""
dataset_builder_desc_of_0730.py
===============================
Build the LlamaFactory alpaca jsonl for the *onlyfans* description delivery (20260730).
Standalone dataset; merged with the av / oneione / pornpics halves only once all exist.

Source : /root/Kokoro2-EU-packed/caption/of_20260720_r18_insert.parquet   (771,809 rows)
Prompt : prompt/prompt_grokv10.txt  -- ONE instruction in THREE languages,
         '\\n\\n\\n'-separated, in the order en, ja, zh.
Images : /root/Kokoro2-EU-packed/onlyfans/{r2_key}     (verified 771,809/771,809 on disk)
Output : jsonl_of_0730/{train,test_unseen,test_unseen_mini}.jsonl

  Base table -- the desc parquet, NOT OF_embedding抽样_20260730/sample.tsv. Those are two
  independent selections and they barely overlap:

      desc parquet (wave 1, tier-selected 20260720) : 771,809, all tier=R18_insert
      embedding sample (SigLIP2-selected 20260730)  : 721,107, only 54,333 R18_insert
      intersection                                  :  48,318
      captioned but not in the embedding sample     : 723,491
      in the embedding sample but not captioned     : 672,789

  Gating this build on the embedding sample would throw away 94% of the annotations for
  a selection whose captions do not exist yet. The embedding sample is the wave-2 work
  list instead -- `--worklist` writes it (see jsonl_of_0730/manifest.tsv).

  coomer -- kept. The OF release note says coomer is a downsampled derivative layer
  (long edge capped 1536/800) and "不得混作主数据集同等质量". That warning is about image
  fidelity, which is what matters for a t2i generation set. This is a captioner: what
  matters is that the image is legible and that the text matches it, and these captions
  were generated from these same downsampled images, so text/image consistency holds.
  800px also sits above what the vision encoder actually consumes. Dropping coomer would
  cost 338,525 rows (44%). --ds makes it reversible without a rebuild, and so does the
  image path itself ('coomer_cut65/...' vs 'onecut65/...').

  Language -- en 79.4% / ja 10.0% / zh 9.9%, the mirror image of av's 80% ja. Each row
  carries exactly one language and the prompt block is the only signal telling the model
  which language to answer in, so pairing desc_lang with its own block is load-bearing.

  Runaway generations are real here in a way they were not for av or oneione: the
  longest ja row is 38,142 chars and the longest zh 34,187, against p99 of 1,751 and
  1,216. MAX_CHARS is doing actual work (60 rows), not just sitting there as a valve.

Filtering (~0.9% of rows):
  - described == False / desc_desc null      (6,017; never annotated)
  - any of the 4 sections empty              (820)
  - desc_lang not in {en, ja, zh}
  - len >= MAX_CHARS / repetition >= MAX_REP
  - image missing under IMG_ROOT              (0 today)

Split: whole *creators* held out (20,296 creators, median 11 images, max 18,232), not
posts. The delivery is ~1 image per post (median 1, max 39 over 666,979 posts), so a
post-level holdout is a row-level holdout -- it leaks nothing back but protects nothing
either. The real near-duplicate risk is one creator's shoot spread across many posts.

    python dataset_builder_desc_of_0730.py --plan
    python dataset_builder_desc_of_0730.py --build
    python dataset_builder_desc_of_0730.py --worklist   # wave-2 caption work list
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
PARQUET = "/root/Kokoro2-EU-packed/caption/of_20260720_r18_insert.parquet"
PROMPT_FILE = os.path.join(HERE, "prompt", "prompt_grokv10.txt")
IMG_ROOT = "/root/Kokoro2-EU-packed/onlyfans"
OUT_DIR = os.path.join(HERE, "jsonl_of_0730")
# Only read by --worklist, to report what wave 2 still has to caption.
SAMPLE_TSV = "/root/Kokoro2-EU-packed/caption/OF_embedding抽样_20260730/sample.tsv"
MANIFEST = os.path.join(OUT_DIR, "manifest.tsv")

LANGS = ("en", "ja", "zh")          # the order the blocks appear in prompt_grokv10.txt
TARGET = 900_000
SPLIT_SEED = 0
SAMPLE_SEED = 7
CREATOR_HOLDOUT_FRAC = 0.02
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
SECTION_FIELDS = ("intent", "foreground_subject", "background_env", "photo_technique")

DS_FILTERS = {
    "all": None,
    "onecut65": {"onecut65"},
    "no-lowres": {"onecut65", "coomer_highres"},
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
    return "\n\n".join(f"{fmt(h)}\n{s.strip()}" for h, s in zip(HEADERS[lang], sections))


def load_rows(args):
    """[(r2_key, creator, lang, output)] after content filtering."""
    if not os.path.exists(PARQUET):
        sys.exit(f"[fatal] {PARQUET} missing.")
    pf = pq.ParquetFile(PARQUET)
    names = set(pf.schema_arrow.names)
    need = {"r2_key", "ds", "creator", "desc_desc", "desc_lang", "described"}
    if not need <= names:
        sys.exit(f"[fatal] {PARQUET} is missing {sorted(need - names)}. Expected the "
                 f"desc_av shape (desc_desc struct + desc_lang + described) keyed by "
                 f"r2_key. Found: {sorted(names)}")
    struct = pf.schema_arrow.field("desc_desc").type
    have = {struct.field(i).name for i in range(struct.num_fields)}
    if not set(SECTION_FIELDS) <= have:
        sys.exit(f"[fatal] desc_desc lacks {sorted(set(SECTION_FIELDS) - have)}; has "
                 f"{sorted(have)}.")

    allowed = DS_FILTERS[args.ds]
    cols = ["r2_key", "ds", "creator", "desc_desc", "desc_lang", "described"]
    drop, kept, seen = Counter(), [], set()
    n = 0
    for b in pf.iter_batches(batch_size=50_000, columns=cols):
        d = b.to_pydict()
        n += len(d["r2_key"])
        for i, key in enumerate(d["r2_key"]):
            if key in seen:
                drop["duplicate_key"] += 1
                continue
            seen.add(key)
            if allowed is not None and d["ds"][i] not in allowed:
                drop[f"ds_excluded"] += 1
                continue
            lang, dd = d["desc_lang"][i], d["desc_desc"][i]
            if not d["described"][i] or dd is None:
                drop["not_described"] += 1
                continue
            if lang not in HEADERS:
                drop["bad_lang"] += 1
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
            kept.append((key, d["creator"][i], lang, out))
    return n, drop, kept


def write_worklist(captioned):
    """The wave-2 caption work list: embedding-sampled, on disk, not yet captioned."""
    if not os.path.exists(SAMPLE_TSV):
        sys.exit(f"[fatal] {SAMPLE_TSV} missing.")
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ("r2_key", "ds", "creator", "post_id", "media_id", "tier", "rating_dom", "cluster")
    todo = on_disk = total = 0
    with open(SAMPLE_TSV, encoding="utf-8") as f, \
         open(MANIFEST, "w", encoding="utf-8") as out:
        header = next(f).rstrip("\n").split("\t")
        missing = [c for c in cols if c not in header]
        if missing:
            sys.exit(f"[fatal] {SAMPLE_TSV} is missing columns {missing}.")
        idx = {c: header.index(c) for c in cols}
        out.write("\t".join(cols + ("local_path",)) + "\n")
        for line in f:
            p = line.rstrip("\n").split("\t")
            key = p[idx["r2_key"]]
            total += 1
            if key in captioned:
                continue
            todo += 1
            path = os.path.join(IMG_ROOT, key)
            if not os.path.exists(path):
                continue
            on_disk += 1
            out.write("\t".join([p[idx[c]] for c in cols] + [path]) + "\n")
    print(f"[worklist] embedding sample={total:,}  already captioned={total - todo:,}  "
          f"still to caption={todo:,}  of those on disk={on_disk:,}", flush=True)
    print(f"[worklist] {MANIFEST}: {on_disk:,} rows", flush=True)


def build(args):
    t0 = time.time()
    prompts = load_prompts()
    n_rows, drop, kept = load_rows(args)
    print(f"[plan] parquet rows={n_rows:,}  ({time.time() - t0:.0f}s)", flush=True)

    missing = [r for r in kept if not os.path.exists(os.path.join(IMG_ROOT, r[0]))]
    if missing:
        gone = {r[0] for r in missing}
        kept = [r for r in kept if r[0] not in gone]
        drop["no_image"] = len(missing)

    print(f"[plan] funnel: parquet={n_rows:,}"
          + "".join(f"  -{k}={v:,}" for k, v in drop.most_common())
          + f"  = usable {len(kept):,} ({len(kept) / n_rows:.2%} of delivery)", flush=True)
    if drop["no_image"]:
        print(f"[plan] NOTE {drop['no_image']:,} rows have no file under {IMG_ROOT}.", flush=True)
        for r in missing[:3]:
            print(f"[plan]      e.g. {r[0]}", flush=True)
    if not kept:
        sys.exit("[fatal] no usable rows.")

    if args.worklist:
        write_worklist({r[0] for r in kept})

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
    ds_n = Counter(r[0].split("/")[0] for r in kept)
    print("[plan] source mix:   "
          + "  ".join(f"{k}={v:,} ({v / len(kept):.1%})" for k, v in sorted(ds_n.items())),
          flush=True)
    for lg in LANGS:
        pool = [len(r[3]) for r in kept if r[2] == lg]
        if pool:
            print(f"[plan]   {lg}: avg output {sum(pool) / len(pool):,.0f} chars", flush=True)

    by_creator = defaultdict(list)
    for r in kept:
        by_creator[r[1]].append(r)
    creators = sorted(by_creator)
    random.seed(SPLIT_SEED)
    random.shuffle(creators)
    n_hold = int(round(len(creators) * CREATOR_HOLDOUT_FRAC))
    test = [r for c in creators[:n_hold] for r in by_creator[c]]
    train = [r for c in creators[n_hold:] for r in by_creator[c]]
    print(f"[plan] creators={len(creators):,}  held out={n_hold:,} "
          f"({CREATOR_HOLDOUT_FRAC:.0%})  -> train={len(train):,}  test_unseen={len(test):,}",
          flush=True)
    hold_mix = Counter(r[2] for r in test)
    print("[plan] test_unseen language mix: "
          + "  ".join(f"{lg}={hold_mix[lg]:,}" for lg in LANGS), flush=True)

    if args.plan:
        key, creator, lg, out = kept[0]
        print(f"[plan] sample -> {key}  creator={creator}  lang={lg}\n"
              f"[plan]   instruction[:80]={('<image>' + prompts[lg])[:80]!r}\n"
              f"[plan]   output[:160]={out[:160]!r}")
        print("[plan] --plan, nothing written.", flush=True)
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    def dump(name, rowset):
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for key, _creator, lang, out in rowset:
                f.write(json.dumps({"system": "",
                                    "instruction": "<image>" + prompts[lang],
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

    # Language-stratified: at 79/10/10 a flat mini would hand ja and zh too few rows to
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
    ap = argparse.ArgumentParser(description="Build the 20260730 onlyfans description dataset.")
    ap.add_argument("--plan", action="store_true", help="report funnel/coverage, write nothing")
    ap.add_argument("--build", action="store_true", help="write jsonl_of_0730/*.jsonl")
    ap.add_argument("--worklist", action="store_true",
                    help="also write jsonl_of_0730/manifest.tsv, the wave-2 caption work list "
                         "(embedding-sampled, on disk, not captioned by this delivery)")
    ap.add_argument("--ds", choices=tuple(DS_FILTERS), default="all",
                    help="source layers to keep. 'all' includes the coomer downsampled "
                         "derivative layer (44%% of rows); see the module docstring.")
    ap.add_argument("--markers", choices=("bold", "plain"), default="bold",
                    help="section marker style in the target text (default bold, '**X**')")
    ap.add_argument("--target", type=int, default=TARGET,
                    help=f"rows to keep before splitting (default {TARGET:,})")
    args = ap.parse_args()
    if not (args.plan or args.build or args.worklist):
        args.plan = True
    build(args)


if __name__ == "__main__":
    main()
