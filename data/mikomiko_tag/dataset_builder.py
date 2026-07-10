#!/usr/bin/env python3
"""
mikomiko_dataset_builder.py
===========================
Build a multimodal image-tag-recognition SFT dataset for LlamaFactory from
`pornstars620_100k-sample-posts_category-tags.csv`.

Task shape: given an image, the model outputs a comma-separated list of tags. Emitted in
LlamaFactory **alpaca** format (instruction/input/output/images).

Two source CSVs joined on image key `{post_id}_{image_name}` (== GEMINI custom_id):
  - GEMINI_CSV  -- per-image tags Gemini judged actually visible  -> the LABEL
  - CATALOG_CSV -- post catalog; `url` gives the webp download link -> the IMAGE
  (CATALOG's own category/post_tag are POST-level, NOT used as labels.)

A Gemini row is usable only if it has no `error`, DID NOT hit the output cap (see
MAX_OUTPUT_TOKENS -- a capped row is truncated or stuck repeating, and 13 such rows sneak a
28,672-char garbage "tag" past every other check), has a URL, and has non-empty tags.

Split -> three sets (see do_plan): train, plus TWO complementary test sets:
  - test_unseen      : whole posts held out  (unseen post/person/scene generalization)
  - test_stratified  : images sampled per tag-frequency band (per-exposure tagging probe)

Two stages (webp download mechanism carried over from the original e621 builder):
  python mikomiko_dataset_builder.py --plan      # parse + split -> *_candidates.jsonl. NO download.
  python mikomiko_dataset_builder.py --download   # fetch webp -> mikomiko_{train,test_unseen,test_stratified}.json

All paths are anchored to this file's directory, so moving the repo does not break it.
"""
import os, io, sys, csv, json, time, queue, random, threading, argparse
import urllib.request
from collections import Counter, defaultdict
from PIL import Image

# Gemini's per-request output cap. A row that hits it did not stop on its own: it was truncated
# mid-list, or (worse) fell into a repetition loop. 166 of 1,240,421 rows hit the cap; 153 emit
# nothing parseable and die on the empty-tags check below, but 13 emit ONE 28,672-char token
# ("aminase" x 4096) that looks like a perfectly legal single tag -- `error` is empty, `tags` is
# non-empty, so both original filters wave it through. 11 of those reached train.jsonl.
MAX_OUTPUT_TOKENS = 4096

# Filled by iter_records(), printed by do_plan() -- a silent filter is a filter nobody audits.
DROPPED = Counter()

# ── paths (anchored to this file: data/mikomiko_tag/) ───────────────────────────
# Two source CSVs, joined on image key `{post_id}_{image_name}` (== GEMINI's custom_id):
#   GEMINI_CSV   -- per-image tags Gemini judged actually visible in each image  -> the LABEL
#   CATALOG_CSV  -- raw pornpic post catalog: gives the download URL per image    -> the IMAGE
# NOTE: CATALOG_CSV's own category/post_tag are POST-level (whole post, not this image), so they
# are NOT used as labels -- using them would teach tags not present in the specific image.
HERE         = os.path.dirname(os.path.abspath(__file__))
GEMINI_CSV   = os.path.join(HERE, "pornpic_tag_recognition_pornpic_tag_full_001.csv")
CATALOG_CSV  = os.path.join(HERE, "pornstars620_100k-sample-posts_category-tags.csv")
IMG_DIR      = os.path.join(HERE, "img")
JSONL_DIR    = os.path.join(HERE, "jsonl")   # all dataset/candidate/mini jsonl live here

# Three output splits (candidate jsonl -> downloaded jsonl). See do_plan() for semantics.
#   train           -- everything not held out
#   test_unseen     -- WHOLE posts held out (unseen post/person/scene generalization)
#   test_stratified -- images sampled by tag-frequency band (per-exposure tagging probe)
SPLITS = {
    "train":           ("train_candidates.jsonl",           "train.jsonl"),
    "test_unseen":     ("test_unseen_candidates.jsonl",     "test_unseen.jsonl"),
    "test_stratified": ("test_stratified_candidates.jsonl", "test_stratified.jsonl"),
}
CAND = {k: os.path.join(JSONL_DIR, v[0]) for k, v in SPLITS.items()}
JSON = {k: os.path.join(JSONL_DIR, v[1]) for k, v in SPLITS.items()}

# ── split knobs ───────────────────────────────────────────────────────────────
SPLIT_SEED        = 0        # deterministic split
LIMIT             = None     # cap total records (set via --limit) for spot checks
# (a) test_unseen: reserve this fraction of whole POSTS (all their images) for the unseen-post test.
POST_HOLDOUT_FRAC = 0.10
# (b) test_stratified: bucket images by the document-frequency of their RAREST tag, then sample up
# to STRAT_PER_BAND images per band from the (non-held-out) pool. Bands are DF ranges [lo, hi).
STRAT_BANDS       = [("very_rare", 1, 10), ("rare", 10, 100), ("mid", 100, 1000),
                     ("common", 1000, 10000), ("very_common", 10000, 10**12)]
STRAT_PER_BAND    = 400

# ── task framing: direct image -> tags ────────────────────────────────────────
# The model sees only the IMAGE and outputs the tags directly (open-vocabulary tagging).
# Label = GEMINI per-image tags. The tagging prompt lives in PROMPT_FILE (editable); if absent
# or empty, DEFAULT_PROMPT is used. Content is "<image>" + prompt (image first, text after).
PROMPT_FILE    = os.path.join(HERE, "prompt.txt")
DEFAULT_PROMPT = ("You are an expert tagger for an adult image board. Look at the image and output "
                  "a comma-separated list of board-style tags describing what is visible: the people "
                  "(ethnicity/nationality, apparent age group, body type, hair), their clothing and "
                  "accessories, exposed body parts, the sexual acts or poses, and the setting. Prefer "
                  "concise established tags. Output only the tags, comma-separated, with no other text.")


def post_of(name):
    """'82534173_9.webp' -> '82534173' (post id is the part before the first underscore)."""
    return name.split("_", 1)[0]


def tag_list(tags_str):
    """'a, b, c' -> ['a','b','c']."""
    return [t.strip() for t in tags_str.split(",") if t.strip()]


_PROMPT = None


def load_prompt():
    """Read the tagging prompt from PROMPT_FILE once; fall back to DEFAULT_PROMPT if empty/missing."""
    global _PROMPT
    if _PROMPT is None:
        content = ""
        if os.path.exists(PROMPT_FILE):
            content = open(PROMPT_FILE, encoding="utf-8").read().strip()
        _PROMPT = content or DEFAULT_PROMPT
    return _PROMPT


def build_instruction():
    """Content = image first, then the tagging prompt."""
    return "<image>" + load_prompt()

# ── download knobs (same as the e621 builder) ──────────────────────────────────
CONCURRENCY  = int(os.environ.get("MIKOMIKO_CONCURRENCY", "16"))  # override for big bulk downloads
TIMEOUT      = 30
RETRIES      = 2
UA           = {"User-Agent": "Mozilla/5.0 (dataset-probe; research)"}

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def clean_tags(raw):
    """Normalize GEMINI's `tags` field (comma- or bracket-list) -> de-duped, comma-joined string.
    Returns '' if nothing usable."""
    if not raw:
        return ""
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    seen, out = set(), []
    for t in raw.split(","):
        t = " ".join(t.split()).strip().strip('"').strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return ", ".join(out)


def load_url_map():
    """Build {image_key: url} from CATALOG_CSV, where image_key == '{post_id}_{image_name}'
    (== GEMINI custom_id). Streamed; ~1.24M entries."""
    url_of = {}
    with open(CATALOG_CSV, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("post_id") or "").strip()
            img = (row.get("image_name") or "").strip()
            url = (row.get("url") or "").strip()
            if pid and img and url:
                url_of[f"{pid}_{img}"] = url
    return url_of


def hit_output_cap(row):
    """True if Gemini stopped because it ran out of budget, not because it was done."""
    try:
        return float(row.get("output_tokens") or 0) >= MAX_OUTPUT_TOKENS
    except (TypeError, ValueError):
        return False


def iter_records():
    """Join GEMINI per-image tags (label) with CATALOG URLs (image). Yields {name,url,tags}
    for images with a Gemini tag list, no error, no output-cap hit, and a matching URL."""
    url_of = load_url_map()
    print(f"[join] catalog urls loaded: {len(url_of)}", flush=True)
    with open(GEMINI_CSV, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            DROPPED["scanned"] += 1
            if (row.get("error") or "").strip():
                DROPPED["error"] += 1
                continue
            if hit_output_cap(row):                         # truncated / repetition loop -> unusable
                DROPPED["output_cap"] += 1
                continue
            name = (row.get("custom_id") or "").strip()   # already '{post_id}_{image_name}'
            url = url_of.get(name)
            if not name or not url:                         # no download link -> skip
                DROPPED["no_url"] += 1
                continue
            tags = clean_tags(row.get("tags") or "")
            if not tags:                                    # Gemini found nothing visible -> skip
                DROPPED["empty_tags"] += 1
                continue
            DROPPED["kept"] += 1
            yield {"name": name, "url": url, "tags": tags}


# ── plan phase: parse + two-test-set split ───────────────────────────────────────
def do_plan(args):
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(JSONL_DIR, exist_ok=True)
    print(f"[plan] label csv = {GEMINI_CSV}  (Gemini per-image tags)")
    print(f"[plan] url   csv = {CATALOG_CSV}  (download URLs)")
    print(f"[plan] test = whole-post holdout {POST_HOLDOUT_FRAC:.0%} + tag-freq stratified "
          f"({STRAT_PER_BAND}/band)  seed={SPLIT_SEED}", flush=True)

    records, seen_names, t0 = [], set(), time.time()
    for rec in iter_records():
        if rec["name"] in seen_names:      # de-dup on unique image filename
            continue
        seen_names.add(rec["name"])
        records.append(rec)
        if LIMIT and len(records) >= LIMIT:
            print(f"[plan] hit --limit={LIMIT}, stopping scan early", flush=True)
            break
        if len(records) % 100000 == 0:
            print(f"  [plan] usable records={len(records)}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"[plan] funnel: scanned={DROPPED['scanned']:,}"
          f"  -error={DROPPED['error']:,}"
          f"  -output_cap={DROPPED['output_cap']:,}"      # truncated / repetition-loop rows
          f"  -no_url={DROPPED['no_url']:,}"
          f"  -empty_tags={DROPPED['empty_tags']:,}"
          f"  = kept {DROPPED['kept']:,} ({len(records):,} after de-dup)", flush=True)

    # ── (a) test_unseen: hold out whole POSTS (no image of a held post ever trains) ──
    by_post = defaultdict(list)
    for r in records:
        by_post[post_of(r["name"])].append(r)
    posts = sorted(by_post)                      # deterministic before shuffle
    random.seed(SPLIT_SEED)
    random.shuffle(posts)
    n_hold = int(round(len(posts) * POST_HOLDOUT_FRAC))
    held = set(posts[:n_hold])
    test_unseen = [r for p in posts[:n_hold] for r in by_post[p]]
    pool        = [r for p in posts[n_hold:] for r in by_post[p]]   # train candidates + stratified test
    print(f"[plan] posts={len(posts)}  held-out posts={len(held)} -> test_unseen imgs={len(test_unseen)}",
          flush=True)

    # ── (b) test_stratified: bucket pool images by their rarest tag's document frequency ──
    df = Counter()
    for r in pool:
        for t in tag_list(r["tags"]):
            df[t] += 1

    def band_of(v):
        for i, (_lbl, lo, hi) in enumerate(STRAT_BANDS):
            if lo <= v < hi:
                return i
        return len(STRAT_BANDS) - 1

    buckets = defaultdict(list)
    for r in pool:
        ts = tag_list(r["tags"])
        if not ts:
            continue
        buckets[band_of(min(df[t] for t in ts))].append(r)   # band by RAREST tag in the image

    random.seed(SPLIT_SEED + 1)
    test_strat, strat_names = [], set()
    for i, (lbl, lo, hi) in enumerate(STRAT_BANDS):
        imgs = buckets.get(i, [])
        random.shuffle(imgs)
        take = imgs[:STRAT_PER_BAND]
        test_strat.extend(take)
        strat_names.update(r["name"] for r in take)
        print(f"  [strat] band {lbl:<12} DF[{lo},{hi}) pool={len(imgs)} -> sampled {len(take)}", flush=True)

    train = [r for r in pool if r["name"] not in strat_names]

    splits = {"train": train, "test_unseen": test_unseen, "test_stratified": test_strat}
    for key, rows in splits.items():
        with open(CAND[key], "w", encoding="utf-8") as f:
            for c in rows:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[plan] total usable={len(records)}  train={len(train)}  "
          f"test_unseen={len(test_unseen)}  test_stratified={len(test_strat)}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[plan] wrote 3 candidate files -> {', '.join(os.path.basename(CAND[k]) for k in splits)}")
    print(f"[plan] >>> next: python mikomiko_dataset_builder.py --download", flush=True)


# ── download phase ──────────────────────────────────────────────────────────────
def fetch(url):
    for _ in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except Exception:
            time.sleep(0.5)
    return None


def worker(cand_q, results, lock, counters):
    while True:
        try:
            c = cand_q.get_nowait()
        except queue.Empty:
            return
        dst = os.path.join(IMG_DIR, c["name"])
        rec = {"tags": c["tags"], "image": dst}
        # resume: valid image already on disk -> keep, skip network
        if os.path.exists(dst):
            try:
                Image.open(dst).verify()
                with lock:
                    results.append(rec)
                    counters["kept"] += 1
                continue
            except Exception:
                pass
        data = fetch(c["url"])
        if data:
            try:
                Image.open(io.BytesIO(data)).verify()   # validate it decodes
                with open(dst, "wb") as fp:
                    fp.write(data)
                with lock:
                    results.append(rec)
                    counters["kept"] += 1
            except Exception:
                pass
        with lock:
            counters["tried"] += 1
            if counters["tried"] % 200 == 0:
                print(f"  [dl] tried={counters['tried']} kept={counters['kept']}", flush=True)


def download_split(cand_path, out_json, tag_label):
    if not os.path.exists(cand_path):
        print(f"[dl] SKIP {tag_label}: {cand_path} missing (run --plan first)", flush=True)
        return 0
    cands = [json.loads(l) for l in open(cand_path, encoding="utf-8")]
    print(f"[dl] {tag_label}: {len(cands)} candidates -> {out_json}", flush=True)

    cand_q = queue.Queue()
    for c in cands:
        cand_q.put(c)
    results, lock = [], threading.Lock()
    counters = {"kept": 0, "tried": 0}
    threads = [threading.Thread(target=worker, args=(cand_q, results, lock, counters))
               for _ in range(CONCURRENCY)]
    for t in threads: t.start()
    for t in threads: t.join()

    random.seed(1234)
    random.shuffle(results)
    # alpaca format, matching the repo's MVTOKEN convention (instruction/input/output/images).
    # instruction = image first + the tagging prompt; output = Gemini per-image tags.
    instr = build_instruction()
    rows = [{
        "instruction": instr,
        "input": "",
        "output": r["tags"],
        "images": [r["image"]],
    } for r in results]
    with open(out_json, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[dl] {tag_label}: kept={len(results)} (tried={counters['tried']}) -> {out_json}", flush=True)
    return len(results)


MINI_EVAL_N = 200   # per-test-set sample used for fast in-loop eval during training


def write_mini_evals():
    """Write a MINI_EVAL_N-sample of each test split (*_mini.jsonl) for fast in-loop eval.
    The full test sets are for final post-training evaluation."""
    for key in ("test_unseen", "test_stratified"):
        src = JSON[key]
        if not os.path.exists(src):
            continue
        lines = open(src, encoding="utf-8").read().splitlines()
        random.seed(42)
        random.shuffle(lines)
        mini = os.path.join(JSONL_DIR, os.path.basename(src).replace(".jsonl", "_mini.jsonl"))
        with open(mini, "w", encoding="utf-8") as f:
            f.write("\n".join(lines[:MINI_EVAL_N]) + "\n")
        print(f"[mini] {os.path.basename(mini)}: {min(MINI_EVAL_N, len(lines))} samples", flush=True)


def do_download(args):
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(JSONL_DIR, exist_ok=True)
    counts = {k: download_split(CAND[k], JSON[k], k) for k in SPLITS}
    write_mini_evals()
    print(f"[dl] DONE " + "  ".join(f"{k}={n}" for k, n in counts.items()), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build pornstar image-tag SFT dataset for LlamaFactory.")
    ap.add_argument("--plan", action="store_true", help="parse CSV + two-test-set split, no download")
    ap.add_argument("--download", action="store_true",
                    help="download images -> {train,test_unseen,test_stratified}.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap total records scanned (spot-check)")
    args = ap.parse_args()
    if args.limit:
        globals()["LIMIT"] = args.limit
    if not (args.plan or args.download):
        args.plan = True   # default to the cheap, reviewable step
    if args.plan:
        do_plan(args)
    if args.download:
        do_download(args)


if __name__ == "__main__":
    main()
