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

# Three output splits (candidate jsonl -> downloaded json). See do_plan() for semantics.
#   train           -- everything not held out
#   test_unseen     -- WHOLE posts held out (unseen post/person/scene generalization)
#   test_stratified -- images sampled by tag-frequency band (per-exposure tagging probe)
SPLITS = {
    "train":           ("train_candidates.jsonl",           "mikomiko_train.json"),
    "test_unseen":     ("test_unseen_candidates.jsonl",     "mikomiko_test_unseen.json"),
    "test_stratified": ("test_stratified_candidates.jsonl", "mikomiko_test_stratified.json"),
}
CAND = {k: os.path.join(HERE, v[0]) for k, v in SPLITS.items()}
JSON = {k: os.path.join(HERE, v[1]) for k, v in SPLITS.items()}

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

# ── task framing: candidate-selection ─────────────────────────────────────────
# The model sees the IMAGE + the post's candidate tags, and must SELECT the ones actually
# visible (mirrors how GEMINI produced the labels). Candidate pool = union of these post-level
# columns; verified that 100% of Gemini labels fall inside category ∪ post_tag (post_tag alone
# covers only ~31%), so BOTH are required.
CANDIDATE_FIELDS = ("category", "post_tag")
# Selection prompt from the colleague; `{tags}` is replaced by the candidate list. Until that
# file exists, DEFAULT_PROMPT is used. Content is "<image>" + prompt  (image first, text after).
PROMPT_FILE    = os.path.join(HERE, "prompt.txt")
DEFAULT_PROMPT = ("Below are candidate tags from this image's source post. Select ONLY the tags "
                  "that are actually visible in this image and output them as a comma-separated "
                  "list.\nCandidate tags: {tags}")


def post_of(name):
    """'82534173_9.webp' -> '82534173' (post id is the part before the first underscore)."""
    return name.split("_", 1)[0]


def tag_list(tags_str):
    """'a, b, c' -> ['a','b','c']."""
    return [t.strip() for t in tags_str.split(",") if t.strip()]


def parse_tags(raw):
    """'[Sexy, Asian, Mom]' or 'Sexy, Asian' -> ordered, de-duped ['Sexy','Asian','Mom']."""
    raw = (raw or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    seen, out = set(), []
    for t in raw.split(","):
        t = " ".join(t.split()).strip().strip('"').strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


_PROMPT = None


def load_prompt():
    """Read the colleague's selection prompt from PROMPT_FILE once; fall back to DEFAULT_PROMPT."""
    global _PROMPT
    if _PROMPT is None:
        content = ""
        if os.path.exists(PROMPT_FILE):
            content = open(PROMPT_FILE, encoding="utf-8").read().strip()
        _PROMPT = content or DEFAULT_PROMPT   # empty/missing file -> default
    return _PROMPT


def build_instruction(candidates):
    """Content = image first, then text. `{tags}` in the prompt is replaced by the candidate list;
    if the prompt has no placeholder, the candidates are appended."""
    p = load_prompt()
    body = p.replace("{tags}", candidates) if "{tags}" in p else f"{p}\n\nCandidate tags: {candidates}"
    return "<image>" + body

# ── download knobs (same as the e621 builder) ──────────────────────────────────
CONCURRENCY  = 16
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


def load_catalog():
    """Build {image_key: (url, candidates)} from CATALOG_CSV, where image_key == '{post_id}_{image_name}'
    (== GEMINI custom_id) and candidates = comma-joined union of CANDIDATE_FIELDS (the selection pool).
    Streamed; ~1.24M entries."""
    catalog = {}
    with open(CATALOG_CSV, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("post_id") or "").strip()
            img = (row.get("image_name") or "").strip()
            url = (row.get("url") or "").strip()
            if not (pid and img and url):
                continue
            seen, cands = set(), []
            for fld in CANDIDATE_FIELDS:
                for t in parse_tags(row.get(fld)):
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        cands.append(t)
            catalog[f"{pid}_{img}"] = (url, ", ".join(cands))
    return catalog


def iter_records():
    """Join GEMINI per-image tags (label) with CATALOG (url + candidate tags). Yields
    {name,url,candidates,tags} for images with a Gemini tag list, no error, a URL, and candidates."""
    catalog = load_catalog()
    print(f"[join] catalog entries loaded: {len(catalog)}", flush=True)
    with open(GEMINI_CSV, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("error") or "").strip():
                continue
            name = (row.get("custom_id") or "").strip()   # already '{post_id}_{image_name}'
            hit = catalog.get(name)
            if not name or not hit:                         # no download link / candidates -> skip
                continue
            url, candidates = hit
            if not candidates:                              # no candidate pool -> can't select
                continue
            tags = clean_tags(row.get("tags") or "")
            if not tags:                                    # Gemini found nothing visible -> skip
                continue
            yield {"name": name, "url": url, "candidates": candidates, "tags": tags}


# ── plan phase: parse + two-test-set split ───────────────────────────────────────
def do_plan(args):
    os.makedirs(IMG_DIR, exist_ok=True)
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
        rec = {"candidates": c["candidates"], "tags": c["tags"], "image": dst}
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
    # instruction = image first + selection prompt with the post's candidate tags injected.
    rows = [{
        "instruction": build_instruction(r["candidates"]),
        "input": "",
        "output": r["tags"],
        "images": [r["image"]],
    } for r in results]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[dl] {tag_label}: kept={len(results)} (tried={counters['tried']}) -> {out_json}", flush=True)
    return len(results)


def do_download(args):
    os.makedirs(IMG_DIR, exist_ok=True)
    counts = {k: download_split(CAND[k], JSON[k], k) for k in SPLITS}
    print(f"[dl] DONE " + "  ".join(f"{k}={n}" for k, n in counts.items()), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build pornstar image-tag SFT dataset for LlamaFactory.")
    ap.add_argument("--plan", action="store_true", help="parse CSV + two-test-set split, no download")
    ap.add_argument("--download", action="store_true",
                    help="download images -> mikomiko_{train,test_unseen,test_stratified}.json")
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
