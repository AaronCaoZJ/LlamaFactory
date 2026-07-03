#!/usr/bin/env python3
"""
mikomiko_dataset_builder.py
===========================
Build a multimodal image-tag-recognition SFT dataset for LlamaFactory from
`pornstars620_100k-sample-posts_category-tags.csv`.

Task shape: given an image, the model outputs a comma-separated list of tags.
The dataset is emitted in LlamaFactory **sharegpt** format (messages + images),
split 80% train / 20% test with a fixed seed.

CSV columns: post_id, image_name, url, category, post_tag
  - `url` carries the full webp download link, so no CDN guessing is needed.
  - the label is category + post_tag merged & de-duplicated (see LABEL_FIELDS).

Two stages (webp download mechanism carried over from the original e621 builder):
  python mikomiko_dataset_builder.py --plan      # parse CSV, split, write *_candidates.jsonl. NO download.
  python mikomiko_dataset_builder.py --download   # fetch webp images, write mikomiko_{train,test}.json.

All paths are anchored to this file's directory, so moving the repo does not break it.
"""
import os, io, sys, csv, json, time, queue, random, threading, argparse
import urllib.request
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
IMG_DIR      = os.path.join(HERE, "img/pornstar")
TRAIN_CAND   = os.path.join(HERE, "train_candidates.jsonl")
TEST_CAND    = os.path.join(HERE, "test_candidates.jsonl")
TRAIN_JSON   = os.path.join(HERE, "mikomiko_train.json")
TEST_JSON    = os.path.join(HERE, "mikomiko_test.json")

# ── dataset knobs ───────────────────────────────────────────────────────────────
TEST_FRAC    = 0.20            # fraction of samples held out as the test set
SPLIT_SEED   = 0              # deterministic train/test split
LIMIT        = None           # cap total records (set via --limit) for spot checks
# Fixed instruction shown to the model for every image (English, matches the tag language).
INSTRUCTION  = ("You are an expert image tagger. Look at the image and output the "
                "descriptive tags as a comma-separated list.")

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


def load_url_map():
    """Build {image_key: url} from CATALOG_CSV, where image_key == '{post_id}_{image_name}'
    (this equals GEMINI's custom_id). Streamed; ~1.24M small entries."""
    url_of = {}
    with open(CATALOG_CSV, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("post_id") or "").strip()
            img = (row.get("image_name") or "").strip()
            url = (row.get("url") or "").strip()
            if pid and img and url:
                url_of[f"{pid}_{img}"] = url
    return url_of


def iter_records():
    """Join GEMINI per-image tags (label) with CATALOG URLs (image). Yields {name,url,tags}
    for images that (a) have a Gemini tag list, (b) no error, and (c) a matching URL."""
    url_of = load_url_map()
    print(f"[join] catalog urls loaded: {len(url_of)}", flush=True)
    with open(GEMINI_CSV, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("error") or "").strip():
                continue
            name = (row.get("custom_id") or "").strip()   # already '{post_id}_{image_name}'
            if not name:
                continue
            url = url_of.get(name)
            if not url:                                     # no download link -> skip
                continue
            tags = clean_tags(row.get("tags") or "")
            if not tags:                                    # Gemini found nothing visible -> skip
                continue
            yield {"name": name, "url": url, "tags": tags}


# ── plan phase: parse + split ───────────────────────────────────────────────────
def do_plan(args):
    os.makedirs(IMG_DIR, exist_ok=True)
    print(f"[plan] label csv = {GEMINI_CSV}  (Gemini per-image tags)")
    print(f"[plan] url   csv = {CATALOG_CSV}  (download URLs)")
    print(f"[plan] split = {int((1-TEST_FRAC)*100)}% train / {int(TEST_FRAC*100)}% test (seed={SPLIT_SEED})")

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

    random.seed(SPLIT_SEED)
    random.shuffle(records)
    n_test = int(round(len(records) * TEST_FRAC))
    test, train = records[:n_test], records[n_test:]

    for path, split in ((TRAIN_CAND, train), (TEST_CAND, test)):
        with open(path, "w", encoding="utf-8") as f:
            for c in split:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[plan] total usable={len(records)}  train={len(train)}  test={len(test)}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[plan] wrote {TRAIN_CAND} and {TEST_CAND}")
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
        # resume: valid image already on disk -> keep, skip network
        if os.path.exists(dst):
            try:
                Image.open(dst).verify()
                with lock:
                    results.append({"tags": c["tags"], "image": dst})
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
                    results.append({"tags": c["tags"], "image": dst})
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
    # alpaca format, matching the repo's MVTOKEN convention
    # (columns: instruction/input/output/images).
    rows = [{
        "instruction": "<image>" + INSTRUCTION,
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
    n_tr = download_split(TRAIN_CAND, TRAIN_JSON, "train")
    n_te = download_split(TEST_CAND, TEST_JSON, "test")
    print(f"[dl] DONE train={n_tr} test={n_te}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build pornstar image-tag SFT dataset for LlamaFactory.")
    ap.add_argument("--plan", action="store_true", help="parse CSV + train/test split, no download")
    ap.add_argument("--download", action="store_true", help="download images -> mikomiko_{train,test}.json")
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
