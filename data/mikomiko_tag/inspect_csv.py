#!/usr/bin/env python3
"""
inspect_csv.py
==============
Observe the TWO raw CSVs that dataset_builder.py joins, WITHOUT downloading anything:

  GEMINI_CSV  = pornpic_tag_recognition_pornpic_tag_full_001.csv   (per-image tags -> the LABEL)
  CATALOG_CSV = pornstars620_100k-sample-posts_category-tags.csv    (post catalog -> the IMAGE url)

Answers: how many posts / images, what tags exist (3 separate tag fields), and how the two
files correspond (join hit-rate + the exact funnel dataset_builder.iter_records() applies).

Streamed, single pass per file. Paths anchored to this file's dir (same as dataset_builder.py).

  python inspect_csv.py                 # full scan (reads both ~400MB files, a few minutes)
  python inspect_csv.py --limit 200000  # quick spot-check: cap rows scanned per file
  python inspect_csv.py --topk 40       # how many top tags to print per vocabulary
"""
import os, sys, csv, json, time, argparse
from collections import Counter, defaultdict

HERE        = os.path.dirname(os.path.abspath(__file__))
GEMINI_CSV  = os.path.join(HERE, "pornpic_tag_recognition_pornpic_tag_full_001.csv")   # LABEL source
CATALOG_CSV = os.path.join(HERE, "pornstars620_100k-sample-posts_category-tags.csv")   # IMAGE url source

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def post_of(name):
    """'82534173_9.webp' -> '82534173' (post id = part before first underscore). Matches builder."""
    return name.split("_", 1)[0]


def split_tags(raw):
    """A tag field may be 'a, b, c' or a bracket-list '[a, b, c]' or a JSON array. -> [tags]."""
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        # try JSON array first (GEMINI raw_response style), else strip brackets and split on comma
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return [str(t).strip() for t in arr if str(t).strip()]
        except Exception:
            raw = raw[1:-1]
    return [t.strip().strip('"').strip() for t in raw.split(",") if t.strip()]


def pct(a, b):
    return f"{(100.0 * a / b):.1f}%" if b else "n/a"


def hist(counter_of_counts, label):
    """Print a compact distribution (min/median/p90/max/mean) of a Counter's VALUES."""
    vals = sorted(counter_of_counts.values())
    if not vals:
        print(f"  {label}: (empty)")
        return
    n = len(vals)
    mean = sum(vals) / n
    p = lambda q: vals[min(n - 1, int(q * n))]
    print(f"  {label}: n={n}  min={vals[0]}  p50={p(0.5)}  p90={p(0.9)}  max={vals[-1]}  mean={mean:.2f}")


def top_vocab(counter, topk, title):
    print(f"\n  [{title}] unique={len(counter):,}  total_occurrences={sum(counter.values()):,}")
    atomic = {t: c for t, c in counter.items() if " " not in t}
    comp   = {t: c for t, c in counter.items() if " " in t}
    print(f"    atomic(1-word) unique={len(atomic):,}   compound(>=2-word) unique={len(comp):,} "
          f"({pct(len(comp), len(counter))} of vocab)")
    df1 = sum(1 for c in counter.values() if c == 1)
    print(f"    tags appearing exactly once (df=1): {df1:,} ({pct(df1, len(counter))} of vocab)")
    print(f"    top {topk} by frequency:")
    for t, c in counter.most_common(topk):
        print(f"      {c:>9,}  {t}")


# ── 1. CATALOG: images, posts, url, post-level category/post_tag ────────────────────────────────
def scan_catalog(limit):
    print("=" * 90)
    print(f"CATALOG_CSV  (image url source)\n  {CATALOG_CSV}")
    print("=" * 90)
    keys          = set()              # '{post_id}_{image_name}'  == GEMINI custom_id
    imgs_per_post = Counter()          # post_id -> #images
    n_rows = n_url = n_dup = 0
    cat_vocab = Counter()              # post-level 'category' tags
    ptag_vocab = Counter()             # post-level 'post_tag' tags
    n_have_cat = n_have_ptag = 0
    t0 = time.time()
    with open(CATALOG_CSV, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            pid = (row.get("post_id") or "").strip()
            img = (row.get("image_name") or "").strip()
            url = (row.get("url") or "").strip()
            if url:
                n_url += 1
            if pid and img:
                key = f"{pid}_{img}"
                if key in keys:
                    n_dup += 1
                else:
                    keys.add(key)
                    imgs_per_post[pid] += 1
            cat = split_tags(row.get("category") or "")
            pt  = split_tags(row.get("post_tag") or "")
            if cat:
                n_have_cat += 1
                cat_vocab.update(cat)
            if pt:
                n_have_ptag += 1
                ptag_vocab.update(pt)
            if limit and n_rows >= limit:
                break
            if n_rows % 200000 == 0:
                print(f"  ...scanned {n_rows:,} rows ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  rows(image entries)         : {n_rows:,}")
    print(f"  unique images (post_id_img) : {len(keys):,}   (duplicate keys skipped: {n_dup:,})")
    print(f"  unique posts                : {len(imgs_per_post):,}")
    print(f"  rows with a url             : {n_url:,} ({pct(n_url, n_rows)})")
    hist(imgs_per_post, "images per post")
    print(f"  rows carrying 'category'    : {n_have_cat:,} ({pct(n_have_cat, n_rows)})  [POST-level, NOT the label]")
    print(f"  rows carrying 'post_tag'    : {n_have_ptag:,} ({pct(n_have_ptag, n_rows)})  [POST-level, NOT the label]")
    return keys, imgs_per_post, cat_vocab, ptag_vocab


# ── 2. GEMINI: per-image tags (the LABEL), errors, coverage ─────────────────────────────────────
def scan_gemini(limit):
    print("\n" + "=" * 90)
    print(f"GEMINI_CSV  (per-image tag / LABEL source)\n  {GEMINI_CSV}")
    print("=" * 90)
    ids           = set()              # custom_id == '{post_id}_{image_name}'
    posts         = set()
    tag_vocab     = Counter()          # per-image tag vocabulary (the actual training label space)
    tags_per_img  = Counter()          # custom_id -> #tags
    model_ver     = Counter()
    n_rows = n_error = n_empty = n_dup = 0
    cost = 0.0
    t0 = time.time()
    with open(GEMINI_CSV, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            cid = (row.get("custom_id") or "").strip()
            if (row.get("error") or "").strip():
                n_error += 1
            if cid:
                if cid in ids:
                    n_dup += 1
                else:
                    ids.add(cid)
                    posts.add(post_of(cid))
            model_ver[(row.get("model_version") or "").strip() or "(none)"] += 1
            try:
                cost += float(row.get("cost_usd") or 0)
            except ValueError:
                pass
            tags = split_tags(row.get("tags") or "")
            if not tags:
                n_empty += 1
            else:
                tag_vocab.update(tags)
                if cid:
                    tags_per_img[cid] = len(tags)
            if limit and n_rows >= limit:
                break
            if n_rows % 200000 == 0:
                print(f"  ...scanned {n_rows:,} rows ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  rows(image judgements)      : {n_rows:,}")
    print(f"  unique custom_id (images)   : {len(ids):,}   (duplicate ids: {n_dup:,})")
    print(f"  unique posts (derived)      : {len(posts):,}")
    print(f"  rows with error (skipped)   : {n_error:,} ({pct(n_error, n_rows)})")
    print(f"  rows with EMPTY tags        : {n_empty:,} ({pct(n_empty, n_rows)})  [Gemini saw nothing -> dropped]")
    print(f"  model_version breakdown     : " + ", ".join(f"{k}={v:,}" for k, v in model_ver.most_common()))
    print(f"  total gemini cost (usd)     : {cost:,.2f}")
    hist(tags_per_img, "tags per image")
    return ids, posts, tag_vocab


# ── 3. correspondence + the exact dataset_builder funnel ────────────────────────────────────────
def correspondence(cat_keys, cat_posts, gem_ids, gem_posts, limit):
    print("\n" + "=" * 90)
    print("CORRESPONDENCE  (how the two files line up on key '{post_id}_{image_name}')")
    print("=" * 90)
    if limit:
        print("  !! --limit is set: the two files are NOT row-aligned, so a capped scan reads DIFFERENT")
        print("     images from each -> image-level join will look ~0. Run WITHOUT --limit for real numbers.\n")
    both = cat_keys & gem_ids
    print(f"  join key                    : GEMINI.custom_id  ==  CATALOG.'{{post_id}}_{{image_name}}'")
    print(f"  images in CATALOG only      : {len(cat_keys - gem_ids):,}   (have url, no Gemini label)")
    print(f"  images in GEMINI only       : {len(gem_ids - cat_keys):,}   (labelled, no download url -> unusable)")
    print(f"  images in BOTH (joinable)   : {len(both):,}   "
          f"({pct(len(both), len(gem_ids))} of gemini, {pct(len(both), len(cat_keys))} of catalog)")
    print(f"  posts in CATALOG            : {len(cat_posts):,}")
    print(f"  posts in GEMINI             : {len(gem_posts):,}")
    print(f"  posts in BOTH               : {len(cat_posts & gem_posts):,}")

    # Reproduce iter_records() funnel EXACTLY: gemini row kept iff (no error) AND (url exists) AND (tags non-empty),
    # then de-dup by custom_id. This is the "usable records" that get split into train/test_unseen/test_stratified.
    print("\n  --- dataset_builder funnel (usable = no-error AND url-match AND non-empty-tags, de-duped) ---")
    kept, seen = 0, set()
    n_no_url = n_empty = n_error = n_rows = 0
    with open(GEMINI_CSV, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            if (row.get("error") or "").strip():
                n_error += 1
                if limit and n_rows >= limit:
                    break
                continue
            cid = (row.get("custom_id") or "").strip()
            if not cid or cid not in cat_keys:
                n_no_url += 1
                if limit and n_rows >= limit:
                    break
                continue
            if not split_tags(row.get("tags") or ""):
                n_empty += 1
                if limit and n_rows >= limit:
                    break
                continue
            if cid not in seen:
                seen.add(cid)
                kept += 1
            if limit and n_rows >= limit:
                break
    print(f"    gemini rows scanned       : {n_rows:,}")
    print(f"    - dropped (error)         : {n_error:,}")
    print(f"    - dropped (no url match)  : {n_no_url:,}")
    print(f"    - dropped (empty tags)    : {n_empty:,}")
    print(f"    = USABLE records (deduped): {kept:,}")
    print(f"      (builder then splits these into train + test_unseen + test_stratified)")


def main():
    ap = argparse.ArgumentParser(description="Inspect the two raw mikomiko CSVs and their correspondence.")
    ap.add_argument("--limit", type=int, default=None, help="cap rows scanned per file (quick spot-check)")
    ap.add_argument("--topk", type=int, default=30, help="top-N tags to print per vocabulary")
    args = ap.parse_args()

    for p in (GEMINI_CSV, CATALOG_CSV):
        if not os.path.exists(p):
            print(f"[error] missing: {p}", file=sys.stderr)
            sys.exit(1)

    cat_keys, imgs_per_post, cat_vocab, ptag_vocab = scan_catalog(args.limit)
    gem_ids, gem_posts, gem_tag_vocab = scan_gemini(args.limit)

    print("\n" + "=" * 90)
    print("TAG VOCABULARIES  (three separate tag fields — only GEMINI 'tags' is the training LABEL)")
    print("=" * 90)
    top_vocab(gem_tag_vocab, args.topk, "GEMINI per-image tags  == LABEL space")
    top_vocab(cat_vocab,     args.topk, "CATALOG 'category'  (POST-level, NOT label)")
    top_vocab(ptag_vocab,    args.topk, "CATALOG 'post_tag'  (POST-level, NOT label)")

    correspondence(cat_keys, set(imgs_per_post), gem_ids, gem_posts, args.limit)
    print("\n[done]")


if __name__ == "__main__":
    main()
