#!/usr/bin/env python3
"""sample_data.py — step 1/3 of the seen/unseen review page pipeline.

Samples N seen (train.jsonl) + N unseen (test_unseen_mini.jsonl) images and joins each image's
POST-level context tags (catalog `category` + `post_tag`) by post_id for side-by-side display.
Output: WORK_DIR/samples.json  [{split,name,post_id,image,gemini,post_tag,category,instruction}]

Usage:
    python sample_data.py [--n 200] [--seed 42] [--work-dir SAVES/viz_review]
Next: ../infer_mikomiko.py --input WORK/samples.json --output WORK/samples_pred.json
(or just run ../infer_tag_2b.sh viz, which drives all three steps)
"""
import argparse, csv, json, os, random, sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parents[4]          # .../LlamaFactory
JSONL_DIR = ROOT / "data/mikomiko_tag/jsonl"
CATALOG_CSV = ROOT / "data/mikomiko_tag/pornstars620_100k-sample-posts_category-tags.csv"
DEFAULT_WORK = ROOT / "saves/qwen3.5-2b/mikomiko/viz_review"


def parse_list(s):
    """'[A, B]' or 'A, B' -> de-duped list of tags."""
    s = (s or "").strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    out, seen = [], set()
    for t in s.split(","):
        t = " ".join(t.split()).strip().strip('"').strip()
        if t and t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24, help="samples per split (default 24 = 6 rows of 4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--jsonl-dir", default=str(JSONL_DIR),
                    help="dir holding test_unseen_mini.jsonl + train.jsonl (default: the v0 jsonl/ set; "
                         "pass data/mikomiko_tag/jsonl_0716 to sample the 0716 set)")
    args = ap.parse_args()
    jsonl_dir = Path(args.jsonl_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    out_path = os.path.join(args.work_dir, "samples.json")
    samples = []

    # unseen: from the 200-image mini set (post-level zero overlap with train)
    rows = [json.loads(l) for l in open(jsonl_dir / "test_unseen_mini.jsonl", encoding="utf-8")]
    random.seed(args.seed)
    for r in random.sample(rows, min(args.n, len(rows))):
        samples.append(dict(split="unseen", image=r["images"][0], gemini=r["output"],
                            instruction=r["instruction"]))

    # seen: sample train.jsonl by line index (single streaming pass; ~1.1M lines)
    train_path = jsonl_dir / "train.jsonl"
    n_train = sum(1 for _ in open(train_path, "rb"))
    random.seed(args.seed)
    want = set(random.sample(range(n_train), args.n * 2))   # oversample: some images may be missing
    picked = []
    with open(train_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in want:
                r = json.loads(line)
                if os.path.exists(r["images"][0]):
                    picked.append(r)
    random.seed(args.seed + 1)
    random.shuffle(picked)
    for r in picked[:args.n]:
        samples.append(dict(split="seen", image=r["images"][0], gemini=r["output"],
                            instruction=r["instruction"]))

    for s in samples:
        s["name"] = os.path.basename(s["image"])            # '82534173_9.webp'
        s["post_id"] = s["name"].split("_", 1)[0]

    # join catalog once, keeping only the sampled posts
    need = {s["post_id"] for s in samples}
    cat_of, pt_of = {}, {}
    with open(CATALOG_CSV, encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("post_id") or "").strip()
            if pid in need and pid not in cat_of:
                cat_of[pid] = parse_list(row.get("category"))
                pt_of[pid] = parse_list(row.get("post_tag"))
                if len(cat_of) == len(need):
                    break
    for s in samples:
        s["category"] = cat_of.get(s["post_id"], [])
        s["post_tag"] = pt_of.get(s["post_id"], [])

    json.dump(samples, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_seen = sum(1 for s in samples if s["split"] == "seen")
    print(f"[sample] total={len(samples)} seen={n_seen} unseen={len(samples)-n_seen} -> {out_path}")
    print(f"[sample] catalog matched: {sum(1 for s in samples if s['category'] or s['post_tag'])}/{len(samples)}")


if __name__ == "__main__":
    main()
