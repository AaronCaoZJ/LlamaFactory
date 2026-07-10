#!/usr/bin/env python3
"""clean_truncated_rows.py — strip Gemini repetition-loop output from the BUILT jsonl.

Why not just re-run `dataset_builder.py --plan`? Because test_stratified is sampled by the document
frequency of each image's rarest tag. Dropping records nudges those df counts, which can move
images between bands and produce a DIFFERENT eval set — every metric ever reported on the old one
would stop being comparable. Surgery keeps train/test/mini membership byte-identical except for the
poisoned labels.

Two failure modes, both found by scanning all splits:
  * whole label is garbage (13 rows) — Gemini hit its 4096-token output cap stuck repeating, and
    emitted one 28,672-char "tag" ("aminase" x 4096) as the entire label. `error` was empty and
    `tags` non-empty, so the original builder's filters waved it through. The row carries no signal
    at all -> DROP THE ROW.
  * one tag is garbage, the rest are fine (1 row) — an in-tag loop like "Big Ass Anal Ass Anal
    Ass Anal ..." (84 chars) sitting next to 10 legitimate tags. It never hit the output cap, so no
    token-budget filter can catch it -> DROP THE TAG, KEEP THE ROW.

dataset_builder.py now rejects rows that hit MAX_OUTPUT_TOKENS, which covers the first mode at the
source. This script cleans artifacts built before that fix, and the second mode, which no
builder-side token check can see.

Everything removed is written to removed_truncated.jsonl (audit trail), never silently discarded.

Usage:
    python clean_truncated_rows.py            # dry run: report only
    python clean_truncated_rows.py --apply
"""
import argparse, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL_DIR = os.path.join(HERE, "jsonl")
REMOVED = os.path.join(JSONL_DIR, "removed_truncated.jsonl")

MAX_TAG_CHARS = 64          # longest legitimate tag is 24 chars ("Skinny Black Hairy Pussy")
# A 1-4 word phrase repeated >=3 times in a row. Catches the loop itself rather than one hard-coded
# word, so the next model that gets stuck repeating something else is caught too.
REPEAT = re.compile(r"\b(\w+(?:\s+\w+){0,3})(?:\s+\1\b){2,}", re.I)
CHAR_LOOP = re.compile(r"(\w{4,}?)\1{3,}")   # 'aminaseaminase...' has no spaces to anchor on

TARGETS = ["train.jsonl", "test_unseen.jsonl", "test_stratified.jsonl",
           "train_candidates.jsonl", "test_unseen_candidates.jsonl", "test_stratified_candidates.jsonl",
           "eval_mini.jsonl", "test_unseen_mini.jsonl", "test_stratified_mini.jsonl"]

LABEL_KEYS = ("output", "tags")   # built dataset files use `output`; candidate files use `tags`


def label_key(rec):
    return next((k for k in LABEL_KEYS if k in rec), None)


def is_bad_tag(tag):
    return len(tag) > MAX_TAG_CHARS or bool(REPEAT.search(tag)) or bool(CHAR_LOOP.search(tag))


def clean_record(rec):
    """-> (action, new_rec, bad_tags). action in {'keep', 'strip', 'drop'}."""
    key = label_key(rec)
    if key is None:
        return "keep", rec, []
    tags = [t.strip() for t in rec[key].split(",") if t.strip()]
    bad = [t for t in tags if is_bad_tag(t)]
    if not bad:
        return "keep", rec, []
    good = [t for t in tags if t not in bad]
    if not good:                                     # nothing left to learn from
        return "drop", rec, bad
    rec = dict(rec, **{key: ", ".join(good)})
    return "strip", rec, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="rewrite the files (default: dry run)")
    args = ap.parse_args()
    tag = "apply" if args.apply else "dry"

    audit, n_dropped, n_stripped = [], 0, 0
    for name in TARGETS:
        path = os.path.join(JSONL_DIR, name)
        if not os.path.exists(path):
            print(f"[skip] {name} (absent)")
            continue

        out_lines, dropped, stripped = [], 0, 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                action, new_rec, bad = clean_record(rec)
                if action == "keep":
                    out_lines.append(line)
                    continue
                audit.append({"file": name, "action": action, "bad_tags": bad, "row": rec})
                if action == "drop":
                    dropped += 1
                else:
                    stripped += 1
                    out_lines.append(json.dumps(new_rec, ensure_ascii=False) + "\n")
        n_dropped += dropped
        n_stripped += stripped
        n_before = len(out_lines) + dropped
        print(f"[{tag}] {name:<36} {n_before:>9,} -> {len(out_lines):>9,}"
              f"   drop {dropped}, strip {stripped}")

        if args.apply and (dropped or stripped):
            tmp = path + ".tmp"                          # write-then-rename: never a half file
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(out_lines)
            os.replace(tmp, path)

    print(f"\n{n_dropped} rows dropped, {n_stripped} rows had a tag stripped.")
    if not args.apply:
        print("Dry run — re-run with --apply to rewrite.")
        return
    if audit:
        with open(REMOVED, "w", encoding="utf-8") as f:
            for r in audit:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Audit trail -> {REMOVED}")
    else:
        print("Nothing to clean — files were already clean.")


if __name__ == "__main__":
    main()
