#!/usr/bin/env python3
"""sample_onlyfans.py — step 1/3 of the OnlyFans (no-ground-truth) review page.

The mikomiko tagger was trained on pornpic board images with Gemini per-image tags. This builds a
review set from a folder of OnlyFans stills, which have NO gold tags at all: the page it feeds is a
distribution check (what does the model say off-distribution?), not a scored eval.

Rows carry `gemini: ""` so infer_mikomiko.py's samples.json path works unchanged and build_html.py
auto-detects the no-gold mode. The instruction is read from data/mikomiko_tag/prompt_tagger.txt — the same
text the dataset builder baked into every training row (verified identical, so prompt parity holds).

Usage:
    python sample_onlyfans.py [--img-dir data/mikomiko_tag/onlyfans] [--n 480] [--work-dir DIR]
Next: bash ../infer_tag_2b.sh viz-onlyfans
"""
import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # .../LlamaFactory
DEFAULT_IMG = ROOT / "data/mikomiko_tag/onlyfans"
PROMPT_FILE = ROOT / "data/mikomiko_tag/prompt_tagger.txt"
DEFAULT_WORK = ROOT / "saves/qwen3.5-2b/mikomiko/viz_onlyfans"
TRAIN_JSONL = ROOT / "data/mikomiko_tag/jsonl/train.jsonl"


def build_train_vocab(cache_path):
    """Set of normalized tags the model was trained to emit (~3.4k). Cached: the scan reads 1.1M
    rows. Used by the page to flag out-of-vocabulary predictions on this off-distribution set."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metrics_mikomiko import norm

    if os.path.exists(cache_path):
        return
    vocab = set()
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            for t in json.loads(line)["output"].split(","):
                if (n := norm(t)):
                    vocab.add(n)
    json.dump(sorted(vocab), open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[vocab] {len(vocab)} unique training tags -> {cache_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img-dir", default=str(DEFAULT_IMG))
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--n", type=int, default=None, help="cap image count (default: all)")
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    instruction = "<image>" + PROMPT_FILE.read_text(encoding="utf-8").strip()
    images = sorted(Path(args.img_dir).glob("*.webp"))
    if args.n:
        images = images[: args.n]
    if not images:
        raise SystemExit(f"[onlyfans] no *.webp under {args.img_dir}")

    samples = []
    for p in images:
        stem = p.stem                                   # 'allchargedup__s0'
        creator, _, shot = stem.rpartition("__")
        samples.append(dict(
            split="onlyfans", instruction=instruction, image=str(p), name=p.name,
            creator=creator or stem, shot=shot or "", gemini="",   # no gold exists
            post_tag=[], category=[],
        ))

    out = os.path.join(args.work_dir, "samples.json")
    json.dump(samples, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    creators = len({s["creator"] for s in samples})
    print(f"[onlyfans] {len(samples)} images from {creators} creators -> {out}")
    build_train_vocab(os.path.join(args.work_dir, "train_tag_vocab.json"))


if __name__ == "__main__":
    main()
