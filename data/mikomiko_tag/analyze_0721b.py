#!/usr/bin/env python3
"""Second pass on effective_20260721_llm_training.parquet:
  - the 6 rows with desc_lang=None: what are they?
  - desc_desc sub-field null/empty rates, and desc_desc.lang vs desc_lang agreement
  - does raw_output always carry the 4 section headers of its language?
  - the >=9000-char tail: degenerate repetition?
  - real token length with the Qwen3.5 tokenizer (sampled) -> pick cutoff_len
"""
import os
import re
import sys
from collections import Counter

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "RAW", "effective_20260721_llm_training.parquet")
PROMPT = os.path.join(HERE, "prompt_grokv8.txt")

# the 4 section headers each language's prompt asks for
HEADERS = {
    "en": ["Creative Intent", "Foreground and Subject",
           "Background and Environment", "Photography Techniques and Visual Presentation"],
    "ja": ["創作意図", "前景と主要な被写体", "背景と周囲の環境", "撮影技法と視覚的表現"],
    "zh": ["创作意图", "前景与主体", "背景与环境", "摄影技术与视觉呈现"],
}
SUBFIELDS = ["intent", "foreground_subject", "background_env", "photo_technique"]


def repetition_score(s, n=40):
    """Fraction of the text covered by the single most frequent n-gram. ~1.0 => a loop."""
    if len(s) < n * 3:
        return 0.0
    grams = Counter(s[i:i + n] for i in range(0, len(s) - n, n))
    top = grams.most_common(1)[0][1]
    return top * n / len(s)


def main():
    pf = pq.ParquetFile(PARQUET)
    cols = ["post_id", "image_name", "desc_lang", "desc_desc", "raw_output"]

    none_rows = []
    sub_missing = Counter()
    lang_mismatch = 0
    header_ok = Counter()
    header_bad_examples = []
    long_rows = []
    sample_texts = []          # (lang, prompt_len, out_len) for tokenizer estimate
    n = 0

    for b in pf.iter_batches(batch_size=100_000, columns=cols):
        d = b.to_pydict()
        for pid, img, lg, dd, raw in zip(d["post_id"], d["image_name"],
                                         d["desc_lang"], d["desc_desc"], d["raw_output"]):
            n += 1
            if lg is None:
                none_rows.append((pid, img, dd, (raw or "")[:400]))
                continue
            dd = dd or {}
            for f in SUBFIELDS:
                if not (dd.get(f) or "").strip():
                    sub_missing[f] += 1
            if (dd.get("lang") or "") != lg:
                lang_mismatch += 1
            hs = HEADERS.get(lg, [])
            hit = sum(1 for h in hs if h in (raw or ""))
            header_ok[(lg, hit)] += 1
            if hit < 4 and len(header_bad_examples) < 6:
                header_bad_examples.append((lg, hit, f"{pid}_{img}", (raw or "")[:200]))
            if raw and len(raw) >= 9000:
                long_rows.append((f"{pid}_{img}", lg, len(raw), repetition_score(raw)))
            if n % 7919 == 0 and len(sample_texts) < 1500:
                sample_texts.append((lg, raw or ""))

    print(f"scanned={n:,}\n")

    print(f"[desc_lang=None] {len(none_rows)} rows")
    for r in none_rows:
        print(f"  {r[0]}_{r[1]}  desc_desc={r[2]}  raw[:400]={r[3]!r}")

    print(f"\n[desc_desc empty sub-fields] {dict(sub_missing)}")
    print(f"[desc_desc.lang != desc_lang] {lang_mismatch:,}")

    print("\n[raw_output section headers present, by lang]")
    for lg in ("en", "ja", "zh"):
        tot = sum(v for (l, _), v in header_ok.items() if l == lg)
        if not tot:
            continue
        row = {h: header_ok.get((lg, h), 0) for h in range(5)}
        print(f"  {lg}: total={tot:,}  4/4={row[4]:,} ({row[4]/tot:.2%})  "
              f"3={row[3]:,} 2={row[2]:,} 1={row[1]:,} 0={row[0]:,}")
    for e in header_bad_examples:
        print(f"    bad: lang={e[0]} hit={e[1]} {e[2]} raw[:200]={e[3]!r}")

    print(f"\n[rows with raw_output >= 9000 chars] {len(long_rows):,}")
    long_rows.sort(key=lambda r: -r[2])
    hi_rep = [r for r in long_rows if r[3] > 0.3]
    print(f"  of which repetition_score>0.3 (likely degenerate loop): {len(hi_rep):,}")
    for r in long_rows[:8]:
        print(f"    {r[0]} lang={r[1]} len={r[2]:,} rep={r[3]:.2f}")

    # ---- tokenizer estimate ----
    print("\n[token length] loading Qwen3.5 tokenizer ...", flush=True)
    try:
        from transformers import AutoTokenizer
        mp = os.path.join(HERE, "..", "..", "..", "hf_download", "models", "Qwen3.5-9B")
        mp = os.path.abspath(mp)
        tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        blocks = open(PROMPT, encoding="utf-8").read().split("\n\n\n")
        pmap = dict(zip(("en", "ja", "zh"), [b.strip() for b in blocks]))
        plen = {k: len(tok(v).input_ids) for k, v in pmap.items()}
        print(f"  prompt tokens per lang: {plen}")
        by_lang = {}
        for lg, raw in sample_texts:
            by_lang.setdefault(lg, []).append(len(tok(raw).input_ids))
        for lg, xs in sorted(by_lang.items()):
            xs.sort()
            q = lambda p: xs[min(len(xs) - 1, int(len(xs) * p))]
            print(f"  {lg}: n={len(xs)} output tokens  p50={q(.5)}  p90={q(.9)}  "
                  f"p99={q(.99)}  max={xs[-1]}  | +prompt p99 = {q(.99) + plen[lg]}")
    except Exception as e:
        print(f"  tokenizer step skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
