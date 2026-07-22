#!/usr/bin/env python3
"""metrics_desc.py — structural health checks for the 4-section description task.

There is deliberately NO similarity-to-gold score here. The target is free prose: a description
can be completely correct and share almost no wording with the reference, so a BLEU/ROUGE number
would mostly measure phrasing luck and would invite reading it as accuracy. What CAN be checked
mechanically, with no judgement call, is whether the output has the shape the prompt asked for:

  lang     did it answer in the language the prompt block asked for?   <- the load-bearing one
  sections are all 4 required section headers present, in order?
  length   how long is it, next to the reference for the same image?
  rep      is it stuck in a repetition loop?
  finish   did it stop on its own, or run into the token cap?

Everything else on the review page (is the description TRUE of the image?) is a human call, which
is what the page's images are for.

Language is measured twice, on purpose, and the two can disagree:
  header_lang  which language's section headers it emitted  (format memorised)
  body_lang    which script the prose itself is written in  (language actually spoken)
A model that emits Chinese headers over an English body has learned the template but not the
switch; only comparing both catches that.

Used by build_html.py; run standalone to print the same table without rebuilding the page:
    python metrics_desc.py --pred WORK/samples_pred.json
"""
import argparse
import json
import re
from collections import Counter

LANGS = ("en", "ja", "zh")
# Keep in sync with data/mikomiko_tag/dataset_builder_desc_0721.py:HEADERS.
HEADERS = {
    "en": ("Creative Intent", "Foreground and Subject",
           "Background and Environment", "Photography Techniques and Visual Presentation"),
    "ja": ("創作意図", "前景と主要な被写体", "背景と周囲の環境", "撮影技法と視覚的表現"),
    "zh": ("创作意图", "前景与主体", "背景与环境", "摄影技术与视觉呈现"),
}
LANG_LABEL = {"en": "English", "ja": "日本語", "zh": "中文", "other": "无法判定", None: "无"}

REP_N = 40                      # same gram size the dataset builder filtered on
REP_BAD = 0.3                   # builder's MAX_REP: rows at/above this were dropped from training

_KANA = re.compile(r"[぀-ヿ]")
_CJK = re.compile(r"[一-鿿]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_ALL_HEADERS = [h for hs in HEADERS.values() for h in hs]
_THINK = re.compile(r"\A\s*<think>(.*?)</think>", re.S)


def strip_think(text):
    """Split a completion into (answer, reasoning). Reasoning is '' when there is none.

    Qwen3.5 is a thinking model. Both models here are served the LlamaFactory prompt, which ends
    right after 'assistant\\n' with no think block -- so the base model opens one ITSELF on every
    single sample, while the SFT model never does. Counting that reasoning as part of the answer
    would corrupt every metric below: it inflates the character count, and because the reasoning
    is usually English it can flip body_lang on a Chinese or Japanese sample.

    The reasoning is not thrown away, only separated -- the page displays it, because "the base
    model spends its output budget thinking" is a finding, not noise. One case has no closing
    tag at all (it reasoned until it hit the token cap and never wrote an answer); that yields
    answer='' , which is exactly right -- there was no answer.

    Returns (answer, reasoning, had_block). had_block is tracked separately from a non-empty
    reasoning string because an EMPTY <think></think> pair is the common case (113/120 base
    samples) and still means the model went into thinking mode.
    """
    text = text or ""
    m = _THINK.search(text)
    if m:
        return text[m.end():].strip(), m.group(1).strip(), True
    if text.lstrip().startswith("<think>"):
        return "", text.lstrip()[len("<think>"):].strip(), True
    return text.strip(), "", False


def repetition_score(s):
    """Fraction of the text covered by its most frequent non-overlapping REP_N-gram.

    Verbatim from the dataset builder so the page's numbers are on the same scale as the
    filter that shaped the training data: healthy prose <0.05, a stuck loop approaches 1.0.
    """
    if len(s) < REP_N * 3:
        return 0.0
    grams = Counter(s[i:i + REP_N] for i in range(0, len(s) - REP_N, REP_N))
    return grams.most_common(1)[0][1] * REP_N / len(s)


def header_lang(text):
    """Which language's 4 headers are ALL present? None if zero or more than one match."""
    hit = [lg for lg, hs in HEADERS.items() if all(h in text for h in hs)]
    return hit[0] if len(hit) == 1 else None


def n_sections(text, lang):
    """How many of `lang`'s 4 required headers appear at all (0-4)."""
    return sum(1 for h in HEADERS[lang] if h in text)


def sections_ordered(text, lang):
    """All 4 headers present AND in the prompt's order."""
    pos = [text.find(h) for h in HEADERS[lang]]
    return all(p >= 0 for p in pos) and pos == sorted(pos)


def body_lang(text):
    """Which language is the PROSE in, ignoring the section headers.

    Headers are removed first: they are language-specific by construction, so leaving them in
    would let a model score as 'Chinese' purely for copying Chinese headers onto English prose --
    exactly the failure this function exists to detect.

    The test is CJK-vs-Latin FIRST, ja-vs-zh second. Doing it the other way round -- "any kana
    means Japanese" -- misfires on the real data: these descriptions quote on-image watermarks
    verbatim, and one English gold row quotes a Japanese studio watermark ("カリビアンコム",
    7 kana against 367 English words), which a kana-first rule calls Japanese. Comparing Latin
    WORDS against CJK CHARACTERS is the right scale, since one CJK char carries roughly what one
    short English word does; only once CJK wins does kana decide Japanese vs Chinese (Chinese
    prose contains no kana at all).
    """
    for h in _ALL_HEADERS:
        text = text.replace(h, " ")
    kana = len(_KANA.findall(text))
    cjk = len(_CJK.findall(text))
    words = len(_LATIN_WORD.findall(text))
    if kana + cjk + words == 0:
        return "other"
    if kana + cjk <= words:                       # Latin prose, whatever it happens to quote
        return "en" if words else "other"
    return "ja" if kana / (kana + cjk) > 0.05 else "zh"


def per_row(sample, tag):
    """Structural metrics for one prediction. `tag` selects pred_<tag>/finish_<tag>/ntok_<tag>."""
    lang = sample["lang"]
    text, think, had_think = strip_think(sample.get(f"pred_{tag}"))
    gold = (sample.get("gold") or "").strip()
    bl = body_lang(text) if text else "other"
    return {
        "empty": not text,
        "think_chars": len(think),
        "has_think": had_think,
        "chars": len(text),
        "gold_chars": len(gold),
        "len_ratio": (len(text) / len(gold)) if gold else 0.0,
        "ntok": sample.get(f"ntok_{tag}", 0),
        "finish": sample.get(f"finish_{tag}", "?"),
        "capped": sample.get(f"finish_{tag}") == "length",
        "body_lang": bl,
        "lang_ok": bl == lang,
        "header_lang": header_lang(text),
        "header_lang_ok": header_lang(text) == lang,
        "n_sections": n_sections(text, lang) if text else 0,
        "sec4": n_sections(text, lang) == 4 if text else False,
        "ordered": sections_ordered(text, lang) if text else False,
        "rep": round(repetition_score(text), 3),
        "rep_bad": repetition_score(text) >= REP_BAD,
    }


def gold_row(sample):
    """The same checks run on the REFERENCE text.

    Two jobs. (1) It calibrates the detector: gold's language is known from the dataset, so
    gold lang_ok below 100% means body_lang() is wrong, not the model. (2) It gives every
    'pred is 800 chars' a same-image reference point instead of an absolute scale nobody has
    intuition for.
    """
    lang = sample["lang"]
    text = (sample.get("gold") or "").strip()
    bl = body_lang(text)
    return {
        "chars": len(text), "body_lang": bl, "lang_ok": bl == lang,
        "header_lang": header_lang(text), "n_sections": n_sections(text, lang),
        "sec4": n_sections(text, lang) == 4, "ordered": sections_ordered(text, lang),
        "rep": round(repetition_score(text), 3),
    }


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def aggregate(ms):
    """Group summary. Rates are percentages over the rows handed in."""
    n = len(ms)
    if not n:
        return None
    pct = lambda f: 100.0 * sum(1 for m in ms if f(m)) / n
    return {
        "n": n,
        "lang_ok": pct(lambda m: m["lang_ok"]),
        "header_ok": pct(lambda m: m.get("header_lang_ok", m.get("header_lang") == m.get("body_lang"))),
        "sec4": pct(lambda m: m["sec4"]),
        "ordered": pct(lambda m: m["ordered"]),
        "chars": _median(m["chars"] for m in ms),
        "gold_chars": _median(m.get("gold_chars", 0) for m in ms),
        "len_ratio": round(sorted(m.get("len_ratio", 0) for m in ms)[n // 2], 2),
        "rep_bad": pct(lambda m: m.get("rep_bad", False)),
        "capped": pct(lambda m: m.get("capped", False)),
        "empty": sum(1 for m in ms if m.get("empty")),
        "think": pct(lambda m: m.get("has_think", False)),
        "think_chars": _median(m.get("think_chars", 0) for m in ms),
    }


def print_table(samples, tags):
    """The same numbers the page's sidebar shows, as text, for the pipeline log."""
    golds = [gold_row(s) for s in samples]
    g_ok = 100.0 * sum(1 for g in golds if g["lang_ok"]) / len(golds)
    g_sec = 100.0 * sum(1 for g in golds if g["sec4"]) / len(golds)
    print(f"\n[metrics] detector calibration on gold (n={len(golds)}): "
          f"body_lang correct {g_ok:.1f}%  ·  4 sections present {g_sec:.1f}%")
    if g_ok < 100.0:
        bad = [(s["name"], s["lang"], g["body_lang"]) for s, g in zip(samples, golds) if not g["lang_ok"]]
        print(f"[metrics] WARNING detector missed on gold: {bad[:5]}")

    hdr = f"{'model':<6} {'group':<14} {'n':>4} {'lang':>7} {'hdr':>7} {'4sec':>7} {'order':>7} " \
          f"{'chars':>7} {'gold':>7} {'x':>5} {'rep!':>6} {'cap':>6} {'think':>7} {'thchar':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for tag in tags:
        ms = [per_row(s, tag) for s in samples]
        groups = [("all", lambda s: True)]
        groups += [(f"{sp}", (lambda sp: lambda s: s["split"] == sp)(sp)) for sp in ("seen", "unseen")]
        groups += [(f"{sp}/{lg}", (lambda sp, lg: lambda s: s["split"] == sp and s["lang"] == lg)(sp, lg))
                   for sp in ("seen", "unseen") for lg in LANGS]
        for label, f in groups:
            a = aggregate([m for m, s in zip(ms, samples) if f(s)])
            if not a:
                continue
            print(f"{tag:<6} {label:<14} {a['n']:>4} {a['lang_ok']:>6.1f}% {a['header_ok']:>6.1f}% "
                  f"{a['sec4']:>6.1f}% {a['ordered']:>6.1f}% {a['chars']:>7} {a['gold_chars']:>7} "
                  f"{a['len_ratio']:>5} {a['rep_bad']:>5.1f}% {a['capped']:>5.1f}% "
                  f"{a['think']:>6.1f}% {a['think_chars']:>7}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="WORK/samples_pred.json")
    ap.add_argument("--tags", default="sft,base")
    args = ap.parse_args()
    samples = json.load(open(args.pred, encoding="utf-8"))
    tags = [t for t in args.tags.split(",") if t and any(f"pred_{t}" in s for s in samples)]
    print_table(samples, tags)


if __name__ == "__main__":
    main()
