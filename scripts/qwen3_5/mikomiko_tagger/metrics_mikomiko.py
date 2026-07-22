#!/usr/bin/env python3
"""
metrics_mikomiko.py — the single scorer for the mikomiko image->tag task.

Every consumer (infer_tag_2b.sh eval/seen, infer_mikomiko.py --score, the review page in visualization/)
goes through per_image() + aggregate() here, so the numbers are identical by construction.
All metrics are on normalized, order-invariant tag SETS.

  KEPT   : micro P/R/F1, macro F1, atomF1 (1-word), compEx (exact >=2-word), compSub (concept
           coverage), tokF1 (word-level) + over-generation diagnostic (pred vs gold tags/img).
  DROPPED: BLEU-4 / ROUGE-* — seq2seq metrics, meaningless for an unordered tag list.

API:  per_image(gold, pred, src) -> per-image counts + tag/word P,R,F1  (one dict per image)
      aggregate(rows)            -> micro/macro/atom/comp/tok F1 over a list of those dicts
      score(...)                 -> the CLI path: read predictions, group, print, persist

CLI:  python metrics_mikomiko.py PRED.jsonl META.jsonl STEP HISTORY.tsv METRICS.json
  PRED : jsonl with {"label","predict"} per line, aligned to META order.
  META : the eval set jsonl (carries "_src" for unseen/stratified grouping).
"""
import json, re, os, sys
from datetime import datetime


def norm(t):
    """lowercase, drop punctuation, collapse whitespace -> canonical tag (order/spacing/case-proof)."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t.strip().lower()).split())


def tagset(s):
    return {n for t in s.split(",") if (n := norm(t))}


def words(ts):
    out = set()
    for t in ts:
        out.update(t.split())
    return out


def prf(tp, npred, ngt):
    p = tp / npred if npred else 0.0
    r = tp / ngt if ngt else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def per_image(gold, pred, src="?"):
    """Score one image. Returns the counts `aggregate` needs plus the per-image tag/word P,R,F1
    the review page shows on each card (`tagP/tagR/tagF1`, `tokP/tokR/tokF1`)."""
    g, p = tagset(gold), tagset(pred)
    ga, pa = {t for t in g if " " not in t}, {t for t in p if " " not in t}
    gc, pc = {t for t in g if " " in t}, {t for t in p if " " in t}
    gw, pw = words(g), words(p)
    tagP, tagR, tagF1 = prf(len(g & p), len(p), len(g))
    tokP, tokR, tokF1 = prf(len(gw & pw), len(pw), len(gw))
    return dict(
        src=src,
        tp=len(g & p), npred=len(p), ngt=len(g),
        atp=len(ga & pa), anp=len(pa), ang=len(ga),
        ctp=len(gc & pc), cnp=len(pc), cng=len(gc),
        cs_tp_r=sum(1 for t in gc if set(t.split()) <= pw),
        cs_tp_p=sum(1 for t in pc if set(t.split()) <= gw),
        ttp=len(gw & pw), tnp=len(pw), tng=len(gw),
        npred_all=len(p), ngt_all=len(g), npred_c=len(pc), ngt_c=len(gc),
        tagP=tagP, tagR=tagR, tagF1=tagF1, tokP=tokP, tokR=tokR, tokF1=tokF1,
        gold_set=g, pred_set=p,
    )


def aggregate(sub):
    """Micro/macro/atom/comp/tok F1 over a list of per_image() dicts. None if the list is empty."""
    if not sub:
        return None
    S = lambda k: sum(r[k] for r in sub)
    mf = lambda a, b, c: prf(S(a), S(b), S(c))
    miP, miR, miF = mf("tp", "npred", "ngt")
    maF = sum(r["tagF1"] for r in sub) / len(sub)
    csP = S("cs_tp_p") / S("cnp") if S("cnp") else 0.0
    csR = S("cs_tp_r") / S("cng") if S("cng") else 0.0
    csF = 2 * csP * csR / (csP + csR) if (csP + csR) else 0.0
    return dict(
        n=len(sub), microP=miP, microR=miR, microF1=miF, macroF1=maF,
        atomF1=mf("atp", "anp", "ang")[2], compF1_exact=mf("ctp", "cnp", "cng")[2],
        compF1_subset=csF, tokF1=mf("ttp", "tnp", "tng")[2],
        macroTokF1=sum(r["tokF1"] for r in sub) / len(sub),
        pred_tpi=S("npred_all") / len(sub), gold_tpi=S("ngt_all") / len(sub),
        pred_cpi=S("npred_c") / len(sub), gold_cpi=S("ngt_c") / len(sub),
    )


def score(pred_path, meta_path, step, history_file, metrics_out):
    preds = [json.loads(l) for l in open(pred_path, encoding="utf-8")]
    try:
        meta = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
    except Exception:
        meta = []
    use_meta = len(meta) == len(preds)
    if not use_meta:
        print(f"[warn] meta({len(meta)}) != preds({len(preds)}) -> group split disabled, ALL-only.")

    rows = [per_image(pr.get("label", ""), pr.get("predict", ""),
                      meta[i].get("_src", "?") if use_meta else "?")
            for i, pr in enumerate(preds)]

    groups = {"ALL": lambda r: True,
              "unseen": lambda r: r["src"] == "unseen",
              "stratified": lambda r: r["src"] == "strat"}
    results = {name: aggregate([r for r in rows if sel(r)]) for name, sel in groups.items()}

    print("\n" + "=" * 118)
    print(f"mikomiko image->tag  eval @ step {step}   (n={len(rows)})")
    print("[dropped] BLEU-4 / ROUGE-* -> IGNORED (seq2seq metrics, invalid for unordered tags)")
    print("=" * 118)
    print(f"{'group':<12}{'n':>5}{'microP':>8}{'microR':>8}{'microF1':>8}{'macroF1':>8}"
          f"{'atomF1':>8}{'compEx':>8}{'compSub':>9}{'tokF1':>8}{'pred/img':>9}{'gold/img':>9}{'pC/img':>8}{'gC/img':>8}")
    for name in groups:
        a = results[name]
        if a:
            print(f"{name:<12}{a['n']:>5}{a['microP']*100:>7.1f}{a['microR']*100:>8.1f}{a['microF1']*100:>8.1f}"
                  f"{a['macroF1']*100:>8.1f}{a['atomF1']*100:>8.1f}{a['compF1_exact']*100:>8.1f}"
                  f"{a['compF1_subset']*100:>9.1f}{a['tokF1']*100:>8.1f}"
                  f"{a['pred_tpi']:>9.1f}{a['gold_tpi']:>9.1f}{a['pred_cpi']:>8.1f}{a['gold_cpi']:>8.1f}")
    print("-" * 118)
    print("compEx=exact composite set match | compSub=composite concept coverage (all words present on other side)")
    print("pred/img vs gold/img = over-generation check (pC/gC = composites/img). Model over-tags if pred>>gold.")

    ts = datetime.now().isoformat(timespec="seconds")
    json.dump({"step": step, "time": ts, "n": len(rows), "groups": results},
              open(metrics_out, "w"), ensure_ascii=False, indent=2)

    new = not os.path.exists(history_file)
    A = results["ALL"]; U = results.get("unseen"); Sg = results.get("stratified")
    with open(history_file, "a") as f:
        if new:
            f.write("step\ttime\tn\tmicroP\tmicroR\tmicroF1\tmacroF1\tatomF1\tcompEx\tcompSub\ttokF1"
                    "\tunseen_F1\tstrat_F1\tpred_tpi\tgold_tpi\n")
        f.write(f"{step}\t{ts}\t{A['n']}\t{A['microP']*100:.1f}\t{A['microR']*100:.1f}\t{A['microF1']*100:.1f}"
                f"\t{A['macroF1']*100:.1f}\t{A['atomF1']*100:.1f}\t{A['compF1_exact']*100:.1f}"
                f"\t{A['compF1_subset']*100:.1f}\t{A['tokF1']*100:.1f}"
                f"\t{(U['microF1']*100 if U else 0):.1f}\t{(Sg['microF1']*100 if Sg else 0):.1f}"
                f"\t{A['pred_tpi']:.1f}\t{A['gold_tpi']:.1f}\n")
    print(f"\n[metrics] history appended -> {history_file}")
    return results


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)
    score(*sys.argv[1:6])
