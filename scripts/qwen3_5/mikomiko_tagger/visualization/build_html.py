#!/usr/bin/env python3
"""build_html.py — final step of the review page pipeline.

Reads WORK_DIR/samples_pred.json and emits a fully self-contained review HTML: 4 cards per row,
thumbnails embedded as base64 JPEG, click to zoom.

Two modes, auto-detected from whether the samples carry gold tags:
  * SCORED (seen/unseen page) — each card lists post tag / category / gemini(gold) / pred with
    hit(TP)/miss(FN)/fp coloring and per-image F1. Scoring is NOT reimplemented here: every number
    comes from ../metrics_mikomiko.py's per_image() / aggregate(), the same functions
    test_mikomiko.sh and infer_mikomiko.py --score use.
  * UNSCORED (onlyfans page) — no gold exists, so F1 is undefined. The page becomes a distribution
    check: tags/atoms/composites per image, and which predicted tags fall outside the ~3.4k-tag
    training vocabulary (the interesting failure mode off-distribution).

Usage:
    python build_html.py [--work-dir SAVES/viz_review] [--out ..._<date>.html]
                         [--model-name "QWEN 3.5 2B"] [--subtitle "Full SFT · ..."]
"""
import argparse, base64, html, io, json, os, sys
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                                  # .../LlamaFactory
sys.path.insert(0, str(HERE.parent))                    # for metrics_mikomiko
from metrics_mikomiko import norm, per_image, aggregate, tagset  # noqa: E402

from PIL import Image  # noqa: E402

DEFAULT_WORK = ROOT / "saves/qwen3.5-2b/mikomiko/viz_review"
THUMB_W, THUMB_H, JPEG_Q = 440, 560, 80
# Training-set reference points (data/mikomiko_tag/csv_inspect_report.txt + the 400-image eval),
# shown next to the OnlyFans numbers so "12 tags/image" has something to be compared against.
TRAIN_GOLD_TPI, EVAL_PRED_TPI, EVAL_PRED_CPI = 10.9, 11.8, 5.6


def thumb_b64(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((THUMB_W, THUMB_H))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_Q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def taglist(s):
    return [t.strip() for t in s.split(",") if t.strip()]


def chips(tags, cls_of):
    return "".join(f'<span class="chip {cls_of(t)}">{html.escape(t)}</span>' for t in tags) or \
           '<span class="chip none">—</span>'


def f1_cls(v):
    return "good" if v >= 0.50 else ("mid" if v >= 0.30 else "bad")


# ── scored card (seen/unseen) ────────────────────────────────────────────────────────────────
def card_scored(s):
    m = s["m"]
    in_gold = lambda t: norm(t) in m["gold_set"]
    in_pred = lambda t: norm(t) in m["pred_set"]
    post_chips = chips(s["post_tag"], lambda t: "ref kept" if in_gold(t) else "ref")
    cat_chips = chips(s["category"], lambda t: "ref kept" if in_gold(t) else "ref")
    gem_chips = chips(taglist(s["gemini"]), lambda t: "hit" if in_pred(t) else "miss")
    pred_chips = chips(taglist(s.get("pred", "")), lambda t: "hit" if in_gold(t) else "fp")
    tf, kf = m["tagF1"], m["tokF1"]
    return f"""
    <article class="card">
      <div class="thumb"><img src="{s['b64']}" alt="{html.escape(s['name'])}" loading="lazy"
           onclick="showLb(this.src)"></div>
      <div class="card-head">
        <span class="pid mono">{html.escape(s['name'])}</span>
        <span class="badges">
          <span class="badge {f1_cls(tf)}" title="tag 级 P={m['tagP']*100:.1f} R={m['tagR']*100:.1f}">tag F1 {tf*100:.1f}</span>
          <span class="badge {f1_cls(kf)}" title="词级 P={m['tokP']*100:.1f} R={m['tokR']*100:.1f}">词 F1 {kf*100:.1f}</span>
        </span>
      </div>
      <div class="tagblock"><h4>post tag <em>{len(s['post_tag'])}</em></h4><div class="chips">{post_chips}</div></div>
      <div class="tagblock"><h4>分类标签 category <em>{len(s['category'])}</em></h4><div class="chips">{cat_chips}</div></div>
      <div class="tagblock"><h4>gemini 标签（gold）<em>{len(taglist(s['gemini']))}</em></h4><div class="chips">{gem_chips}</div></div>
      <div class="tagblock"><h4>pred 标签 <em>{len(taglist(s.get('pred','')))}</em></h4><div class="chips">{pred_chips}</div></div>
    </article>"""


# ── unscored card (onlyfans: no gold) ────────────────────────────────────────────────────────
def card_unscored(s):
    d = s["d"]
    oov = d["oov"]
    tags = taglist(s.get("pred", ""))
    pred_chips = chips(tags, lambda t: "oov" if norm(t) in oov else "plain")
    oov_badge = (f'<span class="badge bad" title="不在训练词表里的 tag">词表外 {len(oov)}</span>'
                 if oov else '<span class="badge good">词表内</span>')
    return f"""
    <article class="card">
      <div class="thumb"><img src="{s['b64']}" alt="{html.escape(s['name'])}" loading="lazy"
           onclick="showLb(this.src)"></div>
      <div class="card-head">
        <span class="pid mono">{html.escape(s['name'])}</span>
        <span class="badges">
          <span class="badge neutral" title="原子 {d['atom']} + 复合 {d['comp']}">{d['n']} tags</span>
          <span class="badge neutral">复合 {d['comp']}</span>
          {oov_badge}
        </span>
      </div>
      <div class="tagblock"><h4>pred 标签 <em>{d['n']}</em></h4><div class="chips">{pred_chips}</div></div>
    </article>"""


# ── page shell ───────────────────────────────────────────────────────────────────────────────
CSS = """
  :root {
    --bg: #f4eee6; --panel: #ffffff; --panel-soft: #fbf7f1;
    --line: #ead9ca; --line-strong: #c8a891;
    --text: #2f2117; --muted: #806958;
    --accent: #96512b; --accent-soft: #f1dfd2;
    --green: #16a34a; --green-soft: #e3f4e8;
    --orange: #c2410c; --orange-soft: #fbe9dd;
    --red: #b91c1c; --red-soft: #fdeaea;
    --slate: #64748b; --slate-soft: #eef1f5;
    --teal: #0891b2;
    --shadow: 0 18px 42px rgba(76, 45, 20, 0.10);
    --radius: 8px; --radius-sm: 6px;
    --mono: Consolas, Menlo, monospace;
    --sans: "Segoe UI", "PingFang SC", "Microsoft YaHei UI", "Noto Sans SC", sans-serif;
  }
  * { box-sizing: border-box; }
  html { color-scheme: light; }
  body {
    margin: 0; min-height: 100vh;
    background: linear-gradient(90deg, rgba(237,221,203,.74), rgba(250,247,243,.7)), var(--bg);
    color: var(--text); font-family: var(--sans); font-variant-numeric: tabular-nums;
  }
  .mono { font-family: var(--mono); }
  .shell { display: grid; grid-template-columns: 392px minmax(0,1fr); gap: 18px; padding: 18px; }
  .sidebar {
    position: sticky; top: 18px; height: calc(100vh - 36px); overflow: auto;
    padding: 22px; border: 1px solid var(--line); border-radius: 20px;
    background: rgba(255,251,246,.96); box-shadow: var(--shadow);
  }
  .brand h1 { margin: 0 0 8px; font-size: 22px; line-height: 1.25; }
  .brand p { margin: 0 0 6px; color: var(--muted); font-size: 13.5px; line-height: 1.55; }
  .stats { width: 100%; margin-top: 14px; border-collapse: collapse; font-variant-numeric: tabular-nums;
           border: 1px solid var(--line); border-radius: var(--radius-sm); overflow: hidden; }
  .stats th, .stats td { padding: 7px 8px; font-size: 12.5px; border-bottom: 1px solid var(--line); }
  .stats thead th { background: var(--accent-soft); color: var(--accent); font-family: var(--mono);
                    font-size: 11.5px; text-align: right; }
  .stats tbody th { text-align: left; color: var(--text); font-weight: 600; background: var(--panel-soft); }
  .stats tbody th em { font-style: normal; font-family: var(--mono); color: var(--muted); font-size: 11px; }
  .stats tbody td { text-align: right; font-family: var(--mono); font-weight: 700; font-size: 13.5px; }
  .stats tbody td.ref { font-weight: 400; color: var(--muted); }
  .stats tbody tr:last-child th, .stats tbody tr:last-child td { border-bottom: none; }
  .panel { margin-top: 18px; display: grid; gap: 8px; padding-top: 14px; border-top: 1px solid var(--line); }
  .panel h2 { margin: 0; font-size: 14px; color: var(--accent); }
  .panel p, .panel li { margin: 0; color: var(--muted); font-size: 12.5px; line-height: 1.6; }
  .panel ul { margin: 0; padding-left: 16px; }
  .panel .mono { font-size: 11.5px; word-break: break-all; }
  .toptags { display: flex; flex-wrap: wrap; gap: 4px; }
  .toptags .chip b { font-weight: 700; color: var(--accent); margin-left: 4px; }
  .jump a { display: inline-block; margin: 2px 6px 2px 0; padding: 4px 10px; border: 1px solid var(--line-strong);
            border-radius: 999px; color: var(--accent); text-decoration: none; font-size: 12.5px; background: var(--accent-soft); }
  main { display: grid; gap: 24px; align-content: start; }
  section { border: 1px solid var(--line); border-radius: 20px; background: rgba(255,251,246,.85);
            box-shadow: var(--shadow); padding: 20px; }
  .sec-head h2 { margin: 0 0 4px; font-size: 19px; }
  .sec-head p { margin: 0 0 4px; color: var(--muted); font-size: 13px; }
  .sec-stats { color: var(--accent); }
  .grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin-top: 14px; }
  @media (max-width: 1500px) { .grid { grid-template-columns: repeat(3, minmax(0,1fr)); } }
  @media (max-width: 1180px) { .grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
  .card { border: 1px solid var(--line); border-radius: var(--radius); background: var(--panel);
          padding: 10px; display: grid; gap: 8px; align-content: start; }
  .thumb { background: var(--panel-soft); border: 1px solid var(--line); border-radius: var(--radius-sm);
           display: flex; align-items: center; justify-content: center; height: 300px; overflow: hidden; }
  .thumb img { max-width: 100%; max-height: 100%; object-fit: contain; cursor: zoom-in; }
  .card-head { display: flex; flex-direction: column; align-items: flex-start; gap: 6px; }
  .pid { font-size: 11.5px; color: var(--muted); word-break: break-all; }
  .badges { display: flex; gap: 6px; white-space: nowrap; }
  .badge { font-family: var(--mono); font-size: 11.5px; font-weight: 700; padding: 3px 8px;
           border-radius: 999px; border: 1px solid transparent; }
  .badge.good { color: var(--green); background: var(--green-soft); border-color: #bfe5cb; }
  .badge.mid  { color: var(--orange); background: var(--orange-soft); border-color: #f0cdb4; }
  .badge.bad  { color: var(--red); background: var(--red-soft); border-color: #f2c5c5; }
  .badge.neutral { color: var(--slate); background: var(--slate-soft); border-color: #dbe1e8; }
  .tagblock h4 { margin: 0 0 4px; font-size: 12px; color: var(--muted); font-weight: 600; }
  .tagblock h4 em { font-style: normal; font-family: var(--mono); color: var(--line-strong); }
  .chips { display: flex; flex-wrap: wrap; gap: 4px; }
  .chip { font-size: 11px; line-height: 1.35; padding: 2px 7px; border-radius: 999px;
          border: 1px solid var(--line); background: var(--panel-soft); color: var(--text); }
  .chip.plain { color: #14532d; background: var(--green-soft); border-color: #a7d9b9; }
  .chip.oov { color: var(--red); background: var(--red-soft); border-color: #f2c5c5; font-weight: 700; }
  .chip.ref { color: var(--slate); background: var(--slate-soft); border-color: #dbe1e8; }
  .chip.ref.kept { border-color: var(--teal); box-shadow: inset 0 -2px 0 rgba(8,145,178,.35); }
  .chip.hit { color: #14532d; background: var(--green-soft); border-color: #a7d9b9; }
  .chip.miss { color: var(--red); background: var(--panel); border: 1px dashed #e2a3a3; }
  .chip.fp { color: var(--orange); background: var(--orange-soft); border-color: #eec4a5; }
  .chip.none { color: var(--muted); }
  .lightbox { position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
              background: rgba(47,33,23,.82); z-index: 50; cursor: zoom-out; }
  .lightbox img { max-width: 92vw; max-height: 92vh; border-radius: 8px; }
  .lightbox.show { display: flex; }
"""

# PAGE carries an inline <script> whose braces would break str.format, so it is filled by
# straight substitution of the @@NAME@@ placeholders below.
PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>@@TITLE@@</title>
<link rel="icon" href="data:,">
<style>@@CSS@@</style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="brand">
      <h1>@@MODEL_NAME@@</h1>
      <p>@@SUBTITLE@@</p>
      <p>@@TAGLINE@@</p>
      @@JUMP@@
    </div>
    @@STATS@@
    @@PANELS@@
  </aside>
  <main>@@SECTIONS@@</main>
</div>
<div class="lightbox" id="lb" onclick="this.classList.remove('show')"><img id="lbimg" src="" alt="zoom"></div>
<script>
function showLb(src) {
  document.getElementById('lbimg').src = src;
  document.getElementById('lb').classList.add('show');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') document.getElementById('lb').classList.remove('show'); });
</script>
</body>
</html>"""


def render_page(out, title, model_name, subtitle, tagline, jump, stats, panels, sections):
    page = PAGE
    for key, val in (("TITLE", html.escape(title)), ("CSS", CSS), ("MODEL_NAME", html.escape(model_name)),
                     ("SUBTITLE", html.escape(subtitle)), ("TAGLINE", tagline), ("JUMP", jump),
                     ("STATS", stats), ("PANELS", panels), ("SECTIONS", sections)):
        page = page.replace(f"@@{key}@@", val)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w", encoding="utf-8").write(page)
    print(f"[html] {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


# ── mode: scored (seen / unseen) ─────────────────────────────────────────────────────────────
def build_scored(samples, args):
    for s in samples:
        s["m"] = per_image(s["gemini"], s.get("pred", ""), s["split"])
    A = {"all": aggregate([s["m"] for s in samples]),
         "seen": aggregate([s["m"] for s in samples if s["split"] == "seen"]),
         "unseen": aggregate([s["m"] for s in samples if s["split"] == "unseen"])}

    sections = ""
    for sp, title, note in (
        ("seen", "seen — 训练集原样样本", "随机抽自 train.jsonl（110.7 万），训练时逐字见过图和标签"),
        ("unseen", "unseen — post 级零重叠", "随机抽自 test_unseen_mini（200），整个 post 未进过训练"),
    ):
        sub = sorted([s for s in samples if s["split"] == sp], key=lambda x: -x["m"]["tagF1"])
        if not sub:
            continue
        a = A[sp]
        sections += f"""
  <section id="{sp}">
    <div class="sec-head">
      <h2>{title} <span class="mono">n={a['n']}</span></h2>
      <p>{note}，按 tag F1 降序</p>
      <p class="mono sec-stats">microF1 {a['microF1']*100:.1f} · macroF1 {a['macroF1']*100:.1f} ·
         atomF1 {a['atomF1']*100:.1f} · tokF1 {a['tokF1']*100:.1f} · pred/gold 每图 {a['pred_tpi']:.1f}/{a['gold_tpi']:.1f}</p>
    </div>
    <div class="grid">{''.join(card_scored(s) for s in sub)}</div>
  </section>"""

    def row(label, a):
        return (f'<tr><th>{label} <em>n={a["n"]}</em></th>'
                f'<td>{a["microF1"]*100:.1f}</td><td>{a["macroF1"]*100:.1f}</td>'
                f'<td>{a["atomF1"]*100:.1f}</td><td>{a["tokF1"]*100:.1f}</td></tr>')

    stats = f"""
    <table class="stats">
      <thead><tr><th></th><th>microF1</th><th>macroF1</th><th>atomF1</th><th>tokF1</th></tr></thead>
      <tbody>
        {row("全部", A["all"])}
        {row("unseen", A["unseen"]) if A["unseen"] else ""}
        {row("seen", A["seen"]) if A["seen"] else ""}
      </tbody>
    </table>"""

    panels = """
    <div class="panel">
      <h2>怎么读</h2>
      <ul>
        <li><b>gold = gemini 标签</b>（与训练标签同源）。tag 级 = 整 tag 精确匹配、词级 = 拆词后匹配，匹配前均做规范化。</li>
        <li><b>microF1</b> 全组 tag 汇总算分；<b>macroF1</b> 每图 F1 取平均；<b>atomF1</b> 只算单词 tag；<b>tokF1</b> 全部拆词，最宽松。</li>
        <li>gemini 行：<span class="chip hit">绿 = 命中</span> <span class="chip miss">虚线红 = 漏检 FN</span>；pred 行：<span class="chip hit">绿 = 命中</span> <span class="chip fp">橙 = 多说 FP</span></li>
        <li>post tag / 分类行是 post 级参考；<span class="chip ref kept">青色下划线</span> = gemini 判为本图 tag。</li>
        <li>徽章：F1 ≥50 绿、30–50 琥珀、&lt;30 红；悬停看 P/R。</li>
      </ul>
    </div>
    <div class="panel">
      <h2>注意</h2>
      <ul>
        <li>每组样本量小、噪声大，定量以 400 张评测为准，本页用于定性看错误形态。</li>
        <li>gold 标注不一致，FP 未必真错、FN 未必真漏，请对照图片判断。</li>
      </ul>
    </div>"""

    jump = '<div class="jump"><a href="#seen">跳到 seen</a><a href="#unseen">跳到 unseen</a></div>'
    render_page(args.out, "Mikomiko Tagger seen/unseen 抽样审阅", args.model_name, args.subtitle,
                "Mikomiko tagger · seen / unseen 抽样审阅", jump, stats, panels, sections)
    for sp in ("seen", "unseen"):
        if (a := A[sp]):
            print(f"[html] {sp:<7} n={a['n']}  microF1={a['microF1']*100:.1f}  macroF1={a['macroF1']*100:.1f}  "
                  f"atomF1={a['atomF1']*100:.1f}  tokF1={a['tokF1']*100:.1f}  "
                  f"pred/gold tpi={a['pred_tpi']:.1f}/{a['gold_tpi']:.1f}")


# ── mode: unscored (onlyfans, no gold) ───────────────────────────────────────────────────────
def build_unscored(samples, args):
    vocab_path = os.path.join(args.work_dir, "train_tag_vocab.json")
    vocab = set(json.load(open(vocab_path, encoding="utf-8"))) if os.path.exists(vocab_path) else set()
    if not vocab:
        print(f"[warn] {vocab_path} missing -> OOV flagging disabled", file=sys.stderr)

    freq = Counter()
    for s in samples:
        p = tagset(s.get("pred", ""))
        atoms = {t for t in p if " " not in t}
        s["d"] = dict(n=len(p), atom=len(atoms), comp=len(p) - len(atoms),
                      oov={t for t in p if vocab and t not in vocab})
        freq.update(p)

    n = len(samples)
    tot = sum(s["d"]["n"] for s in samples)
    tot_oov = sum(len(s["d"]["oov"]) for s in samples)
    imgs_with_oov = sum(1 for s in samples if s["d"]["oov"])
    uniq_oov = len({t for s in samples for t in s["d"]["oov"]})
    tpi, api = tot / n, sum(s["d"]["atom"] for s in samples) / n
    cpi = sum(s["d"]["comp"] for s in samples) / n
    empty = sum(1 for s in samples if s["d"]["n"] == 0)

    stats = f"""
    <table class="stats">
      <thead><tr><th></th><th>本页</th><th>训练/评测参考</th></tr></thead>
      <tbody>
        <tr><th>图片数 <em>creators {len({s['creator'] for s in samples})}</em></th><td>{n}</td><td class="ref">—</td></tr>
        <tr><th>每图 tag 数</th><td>{tpi:.1f}</td><td class="ref">gold {TRAIN_GOLD_TPI} / pred {EVAL_PRED_TPI}</td></tr>
        <tr><th>每图原子 tag</th><td>{api:.1f}</td><td class="ref">—</td></tr>
        <tr><th>每图复合 tag</th><td>{cpi:.1f}</td><td class="ref">pred {EVAL_PRED_CPI}</td></tr>
        <tr><th>唯一 tag 数</th><td>{len(freq)}</td><td class="ref">词表 {len(vocab) or '—'}</td></tr>
        <tr><th>词表外 tag 占比</th><td>{tot_oov/tot*100 if tot else 0:.1f}%</td><td class="ref">{uniq_oov} 种 / {imgs_with_oov} 张图</td></tr>
        <tr><th>空预测</th><td>{empty}</td><td class="ref">—</td></tr>
      </tbody>
    </table>"""

    top = "".join(f'<span class="chip plain">{html.escape(t)}<b>{c}</b></span>'
                  for t, c in freq.most_common(24))
    panels = f"""
    <div class="panel">
      <h2>怎么读</h2>
      <ul>
        <li><b>这批图没有 gold</b>，F1 无从谈起。本页是分布体检：模型在训练分布之外说了什么、说了多少。</li>
        <li>pred 行：<span class="chip plain">绿 = 在训练词表内</span> <span class="chip oov">红 = 词表外</span>（模型没被训过的写法，多为拼写/截断故障）。</li>
        <li>徽章依次是：总 tag 数（悬停看原子/复合拆分）、复合 tag 数、词表外 tag 数。</li>
        <li>训练集 gold 平均 {TRAIN_GOLD_TPI} tag/图，400 张评测里模型平均预测 {EVAL_PRED_TPI} tag/图（复合 {EVAL_PRED_CPI}）——用来判断这里是否异常多说或少说。</li>
      </ul>
    </div>
    <div class="panel">
      <h2>预测最多的 tag</h2>
      <div class="toptags">{top}</div>
    </div>
    <div class="panel">
      <h2>注意</h2>
      <ul>
        <li>OnlyFans 自拍与训练用的 pornpic 图集在构图、画质、水印上都不同属一个分布，指标只反映"模型说了什么"，不代表对错。</li>
        <li>词表外 tag 值得逐个看：训练标签是 3.4k 的闭集，模型本不该造词。</li>
      </ul>
    </div>"""

    sub = sorted(samples, key=lambda s: (s["creator"], s["name"]))
    sections = f"""
  <section id="onlyfans">
    <div class="sec-head">
      <h2>onlyfans — 无 gold 的分布体检 <span class="mono">n={n}</span></h2>
      <p>按 creator 名排序，同一 creator 的多张图相邻，便于看同人一致性</p>
      <p class="mono sec-stats">每图 {tpi:.1f} tags（原子 {api:.1f} / 复合 {cpi:.1f}）·
         唯一 tag {len(freq)} · 词表外 {tot_oov/tot*100 if tot else 0:.1f}%</p>
    </div>
    <div class="grid">{''.join(card_unscored(s) for s in sub)}</div>
  </section>"""

    render_page(args.out, "Mikomiko Tagger OnlyFans 推理审阅", args.model_name, args.subtitle,
                "Mikomiko tagger · onlyfans 无 gold 推理", "", stats, panels, sections)
    print(f"[html] onlyfans n={n}  tags/img={tpi:.1f} (atom {api:.1f} / comp {cpi:.1f})  "
          f"uniq={len(freq)}  OOV={tot_oov}/{tot} ({tot_oov/tot*100 if tot else 0:.1f}%, {uniq_oov} 种)  empty={empty}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--out", default=str(ROOT / f".ai/agent_todo/mikomiko_tagger_seen_unseen_review_{date.today():%Y%m%d}.html"))
    ap.add_argument("--model-name", default="QWEN 3.5 2B", help="sidebar headline")
    ap.add_argument("--subtitle", default="Full SFT · 2.0 epochs · 17296 steps · temperature 0.0",
                    help="sidebar training info; keep in sync with the served checkpoint")
    args = ap.parse_args()

    samples = json.load(open(os.path.join(args.work_dir, "samples_pred.json"), encoding="utf-8"))
    print(f"[html] encoding {len(samples)} thumbnails ...", flush=True)
    for i, s in enumerate(samples, 1):
        s["b64"] = thumb_b64(s["image"])
        if i % 50 == 0 or i == len(samples):
            print(f"  [html] {i}/{len(samples)}", flush=True)

    if any(s.get("gemini") for s in samples):
        build_scored(samples, args)
    else:
        build_unscored(samples, args)


if __name__ == "__main__":
    main()
