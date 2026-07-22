#!/usr/bin/env python3
"""build_html.py — step 3/3: the self-contained grok_desc review page.

Reads WORK_DIR/samples_pred.json and emits one HTML file with everything inside it (thumbnails as
base64 JPEG, all text, all metrics) so it can be sent as an attachment and opened offline.

Layout follows data/mikomiko_tag/jsonl_desc_0721/xhs_1m_category_sample_review_20260708.html: the
data rides along as a JSON <script> block and the page renders itself, so language / split /
"only show me the broken ones" filtering, search and pagination are live rather than baked in.
The sidebar's health table recomputes against the CURRENT filter, which is the point -- "language
correct 97%" over all 120 samples hides that it might be 100/100/91 across en/ja/zh.

Per sample the page puts the image next to gold, the SFT prediction and the untuned base
prediction, in two reading modes:
  并排全文    three columns of full text, for judging fluency and whether it matches the image
  分段对照    the 4 required sections aligned in a row each, for judging structure and coverage

Scoring lives in metrics_desc.py; this file renders, it does not judge.

Usage:
    python build_html.py --work-dir SAVES/viz_desc_0721 --out review.html
"""
import argparse
import base64
import io
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                                   # .../LlamaFactory
sys.path.insert(0, str(HERE))
from metrics_desc import (HEADERS, LANGS, gold_row, header_lang, per_row, strip_think)  # noqa: E402

from PIL import Image  # noqa: E402

DEFAULT_WORK = ROOT / "saves/qwen3.5-9b/mikomiko/viz_desc_0721"
THUMB_W, THUMB_H, JPEG_Q = 420, 540, 80


def thumb_b64(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((THUMB_W, THUMB_H))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_Q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def split_sections(text, lang):
    """Cut a description into its 4 sections. None when the format is not there to cut on.

    Splits on whichever language's headers the text ACTUALLY carries, not on the language it was
    supposed to answer in: a prediction that emitted Chinese headers for an English prompt is a
    finding worth displaying section-by-section, not a parse failure.
    """
    text = (text or "").strip()
    if not text:
        return None
    hs = HEADERS[header_lang(text) or lang]
    pos = [text.find(h) for h in hs]
    if any(p < 0 for p in pos) or pos != sorted(pos):
        return None
    out = []
    for i, (p, h) in enumerate(zip(pos, hs)):
        start = p + len(h)
        end = pos[i + 1] if i + 1 < len(hs) else len(text)
        body = text[start:end]
        body = re.sub(r"^\**\s*\n?", "", body)           # drop the closing ** of the header line
        out.append({"title": h, "body": body.strip()})
    return out


def build_payload(samples, tags, args):
    rows = []
    for i, s in enumerate(samples):
        texts = {}
        g = gold_row(s)
        texts["gold"] = {"text": s.get("gold", ""), "think": "",
                         "sections": split_sections(s.get("gold", ""), s["lang"]),
                         "m": {"chars": g["chars"], "body_lang": g["body_lang"], "lang_ok": g["lang_ok"],
                               "n_sections": g["n_sections"], "sec4": g["sec4"], "ordered": g["ordered"],
                               "rep": g["rep"], "capped": False, "empty": not s.get("gold"),
                               "has_think": False, "think_chars": 0}}
        for t in tags:
            # The displayed text is the ANSWER, with any reasoning split off and shown separately
            # (see metrics_desc.strip_think) -- otherwise the base model's <think> block reads as
            # part of its description and every length on the page is wrong.
            answer, think, _ = strip_think(s.get(f"pred_{t}", ""))
            texts[t] = {"text": answer, "think": think,
                        "sections": split_sections(answer, s["lang"]), "m": per_row(s, t)}
        rows.append({"i": i, "name": s["name"], "post_id": s["post_id"], "split": s["split"],
                     "lang": s["lang"], "img": s["b64"], "texts": texts})

    cols = [{"key": "gold", "label": args.gold_label, "cls": "gold"}]
    cols += [{"key": t, "label": lbl, "cls": t} for t, lbl in
             zip(tags, [args.sft_label, args.base_label])]
    return {
        "title": args.title, "subtitle": args.subtitle, "note": args.note,
        "generated_at": str(date.today()), "columns": cols,
        "lang_order": list(LANGS), "split_order": ["seen", "unseen"],
        "samples": rows, "page_size": args.page_size,
    }


# ── page shell ───────────────────────────────────────────────────────────────────────────────
CSS = """
  :root {
    --bg:#f4eee6; --panel:#fff; --panel-soft:#fbf7f1;
    --line:#ead9ca; --line-strong:#c8a891;
    --text:#2f2117; --muted:#806958;
    --accent:#96512b; --accent-soft:#f1dfd2;
    --green:#16a34a; --green-soft:#e3f4e8;
    --orange:#c2410c; --orange-soft:#fbe9dd;
    --red:#b91c1c; --red-soft:#fdeaea;
    --slate:#64748b; --slate-soft:#eef1f5;
    --blue:#2563eb; --teal:#0891b2; --violet:#9333ea;
    --shadow:0 18px 42px rgba(76,45,20,.10);
    --radius:8px; --radius-sm:6px;
    --mono:Consolas,Menlo,monospace;
    --sans:"Segoe UI","PingFang SC","Microsoft YaHei UI","Noto Sans SC",sans-serif;
  }
  * { box-sizing:border-box; }
  html { color-scheme:light; }
  body { margin:0; min-height:100vh; color:var(--text); font-family:var(--sans);
         font-variant-numeric:tabular-nums;
         background:linear-gradient(90deg,rgba(237,221,203,.74),rgba(250,247,243,.7)),var(--bg); }
  button,input { font:inherit; }
  button { cursor:pointer; }
  button:focus-visible,input:focus-visible { outline:3px solid rgba(37,99,235,.28); outline-offset:2px; }
  .mono { font-family:var(--mono); }

  .shell { min-height:100vh; display:grid; grid-template-columns:392px minmax(0,1fr); gap:18px; padding:18px; }
  .sidebar { position:sticky; top:18px; height:calc(100vh - 36px); overflow:auto; padding:22px;
             border:1px solid var(--line); border-radius:20px; background:rgba(255,251,246,.96);
             box-shadow:var(--shadow); }
  .brand { display:grid; gap:8px; padding-bottom:16px; border-bottom:1px solid var(--line); }
  .brand h1 { margin:0; font-size:22px; line-height:1.25; }
  .brand p { margin:0; color:var(--muted); font-size:13px; line-height:1.55; }
  .meta-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:14px; }
  .metric { border:1px solid var(--line); border-radius:var(--radius-sm); background:var(--panel-soft); padding:10px; }
  .metric .value { display:block; font-size:19px; font-weight:700; }
  .metric .label { display:block; margin-top:3px; color:var(--muted); font-size:12px; }

  .panel { margin-top:18px; display:grid; gap:10px; padding-top:14px; border-top:1px solid var(--line); }
  .panel h2 { margin:0; font-size:14px; color:var(--accent); }
  .panel p, .panel li { margin:0; color:var(--muted); font-size:12.5px; line-height:1.6; }
  .panel ul { margin:0; padding-left:16px; }

  .stats { width:100%; border-collapse:collapse; border:1px solid var(--line);
           border-radius:var(--radius-sm); overflow:hidden; }
  .stats th,.stats td { padding:6px 7px; font-size:12px; border-bottom:1px solid var(--line); }
  .stats thead th { background:var(--accent-soft); color:var(--accent); font-family:var(--mono);
                    font-size:11px; text-align:right; }
  .stats thead th:first-child { text-align:left; }
  .stats tbody th { text-align:left; font-weight:600; background:var(--panel-soft); white-space:nowrap; }
  .stats tbody td { text-align:right; font-family:var(--mono); font-weight:700; font-size:12.5px; }
  .stats tbody tr:last-child th,.stats tbody tr:last-child td { border-bottom:none; }
  .stats td.good { color:var(--green); } .stats td.mid { color:var(--orange); } .stats td.bad { color:var(--red); }
  .stats td.plain { font-weight:400; color:var(--muted); }

  .btn-grid { display:grid; gap:8px; }
  .btn-grid.two { grid-template-columns:1fr 1fr; }
  .btn-grid.three { grid-template-columns:repeat(3,1fr); }
  .pill { min-height:38px; border:1px solid var(--line); border-radius:999px; background:#fff;
          color:var(--text); padding:7px 10px; text-align:center; font-size:12.5px;
          transition:border-color .18s ease,background .18s ease; }
  .pill:hover,.pill.active { border-color:var(--accent); background:var(--accent-soft); }
  .mode-button { border:1px solid var(--line); border-radius:var(--radius-sm); background:#fff;
                 color:var(--text); padding:9px 10px; text-align:left; }
  .mode-button strong { display:block; font-size:13px; }
  .mode-button span { display:block; margin-top:3px; color:var(--muted); font-size:11.5px; line-height:1.4; }
  .mode-button:hover,.mode-button.active { border-color:var(--accent); background:var(--accent-soft); }
  .search input { width:100%; min-height:40px; padding:8px 12px; border:1px solid var(--line);
                  border-radius:999px; background:#fff; }
  .page-row { display:grid; grid-template-columns:1fr auto 1fr; gap:8px; align-items:center; }
  .page-jump { display:flex; gap:6px; align-items:center; }
  .page-input { width:70px; min-height:38px; padding:6px 8px; border:1px solid var(--line);
                border-radius:999px; text-align:center; }
  .pill:disabled { cursor:not-allowed; color:#b59d8b; background:var(--panel-soft); border-color:var(--line); }
  .page-summary { color:var(--muted); font-size:12px; }

  main { display:grid; gap:18px; align-content:start; }
  .hero { display:flex; justify-content:space-between; align-items:center; gap:16px;
          border:1px solid var(--line); border-radius:20px; background:rgba(255,251,246,.85);
          box-shadow:var(--shadow); padding:18px 20px; }
  .hero h2 { margin:0 0 4px; font-size:18px; }
  .hero p { margin:0; color:var(--muted); font-size:13px; }
  .status-pill { white-space:nowrap; padding:8px 14px; border-radius:999px; background:var(--accent-soft);
                 color:var(--accent); font-family:var(--mono); font-size:12.5px; font-weight:700; }

  .sample-row { display:grid; grid-template-columns:300px minmax(0,1fr); gap:16px;
                border:1px solid var(--line); border-left:5px solid var(--color,var(--slate));
                border-radius:16px; background:rgba(255,251,246,.9); box-shadow:var(--shadow); padding:16px; }
  @media (max-width:1100px) { .sample-row { grid-template-columns:minmax(0,1fr); } }
  .row-media { display:grid; gap:10px; align-content:start; position:sticky; top:18px; }
  .thumb { background:var(--panel-soft); border:1px solid var(--line); border-radius:var(--radius-sm);
           display:flex; align-items:center; justify-content:center; overflow:hidden; }
  .thumb img { max-width:100%; max-height:420px; object-fit:contain; cursor:zoom-in; display:block; }
  .sample-title strong { display:block; font-size:13px; word-break:break-all; }
  .sample-title span { display:block; margin-top:2px; color:var(--muted); font-size:11.5px; font-family:var(--mono); }
  .chips { display:flex; flex-wrap:wrap; gap:5px; }
  .chip { font-size:11px; line-height:1.4; padding:3px 8px; border-radius:999px;
          border:1px solid var(--line); background:var(--panel-soft); color:var(--text); }
  .chip.good { color:#14532d; background:var(--green-soft); border-color:#a7d9b9; }
  .chip.bad { color:var(--red); background:var(--red-soft); border-color:#f2c5c5; font-weight:700; }
  .chip.mid { color:var(--orange); background:var(--orange-soft); border-color:#eec4a5; }
  .chip.neutral { color:var(--slate); background:var(--slate-soft); border-color:#dbe1e8; }
  .minitable { width:100%; border-collapse:collapse; font-size:11.5px; }
  .minitable th,.minitable td { padding:4px 5px; border-bottom:1px solid var(--line); text-align:right; }
  .minitable th:first-child,.minitable td:first-child { text-align:left; }
  .minitable thead th { color:var(--muted); font-weight:600; font-size:11px; }
  .minitable td { font-family:var(--mono); }
  .minitable tr:last-child td,.minitable tr:last-child th { border-bottom:none; }

  .cols { display:grid; gap:12px; grid-template-columns:repeat(var(--ncol,3),minmax(0,1fr)); }
  @media (max-width:1500px) { .cols { grid-template-columns:minmax(0,1fr); } }
  .textbox { border:1px solid var(--line); border-radius:var(--radius-sm); background:var(--panel);
             display:grid; grid-template-rows:auto 1fr; overflow:hidden; }
  .textbox > h4 { margin:0; padding:8px 10px; font-size:12.5px; background:var(--panel-soft);
                  border-bottom:1px solid var(--line); display:flex; justify-content:space-between;
                  align-items:center; gap:8px; }
  .textbox.gold > h4 { background:var(--accent-soft); color:var(--accent); }
  .textbox.sft > h4 { background:#e8f0fe; color:var(--blue); }
  .textbox.base > h4 { background:var(--slate-soft); color:var(--slate); }
  .body { padding:10px 12px; font-size:12.5px; line-height:1.75; max-height:640px; overflow:auto;
          white-space:pre-wrap; word-break:break-word; }
  .body h5 { margin:12px 0 4px; font-size:12.5px; color:var(--accent); }
  .body h5:first-child { margin-top:0; }
  .body .warn { color:var(--red); font-size:11.5px; }
  .body strong { font-weight:700; }
  details.think { margin:0 0 10px; border:1px dashed var(--line-strong); border-radius:var(--radius-sm);
                  background:var(--slate-soft); padding:6px 9px; }
  details.think > summary { cursor:pointer; color:var(--slate); font-size:11.5px; }
  details.think > div { margin-top:8px; padding-top:8px; border-top:1px dashed var(--line-strong);
                        color:var(--muted); font-size:11.5px; line-height:1.65; max-height:260px; overflow:auto; }
  .secgrid { display:grid; gap:10px; }
  .secrow { border:1px solid var(--line); border-radius:var(--radius-sm); overflow:hidden; }
  .secrow > h4 { margin:0; padding:7px 10px; font-size:12.5px; background:var(--accent-soft);
                 color:var(--accent); border-bottom:1px solid var(--line); }
  .secrow .cols { gap:0; grid-template-columns:repeat(var(--ncol,3),minmax(0,1fr)); }
  @media (max-width:1500px) { .secrow .cols { grid-template-columns:minmax(0,1fr); } }
  .seccell { border-right:1px solid var(--line); }
  .seccell:last-child { border-right:none; }
  .seccell > h5 { margin:0; padding:6px 10px; font-size:11.5px; color:var(--muted);
                  background:var(--panel-soft); border-bottom:1px solid var(--line); }
  .seccell .body { max-height:340px; }
  .empty { padding:14px; color:var(--muted); font-size:13px; text-align:center; }

  .lightbox { position:fixed; inset:0; display:none; align-items:center; justify-content:center;
              background:rgba(47,33,23,.85); z-index:50; cursor:zoom-out; }
  .lightbox img { max-width:94vw; max-height:94vh; border-radius:8px; }
  .lightbox.show { display:flex; }
"""

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
<script id="app-data" type="application/json">@@DATA@@</script>
<div class="shell">
  <aside class="sidebar">
    <section class="brand">
      <h1 id="appTitle"></h1>
      <p id="appSubtitle"></p>
      <div class="meta-grid">
        <div class="metric"><span class="value" id="mSamples">0</span><span class="label">抽样图片</span></div>
        <div class="metric"><span class="value" id="mMatched">0</span><span class="label">当前筛选</span></div>
        <div class="metric"><span class="value" id="mLangs">3</span><span class="label">语言（各等量）</span></div>
        <div class="metric"><span class="value" id="mDate">—</span><span class="label">生成日期</span></div>
      </div>
    </section>

    <section class="panel">
      <h2>结构体检（随筛选实时重算）</h2>
      <div id="statsBox"></div>
      <p id="statsNote"></p>
    </section>

    <section class="panel">
      <h2>展示模式</h2>
      <div id="modeToggle" class="btn-grid">
        <button type="button" class="mode-button active" data-mode="full">
          <strong>并排全文</strong><span>gold / SFT / 基座 三栏整篇对读，看文风和是否贴合图片。</span>
        </button>
        <button type="button" class="mode-button" data-mode="sections">
          <strong>分段对照</strong><span>按 4 个必需段落逐段横向对齐，看结构和每段覆盖度。</span>
        </button>
      </div>
    </section>

    <section class="panel">
      <h2>语言</h2>
      <div id="langButtons" class="btn-grid two"></div>
      <h2 style="margin-top:6px">数据划分</h2>
      <div id="splitButtons" class="btn-grid three"></div>
      <h2 style="margin-top:6px">只看异常</h2>
      <div id="flagButtons" class="btn-grid two"></div>
    </section>

    <section class="panel">
      <h2>搜索</h2>
      <div class="search"><input id="searchInput" type="search" placeholder="搜索文件名 / post_id / 正文"></div>
    </section>

    <section class="panel">
      <h2>分页</h2>
      <div class="page-summary" id="pageSummary"></div>
      <div class="page-row">
        <button id="prevPage" class="pill" type="button">上一页</button>
        <div class="page-jump">
          <input id="pageInput" class="page-input" type="number" min="1" value="1" aria-label="跳到页">
          <button id="pageJump" class="pill" type="button">Go</button>
        </div>
        <button id="nextPage" class="pill" type="button">下一页</button>
      </div>
    </section>

    <section class="panel">
      <h2>怎么读</h2>
      <ul>
        <li><b>语言</b>是这份数据的命门：每张图只有一种语言，prompt 里的语言块是唯一开关。徽章显示实测语种，与要求不符标红。</li>
        <li><b>段数</b>是 prompt 点名要的 4 个小标题；缺段说明格式没学会。<b>正文语种</b>与<b>标题语种</b>分开判：只抄对标题、正文却换了语言，是两回事。</li>
        <li><b>撞上限</b>= 生成到 token 上限被截断，不是模型自己收尾。</li>
        <li><b>think 块</b>：两个模型喂的是同一份无-think 提示词，基座每条都会自己开一个
            <code>&lt;think&gt;</code>（微调后一条都没有）。它被折叠单独展示、<b>不计入字数</b>，
            否则思考内容会被当成描述正文，把长度和语种判定全带偏。</li>
        <li><b>字数比</b>= 预测字数 ÷ 同一张图 gold 字数。语言之间字数不可比（中文 gold 中位数 787 字、英文 2635），所以只看同图比值。</li>
        <li>gold 只是众多合法答案之一，<b>没有</b>做与 gold 的相似度打分：措辞不同不等于说错，判对错请看图。</li>
      </ul>
    </section>
  </aside>

  <main>
    <section class="hero">
      <div>
        <h2>逐图对读：gold / 微调后 / 未微调基座</h2>
        <p id="heroNote"></p>
      </div>
      <div id="statusPill" class="status-pill">0</div>
    </section>
    <div id="rows"></div>
  </main>
</div>
<div class="lightbox" id="lb" onclick="this.classList.remove('show')"><img id="lbimg" src="" alt="zoom"></div>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("app-data").textContent);
  const SAMPLES = DATA.samples;
  const COLS = DATA.columns;
  const PRED_COLS = COLS.filter(function (c) { return c.key !== "gold"; });
  const LANG_LABEL = { en: "English", ja: "日本語", zh: "中文", other: "无法判定" };

  const state = { lang: "all", split: "all", flag: "none", query: "", mode: "full", page: 1,
                  pageSize: DATA.page_size || 5 };

  const $ = function (id) { return document.getElementById(id); };
  function esc(v) {
    return String(v == null ? "" : v).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
  }
  function md(text) {                       // the only markup these outputs use is **bold**
    return esc(text).replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
  }
  function langLabel(l) { return LANG_LABEL[l] || l || "无"; }
  function median(xs) {
    if (!xs.length) return 0;
    const s = xs.slice().sort(function (a, b) { return a - b; });
    return s[Math.floor(s.length / 2)];
  }

  // ── filtering ────────────────────────────────────────────────────────────────────────────
  function anyPred(row, f) { return PRED_COLS.some(function (c) { return f(row.texts[c.key].m); }); }
  const FLAGS = {
    none: function () { return true; },
    lang: function (r) { return anyPred(r, function (m) { return !m.lang_ok; }); },
    sec: function (r) { return anyPred(r, function (m) { return !m.sec4; }); },
    cap: function (r) { return anyPred(r, function (m) { return m.capped || m.rep_bad || m.empty; }); },
  };
  function blob(row) {
    return [row.name, row.post_id, COLS.map(function (c) { return row.texts[c.key].text; }).join(" ")]
      .join(" ").toLowerCase();
  }
  function filtered() {
    const q = state.query.trim().toLowerCase();
    return SAMPLES.filter(function (r) {
      if (state.lang !== "all" && r.lang !== state.lang) return false;
      if (state.split !== "all" && r.split !== state.split) return false;
      if (!FLAGS[state.flag](r)) return false;
      if (q && blob(r).indexOf(q) < 0) return false;
      return true;
    });
  }

  // ── sidebar health table, recomputed against the current filter ───────────────────────────
  function agg(rows, key) {
    const ms = rows.map(function (r) { return r.texts[key].m; });
    const n = ms.length;
    if (!n) return null;
    const pct = function (f) { return 100 * ms.filter(f).length / n; };
    return {
      n: n,
      lang: pct(function (m) { return m.lang_ok; }),
      sec4: pct(function (m) { return m.sec4; }),
      chars: median(ms.map(function (m) { return m.chars; })),
      ratio: key === "gold" ? null : median(ms.map(function (m) { return m.len_ratio; })),
      bad: key === "gold" ? 0 : pct(function (m) { return m.capped || m.rep_bad || m.empty; }),
      think: key === "gold" ? null : pct(function (m) { return m.has_think; }),
    };
  }
  function cls(v, hi, mid) { return v >= hi ? "good" : (v >= mid ? "mid" : "bad"); }
  function renderStats(rows) {
    const body = COLS.map(function (c) {
      const a = agg(rows, c.key);
      if (!a) return "";
      return "<tr><th>" + esc(c.label) + "</th>" +
        '<td class="' + cls(a.lang, 99, 90) + '">' + a.lang.toFixed(0) + "%</td>" +
        '<td class="' + cls(a.sec4, 99, 90) + '">' + a.sec4.toFixed(0) + "%</td>" +
        '<td class="plain">' + a.chars + "</td>" +
        '<td class="plain">' + (a.ratio == null ? "—" : a.ratio.toFixed(2) + "x") + "</td>" +
        '<td class="' + (c.key === "gold" ? "plain" : (a.bad > 0 ? "bad" : "good")) + '">' +
          (c.key === "gold" ? "—" : a.bad.toFixed(0) + "%") + "</td>" +
        '<td class="plain">' + (a.think == null ? "—" : a.think.toFixed(0) + "%") + "</td></tr>";
    }).join("");
    $("statsBox").innerHTML =
      '<table class="stats"><thead><tr><th>n=' + rows.length +
      "</th><th>语言</th><th>4段</th><th>字数</th><th>比</th><th>异常</th><th>think</th></tr></thead>" +
      "<tbody>" + body + "</tbody></table>";
    $("statsNote").innerHTML = "语言 = 正文语种与 prompt 要求一致的比例；4段 = 4 个必需小标题齐全；" +
      "字数 = 中位字符数（<b>不含 think 块</b>）；比 = 同图 预测/gold 字数中位数；" +
      "异常 = 撞 token 上限 / 重复循环 / 无正文；think = 输出里自带 &lt;think&gt; 块的比例。" +
      "gold 行是同一套判据跑在参考文本上的<b>标定线</b>，它不是 100% 就说明判据本身有问题。";
  }

  // ── rendering ────────────────────────────────────────────────────────────────────────────
  function chip(text, kind) { return '<span class="chip ' + (kind || "") + '">' + esc(text) + "</span>"; }
  function predChips(row) {
    return PRED_COLS.map(function (c) {
      const m = row.texts[c.key].m;
      const bits = [c.label + "：" +
        (m.lang_ok ? "语种✓" : "语种✗ " + langLabel(m.body_lang)) +
        " · " + m.n_sections + "/4段"];
      if (m.capped) bits.push("撞上限");
      if (m.rep_bad) bits.push("重复循环");
      if (m.empty) bits.push("无正文");
      if (m.think_chars) bits.push("think " + m.think_chars + " 字");
      const bad = !m.lang_ok || !m.sec4 || m.capped || m.rep_bad || m.empty;
      return chip(bits.join(" · "), bad ? "bad" : "good");
    }).join("");
  }
  function miniTable(row) {
    const head = "<tr><th></th>" + COLS.map(function (c) { return "<th>" + esc(c.label.split(" ")[0]) + "</th>"; }).join("") + "</tr>";
    function line(label, f) {
      return "<tr><th>" + label + "</th>" +
        COLS.map(function (c) { return "<td>" + f(row.texts[c.key].m, c.key) + "</td>"; }).join("") + "</tr>";
    }
    return '<table class="minitable"><thead>' + head + "</thead><tbody>" +
      line("正文语种", function (m) { return langLabel(m.body_lang); }) +
      line("标题语种", function (m) { return m.header_lang ? langLabel(m.header_lang) : "无"; }) +
      line("段数", function (m) { return m.n_sections + "/4"; }) +
      line("字符", function (m) { return m.chars; }) +
      line("token", function (m, k) { return k === "gold" ? "—" : (m.ntok || 0); }) +
      line("重复度", function (m) { return (m.rep != null ? m.rep : 0).toFixed(2); }) +
      "</tbody></table>";
  }
  // A <think> block is the base model's, never the SFT model's: served the LlamaFactory prompt
  // (which ends right after "assistant\\n"), the base opens one itself on every sample. It is
  // shown, collapsed, rather than deleted -- on 6 samples it swallows thousands of characters of
  // the output budget, and on one it swallows the whole answer.
  function thinkBlock(e) {
    if (!e.m.has_think) return "";
    const n = e.think.length;
    const head = n ? "think 块（" + n + " 字，未计入上面的字数）" : "空 think 块（0 字）";
    return '<details class="think"><summary>' + head + "</summary>" +
      (n ? "<div>" + md(e.think) + "</div>" : "") + "</details>";
  }
  function textBox(row, col) {
    const e = row.texts[col.key];
    const m = e.m;
    const flag = col.key === "gold" ? "" :
      (m.empty ? '<span class="chip bad">无正文</span>' :
        (m.capped ? '<span class="chip bad">截断</span>' : ""));
    const bodyHtml = e.text
      ? md(e.text)
      : '<span class="warn">（这一栏没有正文：模型把 token 预算全用在思考上了，见下方 think 块）</span>';
    return '<div class="textbox ' + col.cls + '"><h4><span>' + esc(col.label) + "</span><span>" +
      flag + '<span class="chip neutral">' + m.chars + " 字</span></span></h4>" +
      '<div class="body">' + thinkBlock(e) + bodyHtml + "</div></div>";
  }
  function fullMode(row) {
    return '<div class="cols" style="--ncol:' + COLS.length + '">' +
      COLS.map(function (c) { return textBox(row, c); }).join("") + "</div>";
  }
  function sectionMode(row) {
    const unparsed = COLS.filter(function (c) { return row.texts[c.key].text && !row.texts[c.key].sections; });
    const n = 4;
    let html = "";
    for (let i = 0; i < n; i++) {
      const cells = COLS.map(function (c) {
        const secs = row.texts[c.key].sections;
        const s = secs ? secs[i] : null;
        const body = s ? md(s.body) : '<span class="warn">（这一栏没有可切分的 4 段结构，见下方整篇）</span>';
        return '<div class="seccell"><h5>' + esc(c.label) + (s ? " · " + s.body.length + " 字" : "") +
          '</h5><div class="body">' + body + "</div></div>";
      }).join("");
      const title = (row.texts.gold.sections && row.texts.gold.sections[i])
        ? row.texts.gold.sections[i].title : "段 " + (i + 1);
      html += '<div class="secrow"><h4>' + esc(title) + '</h4><div class="cols" style="--ncol:' +
        COLS.length + '">' + cells + "</div></div>";
    }
    if (unparsed.length) {
      html += '<div class="secrow"><h4>未按 4 段格式输出的整篇原文</h4><div class="cols" style="--ncol:' +
        unparsed.length + '">' + unparsed.map(function (c) { return textBox(row, c); }).join("") + "</div></div>";
    }
    return '<div class="secgrid">' + html + "</div>";
  }
  function sampleRow(row) {
    const color = row.split === "seen" ? "var(--violet)" : "var(--teal)";
    return '<article class="sample-row" style="--color:' + color + '">' +
      '<div class="row-media">' +
        '<div class="thumb"><img loading="lazy" src="' + row.img + '" alt="' + esc(row.name) +
          '" onclick="showLb(this.src)"></div>' +
        '<div class="sample-title"><strong>' + esc(row.name) + "</strong><span>#" + (row.i + 1) +
          " · post " + esc(row.post_id) + "</span></div>" +
        '<div class="chips">' + chip(row.split === "seen" ? "seen（训练见过）" : "unseen（post 级零重叠）", "neutral") +
          chip(langLabel(row.lang), "neutral") + "</div>" +
        '<div class="chips">' + predChips(row) + "</div>" +
        miniTable(row) +
      "</div>" +
      '<div class="row-detail">' + (state.mode === "full" ? fullMode(row) : sectionMode(row)) + "</div>" +
    "</article>";
  }

  // ── controls ─────────────────────────────────────────────────────────────────────────────
  function buttons(el, items, current, attr) {
    el.innerHTML = items.map(function (it) {
      return '<button type="button" class="pill' + (current === it.key ? " active" : "") +
        '" data-' + attr + '="' + esc(it.key) + '">' + esc(it.label) + "</button>";
    }).join("");
  }
  function counts(key, val) {
    return SAMPLES.filter(function (r) { return val === "all" || r[key] === val; }).length;
  }
  function renderControls() {
    buttons($("langButtons"), [{ key: "all", label: "全部 (" + SAMPLES.length + ")" }].concat(
      DATA.lang_order.map(function (l) { return { key: l, label: langLabel(l) + " (" + counts("lang", l) + ")" }; })),
      state.lang, "lang");
    buttons($("splitButtons"), [{ key: "all", label: "全部" }].concat(
      DATA.split_order.map(function (s) { return { key: s, label: s + " (" + counts("split", s) + ")" }; })),
      state.split, "split");
    buttons($("flagButtons"), [
      { key: "none", label: "不筛" },
      { key: "lang", label: "语种不符 (" + SAMPLES.filter(FLAGS.lang).length + ")" },
      { key: "sec", label: "缺段 (" + SAMPLES.filter(FLAGS.sec).length + ")" },
      { key: "cap", label: "截断/重复/空 (" + SAMPLES.filter(FLAGS.cap).length + ")" },
    ], state.flag, "flag");
    Array.prototype.forEach.call($("modeToggle").querySelectorAll("[data-mode]"), function (b) {
      b.classList.toggle("active", b.getAttribute("data-mode") === state.mode);
    });
  }
  function render() {
    const rows = filtered();
    const totalPages = Math.max(1, Math.ceil(rows.length / state.pageSize));
    if (state.page > totalPages) state.page = totalPages;
    if (state.page < 1) state.page = 1;
    const start = (state.page - 1) * state.pageSize;
    const page = rows.slice(start, start + state.pageSize);

    renderControls();
    renderStats(rows);
    $("mMatched").textContent = rows.length;
    $("statusPill").textContent = rows.length + " 张匹配 · 第 " + state.page + "/" + totalPages + " 页";
    $("pageSummary").textContent = "第 " + state.page + " / " + totalPages + " 页 · 每页 " +
      state.pageSize + " 张 · 共 " + rows.length + " 张";
    $("pageInput").value = state.page;
    $("pageInput").max = totalPages;
    $("prevPage").disabled = state.page <= 1;
    $("nextPage").disabled = state.page >= totalPages;
    $("rows").innerHTML = page.length ? page.map(sampleRow).join("")
      : '<div class="hero"><div class="empty">没有匹配样本，放宽筛选或清空搜索。</div></div>';
    window.scrollTo({ top: 0 });
  }

  $("appTitle").textContent = DATA.title;
  $("appSubtitle").textContent = DATA.subtitle;
  $("heroNote").textContent = DATA.note;
  $("mSamples").textContent = SAMPLES.length;
  $("mDate").textContent = DATA.generated_at.slice(5);

  function bind(id, attr, key) {
    $(id).addEventListener("click", function (e) {
      const b = e.target.closest("[data-" + attr + "]");
      if (!b) return;
      state[key] = b.getAttribute("data-" + attr);
      if (key !== "mode") state.page = 1;
      render();
    });
  }
  bind("langButtons", "lang", "lang");
  bind("splitButtons", "split", "split");
  bind("flagButtons", "flag", "flag");
  bind("modeToggle", "mode", "mode");
  $("searchInput").addEventListener("input", function () {
    state.query = this.value; state.page = 1; render();
  });
  $("prevPage").addEventListener("click", function () { state.page -= 1; render(); });
  $("nextPage").addEventListener("click", function () { state.page += 1; render(); });
  $("pageJump").addEventListener("click", function () { state.page = Number($("pageInput").value || 1); render(); });
  $("pageInput").addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); state.page = Number(this.value || 1); render(); }
  });
  render();
}());
function showLb(src) {
  document.getElementById("lbimg").src = src;
  document.getElementById("lb").classList.add("show");
}
document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") document.getElementById("lb").classList.remove("show");
});
</script>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--pred", default=None, help="default WORK_DIR/samples_pred.json")
    ap.add_argument("--out", default=None,
                    help=f"default WORK_DIR/mikomiko_grok_desc_review_{date.today():%Y%m%d}.html")
    ap.add_argument("--tags", default="sft,base")
    ap.add_argument("--title", default="Mikomiko 描述模型 抽样审阅")
    ap.add_argument("--subtitle", default="Qwen3.5-9B · 全参 SFT · 1 epoch")
    ap.add_argument("--note", default="")
    ap.add_argument("--gold-label", default="gold 参考")
    ap.add_argument("--sft-label", default="SFT 微调后")
    ap.add_argument("--base-label", default="未微调基座")
    ap.add_argument("--page-size", type=int, default=5)
    args = ap.parse_args()

    pred = args.pred or os.path.join(args.work_dir, "samples_pred.json")
    out = args.out or os.path.join(args.work_dir,
                                   f"mikomiko_grok_desc_review_{date.today():%Y%m%d}.html")
    samples = json.load(open(pred, encoding="utf-8"))
    tags = [t for t in args.tags.split(",") if t and any(f"pred_{t}" in s for s in samples)]
    if not tags:
        raise SystemExit(f"[fatal] no pred_* fields in {pred}; run infer_desc.py first")

    print(f"[html] encoding {len(samples)} thumbnails ...", flush=True)
    for i, s in enumerate(samples, 1):
        s["b64"] = thumb_b64(s["image"])
        if i % 30 == 0 or i == len(samples):
            print(f"  [html] {i}/{len(samples)}", flush=True)

    payload = build_payload(samples, tags, args)
    # JSON inside <script>: the ONLY sequence that can break out is "</", so escape just that.
    # (base64-wrapping the payload instead would inflate the embedded JPEGs by another third.)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    page = PAGE
    for k, v in (("TITLE", args.title), ("CSS", CSS), ("DATA", data)):
        page = page.replace(f"@@{k}@@", v)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w", encoding="utf-8").write(page)
    print(f"[html] {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

    from metrics_desc import print_table
    print_table(samples, tags)


if __name__ == "__main__":
    main()
