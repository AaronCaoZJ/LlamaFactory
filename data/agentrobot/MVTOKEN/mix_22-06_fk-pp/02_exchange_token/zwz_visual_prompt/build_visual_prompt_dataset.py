#!/usr/bin/env python3
"""Build the visual-prompt training set from 02_exchange_token/rollout_lite_plus_zwz_0723_0724_0802_long_collect.json.

Two changes per sample versus the base set:

  1. a THIRD image is appended - the rollout's ``visual_prompt.png`` (the first agentview frame with
     a green box round the object to pick and a red cross on the release spot), so the media prefix
     becomes ``<image><image><image>`` and images are [agentview, wrist, visual_prompt];
  2. the instruction is re-rendered from ``prompt_v3_visual_prompt.txt``, which introduces that third
     view and says what the annotations mean. ``{task}`` / ``{recent_moves}`` are parsed back out of
     the base instruction, so nothing else about the sample changes.

Only rollouts that actually HAVE a usable annotation survive. The filter is driven by
``visual_prompt_qa.json`` / ``visual_prompt_exclusions.json`` rather than a hard-coded list:

  keep    category_verdict in {ok, reannotated, fixed, loose}
  drop    anything listed in visual_prompt_exclusions.json (no annotation file exists)
  drop    the whole zwz_0802_long_collect root - long-horizon multi-step collect, a single source
          box plus one goal cross cannot express it (per user decision, 2026-08-10)

``_horizon_flip`` rollouts never appear because the base set does not contain them.

Writes rollout_lite_visual_prompt.json next to this script and prints the per-category breakdown
that SAMPLES.md is built from.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MV = HERE.parents[2]
LF_DATA = MV.parents[1]
BASE = MV / "mix_22-06_fk-pp/02_exchange_token/rollout_lite_plus_zwz_0723_0724_0802_long_collect.json"
# Two prompt variants over the SAME sample set, so the pair is a clean ablation:
#   task    keeps the per-episode task sentence ("pick up the banana and place it on the plate").
#           The text alone already says what to do, so the model can solve most of the episode
#           without ever reading the green box -- the visual prompt is a redundant channel.
#   notask  replaces it with one constant sentence that only refers to the marks ("pick up the
#           object marked by the green box and release it at the red cross"). The instruction
#           text is now identical in every sample, so WHICH object and WHERE it goes can only
#           come from the third image. This is the setting that actually tests the visual prompt.
VARIANTS = {
    "task": (HERE / "prompt_v3_visual_prompt.txt", HERE / "rollout_lite_visual_prompt.json"),
    "notask": (
        HERE / "prompt_v3_visual_prompt_notask.txt",
        HERE / "rollout_lite_visual_prompt_notask.json",
    ),
}
REPORT = HERE / "SAMPLES.md"

KEEP_VERDICTS = {"ok", "reannotated", "fixed", "loose"}
DROP_ROOTS = {"zwz_0802_long_collect"}
MEDIA_PREFIX = "<image><image><image>"

ROLLOUT_RE = re.compile(r"/MVTOKEN/(.+?)/(?:agentview|wrist)/")
MEDIA_RE = re.compile(r"^(?:<image>|<video>)+")
TASK_RE = re.compile(r"^Task:\s*(?P<task>.+?)\s*$", re.MULTILINE)
RECENT_RE = re.compile(r"^Recent(?: previous)? moves, newest first:\s*(?P<recent>.*?)\s*$", re.MULTILINE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write both dataset jsons + SAMPLES.md.")
    args = parser.parse_args()

    templates = {
        name: path.read_text(encoding="utf-8").rstrip("\n") for name, (path, _) in VARIANTS.items()
    }
    qa = {e["rollout"]: e for e in json.loads((MV / "visual_prompt_qa.json").read_text(encoding="utf-8"))}
    excluded = json.loads((MV / "visual_prompt_exclusions.json").read_text(encoding="utf-8"))["rollouts"]
    base = json.loads(BASE.read_text(encoding="utf-8"))

    out: dict[str, list[dict]] = {name: [] for name in VARIANTS}
    kept_r, kept_s, dropped_r, dropped_s = Counter(), Counter(), Counter(), Counter()
    kept_v: dict[str, Counter] = {}  # category -> verdict counts, over the KEPT rollouts only
    seen: set[str] = set()

    for sample in base:
        rel = ROLLOUT_RE.search(sample["images"][0]).group(1)
        root = rel.split("/")[0]
        if root in DROP_ROOTS:
            reason = f"{root} (multi-step long-collect, not visual-prompt shaped)"
        elif rel in excluded:
            reason = f"excluded: {excluded[rel]}"
        elif qa.get(rel, {}).get("category_verdict") not in KEEP_VERDICTS:
            reason = f"verdict: {qa.get(rel, {}).get('category_verdict', 'no annotation')}"
        else:
            reason = None
        category = qa[rel]["category"] if rel in qa else "/".join(rel.split("/")[:-1])
        if reason is not None:
            dropped_s[(category, reason)] += 1
            if rel not in seen:
                dropped_r[(category, reason)] += 1
                seen.add(rel)
            continue
        if rel not in seen:
            kept_r[category] += 1
            kept_v.setdefault(category, Counter())[qa[rel]["category_verdict"]] += 1
            seen.add(rel)
        kept_s[category] += 1

        body = MEDIA_RE.sub("", sample["instruction"])
        task = TASK_RE.search(body).group("task")
        recent = RECENT_RE.search(body).group("recent")
        vp = str(LF_DATA / "agentrobot/MVTOKEN" / rel / "visual_prompt.png")
        if not Path(vp).exists():
            raise FileNotFoundError(vp)
        images = [sample["images"][0], sample["images"][1], vp]
        for name, template in templates.items():
            # `task` is simply unused by the notask template -- str.format ignores extra kwargs.
            out[name].append(
                {
                    "instruction": MEDIA_PREFIX + template.format(task=task, recent_moves=recent),
                    "input": sample.get("input", ""),
                    "output": sample["output"],
                    "images": list(images),
                }
            )

    print(f"{'category':46s} {'rollouts':>8s} {'samples':>8s}")
    for c in sorted(kept_r):
        print(f"{c:46s} {kept_r[c]:8d} {kept_s[c]:8d}")
    print(f"{'TOTAL KEPT':46s} {sum(kept_r.values()):8d} {sum(kept_s.values()):8d}\n")
    print(f"{'dropped category':46s} {'rollouts':>8s} {'samples':>8s}  reason")
    for key in sorted(dropped_r):
        c, reason = key
        print(f"{c:46s} {dropped_r[key]:8d} {dropped_s[key]:8d}  {reason}")
    print(f"{'TOTAL DROPPED':46s} {sum(dropped_r.values()):8d} {sum(dropped_s.values()):8d}")

    if args.write:
        print()
        for name, (_, out_path) in VARIANTS.items():
            out_path.write_text(
                json.dumps(out[name], ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
            )
            print(f"wrote {out_path.name} ({len(out[name])} samples, variant={name})")
        write_report(kept_r, kept_s, kept_v, dropped_r, dropped_s)
        print(f"wrote {REPORT.name}")


# visual_prompt_qa.json 里每条标注的人工质检结论；这里只出现"用了"的那几种
VERDICT_CN = {
    "ok": "自动生成后抽查通过",
    "reannotated": "整类重标过（2026-08-09，annotation_version 3）",
    "fixed": "人工逐条改过框",
    "loose": "语义对但框偏松（padding/偏移约 10px）",
    "mostly-ok": "大体可用，target 有时圈住一组物体",
    "out_of_scope": "判定不做视觉提示标注",
}
REASON_CN = {
    "no_grasp_no_release_not_a_pick_place_trajectory": "整条轨迹没有 GRASP 也没有 RELEASE，不是抓放任务",
    "no_release_object_is_never_placed": "有 GRASP 没有 RELEASE，物体从没被放下，不存在放置目标",
    "no_grasp_object_is_never_picked": "有 RELEASE 没有 GRASP，是放置/退回片段，物体从没被抓起",
    "empty_actions": "actions.jsonl 为空",
    "task_not_suited_for_visual_prompt": "任务没有单一可见的放置落点，一个框加一个十字表达不了",
}


def write_report(kept_r, kept_s, kept_v, dropped_r, dropped_s) -> None:
    """重新生成 SAMPLES.md，保证清单不会和数据集本身漂移。"""

    def verdict_cell(cat: str) -> str:
        c = kept_v.get(cat, Counter())
        return "、".join(f"{VERDICT_CN.get(k, k)} {n} 条" for k, n in c.most_common()) or "-"

    def reason_cell(reason: str) -> str:
        if reason.startswith("excluded: "):
            key = reason[len("excluded: ") :]
            return f"已排除：{REASON_CN.get(key, key)}"
        return "0802 long collect 整批不做：long-horizon 多步收集，中间有多个 subgoal"

    total_kept_r, total_kept_s = sum(kept_r.values()), sum(kept_s.values())
    total_drop_r, total_drop_s = sum(dropped_r.values()), sum(dropped_s.values())

    lines = [
        "# Visual prompt 实验 — 样本清单",
        "",
        "由 `build_visual_prompt_dataset.py --write` 自动生成，**不要手工编辑**。",
        "",
        f"**{total_kept_s} 条样本 / {total_kept_r} 个 rollout**，两个 prompt 变体共用同一批样本：",
        "",
        "| 变体 | 数据集文件 | prompt | `Task:` 那行 |",
        "|---|---|---|---|",
        f"| `task` | `{VARIANTS['task'][1].name}` | `{VARIANTS['task'][0].name}` | 每条不同，是真实任务描述 |",
        f"| `notask` | `{VARIANTS['notask'][1].name}` | `{VARIANTS['notask'][0].name}` | 每条相同，只指向绿框和红叉 |",
        "",
        "`notask` 是真正检验 visual prompt 的那个设置：文本在所有样本里完全一样，"
        "「抓哪个、放哪」只能从第三张图里读。`task` 变体保留下来做对照。",
        f"来源是 `{BASE.name}`（{total_kept_s + total_drop_s} 条 / "
        f"{total_kept_r + total_drop_r} 个 rollout），该基础集本身就不含 `_horizon_flip` 数据。",
        "",
        "每条样本带三张图 —— `[agentview, wrist, visual_prompt]`，prompt 用 `prompt_v3_visual_prompt.txt`。",
        "`visual_prompt.png` 是首帧 agentview，绿框标出要抓的物体，红十字标出松手的位置。",
        "",
        "## 用了的样本",
        "",
        "「质检结论」来自 `MVTOKEN/visual_prompt_qa.json` 的 `category_verdict` 字段，"
        "记录的是这条标注**当初是怎么来的、人工看过之后的结论**（含义见文末对照表）；",
        "这里只统计实际进了数据集的那些 rollout。",
        "",
        "| 类别 | rollout | 样本 | 质检结论 |",
        "|---|---:|---:|---|",
    ]
    for c in sorted(kept_r):
        lines.append(f"| `{c}` | {kept_r[c]} | {kept_s[c]} | {verdict_cell(c)} |")
    lines += [
        f"| **合计** | **{total_kept_r}** | **{total_kept_s}** | |",
        "",
        "## 没用的样本",
        "",
        "| 类别 | rollout | 样本 | 原因 |",
        "|---|---:|---:|---|",
    ]
    for key in sorted(dropped_r):
        c, reason = key
        lines.append(f"| `{c}` | {dropped_r[key]} | {dropped_s[key]} | {reason_cell(reason)} |")
    lines += [
        f"| **合计** | **{total_drop_r}** | **{total_drop_s}** | |",
        "",
        "## 筛选规则",
        "",
        "规则读 `MVTOKEN/visual_prompt_qa.json` 和 `MVTOKEN/visual_prompt_exclusions.json`，没有硬编码名单：",
        "",
        f"- 保留 `category_verdict` ∈ {{{', '.join(sorted(KEEP_VERDICTS))}}}；",
        "- 丢掉 `visual_prompt_exclusions.json` 里登记的（这些 rollout 根本没有标注文件 —— "
        "轨迹缺 GRASP、缺 RELEASE，或者任务本身表达不了）；",
        f"- 丢掉整个 `{sorted(DROP_ROOTS)[0]}` root —— long-horizon 多步收集，中间有多个 subgoal，"
        "一个源框加一个目标十字表达不了（2026-08-10 决定）。",
        "",
        "**`loose` 保留了**：这些框（全是 `0622/stack_white_bowl`）偏移最多约 10px，但仍然落在正确的物体上，"
        "而绿框在这里要传达的就是「哪个物体」—— 模型输出的是动作 token，不是框。",
        "想要严格集就把构建脚本里 `KEEP_VERDICTS` 的 `loose` 去掉重跑。",
        "",
        "`_horizon_flip` 不会出现，因为基础集本来就没有；训练也不使用翻转数据。",
        "",
        "## 质检结论对照表",
        "",
        "| verdict | 含义 |",
        "|---|---|",
    ]
    for k in ("ok", "reannotated", "fixed", "loose"):
        lines.append(f"| `{k}` | {VERDICT_CN[k]} |")
    lines += [
        f"| `mostly-ok` / `out_of_scope` | {VERDICT_CN['mostly-ok']} / {VERDICT_CN['out_of_scope']}；"
        "只出现在 0802 long collect，本实验没用到 |",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
