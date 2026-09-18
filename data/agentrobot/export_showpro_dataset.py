#!/usr/bin/env python3
"""Export a trained-on sample set into a self-contained, release-ready dataset folder.

训练用的 rollout_lite.json 里是**绝对图片路径**，指向散落在多处的采集目录（真机那份横跨
5 个批次目录）。要把数据发出去就得把图收拢到一处、路径改成相对的、集编号统一。

产出结构：

    <out>/
    ├── rollouts.json         训练样本，images 改为相对本文件的路径
    ├── episodes.jsonl        每集一行的索引：机体、任务、步数、来源
    └── metadata/
        ├── rollout_000/
        │   ├── actions.jsonl   采集时的逐步记录
        │   ├── agentview/      只含被样本引用到的帧
        │   └── wrist/
        └── rollout_001/ ...

集目录**扁平铺开并重新编号**（rollout_000, rollout_001, ...），不按采集批次或任务分子目录 ——
发布的数据集不该把内部的批次划分暴露成目录结构。编号按源路径字典序分配，所以同一份输入
重跑得到的编号完全一致。

扁平化会丢掉"这一集是哪个机体采的"，而这件事关系到能不能正确用这份数据（Piper 的
FWD/BACK 是交换过的，部署要换回来），所以它连同任务文本一起写进 episodes.jsonl。

只复制**被引用到的**帧：采集目录里常有没进训练集的帧（清洗掉的复位步等），发数据集时不该带。
visualization.mp4 和 metadata.json 不复制（后者的有用字段已提取进 episodes.jsonl）。

Usage:
    python data/agentrobot/export_showpro_dataset.py \
        --samples data/agentrobot/MVTOKEN/mix_22-06_fk-pp/02_exchange_token/rollout_lite.json \
        --out     .../Show-Pro-data/real
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


TASK_RE = re.compile(r"^Task:\s*(.+?)\s*$", re.M)


def _load(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    return data if isinstance(data, list) else [data]


def _episode_dir(image_path: str) -> str:
    """.../rollout_XXX/agentview/0000.png -> .../rollout_XXX"""
    return str(Path(image_path).parent.parent)


def _episode_meta(src_dir: Path) -> dict:
    """从采集端 metadata.json 挑出值得留进索引的字段。

    集目录扁平化后，"这集是哪个机体采的""是哪个仿真任务"就只能靠索引了 —— 而机体这件事
    直接决定数据能不能被正确使用（Piper 的 FWD/BACK 是交换过的）。字段不存在就不写，所以
    真机那份只会有 embodiment，仿真那份会多出 simulator / env_id / seed。
    """
    out: dict = {}
    meta = src_dir / "metadata.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
        except (json.JSONDecodeError, OSError):
            m = {}
        if m.get("robot"):
            out["embodiment"] = str(m["robot"])
        if m.get("sim"):
            out["simulator"] = str(m["sim"])
        if m.get("env_id"):
            out["env_id"] = str(m["env_id"])
        if m.get("seed") is not None:
            out["seed"] = m["seed"]
    out.setdefault("embodiment", "piper" if "piper" in str(src_dir).lower() else "franka")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="把训练样本集导出成可发布的数据集目录。")
    ap.add_argument("--samples", type=Path, required=True, help="rollout_lite.json")
    ap.add_argument("--out", type=Path, required=True, help="输出目录")
    ap.add_argument("--dry-run", action="store_true", help="只报告将要做什么，不落盘")
    args = ap.parse_args()

    samples = _load(args.samples)
    all_imgs = [p for s in samples for p in s.get("images", [])]
    if not all_imgs:
        raise SystemExit("ERROR: 样本里没有 images 字段")

    missing = [p for p in set(all_imgs) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} 张源图不存在，例如 {missing[0]}")

    # ── 分配扁平编号（按源路径字典序，保证可复现）──────────────────────────
    src_dirs = sorted({_episode_dir(p) for p in all_imgs})
    width = max(3, len(str(len(src_dirs) - 1)))
    new_id = {d: f"rollout_{i:0{width}d}" for i, d in enumerate(src_dirs)}
    print(f"样本 {len(samples)} 条 | 唯一图片 {len(set(all_imgs))} 张 | 集 {len(src_dirs)} 个")
    print(f"编号: {new_id[src_dirs[0]]} .. {new_id[src_dirs[-1]]}")

    if args.dry_run:
        print(f"[dry-run] 示例: {src_dirs[0]}\n      ->  metadata/{new_id[src_dirs[0]]}/")
        return

    meta_dir = args.out / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ── 复制图片 ─────────────────────────────────────────────────────────────
    def remap(src: str) -> str:
        p = Path(src)
        return str(Path("metadata") / new_id[_episode_dir(src)] / p.parent.name / p.name)

    uniq = sorted(set(all_imgs))
    for i, src in enumerate(uniq, 1):
        dst = args.out / remap(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        if i % 2000 == 0:
            print(f"  已复制 {i}/{len(uniq)} 张", flush=True)
    print(f"图片复制完成 {len(uniq)} 张")

    # ── 每集的 actions.jsonl 与索引 ──────────────────────────────────────────
    steps_per_ep = {d: len({p for p in all_imgs if _episode_dir(p) == d}) // 2 for d in src_dirs}
    samples_per_ep: dict[str, int] = {}
    task_per_ep: dict[str, str] = {}
    for s in samples:
        d = _episode_dir(s["images"][0])
        samples_per_ep[d] = samples_per_ep.get(d, 0) + 1
        if d not in task_per_ep:
            m = TASK_RE.search(s["instruction"])
            task_per_ep[d] = m.group(1) if m else ""

    episodes, n_act = [], 0
    for d in src_dirs:
        src = Path(d)
        act = src / "actions.jsonl"
        if act.exists():
            shutil.copy2(act, meta_dir / new_id[d] / "actions.jsonl")
            n_act += 1
        episodes.append({
            "episode": new_id[d],
            **_episode_meta(src),
            "task": task_per_ep.get(d, ""),
            "num_frames": steps_per_ep[d],
            "num_samples": samples_per_ep.get(d, 0),
        })
    print(f"actions.jsonl 复制完成 {n_act}/{len(src_dirs)} 个")

    (args.out / "episodes.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes), encoding="utf-8")
    print(f"写出 episodes.jsonl（{len(episodes)} 集）")

    # ── 重写样本路径 ─────────────────────────────────────────────────────────
    out_samples = [dict(s, images=[remap(p) for p in s["images"]]) for s in samples]
    (args.out / "rollouts.json").write_text(
        json.dumps(out_samples, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 rollouts.json（{len(out_samples)} 条）")

    # ── 校验 ─────────────────────────────────────────────────────────────────
    bad = [p for s in out_samples for p in s["images"] if not (args.out / p).exists()]
    if bad:
        raise SystemExit(f"ERROR: {len(bad)} 条重写后的路径解析不到，例如 {bad[0]}")
    print(f"校验通过：{sum(len(s['images']) for s in out_samples)} 条相对路径全部可解析")


if __name__ == "__main__":
    main()
