#!/usr/bin/env python3
"""Clean the 0627 grasp/release raw rollouts into a single 0627_cleaned set.

The 0627 raw data (``hf_download/datasets/MVTOKEN_RAW/0627``) has two folders of teleop
rollouts that each repeatedly demonstrate ONE gripper skill, using the OPPOSITE gripper
action only as a reset between demonstrations:

  * ``grasp/``   — demonstrates GRASP; the RELEASE steps merely reopen the gripper so the
                   operator can grasp again. All grasp-skill frames are gripper-open.
  * ``release/`` — demonstrates RELEASE; the GRASP steps merely reclose the gripper so the
                   operator can release again.

Cleaning drops those reset steps so each folder keeps only its target skill's frames:

  * ``grasp/``   -> remove every ``RELEASE`` step
  * ``release/`` -> remove every ``GRASP`` step

Each cleaned rollout keeps the original layout (``actions.jsonl`` + ``agentview/`` +
``wrist/`` + ``metadata.json``) with the kept steps RE-INDEXED contiguously (0000, 0001, ...)
and their images copied under the new names. ``visualization.mp4`` is skipped (it shows the
un-cleaned sequence). The source images are only read, never modified.

The two source folders use disjoint rollout ids (grasp 000-007, release 008-012), so they
merge into ``--out-dir`` without collision.

Usage:
    python data/agentrobot/clean_grasp_release.py \\
        --grasp-dir   /workspace1/zhijun/hf_download/datasets/MVTOKEN_RAW/0627/grasp \\
        --release-dir /workspace1/zhijun/hf_download/datasets/MVTOKEN_RAW/0627/release \\
        --out-dir     data/agentrobot/MVTOKEN/0627_cleaned
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _read_steps(actions_path: Path) -> list[dict]:
    steps: list[dict] = []
    with open(actions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return steps


def clean_rollout(rollout_dir: Path, drop_token: str, out_dir: Path) -> tuple[int, int]:
    """Clean one rollout into ``out_dir``; return (kept, removed) step counts."""
    actions_path = rollout_dir / "actions.jsonl"
    if not actions_path.exists():
        print(f"[skip] no actions.jsonl in {rollout_dir}", file=sys.stderr)
        return (0, 0)

    steps = _read_steps(actions_path)
    kept = [s for s in steps if s.get("token") != drop_token]
    removed = len(steps) - len(kept)
    if not kept:
        print(f"[skip] {rollout_dir.name}: nothing left after dropping {drop_token}", file=sys.stderr)
        return (0, removed)

    (out_dir / "agentview").mkdir(parents=True, exist_ok=True)
    (out_dir / "wrist").mkdir(parents=True, exist_ok=True)

    new_steps: list[dict] = []
    for new_idx, step in enumerate(kept):
        new_name = f"{new_idx:04d}.png"
        for cam in ("agentview", "wrist"):
            src = rollout_dir / step[cam]
            shutil.copy2(src, out_dir / cam / new_name)
        step = dict(step)
        step["step"] = new_idx
        step["agentview"] = f"agentview/{new_name}"
        step["wrist"] = f"wrist/{new_name}"
        new_steps.append(step)

    with open(out_dir / "actions.jsonl", "w", encoding="utf-8") as f:
        for step in new_steps:
            f.write(json.dumps(step, ensure_ascii=False) + "\n")

    # Carry metadata.json forward with the token list / num_steps refreshed and a provenance
    # note of what was removed; other fields (serials, step_m, z_floor, ...) are preserved.
    meta_path = rollout_dir / "metadata.json"
    meta: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    meta["num_steps"] = len(new_steps)
    meta["tokens"] = [s["token"] for s in new_steps]
    meta["cleaned_removed_token"] = drop_token
    meta["cleaned_removed_count"] = removed
    meta["cleaned_source"] = str(rollout_dir)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return (len(new_steps), removed)


def process_folder(src_dir: Path, drop_token: str, out_root: Path) -> None:
    rollouts = sorted(d for d in src_dir.iterdir() if d.is_dir() and (d / "actions.jsonl").exists())
    if not rollouts:
        print(f"[warn] no rollouts found under {src_dir}", file=sys.stderr)
        return
    print(f"[{src_dir.name}] {len(rollouts)} rollout(s); dropping all {drop_token} steps")
    for rollout_dir in rollouts:
        out_dir = out_root / rollout_dir.name
        if out_dir.exists():
            print(f"[skip] {out_dir} already exists (id collision?)", file=sys.stderr)
            continue
        kept, removed = clean_rollout(rollout_dir, drop_token, out_dir)
        print(f"  {rollout_dir.name}: kept {kept}, removed {removed} {drop_token}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean 0627 grasp/release rollouts.")
    parser.add_argument("--grasp-dir", type=Path, required=True, help="Raw grasp/ folder (RELEASE dropped).")
    parser.add_argument("--release-dir", type=Path, required=True, help="Raw release/ folder (GRASP dropped).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output 0627_cleaned folder.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    process_folder(args.grasp_dir, drop_token="RELEASE", out_root=args.out_dir)
    process_folder(args.release_dir, drop_token="GRASP", out_root=args.out_dir)

    total = sorted(d.name for d in args.out_dir.iterdir() if d.is_dir())
    print(f"\nWrote {len(total)} cleaned rollout(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
