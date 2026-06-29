#!/usr/bin/env python3
"""Merge several LLaMA-Factory rollout sample files into one.

Each input is a converter output from ``rollout_to_llamafactory.py`` -- a list of
``{instruction, input, output, images}`` samples (JSON array or JSON Lines). Samples carry
ABSOLUTE image paths, so merging is a plain concatenation in the given order (no path
rewriting). Used to mix datasets, e.g. the 0622 and 0627_cleaned v3 lite sets ->
MVTOKEN/mix_22_27/rollout_lite.json.

Usage:
    python data/agentrobot/merge_rollouts.py \\
        data/agentrobot/MVTOKEN/0622/v3/rollout_lite.json \\
        data/agentrobot/MVTOKEN/0627_cleaned/v3/rollout_lite.json \\
        --output data/agentrobot/MVTOKEN/mix_22_27/rollout_lite.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_samples(path: Path) -> list[dict]:
    """Load a rollout file as a list of samples (JSON array or JSON Lines)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        # A single JSON object -> wrap it.
        return [data]
    except json.JSONDecodeError:
        # Fall back to JSON Lines.
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate rollout sample files into one.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input rollout JSON/JSONL files.")
    parser.add_argument("--output", type=Path, required=True, help="Merged output path.")
    parser.add_argument(
        "--jsonl", action="store_true", default=False,
        help="Write JSON Lines instead of a JSON array.",
    )
    args = parser.parse_args()

    merged: list[dict] = []
    for path in args.inputs:
        if not path.exists():
            print(f"[error] input not found: {path}", file=sys.stderr)
            sys.exit(1)
        samples = _load_samples(path)
        print(f"  {path}: {len(samples)} samples")
        merged.extend(samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        if args.jsonl:
            for sample in merged:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(merged)} samples -> {args.output}")


if __name__ == "__main__":
    main()
