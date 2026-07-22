#!/usr/bin/env python3
"""Register converted RoboVQA JSONL files in data/dataset_info.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DATASET_INFO = Path("data/dataset_info.json")


def dataset_entry(file_name: str, num_samples: int | None) -> dict:
    entry = {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "videos": "videos",
        },
    }
    if num_samples is not None:
        entry["num_samples"] = num_samples
    return entry


def add_entry(info: dict, name: str, file_name: str, num_samples: int | None, overwrite: bool) -> None:
    if name in info and not overwrite:
        print(f"[register] keep existing entry: {name}")
        return
    info[name] = dataset_entry(file_name, num_samples)
    print(f"[register] set {name} -> {file_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--only", choices=["reasoning", "understanding", "both"], default="reasoning")
    parser.add_argument("--name-suffix", default="", help="Suffix used by convert_to_llamafactory.py.")
    parser.add_argument("--reasoning-num-samples", type=int, default=None)
    parser.add_argument("--understanding-num-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing RoboVQA entries.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suffix = args.name_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    path = args.dataset_info
    info = json.loads(path.read_text(encoding="utf-8"))

    if args.only in {"reasoning", "both"}:
        add_entry(
            info,
            f"robovqa_reasoning_lf{suffix}",
            f"robovqa/robovqa_reasoning_lf{suffix}.jsonl",
            args.reasoning_num_samples,
            args.overwrite,
        )
    if args.only in {"understanding", "both"}:
        add_entry(
            info,
            f"robovqa_understanding_lf{suffix}",
            f"robovqa/robovqa_understanding_lf{suffix}.jsonl",
            args.understanding_num_samples,
            args.overwrite,
        )

    if args.dry_run:
        print(json.dumps(info, ensure_ascii=False, indent=2)[:4000])
        print("[register] dry-run only; no file written")
        return

    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[register] wrote {path}")


if __name__ == "__main__":
    main()
