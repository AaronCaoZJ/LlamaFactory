#!/usr/bin/env python3
"""Mix two cleaned mikomiko_tag train jsonl files into one training file.

The default inputs are the two cleaned training sets already present in this
directory:
  - jsonl/cleaned/train.jsonl
  - jsonl_0716/train.jsonl

The merge is streaming and does not load either file into memory. Records are
written in round-robin blocks so the final file is physically mixed instead of
just appended.

Usage:
    python mix_train_jsonl.py
    python mix_train_jsonl.py --output /path/to/mix_train.jsonl
    python mix_train_jsonl.py --block-size 8
"""

from __future__ import annotations

import argparse
import os
from typing import Iterator, Sequence, Tuple


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUTS = (
    os.path.join(HERE, "jsonl", "cleaned", "train.jsonl"),
    os.path.join(HERE, "jsonl_0716", "train.jsonl"),
)
DEFAULT_OUTPUT = os.path.join(HERE, "mix_train.jsonl")


def resolve_input(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "train.jsonl")
        if os.path.exists(candidate):
            return candidate
    return path


def mixed_lines(paths: Sequence[str], block_size: int) -> Iterator[Tuple[int, str]]:
    handles = [open(path, encoding="utf-8") for path in paths]
    try:
        active = [True] * len(handles)
        while any(active):
            for index, handle in enumerate(handles):
                if not active[index]:
                    continue
                emitted = 0
                while emitted < block_size:
                    line = handle.readline()
                    if not line:
                        active[index] = False
                        break
                    emitted += 1
                    yield index, line
    finally:
        for handle in handles:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        help="input jsonl file or directory; may be passed multiple times",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"output jsonl file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="number of consecutive rows to take from each source before switching",
    )
    args = parser.parse_args()

    inputs = [resolve_input(path) for path in (args.inputs or list(DEFAULT_INPUTS))]
    if len(inputs) < 2:
        raise SystemExit("[fatal] need at least two input files")
    for path in inputs:
        if not os.path.exists(path):
            raise SystemExit(f"[fatal] missing input: {path}")
    if args.block_size < 1:
        raise SystemExit("[fatal] --block-size must be >= 1")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    counts = [0] * len(inputs)
    total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for source_index, line in mixed_lines(inputs, args.block_size):
            out.write(line)
            counts[source_index] += 1
            total += 1

    print("[mix] inputs:")
    for path, count in zip(inputs, counts):
        print(f"[mix]   {path} -> {count:,} rows")
    print(f"[mix] output: {args.output} -> {total:,} rows")


if __name__ == "__main__":
    main()