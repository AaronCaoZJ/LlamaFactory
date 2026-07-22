#!/usr/bin/env python3
"""Inspect prompt bodies embedded in a LLaMA-Factory rollout file.

The rollout samples store the full rendered prompt in ``instruction``. This
script parses those instructions and reports:

- number of samples
- number of unique full prompt bodies (after removing <image>/<video> markers)
- number of unique task strings
- task -> counts / recent-move variants / example prompt mapping

Use ``--include-prompt-bodies`` when you want every distinct prompt body copied
verbatim into the JSON report. For large rollouts this can be thousands of
entries, because ``Recent moves`` is part of the prompt.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MEDIA_PREFIX_RE = re.compile(r"^(?:(?:<image>)|(?:<video>))+")
TASK_RE = re.compile(r"^Task:\s*(?P<task>.+?)\s*$", re.MULTILINE)
RECENT_RE = re.compile(
    r"^Recent(?:\s+(?:previous|past|historical))?\s+moves,\s+newest\s+first:\s*(?P<recent>.*?)\s*$",
    re.MULTILINE,
)


def load_samples(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON array or JSON Lines samples.")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"sample {idx} is not a JSON object.")
    return data


def strip_media_prefix(instruction: str) -> tuple[str, str]:
    match = MEDIA_PREFIX_RE.match(instruction)
    if match is None:
        return "", instruction
    return match.group(0), instruction[match.end():]


def parse_prompt_body(body: str, idx: int) -> tuple[str, str]:
    task_match = TASK_RE.search(body)
    recent_match = RECENT_RE.search(body)
    if task_match is None:
        raise ValueError(f"sample {idx}: cannot find a Task line.")
    if recent_match is None:
        raise ValueError(f"sample {idx}: cannot find a Recent moves line.")
    return task_match.group("task"), recent_match.group("recent")


def image_root(sample: dict[str, Any]) -> str:
    images = sample.get("images")
    if not isinstance(images, list) or not images:
        return ""

    image = str(images[0])
    if "/MVTOKEN/" in image:
        image = image.split("/MVTOKEN/", 1)[1]
    parts = image.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else image


def build_report(samples: list[dict[str, Any]], include_prompt_bodies: bool) -> dict[str, Any]:
    prompt_counts: Counter[str] = Counter()
    prompt_meta: dict[str, dict[str, Any]] = {}
    task_counts: Counter[str] = Counter()
    task_recent_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_root_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_example_prompt: dict[str, str] = {}
    media_prefix_counts: Counter[str] = Counter()

    for idx, sample in enumerate(samples):
        instruction = sample.get("instruction")
        if not isinstance(instruction, str):
            raise TypeError(f"sample {idx}: missing string instruction.")

        media_prefix, body = strip_media_prefix(instruction)
        task, recent_moves = parse_prompt_body(body, idx)
        root = image_root(sample)

        prompt_counts[body] += 1
        task_counts[task] += 1
        task_recent_counts[task][recent_moves] += 1
        task_root_counts[task][root] += 1
        media_prefix_counts[media_prefix] += 1
        task_example_prompt.setdefault(task, body)

        if body not in prompt_meta:
            prompt_meta[body] = {
                "first_index": idx,
                "media_prefix": media_prefix,
                "task": task,
                "recent_moves": recent_moves,
                "image_root": root,
            }

    tasks = []
    for task, count in task_counts.most_common():
        recent_counts = task_recent_counts[task]
        root_counts = task_root_counts[task]
        tasks.append(
            {
                "task": task,
                "count": count,
                "unique_recent_moves": len(recent_counts),
                "top_recent_moves": [
                    {"recent_moves": recent, "count": recent_count}
                    for recent, recent_count in recent_counts.most_common(10)
                ],
                "image_roots": [
                    {"root": root, "count": root_count}
                    for root, root_count in root_counts.most_common()
                    if root
                ],
                "example_prompt_body": task_example_prompt[task],
            }
        )

    report: dict[str, Any] = {
        "total_samples": len(samples),
        "unique_prompt_bodies": len(prompt_counts),
        "unique_tasks": len(task_counts),
        "media_prefixes": [
            {"prefix": prefix, "count": count}
            for prefix, count in media_prefix_counts.most_common()
        ],
        "tasks": tasks,
    }

    if include_prompt_bodies:
        report["prompt_bodies"] = [
            {
                "id": i,
                "count": count,
                **prompt_meta[body],
                "prompt_body": body,
            }
            for i, (body, count) in enumerate(prompt_counts.most_common())
        ]

    return report


def print_text_summary(report: dict[str, Any], print_example_prompts: bool) -> None:
    print(f"total_samples: {report['total_samples']}")
    print(f"unique_prompt_bodies: {report['unique_prompt_bodies']}")
    print(f"unique_tasks: {report['unique_tasks']}")
    print()
    print("tasks:")
    for item in report["tasks"]:
        print(f"- count={item['count']} unique_recent_moves={item['unique_recent_moves']} :: {item['task']}")
        roots = ", ".join(f"{r['root']} ({r['count']})" for r in item["image_roots"])
        if roots:
            print(f"  image_roots: {roots}")
        if print_example_prompts:
            print("  example_prompt_body:")
            print(_indent(item["example_prompt_body"], "    "))


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input rollout JSON/JSONL.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--include-prompt-bodies",
        action="store_true",
        help="Include every unique full prompt body verbatim in the JSON report.",
    )
    parser.add_argument(
        "--print-example-prompts",
        action="store_true",
        help="Print one full prompt body per task in the text summary.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON report to stdout instead of the compact text summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input)
    report = build_report(samples, include_prompt_bodies=args.include_prompt_bodies)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote report -> {args.output}")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.output is None:
        print_text_summary(report, print_example_prompts=args.print_example_prompts)


if __name__ == "__main__":
    main()
