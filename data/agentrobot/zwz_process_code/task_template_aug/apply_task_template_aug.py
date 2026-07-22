#!/usr/bin/env python3
"""Apply fixed task-description templates to a LLaMA-Factory rollout JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/agentrobot/MVTOKEN/mix_22-06_fk-pp/02_exchange_token/rollout_lite.json")
DEFAULT_TEMPLATES = Path(__file__).with_name("task_templates.json")
TASK_RE = re.compile(r"(^Task:\s*)(.+)$", re.MULTILINE)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def extract_task(instruction: str) -> str:
    match = TASK_RE.search(instruction)
    if not match:
        raise ValueError("instruction has no 'Task:' line")
    return match.group(2).strip()


def replace_task(instruction: str, new_task: str) -> str:
    replaced, count = TASK_RE.subn(lambda match: f"{match.group(1)}{new_task}", instruction, count=1)
    if count != 1:
        raise ValueError("instruction has no replaceable 'Task:' line")
    return replaced


def rollout_key(sample: dict[str, Any], idx: int) -> str:
    images = sample.get("images") or []
    if images:
        path = Path(str(images[0]))
        parts = path.parts
        for part_idx, part in enumerate(parts):
            if part.startswith("rollout_"):
                return str(Path(*parts[: part_idx + 1]))
        return str(path.parent)
    return f"sample:{idx}"


def rollout_suffix(sample: dict[str, Any], idx: int) -> str:
    images = sample.get("images") or []
    if not images:
        return rollout_key(sample, idx)

    path = Path(str(images[0]))
    parts = path.parts
    try:
        start = parts.index("MVTOKEN") + 1
    except ValueError:
        start = 0

    for part_idx, part in enumerate(parts[start:], start=start):
        if part.startswith("rollout_"):
            begin = start if start < part_idx else max(0, part_idx - 2)
            return str(Path(*parts[begin : part_idx + 1]))

    return str(path.parent)


def resolve_variants(
    *,
    task: str,
    sample: dict[str, Any],
    idx: int,
    templates: dict[str, Any],
) -> tuple[list[str] | None, str | None]:
    spec = templates.get(task)
    if spec is None:
        return None, None

    if isinstance(spec, list):
        return spec, task

    if not isinstance(spec, dict):
        raise ValueError(f"template spec for task {task!r} must be a list or object")

    suffix = rollout_suffix(sample, idx)
    by_rollout = spec.get("by_rollout") or {}
    variants = by_rollout.get(suffix)
    if variants is not None:
        return variants, f"{task} [{suffix}]"

    variants = spec.get("default")
    if variants is not None:
        return variants, f"{task} [default]"

    raise ValueError(f"template spec for task {task!r} has no default variants for rollout {suffix!r}")


def validate_variants(task: str, group_key: str, variants: Any) -> list[str]:
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"template variants for {group_key!r} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in variants):
        raise ValueError(f"template variants for {group_key!r} must all be non-empty strings")
    return variants


def choose_variant(
    *,
    assignment: str,
    assignment_key: str,
    sample: dict[str, Any],
    idx: int,
    variants: list[str],
    sample_counts: Counter[str],
    rollout_variant: dict[tuple[str, str], int],
    rollout_counts: Counter[str],
) -> int:
    if assignment == "sample":
        variant_idx = sample_counts[assignment_key] % len(variants)
        sample_counts[assignment_key] += 1
        return variant_idx

    if assignment == "hash":
        key = json.dumps(
            {
                "task": assignment_key,
                "images": sample.get("images") or [],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % len(variants)

    key = rollout_key(sample, idx)
    map_key = (assignment_key, key)
    if map_key not in rollout_variant:
        rollout_variant[map_key] = rollout_counts[assignment_key] % len(variants)
        rollout_counts[assignment_key] += 1
    return rollout_variant[map_key]


def apply_templates(samples: list[dict[str, Any]], templates: dict[str, Any], assignment: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_counts: Counter[str] = Counter()
    variant_counts: dict[str, Counter[int]] = defaultdict(Counter)
    rollout_counts: Counter[str] = Counter()
    rollout_variant: dict[tuple[str, str], int] = {}
    output: list[dict[str, Any]] = []
    missing: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()

    for idx, sample in enumerate(samples):
        instruction = sample.get("instruction")
        if not isinstance(instruction, str):
            raise ValueError(f"sample {idx}: instruction is not a string")

        task = extract_task(instruction)
        task_counts[task] += 1
        variants, group_key = resolve_variants(task=task, sample=sample, idx=idx, templates=templates)
        if not variants:
            missing[task] += 1
            continue
        assert group_key is not None
        variants = validate_variants(task, group_key, variants)

        variant_idx = choose_variant(
            assignment=assignment,
            assignment_key=group_key,
            sample=sample,
            idx=idx,
            variants=variants,
            sample_counts=sample_counts,
            rollout_variant=rollout_variant,
            rollout_counts=rollout_counts,
        )
        variant_counts[group_key][variant_idx] += 1
        updated = dict(sample)
        updated["instruction"] = replace_task(instruction, variants[variant_idx])
        output.append(updated)

    if missing:
        lines = [f"{count} x {task}" for task, count in missing.most_common()]
        raise ValueError("missing templates for tasks:\n" + "\n".join(lines))

    stats = {
        "samples": len(samples),
        "output_samples": len(output),
        "unique_tasks": len(task_counts),
        "unique_variant_groups": len(variant_counts),
        "assignment": assignment,
        "variant_counts": {
            group_key: {str(idx): count for idx, count in counts.items()} for group_key, counts in sorted(variant_counts.items())
        },
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument(
        "--assignment",
        choices=["rollout", "sample", "hash"],
        default="sample",
        help="How to assign the 6 variants. 'sample' maximizes wording diversity; 'rollout' keeps one wording per rollout.",
    )
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_task_template_aug{input_path.suffix}")
    )

    samples = load_json(input_path)
    if not isinstance(samples, list):
        raise ValueError(f"expected a JSON array in {input_path}")

    templates = load_json(args.templates)
    augmented, stats = apply_templates(samples, templates, args.assignment)

    print(json.dumps({k: v for k, v in stats.items() if k != "variant_counts"}, ensure_ascii=False, indent=2))
    print("[stats] per-task variant counts:")
    for task, counts in stats["variant_counts"].items():
        rendered = ", ".join(f"{idx}:{count}" for idx, count in sorted(counts.items(), key=lambda item: int(item[0])))
        print(f"  {task}: {rendered}")

    if args.dry_run:
        print(f"[dry-run] would write: {output_path}")
        return

    write_json(output_path, augmented)
    print(f"[write] {len(augmented)} samples -> {output_path}")

    if args.stats_output is not None:
        write_json(args.stats_output, stats)
        print(f"[write] stats -> {args.stats_output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
