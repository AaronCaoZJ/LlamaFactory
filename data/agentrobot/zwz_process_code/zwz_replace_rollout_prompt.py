#!/usr/bin/env python3
"""Replace the rendered prompt inside rollout_lite.json samples.

The script preserves sample-specific media tokens, image paths, labels, and all
other fields. It only rebuilds ``instruction`` from a template using fields
parsed from the existing instruction:

  {task}          from the ``Task: ...`` line
  {recent_moves}  from the ``Recent moves, newest first: ...`` line

This is useful when a rollout has the correct images/actions but should be
trained with a different prompt wording.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Any


MEDIA_PREFIX_RE = re.compile(r"^(?:(?:<image>)|(?:<video>))+")
TASK_RE = re.compile(r"^Task:\s*(?P<task>.+?)\s*$", re.MULTILINE)
RECENT_RE = re.compile(r"^Recent moves, newest first:\s*(?P<recent>.*?)\s*$", re.MULTILINE)
GRIPPER_RE = re.compile(r"^Gripper now:\s*(?P<gripper>.+?)\s*$", re.MULTILINE)

ALLOWED_TEMPLATE_FIELDS = {"task", "recent_moves", "gripper_state"}
REQUIRED_TEMPLATE_FIELDS = {"task", "recent_moves"}


def _load_samples(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON array or JSON Lines samples.")
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise TypeError(f"sample {idx} is not a JSON object.")
    return data


def _template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name:
            # Reject complex formatting expressions early; this is intentionally
            # just a prompt template, not a mini data language.
            root = re.split(r"[.[]", field_name, maxsplit=1)[0]
            fields.add(root)
    return fields


def _load_template(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        template = args.prompt_file.read_text(encoding="utf-8")
    else:
        template = args.prompt

    # The converter historically used prompt files without a meaningful trailing
    # newline. Dropping only final newlines keeps accidental editor EOF newlines
    # from becoming part of every instruction.
    return template.rstrip("\n")


def _split_instruction(instruction: str, idx: int) -> tuple[str, str]:
    match = MEDIA_PREFIX_RE.match(instruction)
    if match is None:
        raise ValueError(f"sample {idx}: instruction does not start with <image>/<video> tokens.")
    return match.group(0), instruction[match.end():]


def _extract_context(body: str, idx: int, default_gripper_state: str | None) -> dict[str, str]:
    task_match = TASK_RE.search(body)
    recent_match = RECENT_RE.search(body)
    if task_match is None:
        raise ValueError(f"sample {idx}: cannot find a 'Task:' line in instruction.")
    if recent_match is None:
        raise ValueError(f"sample {idx}: cannot find a 'Recent moves, newest first:' line in instruction.")

    values = {
        "task": task_match.group("task"),
        "recent_moves": recent_match.group("recent"),
    }

    gripper_match = GRIPPER_RE.search(body)
    if gripper_match is not None:
        values["gripper_state"] = gripper_match.group("gripper")
    elif default_gripper_state is not None:
        values["gripper_state"] = default_gripper_state

    return values


def _replace_prompts(
    samples: list[dict[str, Any]],
    template: str,
    default_gripper_state: str | None,
) -> list[dict[str, Any]]:
    fields = _template_fields(template)
    unknown = fields - ALLOWED_TEMPLATE_FIELDS
    if unknown:
        raise ValueError(f"unsupported template field(s): {', '.join(sorted(unknown))}")

    missing_required = REQUIRED_TEMPLATE_FIELDS - fields
    if missing_required:
        raise ValueError(
            "template should preserve per-sample fields: "
            + ", ".join(f"{{{name}}}" for name in sorted(missing_required))
        )

    new_samples: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        instruction = sample.get("instruction")
        if not isinstance(instruction, str):
            raise TypeError(f"sample {idx}: missing string 'instruction'.")

        media_prefix, body = _split_instruction(instruction, idx)
        values = _extract_context(body, idx, default_gripper_state)
        missing_for_format = fields - values.keys()
        if missing_for_format:
            raise ValueError(
                f"sample {idx}: template needs {sorted(missing_for_format)}, "
                "but the existing instruction does not contain it."
            )

        replaced = dict(sample)
        replaced["instruction"] = media_prefix + template.format(**values)
        new_samples.append(replaced)

    return new_samples


def _write_samples(path: Path, samples: list[dict[str, Any]], jsonl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if jsonl:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            json.dump(samples, f, ensure_ascii=False, indent=2)
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input rollout JSON/JSONL.")
    parser.add_argument("--output", type=Path, help="Output rollout JSON/JSONL.")

    template_group = parser.add_mutually_exclusive_group(required=True)
    template_group.add_argument("--prompt-file", type=Path, help="Prompt template text file.")
    template_group.add_argument("--prompt", help="Prompt template string.")

    parser.add_argument(
        "--default-gripper-state",
        choices=("open", "closed"),
        help="Fallback value for templates that include {gripper_state}.",
    )
    parser.add_argument("--jsonl", action="store_true", help="Write JSON Lines instead of a JSON array.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first rewritten instruction only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = _load_template(args)
    samples = _load_samples(args.input)
    new_samples = _replace_prompts(samples, template, args.default_gripper_state)

    if args.dry_run:
        if new_samples:
            print(new_samples[0]["instruction"])
        print(f"\n[dry-run] rewritten samples: {len(new_samples)}", file=sys.stderr)
        return

    if args.output is None:
        raise SystemExit("--output is required unless --dry-run is used.")
    _write_samples(args.output, new_samples, args.jsonl)
    print(f"Wrote {len(new_samples)} samples -> {args.output}")


if __name__ == "__main__":
    main()
