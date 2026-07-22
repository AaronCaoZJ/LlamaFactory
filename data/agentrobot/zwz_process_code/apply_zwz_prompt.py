#!/usr/bin/env python3
"""Apply ZWZ prompt templates to rollout_lite.json files.

This keeps every sample's labels, images, and other metadata unchanged. Only
``instruction`` is rebuilt:

1. Parse the old instruction's media prefix, usually ``<image><image>``.
2. Parse the per-sample task from ``Task: ...``.
3. Parse the per-sample recent moves from either:
   - ``Recent moves, newest first: ...``
   - ``Recent previous moves, newest first: ...``
4. Render a ZWZ prompt template with ``{task}`` and ``{recent_moves}``.
5. Write the new instruction as ``media_prefix + rendered_prompt``.

Two dataset-specific modes are provided for the MVTOKEN mix:

  mixed-piper-franka : piper samples use prompt_piper_zwz.txt, franka samples use
                       prompt_franka_zwz.txt. Piper is detected from image paths
                       containing 0705_piper or 0706_piper.
  franka-only        : all samples use prompt_franka_zwz.txt, and the script
                       errors out if any piper sample is present.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = SCRIPT_DIR / "prompt_piper_zwz.txt"
DEFAULT_PIPER_PROMPT = SCRIPT_DIR / "prompt_piper_zwz.txt"
DEFAULT_FRANKA_PROMPT = SCRIPT_DIR / "prompt_franka_zwz.txt"
DEFAULT_OUTPUT_NAME = "rollout_lite_zwz.json"
PIPER_SOURCE_MARKERS = ("0705_piper", "0706_piper")

MEDIA_PREFIX_RE = re.compile(r"^(?:(?:<image>)|(?:<video>))+")
TASK_RE = re.compile(r"^Task:\s*(?P<task>.+?)\s*$", re.MULTILINE)
RECENT_RE = re.compile(
    r"^Recent(?:\s+(?:previous|past|historical))?\s+moves,\s+newest\s+first:\s*(?P<recent>.*?)\s*$",
    re.MULTILINE,
)

ALLOWED_FIELDS = {"task", "recent_moves"}
REQUIRED_FIELDS = {"task", "recent_moves"}


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
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise TypeError(f"sample {idx} is not a JSON object.")
    return data


def template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name:
            root = re.split(r"[.[]", field_name, maxsplit=1)[0]
            fields.add(root)
    return fields


def validate_template(template: str, prompt_path: Path) -> None:
    fields = template_fields(template)
    unknown = fields - ALLOWED_FIELDS
    if unknown:
        raise ValueError(
            f"{prompt_path} has unsupported field(s): {', '.join(sorted(unknown))}. "
            "Only {task} and {recent_moves} are supported."
        )

    missing = REQUIRED_FIELDS - fields
    if missing:
        raise ValueError(
            f"{prompt_path} must contain: " + ", ".join(f"{{{name}}}" for name in sorted(missing))
        )


def split_instruction(instruction: str, idx: int) -> tuple[str, str]:
    match = MEDIA_PREFIX_RE.match(instruction)
    if match is None:
        raise ValueError(f"sample {idx}: instruction does not start with <image>/<video> tokens.")
    return match.group(0), instruction[match.end():]


def parse_context(body: str, idx: int) -> dict[str, str]:
    task_match = TASK_RE.search(body)
    recent_match = RECENT_RE.search(body)
    if task_match is None:
        raise ValueError(f"sample {idx}: cannot find a 'Task:' line.")
    if recent_match is None:
        raise ValueError(f"sample {idx}: cannot find a 'Recent moves' line.")

    return {
        "task": task_match.group("task"),
        "recent_moves": recent_match.group("recent"),
    }


def sample_source(sample: dict[str, Any], idx: int) -> str:
    images = sample.get("images")
    if not isinstance(images, list) or not images:
        raise TypeError(f"sample {idx}: missing non-empty 'images' list.")

    image_text = " ".join(str(image) for image in images)
    if any(marker in image_text for marker in PIPER_SOURCE_MARKERS):
        return "piper"
    return "franka"


def apply_prompt(samples: list[dict[str, Any]], template: str) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        instruction = sample.get("instruction")
        if not isinstance(instruction, str):
            raise TypeError(f"sample {idx}: missing string 'instruction'.")

        media_prefix, body = split_instruction(instruction, idx)
        values = parse_context(body, idx)

        item = dict(sample)
        item["instruction"] = media_prefix + template.format(**values)
        rewritten.append(item)

    return rewritten


def apply_mixed_prompts(
    samples: list[dict[str, Any]],
    piper_template: str,
    franka_template: str,
    *,
    franka_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rewritten: list[dict[str, Any]] = []
    counts = {"piper": 0, "franka": 0}

    for idx, sample in enumerate(samples):
        instruction = sample.get("instruction")
        if not isinstance(instruction, str):
            raise TypeError(f"sample {idx}: missing string 'instruction'.")

        source = sample_source(sample, idx)
        if franka_only and source != "franka":
            raise ValueError(
                f"sample {idx}: franka-only mode found a piper sample: {sample.get('images')}"
            )

        media_prefix, body = split_instruction(instruction, idx)
        values = parse_context(body, idx)
        template = piper_template if source == "piper" else franka_template

        item = dict(sample)
        item["instruction"] = media_prefix + template.format(**values)
        rewritten.append(item)
        counts[source] += 1

    return rewritten, counts


def write_samples(path: Path, samples: list[dict[str, Any]], jsonl: bool) -> None:
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
    parser.add_argument(
        "--mode",
        choices=("single", "mixed-piper-franka", "franka-only"),
        default="single",
        help=(
            "Prompt application mode. 'single' applies --prompt-file to every sample; "
            "'mixed-piper-franka' chooses piper/franka prompts from image paths; "
            "'franka-only' applies the franka prompt and rejects piper samples."
        ),
    )
    parser.add_argument("--output", type=Path, help="Output rollout JSON/JSONL.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT,
        help=f"Single-mode prompt template path. Default: {DEFAULT_PROMPT}",
    )
    parser.add_argument(
        "--piper-prompt-file",
        type=Path,
        default=DEFAULT_PIPER_PROMPT,
        help=f"Piper prompt template path for mixed mode. Default: {DEFAULT_PIPER_PROMPT}",
    )
    parser.add_argument(
        "--franka-prompt-file",
        type=Path,
        default=DEFAULT_FRANKA_PROMPT,
        help=f"Franka prompt template path for mixed/franka-only modes. Default: {DEFAULT_FRANKA_PROMPT}",
    )
    parser.add_argument("--jsonl", action="store_true", help="Write JSON Lines instead of a JSON array.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first rewritten instruction only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input)

    if args.mode == "single":
        template = args.prompt_file.read_text(encoding="utf-8").rstrip("\n")
        validate_template(template, args.prompt_file)
        rewritten = apply_prompt(samples, template)
        counts = {"all": len(rewritten)}
    else:
        piper_template = args.piper_prompt_file.read_text(encoding="utf-8").rstrip("\n")
        franka_template = args.franka_prompt_file.read_text(encoding="utf-8").rstrip("\n")
        validate_template(piper_template, args.piper_prompt_file)
        validate_template(franka_template, args.franka_prompt_file)
        rewritten, counts = apply_mixed_prompts(
            samples,
            piper_template,
            franka_template,
            franka_only=args.mode == "franka-only",
        )

    if args.dry_run:
        if rewritten:
            print(rewritten[0]["instruction"])
        print(f"\n[dry-run] rewritten samples: {len(rewritten)} counts={counts}", file=sys.stderr)
        return

    output = args.output if args.output is not None else args.input.with_name(DEFAULT_OUTPUT_NAME)

    write_samples(output, rewritten, args.jsonl)
    print(f"Wrote {len(rewritten)} samples -> {output}")
    print("Counts: " + ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
