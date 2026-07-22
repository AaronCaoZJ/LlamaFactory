#!/usr/bin/env python3
"""Convert RoboVQA JSON arrays into LLaMA-Factory video JSONL files.

The raw RoboVQA files are huge JSON arrays. This script streams them with ijson and
writes Alpaca-style samples:

  {"instruction": "<video>...", "input": "", "output": "...", "videos": ["clips/x.mp4"]}

Use the full assistant content for reasoning by default, preserving <think> and <answer>.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_RAW_DIR = Path("/storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA")
DEFAULT_OUTPUT_DIR = Path("data/robovqa")
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"</answer>", re.IGNORECASE)
ANSWER_OR_THINK_TAG_RE = re.compile(r"</?(?:answer|think)>", re.IGNORECASE)
ANSWER_FORMAT_RE = re.compile(
    r"\s*Please answer the question in the following format:\s*"
    r"<think>\s*your reasoning\s*</think>\s*"
    r"<answer>\s*your answer\s*</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)


def extract_answer_only(text: str) -> str:
    """Extract the final answer block, tolerating messy raw traces."""
    openings = list(ANSWER_OPEN_RE.finditer(text))
    if openings:
        answer = text[openings[-1].end() :]
        answer = ANSWER_CLOSE_RE.split(answer, maxsplit=1)[0]
    else:
        matches = list(ANSWER_RE.finditer(text))
        answer = matches[-1].group(1) if matches else text

    return ANSWER_OR_THINK_TAG_RE.sub("", answer).strip()


def require_ijson():
    try:
        import ijson  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ERROR: ijson is required for streaming the RoboVQA JSON files. "
            "Install it in the active environment, then rerun this script."
        ) from exc

    return ijson


def parse_shards(value: str) -> list[int]:
    shards: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            shards.extend(range(int(left), int(right) + 1))
        else:
            shards.append(int(part))

    bad = [shard for shard in shards if shard < 0 or shard > 4]
    if bad:
        raise argparse.ArgumentTypeError(f"RoboVQA reasoning shards must be in 0..4, got {bad}")
    return list(dict.fromkeys(shards))


def iter_json_array(path: Path):
    ijson = require_ijson()
    with path.open("rb") as f:
        yield from ijson.items(f, "item")


def get_turns(obj: dict) -> tuple[str, str, str]:
    conversations = obj.get("conversations") or []
    if len(conversations) < 3:
        raise ValueError("expected at least system/user/assistant turns")
    system = str(conversations[0].get("content") or "")
    user = str(conversations[1].get("content") or "")
    assistant = str(conversations[2].get("content") or "")
    return system, user, assistant


def build_sample(
    obj: dict,
    *,
    answer_mode: str,
    include_system: bool,
    validate_media: bool,
    skip_missing_media: bool,
    clean_answer_only_prompt: bool,
    raw_dir: Path,
) -> tuple[dict | None, bool]:
    system, user, assistant = get_turns(obj)
    video = str(obj.get("video") or "")
    if not video:
        raise ValueError("sample is missing the video field")

    media_missing = validate_media and not (raw_dir / video).is_file()
    if media_missing and skip_missing_media:
        return None, True

    if answer_mode == "answer-only":
        assistant = extract_answer_only(assistant)
        if clean_answer_only_prompt:
            user = ANSWER_FORMAT_RE.sub("", user).rstrip()

    instruction = "<video>" + user
    if include_system and system:
        instruction = f"{system}\n\n{instruction}"

    return {
        "instruction": instruction,
        "input": "",
        "output": assistant,
        "videos": [video],
    }, media_missing


def output_path_for(kind: str, output_dir: Path, name_suffix: str) -> Path:
    suffix = name_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    return output_dir / f"robovqa_{kind}_lf{suffix}.jsonl"


def convert_files(
    paths: Iterable[Path],
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen = 0
    missing_media = 0
    skipped_missing = 0

    with output_path.open("w", encoding="utf-8") as out:
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            print(f"[convert] reading {path}")
            for obj in iter_json_array(path):
                seen += 1
                if args.max_samples is not None and written >= args.max_samples:
                    break
                sample, media_missing = build_sample(
                    obj,
                    answer_mode=args.answer_mode,
                    include_system=args.include_system,
                    validate_media=args.validate_media,
                    skip_missing_media=args.skip_missing_media,
                    clean_answer_only_prompt=args.clean_answer_only_prompt,
                    raw_dir=args.raw_dir,
                )
                if media_missing:
                    missing_media += 1
                if sample is None:
                    skipped_missing += 1
                    continue
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
            if args.max_samples is not None and written >= args.max_samples:
                break

    print(f"[convert] wrote          : {written}")
    print(f"[convert] raw rows seen  : {seen}")
    print(f"[convert] output         : {output_path}")
    if args.validate_media:
        print(f"[convert] missing media  : {missing_media}")
        print(f"[convert] skipped missing: {skipped_missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kind", choices=["reasoning", "understanding", "both"], default="reasoning")
    parser.add_argument("--reasoning-shards", type=parse_shards, default=parse_shards("0-4"))
    parser.add_argument("--name-suffix", default="", help="Suffix for output file names, e.g. 20k -> *_20k.jsonl.")
    parser.add_argument("--max-samples", type=int, default=None, help="Write at most this many samples per selected kind.")
    parser.add_argument(
        "--answer-mode",
        choices=["full", "answer-only"],
        default="full",
        help="For reasoning, keep full <think>/<answer> output or train only on the parsed answer.",
    )
    parser.add_argument(
        "--clean-answer-only-prompt",
        action="store_true",
        help="When using answer-only targets, remove the user request for <think>/<answer> formatting.",
    )
    parser.add_argument("--include-system", action="store_true", help="Prefix the system turn into instruction text.")
    parser.add_argument("--validate-media", action="store_true", help="Check that each referenced mp4 exists.")
    parser.add_argument(
        "--skip-missing-media",
        action="store_true",
        help="When validating media, skip samples whose mp4 is missing instead of writing them.",
    )
    args = parser.parse_args()
    args.raw_dir = args.raw_dir.resolve()

    selected: list[tuple[str, list[Path]]] = []
    if args.kind in {"reasoning", "both"}:
        selected.append(
            (
                "reasoning",
                [args.raw_dir / f"robovqa_reasoning_{shard}.json" for shard in args.reasoning_shards],
            )
        )
    if args.kind in {"understanding", "both"}:
        selected.append(("understanding", [args.raw_dir / "robovqa_understanding.json"]))

    for kind, paths in selected:
        output_path = output_path_for(kind, args.output_dir, args.name_suffix)
        convert_files(paths, output_path, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
