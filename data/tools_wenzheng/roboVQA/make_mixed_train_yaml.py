#!/usr/bin/env python3
"""Create a Qwen3.5 VLM training yaml that mixes current robot data with RoboVQA."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_BASE_CONFIG = Path(
    "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/"
    "qwen3_5_9b_03_just_mix_zwz_new_prompt_add_horizon_flip.yaml"
)
DEFAULT_OUTPUT_CONFIG = Path(
    "examples/train_lora/qwen3_5_9b/mix_22-06_fk-pp/"
    "qwen3_5_9b_03_just_mix_zwz_new_prompt_add_horizon_flip_plus_robovqa.yaml"
)
DEFAULT_RAW_DIR = Path("/storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA")


def replace_or_append(lines: list[str], key: str, value: str, insert_after: str | None = None) -> list[str]:
    pattern = re.compile(rf"^({re.escape(key)}\s*:).*$")
    replaced = False
    out: list[str] = []
    for line in lines:
        if pattern.match(line):
            out.append(f"{key}: {value}\n")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        new_line = f"{key}: {value}\n"
        if insert_after is not None:
            for idx, line in enumerate(out):
                if line.startswith(f"{insert_after}:"):
                    out.insert(idx + 1, new_line)
                    break
            else:
                out.append(new_line)
        else:
            out.append(new_line)
    return out


def parse_scalar(lines: list[str], key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.*?)\s*(?:#.*)?$")
    for line in lines:
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def ensure_video_keys(lines: list[str], video_max_pixels: int, video_min_pixels: int, video_fps: float, video_maxlen: int) -> list[str]:
    values = {
        "video_max_pixels": str(video_max_pixels),
        "video_min_pixels": str(video_min_pixels),
        "video_fps": str(video_fps),
        "video_maxlen": str(video_maxlen),
    }
    existing = {key for key in values if parse_scalar(lines, key) is not None}
    for key in existing:
        lines = replace_or_append(lines, key, values[key])

    missing = [key for key in values if key not in existing]
    if not missing:
        return lines

    insert_at = 0
    for idx, line in enumerate(lines):
        if line.startswith("model_name_or_path:"):
            insert_at = idx + 1
            break

    block = [f"{key}: {values[key]}\n" for key in missing]
    return lines[:insert_at] + block + lines[insert_at:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--robovqa-dataset", default="robovqa_reasoning_lf")
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--video-max-pixels", type=int, default=65536)
    parser.add_argument("--video-min-pixels", type=int, default=1024)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-maxlen", type=int, default=64)
    parser.add_argument("--output-dir-suffix", default="_plus_robovqa")
    parser.add_argument("--run-name-suffix", default="-plus-robovqa")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lines = args.base_config.read_text(encoding="utf-8").splitlines(keepends=True)
    robot_dataset = parse_scalar(lines, "dataset")
    if not robot_dataset:
        raise SystemExit(f"ERROR: could not find dataset: in {args.base_config}")

    mixed_dataset = f"{robot_dataset},{args.robovqa_dataset}"
    lines = replace_or_append(lines, "dataset", mixed_dataset)
    lines = replace_or_append(lines, "media_dir", str(args.media_dir), insert_after="dataset")
    lines = replace_or_append(lines, "cutoff_len", str(args.cutoff_len))
    lines = ensure_video_keys(lines, args.video_max_pixels, args.video_min_pixels, args.video_fps, args.video_maxlen)

    old_output_dir = parse_scalar(lines, "output_dir")
    if old_output_dir:
        lines = replace_or_append(lines, "output_dir", old_output_dir + args.output_dir_suffix)
    old_run_name = parse_scalar(lines, "run_name")
    if old_run_name:
        lines = replace_or_append(lines, "run_name", old_run_name + args.run_name_suffix)

    text = "".join(lines)
    if args.dry_run:
        print(text)
        return

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(text, encoding="utf-8")
    print(f"[yaml] wrote {args.output_config}")
    print(f"[yaml] dataset: {mixed_dataset}")


if __name__ == "__main__":
    main()
