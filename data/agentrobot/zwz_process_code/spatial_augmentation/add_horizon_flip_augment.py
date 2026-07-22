#!/usr/bin/env python3
"""Build left-right mirrored rollout samples for two-view robot data."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2


DEFAULT_INPUT = Path("data/agentrobot/MVTOKEN/mix_22-06_fk-pp/03_just_mix/rollout_lite_zwz_new_prompt.json")
DATA_MARKER = Path("data/agentrobot/MVTOKEN")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
LEFT_RIGHT_TOKEN_RE = re.compile(r"\bMV_(LEFT|RIGHT)\b")
HISTORY_RE = re.compile(r"(Recent (?:previous )?moves, newest first: )([^\n]*)")


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "data/agentrobot").exists():
            return parent
    raise RuntimeError("could not find LlamaFactory repository root")


REPO_ROOT = find_repo_root()


def swap_left_right_token(token: str) -> str:
    if token == "MV_LEFT":
        return "MV_RIGHT"
    if token == "MV_RIGHT":
        return "MV_LEFT"
    return token


def swap_left_right_tokens_in_text(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return swap_left_right_token(f"MV_{match.group(1)}")

    return LEFT_RIGHT_TOKEN_RE.sub(repl, text)


def swap_history_only(instruction: str) -> str:
    """Swap MV_LEFT/MV_RIGHT only in the rendered move history line."""

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{swap_left_right_tokens_in_text(match.group(2))}"

    return HISTORY_RE.sub(repl, instruction)


def swap_action_tokens(value: Any) -> Any:
    if isinstance(value, str):
        return swap_left_right_token(value)
    if isinstance(value, list):
        return [swap_action_tokens(item) for item in value]
    if isinstance(value, dict):
        return {key: swap_action_tokens(item) for key, item in value.items()}
    return value


def load_samples(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array in {path}, got {type(data).__name__}")
    return data


def write_samples(path: Path, samples: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def is_flipped_sample(sample: dict[str, Any]) -> bool:
    images = sample.get("images") or []
    return any("_horizon_flip" in str(path) for path in images)


def marker_relative(path: Path) -> Path:
    parts = path.parts
    marker_parts = DATA_MARKER.parts
    for idx in range(len(parts) - len(marker_parts) + 1):
        if parts[idx : idx + len(marker_parts)] == marker_parts:
            return Path(*parts[idx:])
    raise ValueError(f"path is not under {DATA_MARKER}: {path}")


def local_path_from_sample_path(path: str) -> Path:
    return REPO_ROOT / marker_relative(Path(path))


def rollout_dir_from_image(path: Path) -> Path:
    for parent in path.parents:
        if parent.name.startswith("rollout_"):
            return parent
    raise ValueError(f"could not find rollout_* directory for image: {path}")


def flipped_rollout_dir(source_rollout: Path) -> Path:
    if source_rollout.name.endswith("_horizon_flip"):
        return source_rollout
    return source_rollout.with_name(f"{source_rollout.name}_horizon_flip")


def flip_image_lr(src: Path, dst: Path) -> None:
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(src)
    flipped = cv2.flip(image, 1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), flipped):
        raise RuntimeError(f"failed to write image: {dst}")


def flip_video_lr(src: Path, dst: Path) -> None:
    """Mirror a rollout visualization video and write browser-playable H.264 MP4."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write playable flipped visualization videos")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(f".{dst.stem}.h264.tmp{dst.suffix}")
    if tmp_dst.exists():
        tmp_dst.unlink()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-map",
                "0:v:0",
                "-vf",
                "hflip",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                "20",
                str(tmp_dst),
            ],
            check=True,
        )
        os.replace(tmp_dst, dst)
    finally:
        tmp_dst.unlink(missing_ok=True)


def copy_and_flip_rollout(source_rollout: Path, target_rollout: Path, overwrite: bool) -> bool:
    if target_rollout.exists() and not overwrite:
        return False

    target_rollout.mkdir(parents=True, exist_ok=True)
    for src in source_rollout.rglob("*"):
        rel = src.relative_to(source_rollout)
        dst = target_rollout / rel
        if src.is_dir():
            dst.mkdir(exist_ok=True)
            continue

        if src.suffix.lower() in IMAGE_SUFFIXES:
            flip_image_lr(src, dst)
        elif src.name == "actions.jsonl":
            flip_actions_jsonl(src, dst)
        elif src.name == "metadata.json":
            flip_metadata_json(src, dst)
        elif src.suffix.lower() in VIDEO_SUFFIXES:
            flip_video_lr(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return True


def flip_actions_jsonl(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as f_in, open(dst, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            row = json.loads(line)
            if "token" in row:
                row["token"] = swap_left_right_token(row["token"])
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")


def flip_metadata_json(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = dict(data)
        if isinstance(data.get("rollout"), str) and not data["rollout"].endswith("_horizon_flip"):
            data["rollout"] = f"{data['rollout']}_horizon_flip"
        if "tokens" in data:
            data["tokens"] = swap_action_tokens(data["tokens"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def flipped_json_image_path(image_path: str) -> str:
    local_image = local_path_from_sample_path(image_path)
    source_rollout = rollout_dir_from_image(local_image)
    target_rollout = flipped_rollout_dir(source_rollout)
    flipped_image = target_rollout / local_image.relative_to(source_rollout)
    return str(flipped_image)


def make_flipped_sample(sample: dict[str, Any]) -> dict[str, Any]:
    images = sample.get("images")
    if not isinstance(images, list) or len(images) != 2:
        raise ValueError(f"expected exactly two images in sample, got {images!r}")

    flipped = dict(sample)
    flipped["images"] = [flipped_json_image_path(str(path)) for path in images]
    if isinstance(flipped.get("instruction"), str):
        flipped["instruction"] = swap_history_only(flipped["instruction"])
    if isinstance(flipped.get("output"), str):
        flipped["output"] = swap_left_right_token(flipped["output"])
    return flipped


def collect_rollout_pairs(samples: list[dict[str, Any]]) -> dict[Path, Path]:
    pairs: dict[Path, Path] = {}
    for idx, sample in enumerate(samples):
        images = sample.get("images")
        if not isinstance(images, list) or len(images) != 2:
            raise ValueError(f"sample {idx}: expected exactly two images")
        for image_path in images:
            local_image = local_path_from_sample_path(str(image_path))
            if not local_image.exists():
                raise FileNotFoundError(local_image)
            source_rollout = rollout_dir_from_image(local_image)
            pairs[source_rollout] = flipped_rollout_dir(source_rollout)
    return pairs


def augment_dataset(input_path: Path, output_path: Path, overwrite_rollouts: bool, dry_run: bool) -> None:
    samples = load_samples(input_path)
    original_samples = [sample for sample in samples if not is_flipped_sample(sample)]
    flipped_count_in_input = len(samples) - len(original_samples)
    rollout_pairs = collect_rollout_pairs(original_samples)

    print(f"input samples: {len(samples)}")
    print(f"base samples : {len(original_samples)}")
    print(f"existing flipped samples ignored for rebuild: {flipped_count_in_input}")
    print(f"unique rollout dirs to mirror: {len(rollout_pairs)}")
    print(f"output samples: {len(original_samples) * 2}")
    print(f"output json   : {output_path}")
    if dry_run:
        for source, target in list(rollout_pairs.items())[:12]:
            print(f"  {source.relative_to(REPO_ROOT)} -> {target.relative_to(REPO_ROOT)}")
        if len(rollout_pairs) > 12:
            print(f"  ... {len(rollout_pairs) - 12} more")
        return

    written = 0
    reused = 0
    for source, target in rollout_pairs.items():
        if copy_and_flip_rollout(source, target, overwrite=overwrite_rollouts):
            written += 1
        else:
            reused += 1

    augmented: list[dict[str, Any]] = []
    for sample in original_samples:
        augmented.append(sample)
        augmented.append(make_flipped_sample(sample))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_samples(output_path, augmented)
    print(f"written rollout dirs: {written}")
    print(f"reused rollout dirs : {reused}")
    print(f"wrote dataset       : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source rollout_lite JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON. Defaults to '<input stem>_add_horizon_flip<ext>' next to --input.",
    )
    parser.add_argument(
        "--overwrite-rollouts",
        action="store_true",
        help="Rewrite existing *_horizon_flip rollout files instead of reusing them.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing files.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_add_horizon_flip{input_path.suffix}")
    )
    try:
        augment_dataset(input_path, output_path, overwrite_rollouts=args.overwrite_rollouts, dry_run=args.dry_run)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
