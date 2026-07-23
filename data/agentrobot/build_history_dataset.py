#!/usr/bin/env python3
"""Build visual-history samples from an existing rollout_lite.json."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROLLOUT_RE = re.compile(r"(.*/rollout_[^/]+)/([^/]+)/([0-9]+)\.png$")


def _strip_image_tokens(text: str) -> str:
    return text.replace("<image>", "").strip()


def _image_tokens(num_images: int) -> str:
    return "<image>" * num_images


def _history_prefix(history_frames: int) -> str:
    """TRAINING CONTRACT -- the deployed prompt must reproduce this text byte-for-byte.

    For history_frames=2 (the only case actually trained) the deployment copy is the prompt
    file ``AgentRobot/prompts/v3/history2_mvtoken_generator_lite.txt`` (this preamble + the v3
    body). Change this function and that file goes stale: re-render it and re-verify against
    the emitted dataset. ``AgentRobot/vlm/mvtoken_roles.py:_history_prefix`` is the legacy
    runtime copy, kept only for other history_frames values.
    """
    num_pairs = history_frames + 1
    lines = [
        "Temporal visual history is provided before the current observation.",
        "Images are ordered from oldest to newest as camera pairs:",
    ]
    for idx in range(num_pairs):
        role = "current state" if idx == num_pairs - 1 else "history"
        lines.append(f"- Pair {idx + 1}: Agentview then Wristview, {role}")

    lines.extend(
        [
            "",
            "Use history to resolve occlusion and completion.",
            "Predict the next action for the final/current state only.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_rollout_step(sample: dict[str, Any]) -> tuple[str, int]:
    images = sample.get("images") or []
    if not images:
        raise ValueError("Sample has no images.")

    match = ROLLOUT_RE.match(images[0])
    if not match:
        raise ValueError(f"Cannot parse rollout/step from image path: {images[0]}")

    return match.group(1), int(match.group(3))


def _group_samples(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for idx, sample in enumerate(samples):
        rollout, step = _parse_rollout_step(sample)
        item = dict(sample)
        item["_idx"] = idx
        item["_rollout"] = rollout
        item["_step"] = step
        grouped.setdefault(rollout, []).append(item)

    for items in grouped.values():
        # Preserve duplicate final-frame samples such as RELEASE and DONE.
        items.sort(key=lambda item: (item["_step"], item["_idx"]))

    return grouped


def _frame_by_step(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    for item in items:
        frames.setdefault(item["_step"], item)
    return frames


def _select_history_frames(
    item: dict[str, Any],
    frame_lookup: dict[int, dict[str, Any]],
    history_frames: int,
) -> list[dict[str, Any]]:
    steps = sorted(frame_lookup)
    current_step = item["_step"]
    current_pos = max(pos for pos, step in enumerate(steps) if step <= current_step)

    selected = []
    for offset in range(history_frames, -1, -1):
        pos = max(0, current_pos - offset)
        selected.append(frame_lookup[steps[pos]])
    return selected


def build_history_dataset(
    samples: list[dict[str, Any]],
    history_frames: int,
) -> list[dict[str, Any]]:
    if history_frames < 0:
        raise ValueError("history_frames must be non-negative.")

    grouped = _group_samples(samples)
    prefix = _history_prefix(history_frames) if history_frames else ""
    new_samples: list[dict[str, Any]] = []

    for items in grouped.values():
        frame_lookup = _frame_by_step(items)
        for item in items:
            selected = _select_history_frames(item, frame_lookup, history_frames)
            images: list[str] = []
            for frame in selected:
                images.extend(frame["images"])

            body = _strip_image_tokens(item["instruction"])
            instruction = _image_tokens(len(images)) + (prefix + body if prefix else body)
            new_samples.append(
                {
                    "instruction": instruction,
                    "input": item.get("input", ""),
                    "output": item["output"],
                    "images": images,
                }
            )

    return new_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input rollout_lite.json.")
    parser.add_argument("--output", required=True, type=Path, help="Output history JSON.")
    parser.add_argument("--history-frames", type=int, default=2)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    with args.input.open() as f:
        samples = json.load(f)

    new_samples = build_history_dataset(samples, history_frames=args.history_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(new_samples, f, indent=args.indent)
        f.write("\n")

    image_counts = sorted({len(sample["images"]) for sample in new_samples})
    image_token_counts = sorted({sample["instruction"].count("<image>") for sample in new_samples})
    print(f"input_samples: {len(samples)}")
    print(f"output_samples: {len(new_samples)}")
    print(f"history_frames: {args.history_frames}")
    print(f"image_counts: {image_counts}")
    print(f"image_token_counts: {image_token_counts}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
