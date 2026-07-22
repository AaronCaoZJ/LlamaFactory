#!/usr/bin/env python3
"""Add Franka-only prompt-rule augmentation by swapping MV_FWD/MV_BACK semantics."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "data/agentrobot/MVTOKEN/mix_22-06_fk-pp/03_just_mix/"
    "rollout_lite_zwz_new_prompt_add_horizon_flip.json"
)

HISTORY_RE = re.compile(r"(Recent (?:previous )?moves, newest first: )([^\n]*)")
FWD_BACK_TOKEN_RE = re.compile(r"\bMV_(FWD|BACK)\b")

AGENT_FWD = "- MV_FWD: move the end effector closer to the AgentView camera, away from the robot body."
AGENT_BACK = "- MV_BACK: move the end effector farther from the AgentView camera, toward the robot body."
AGENT_FWD_AUG = "- MV_FWD: move the end effector farther from the AgentView camera, toward the robot body."
AGENT_BACK_AUG = "- MV_BACK: move the end effector closer to the AgentView camera, away from the robot body."

WRIST_FWD = "- MV_FWD: move the gripper toward the bottom of the WristView image, away from the robot body."
WRIST_BACK = "- MV_BACK: move the gripper toward the top of the WristView image, toward the robot body."
WRIST_FWD_AUG = "- MV_FWD: move the gripper toward the top of the WristView image, toward the robot body."
WRIST_BACK_AUG = "- MV_BACK: move the gripper toward the bottom of the WristView image, away from the robot body."


def swap_fwd_back_token(token: str) -> str:
    if token == "MV_FWD":
        return "MV_BACK"
    if token == "MV_BACK":
        return "MV_FWD"
    return token


def swap_fwd_back_tokens_in_text(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return swap_fwd_back_token(f"MV_{match.group(1)}")

    return FWD_BACK_TOKEN_RE.sub(repl, text)


def swap_history_only(instruction: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{swap_fwd_back_tokens_in_text(match.group(2))}"

    return HISTORY_RE.sub(repl, instruction)


def is_franka_sample(sample: dict[str, Any]) -> bool:
    images = " ".join(str(path) for path in sample.get("images", []))
    return "piper" not in images


def is_rule_augmented(sample: dict[str, Any]) -> bool:
    instruction = sample.get("instruction", "")
    return is_franka_sample(sample) and AGENT_FWD_AUG in instruction and WRIST_FWD_AUG in instruction


def reverse_fwd_back_definition_rules(instruction: str) -> str:
    expected_lines = [AGENT_FWD, AGENT_BACK, WRIST_FWD, WRIST_BACK]
    missing = [line for line in expected_lines if line not in instruction]
    if missing:
        raise ValueError("missing expected Franka MV_FWD/MV_BACK definition line: " + missing[0])

    replacements = [
        (AGENT_FWD, "__AGENT_FWD__"),
        (AGENT_BACK, "__AGENT_BACK__"),
        (WRIST_FWD, "__WRIST_FWD__"),
        (WRIST_BACK, "__WRIST_BACK__"),
    ]
    for old, tmp in replacements:
        instruction = instruction.replace(old, tmp)

    return (
        instruction
        .replace("__AGENT_FWD__", AGENT_FWD_AUG)
        .replace("__AGENT_BACK__", AGENT_BACK_AUG)
        .replace("__WRIST_FWD__", WRIST_FWD_AUG)
        .replace("__WRIST_BACK__", WRIST_BACK_AUG)
    )


def make_rule_augmented_sample(sample: dict[str, Any]) -> dict[str, Any]:
    augmented = dict(sample)
    augmented["instruction"] = reverse_fwd_back_definition_rules(sample["instruction"])
    augmented["instruction"] = swap_history_only(augmented["instruction"])
    augmented["output"] = swap_fwd_back_token(str(sample["output"]))
    return augmented


def load_samples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array in {path}")
    return data


def write_samples(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def build_dataset(input_path: Path, output_path: Path, dry_run: bool) -> None:
    samples = load_samples(input_path)
    base_samples = [sample for sample in samples if not is_rule_augmented(sample)]
    already_augmented = len(samples) - len(base_samples)
    franka_samples = [sample for sample in base_samples if is_franka_sample(sample)]
    piper_samples = [sample for sample in base_samples if not is_franka_sample(sample)]
    augmented_franka_samples = [make_rule_augmented_sample(sample) for sample in franka_samples]
    output_samples = base_samples + augmented_franka_samples

    print(f"input samples: {len(samples)}")
    print(f"base samples : {len(base_samples)}")
    print(f"piper samples: {len(piper_samples)}")
    print(f"franka samples: {len(franka_samples)}")
    print(f"existing augment-rule samples ignored for rebuild: {already_augmented}")
    print(f"new augment-rule samples: {len(augmented_franka_samples)}")
    print(f"output samples: {len(output_samples)}")
    print(f"output json   : {output_path}")

    if not dry_run:
        write_samples(output_path, output_samples)
        print(f"wrote dataset : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSON with base + horizon-flip samples.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON. Defaults to '<input stem>_add_augment_rules<ext>' next to --input.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing files.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_add_augment_rules{input_path.suffix}")
    )

    try:
        build_dataset(input_path, output_path, dry_run=args.dry_run)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
