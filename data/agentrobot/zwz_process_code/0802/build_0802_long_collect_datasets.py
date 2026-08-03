#!/usr/bin/env python3
"""Build 0802 long-collect training datasets.

Three datasets are generated from the existing 02_exchange_token + zwz_0723/0724
setting:

1. Add all zwz_0802_long_collect tasks.
2. Add zwz_0802_long_collect but exclude task_3, the mixed sort task.
3. Add zwz_0802_long_collect but exclude task_3 and task_4_1.
"""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
LF_ROOT = SCRIPT_DIR.parents[3]
MVTOKEN_ROOT = LF_ROOT / "data" / "agentrobot" / "MVTOKEN"
HELPER_PATH = SCRIPT_DIR.parent / "0724" / "build_plan_0724_datasets.py"

BASE_EXCHANGE_PLUS_0723_0724_JSON = (
    MVTOKEN_ROOT
    / "mix_22-06_fk-pp"
    / "02_exchange_token"
    / "rollout_lite_plus_zwz_0723_0724.json"
)
OUT_EXCHANGE_DIR = MVTOKEN_ROOT / "mix_22-06_fk-pp" / "02_exchange_token"
ZWZ_0802_ROOT = MVTOKEN_ROOT / "zwz_0802_long_collect"
PROMPT_V4_FRANKA = LF_ROOT.parent / "AgentRobot" / "prompts" / "v4" / "franka_mvtoken_lite.txt"

OUT_ALL = OUT_EXCHANGE_DIR / "rollout_lite_plus_zwz_0723_0724_0802_long_collect.json"
OUT_NO_TASK3 = (
    OUT_EXCHANGE_DIR
    / "rollout_lite_plus_zwz_0723_0724_0802_long_collect_no_task3.json"
)
OUT_NO_TASK3_TASK4_1 = (
    OUT_EXCHANGE_DIR
    / "rollout_lite_plus_zwz_0723_0724_0802_long_collect_no_task3_task4_1.json"
)

TASK_PROMPTS = {
    "task_2": "Pick up the letter O and place it to the right of the letter H.",
    "task_3": (
        "Sort the mixed letters and blocks: put the letters into the left plate "
        "and the blocks into the right plate."
    ),
    "task_4_1": (
        "Collect all letters into the left plate, and then collect all blocks "
        "into the right plate."
    ),
    "task_4_2": (
        "Collect all letters into the left plate, and then collect all blocks "
        "into the right plate."
    ),
    "task_5": (
        "Collect all blocks into the left plate, and then collect all letters "
        "into the right plate."
    ),
}

TASK_VARIANTS = {
    "task_2": "single spatial relation",
    "task_3": "mixed sort, no explicit order",
    "task_4_1": "letters then blocks, random",
    "task_4_2": "letters then blocks, close_2_gripper_first",
    "task_5": "blocks then letters, close_2_gripper_first",
}


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("build_plan_0724_datasets", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper from {HELPER_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert_0802(
    *,
    converter: Any,
    helper: Any,
    prompt_template: str,
    excluded_tasks: set[str],
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], int]:
    samples: list[dict[str, Any]] = []
    per_task_key: Counter[str] = Counter()
    per_task_prompt: Counter[str] = Counter()
    rollout_count = 0

    task_dirs = sorted(p for p in ZWZ_0802_ROOT.iterdir() if p.is_dir() and p.name.startswith("task_"))
    for task_dir in task_dirs:
        if task_dir.name in excluded_tasks:
            continue
        try:
            task_prompt = TASK_PROMPTS[task_dir.name]
        except KeyError as exc:
            raise KeyError(f"missing canonical prompt for {task_dir}") from exc

        rollout_dirs = helper.rollout_dirs(task_dir)
        rollout_count += len(rollout_dirs)
        for rollout_dir in rollout_dirs:
            converted = converter.convert_rollout(
                rollout_dir,
                prompt_template=prompt_template,
                mode="lite",
                task_override=task_prompt,
            )
            samples.extend(converted)
            per_task_key[task_dir.name] += len(converted)
            per_task_prompt[task_prompt] += len(converted)

    return samples, per_task_key, per_task_prompt, rollout_count


def build_dataset(
    *,
    helper: Any,
    base_samples: list[dict[str, Any]],
    zwz_samples: list[dict[str, Any]],
    output_path: Path,
    stats: dict[str, Any],
) -> None:
    samples = base_samples + zwz_samples
    helper.normalize_mvtoken_media_paths(samples)
    helper.write_json(output_path, samples)
    helper.write_stats(output_path, {"total_samples": len(samples), **stats})
    print(f"wrote {len(samples)} samples -> {output_path}")


def main() -> None:
    helper = load_helper()
    converter = helper.load_rollout_converter()

    base_samples = helper.load_json(BASE_EXCHANGE_PLUS_0723_0724_JSON)
    prompt_v4_franka = PROMPT_V4_FRANKA.read_text(encoding="utf-8").rstrip("\n")
    prompt_info = helper.prompt_fingerprint(PROMPT_V4_FRANKA)

    zwz_all, per_task_all, per_prompt_all, rollouts_all = convert_0802(
        converter=converter,
        helper=helper,
        prompt_template=prompt_v4_franka,
        excluded_tasks=set(),
    )
    zwz_no_task3, per_task_no_task3, per_prompt_no_task3, rollouts_no_task3 = convert_0802(
        converter=converter,
        helper=helper,
        prompt_template=prompt_v4_franka,
        excluded_tasks={"task_3"},
    )
    (
        zwz_no_task3_task4_1,
        per_task_no_task3_task4_1,
        per_prompt_no_task3_task4_1,
        rollouts_no_task3_task4_1,
    ) = convert_0802(
        converter=converter,
        helper=helper,
        prompt_template=prompt_v4_franka,
        excluded_tasks={"task_3", "task_4_1"},
    )

    common_stats = {
        "base_samples": len(base_samples),
        "base_path": str(BASE_EXCHANGE_PLUS_0723_0724_JSON),
        "zwz_0802_root": str(ZWZ_0802_ROOT),
        "new_zwz_prompt": {
            "kind": "agentrobot_v4_franka_lite_with_0802_canonical_task_prompts",
            **prompt_info,
        },
        "task_prompts": TASK_PROMPTS,
        "task_variants": TASK_VARIANTS,
    }

    build_dataset(
        helper=helper,
        base_samples=base_samples,
        zwz_samples=zwz_all,
        output_path=OUT_ALL,
        stats={
            **common_stats,
            "zwz_0802_samples": len(zwz_all),
            "zwz_0802_rollouts": rollouts_all,
            "zwz_0802_per_task": dict(sorted(per_task_all.items())),
            "zwz_0802_per_prompt": dict(sorted(per_prompt_all.items())),
            "excluded_tasks": [],
        },
    )
    build_dataset(
        helper=helper,
        base_samples=base_samples,
        zwz_samples=zwz_no_task3,
        output_path=OUT_NO_TASK3,
        stats={
            **common_stats,
            "zwz_0802_samples": len(zwz_no_task3),
            "zwz_0802_rollouts": rollouts_no_task3,
            "zwz_0802_per_task": dict(sorted(per_task_no_task3.items())),
            "zwz_0802_per_prompt": dict(sorted(per_prompt_no_task3.items())),
            "excluded_tasks": ["task_3"],
        },
    )
    build_dataset(
        helper=helper,
        base_samples=base_samples,
        zwz_samples=zwz_no_task3_task4_1,
        output_path=OUT_NO_TASK3_TASK4_1,
        stats={
            **common_stats,
            "zwz_0802_samples": len(zwz_no_task3_task4_1),
            "zwz_0802_rollouts": rollouts_no_task3_task4_1,
            "zwz_0802_per_task": dict(sorted(per_task_no_task3_task4_1.items())),
            "zwz_0802_per_prompt": dict(sorted(per_prompt_no_task3_task4_1.items())),
            "excluded_tasks": ["task_3", "task_4_1"],
        },
    )


if __name__ == "__main__":
    main()
