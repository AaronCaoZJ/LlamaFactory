#!/usr/bin/env python3
"""Build the four training datasets requested by MVTOKEN/plan_0724.md."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
LF_ROOT = SCRIPT_DIR.parents[3]
MVTOKEN_ROOT = LF_ROOT / "data" / "agentrobot" / "MVTOKEN"
PROCESS_ROOT = LF_ROOT / "data" / "agentrobot" / "zwz_process_code"

BASE_EXCHANGE_JSON = MVTOKEN_ROOT / "mix_22-06_fk-pp" / "02_exchange_token" / "rollout_lite.json"
BASE_JUST_MIX_JSON = MVTOKEN_ROOT / "mix_22-06_fk-pp" / "03_just_mix" / "rollout_lite.json"
OUT_EXCHANGE_DIR = MVTOKEN_ROOT / "mix_22-06_fk-pp" / "02_exchange_token"
OUT_JUST_MIX_DIR = MVTOKEN_ROOT / "mix_22-06_fk-pp" / "03_just_mix"

ZWZ_0723_ROOT = MVTOKEN_ROOT / "zwz_0723"
ZWZ_0724_ROOT = MVTOKEN_ROOT / "zwz_0724"
ZWZ_0724_EASY_ROOT = ZWZ_0724_ROOT / "easy_spatial_left_right"

PROMPT_V4_FRANKA = LF_ROOT.parent / "AgentRobot" / "prompts" / "v4" / "franka_mvtoken_lite.txt"
PROMPT_V2_FRANKA = PROCESS_ROOT / "prompt_franka_zwz_v2.txt"
PROMPT_V2_PIPER = PROCESS_ROOT / "prompt_piper_zwz_v2.txt"

ROBOVQA_REASONING = LF_ROOT / "data" / "robovqa" / "robovqa_reasoning_lf_ans6k.jsonl"
ROBOVQA_UNDERSTANDING = LF_ROOT / "data" / "robovqa" / "robovqa_understanding_lf_500.jsonl"

OUT_1 = OUT_EXCHANGE_DIR / "rollout_lite_plus_zwz_0724_easy_spatial.json"
OUT_2 = OUT_EXCHANGE_DIR / "rollout_lite_plus_zwz_0723_0724.json"
OUT_3 = OUT_JUST_MIX_DIR / "rollout_lite_plus_zwz_0723_0724_v2_prompt.json"
OUT_4 = OUT_JUST_MIX_DIR / "rollout_lite_plus_zwz_0723_0724_v2_prompt_plus_robovqa_clean_ans6k_under500.json"

PIPER_SOURCE_MARKERS = ("0705_piper", "0706_piper")
MEDIA_PREFIX_RE = re.compile(r"^(?:(?:<image>)|(?:<video>))+")
TASK_RE = re.compile(r"^Task:\s*(?P<task>.+?)\s*$", re.MULTILINE)
RECENT_RE = re.compile(
    r"^Recent(?:\s+(?:previous|past|historical))?\s+moves,\s+newest\s+first:\s*(?P<recent>.*?)\s*$",
    re.MULTILINE,
)
ALLOWED_TEMPLATE_FIELDS = {"task", "recent_moves"}


def load_rollout_converter() -> Any:
    path = LF_ROOT / "data" / "agentrobot" / "rollout_to_llamafactory.py"
    spec = importlib.util.spec_from_file_location("rollout_to_llamafactory", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import converter from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def load_task_txt(path: Path) -> dict[str, str]:
    task_map: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, _, task = line.partition(":")
        task_map[key.strip()] = task.strip()
    return task_map


def rollout_dirs(search_root: Path) -> list[Path]:
    return sorted(p.parent for p in search_root.rglob("actions.jsonl") if p.parent.name.startswith("rollout_"))


def task_for_rollout(rollout_dir: Path, task_root: Path, task_map: dict[str, str]) -> str:
    task_dir = rollout_dir.parent
    rel_key = task_dir.relative_to(task_root).as_posix()
    if rel_key in task_map:
        return task_map[rel_key]
    if task_dir.name in task_map:
        return task_map[task_dir.name]
    raise KeyError(f"no task.txt entry for {task_dir} (tried {rel_key!r} and {task_dir.name!r})")


def convert_zwz(
    *,
    converter: Any,
    search_root: Path,
    task_root: Path,
    task_map: dict[str, str],
    prompt_template: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    samples: list[dict[str, Any]] = []
    per_task: Counter[str] = Counter()

    for rollout_dir in rollout_dirs(search_root):
        task = task_for_rollout(rollout_dir, task_root, task_map)
        converted = converter.convert_rollout(
            rollout_dir,
            prompt_template=prompt_template,
            mode="lite",
            task_override=task,
        )
        samples.extend(converted)
        per_task[task] += len(converted)

    return samples, per_task


def validate_template(template: str, path: Path) -> None:
    fields: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name:
            fields.add(re.split(r"[.[]", field_name, maxsplit=1)[0])

    unknown = fields - ALLOWED_TEMPLATE_FIELDS
    if unknown:
        raise ValueError(f"{path} has unsupported fields: {', '.join(sorted(unknown))}")

    missing = ALLOWED_TEMPLATE_FIELDS - fields
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")


def parse_prompt_context(instruction: str, idx: int) -> tuple[str, str, str]:
    media_match = MEDIA_PREFIX_RE.match(instruction)
    if media_match is None:
        raise ValueError(f"sample {idx}: instruction does not start with media placeholders")

    body = instruction[media_match.end():]
    task_match = TASK_RE.search(body)
    recent_match = RECENT_RE.search(body)
    if task_match is None:
        raise ValueError(f"sample {idx}: cannot find Task line")
    if recent_match is None:
        raise ValueError(f"sample {idx}: cannot find Recent moves line")

    return media_match.group(0), task_match.group("task"), recent_match.group("recent")


def sample_source(sample: dict[str, Any]) -> str:
    images = sample.get("images") or []
    image_text = " ".join(str(image) for image in images)
    if any(marker in image_text for marker in PIPER_SOURCE_MARKERS):
        return "piper"
    return "franka"


def rewrite_v2_prompt(
    samples: list[dict[str, Any]],
    *,
    franka_template: str,
    piper_template: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rewritten: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    for idx, sample in enumerate(samples):
        instruction = sample.get("instruction")
        if not isinstance(instruction, str):
            raise TypeError(f"sample {idx}: instruction is not a string")

        media_prefix, task, recent_moves = parse_prompt_context(instruction, idx)
        source = sample_source(sample)
        template = piper_template if source == "piper" else franka_template

        item = dict(sample)
        item["instruction"] = media_prefix + template.format(task=task, recent_moves=recent_moves)
        rewritten.append(item)
        source_counts[source] += 1

    return rewritten, source_counts


def with_media_keys(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for sample in samples:
        item = dict(sample)
        item.setdefault("images", [])
        item.setdefault("videos", [])
        normalized.append(item)
    return normalized


def write_stats(path: Path, stats: dict[str, Any]) -> None:
    write_json(path.with_suffix(path.suffix + ".stats.json"), stats)


def prompt_fingerprint(path: Path) -> dict[str, str | int]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def main() -> None:
    converter = load_rollout_converter()

    base_exchange_samples = load_json(BASE_EXCHANGE_JSON)
    base_just_mix_samples = load_json(BASE_JUST_MIX_JSON)
    prompt_v4_franka = PROMPT_V4_FRANKA.read_text(encoding="utf-8").rstrip("\n")
    prompt_v2_franka = PROMPT_V2_FRANKA.read_text(encoding="utf-8").rstrip("\n")
    prompt_v2_piper = PROMPT_V2_PIPER.read_text(encoding="utf-8").rstrip("\n")
    validate_template(prompt_v2_franka, PROMPT_V2_FRANKA)
    validate_template(prompt_v2_piper, PROMPT_V2_PIPER)
    legacy_prompt_info = prompt_fingerprint(PROMPT_V4_FRANKA)
    v2_prompt_info = {
        "franka": prompt_fingerprint(PROMPT_V2_FRANKA),
        "piper": prompt_fingerprint(PROMPT_V2_PIPER),
    }

    task_0723 = load_task_txt(ZWZ_0723_ROOT / "task.txt")
    task_0724 = load_task_txt(ZWZ_0724_ROOT / "task.txt")

    zwz_0724_easy, per_task_0724_easy = convert_zwz(
        converter=converter,
        search_root=ZWZ_0724_EASY_ROOT,
        task_root=ZWZ_0724_ROOT,
        task_map=task_0724,
        prompt_template=prompt_v4_franka,
    )
    zwz_0724_all, per_task_0724_all = convert_zwz(
        converter=converter,
        search_root=ZWZ_0724_ROOT,
        task_root=ZWZ_0724_ROOT,
        task_map=task_0724,
        prompt_template=prompt_v4_franka,
    )
    zwz_0723_all, per_task_0723 = convert_zwz(
        converter=converter,
        search_root=ZWZ_0723_ROOT,
        task_root=ZWZ_0723_ROOT,
        task_map=task_0723,
        prompt_template=prompt_v4_franka,
    )

    exp1 = base_exchange_samples + zwz_0724_easy
    exp2 = base_exchange_samples + zwz_0723_all + zwz_0724_all
    exp3_source = base_just_mix_samples + zwz_0723_all + zwz_0724_all
    exp3, exp3_source_counts = rewrite_v2_prompt(
        exp3_source,
        franka_template=prompt_v2_franka,
        piper_template=prompt_v2_piper,
    )
    robovqa_reasoning = load_jsonl(ROBOVQA_REASONING)
    robovqa_understanding = load_jsonl(ROBOVQA_UNDERSTANDING)
    exp4 = with_media_keys(exp3) + with_media_keys(robovqa_reasoning) + with_media_keys(robovqa_understanding)

    outputs = [
        (
            OUT_1,
            exp1,
            {
                "total_samples": len(exp1),
                "base_samples": len(base_exchange_samples),
                "base_path": str(BASE_EXCHANGE_JSON),
                "zwz_0724_easy_samples": len(zwz_0724_easy),
                "zwz_0724_easy_rollouts": len(rollout_dirs(ZWZ_0724_EASY_ROOT)),
                "zwz_0724_easy_per_task": dict(sorted(per_task_0724_easy.items())),
                "new_zwz_prompt": {
                    "kind": "legacy_agentrobot_v4_franka_lite",
                    **legacy_prompt_info,
                },
            },
        ),
        (
            OUT_2,
            exp2,
            {
                "total_samples": len(exp2),
                "base_samples": len(base_exchange_samples),
                "base_path": str(BASE_EXCHANGE_JSON),
                "zwz_0723_samples": len(zwz_0723_all),
                "zwz_0723_rollouts": len(rollout_dirs(ZWZ_0723_ROOT)),
                "zwz_0723_per_task": dict(sorted(per_task_0723.items())),
                "zwz_0724_samples": len(zwz_0724_all),
                "zwz_0724_rollouts": len(rollout_dirs(ZWZ_0724_ROOT)),
                "zwz_0724_per_task": dict(sorted(per_task_0724_all.items())),
                "new_zwz_prompt": {
                    "kind": "legacy_agentrobot_v4_franka_lite",
                    **legacy_prompt_info,
                },
            },
        ),
        (
            OUT_3,
            exp3,
            {
                "total_samples": len(exp3),
                "source_counts": dict(sorted(exp3_source_counts.items())),
                "base_samples": len(base_just_mix_samples),
                "base_path": str(BASE_JUST_MIX_JSON),
                "zwz_0723_samples": len(zwz_0723_all),
                "zwz_0724_samples": len(zwz_0724_all),
                "prompt": {
                    "kind": "zwz_v2_mixed_by_source",
                    **v2_prompt_info,
                },
            },
        ),
        (
            OUT_4,
            exp4,
            {
                "total_samples": len(exp4),
                "robot_v2_samples": len(exp3),
                "robovqa_reasoning_samples": len(robovqa_reasoning),
                "robovqa_understanding_samples": len(robovqa_understanding),
                "source_counts": {
                    **dict(sorted(exp3_source_counts.items())),
                    "robovqa_reasoning": len(robovqa_reasoning),
                    "robovqa_understanding": len(robovqa_understanding),
                },
                "media_columns": ["images", "videos"],
                "robot_prompt": {
                    "kind": "zwz_v2_mixed_by_source",
                    **v2_prompt_info,
                },
            },
        ),
    ]

    for path, samples, stats in outputs:
        write_json(path, samples)
        write_stats(path, stats)
        print(f"wrote {len(samples)} samples -> {path}")


if __name__ == "__main__":
    main()
