#!/usr/bin/env python3
"""Build the 0807 chazuo (plug-into-socket) post-training dataset.

Small, single-task dataset: only ``MVTOKEN/zwz_0807_chazuo`` -- 11 rollouts, no mixing with
02_exchange_token / zwz_0723 / 0724 / 0802. It is meant to be post-trained ON TOP of
``saves/qwen3.5-9b/robot/mix_22-06_fk-pp/02_exchange_token_base``, not from scratch, so it
deliberately reuses that run's exact contract:

  * prompt   -- AgentRobot ``prompts/v3/mvtoken_generator_lite.txt`` (the "normal" 2-image v3
                prompt; byte-identical to ``prompts/v4/franka_mvtoken_lite.txt``, and it renders
                to exactly the instruction found in 02_exchange_token/rollout_lite.json).
  * shape    -- ``<image><image>`` + [agentview, wrist], one action token per sample.
  * history  -- last 5 MV_* tokens, newest first (convert_rollout's RECENT_WINDOW).

Post-processing, from ``MVTOKEN/zwz_0807_chazuo/说明.txt``:

    全部样本最后的move up都需要删掉

i.e. every rollout ends with a run of MV_UP -- the operator retracting the gripper after the
plug is seated. That retreat is not part of the task, so the trailing MV_UP run is dropped and
the synthesized terminal DONE lands on the last MV_DOWN frame (= plug inserted) instead.
Only the TRAILING run is dropped; a mid-trajectory MV_UP (a re-approach) is kept, and the stats
file records both counts so a mismatch is visible.

The trimming is done by re-emitting a trimmed ``actions.jsonl`` into a temp dir with absolute
image paths, then handing that to the unmodified ``convert_rollout`` -- so history, the DONE
synthesis and the sample shape all stay bit-identical to every other MVTOKEN dataset.

Run:
    python data/agentrobot/zwz_process_code/0807/build_0807_chazuo_dataset.py
Output:
    data/agentrobot/MVTOKEN/zwz_0807_chazuo/rollout_lite_chazuo.json (+ .stats.json)
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
LF_ROOT = SCRIPT_DIR.parents[3]
MVTOKEN_ROOT = LF_ROOT / "data" / "agentrobot" / "MVTOKEN"
HELPER_PATH = SCRIPT_DIR.parent / "0724" / "build_plan_0724_datasets.py"

CHAZUO_ROOT = MVTOKEN_ROOT / "zwz_0807_chazuo"
OUT_JSON = CHAZUO_ROOT / "rollout_lite_chazuo.json"

# The v3 "normal" prompt -- the one 02_exchange_token_base was trained with. Post-training on a
# different prompt would waste the base LoRA, so this path is not a knob.
PROMPT_V3 = LF_ROOT.parent / "AgentRobot" / "prompts" / "v3" / "mvtoken_generator_lite.txt"

# Single constant task line for all 11 rollouts (说明.txt: 把插头插入插座中). English, because
# every task string the base LoRA ever saw is English -- a Chinese task line would be OOD text
# in an otherwise English prompt. Must match TASK in AgentRobot's launcher byte-for-byte.
TASK_PROMPT = "Insert the plug into the socket."

# The retreat motion to strip from the tail of every episode.
TRAILING_TOKEN = "MV_UP"


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("build_plan_0724_datasets", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper from {HELPER_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_actions(rollout_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (rollout_dir / "actions.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def trim_trailing(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop the trailing run of TRAILING_TOKEN. Returns (kept_steps, n_dropped)."""
    end = len(steps)
    while end > 0 and steps[end - 1]["token"] == TRAILING_TOKEN:
        end -= 1
    if end == 0:
        raise ValueError("every step is a trailing MV_UP -- nothing left to train on")
    return steps[:end], len(steps) - end


def materialize_trimmed(steps: list[dict[str, Any]], rollout_dir: Path, dest: Path) -> Path:
    """Write a trimmed actions.jsonl whose image paths are absolute, so convert_rollout's
    ``rollout_dir / step["agentview"]`` resolves back to the real frames."""
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "actions.jsonl").open("w", encoding="utf-8") as f:
        for step in steps:
            row = dict(step)
            row["agentview"] = str((rollout_dir / step["agentview"]).resolve())
            row["wrist"] = str((rollout_dir / step["wrist"]).resolve())
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dest


def main() -> None:
    helper = load_helper()
    converter = helper.load_rollout_converter()

    prompt_template = PROMPT_V3.read_text(encoding="utf-8").rstrip("\n")
    helper.validate_template(prompt_template, PROMPT_V3)

    rollout_dirs = helper.rollout_dirs(CHAZUO_ROOT)
    if not rollout_dirs:
        raise SystemExit(f"no rollouts under {CHAZUO_ROOT}")

    samples: list[dict[str, Any]] = []
    per_rollout: dict[str, dict[str, int]] = {}
    token_counts: Counter[str] = Counter()

    with tempfile.TemporaryDirectory(prefix="chazuo_trimmed_") as tmp:
        for rollout_dir in rollout_dirs:
            steps = read_actions(rollout_dir)
            kept, dropped = trim_trailing(steps)
            staged = materialize_trimmed(kept, rollout_dir, Path(tmp) / rollout_dir.name)

            converted = converter.convert_rollout(
                staged,
                prompt_template=prompt_template,
                mode="lite",
                task_override=TASK_PROMPT,
            )
            samples.extend(converted)
            token_counts.update(s["output"] for s in converted)
            per_rollout[rollout_dir.name] = {
                "steps_recorded": len(steps),
                "trailing_mv_up_dropped": dropped,
                "steps_kept": len(kept),
                # samples = kept steps + 1 synthesized DONE
                "samples": len(converted),
                # sanity: if these differ, MV_UP also appears mid-trajectory
                "mv_up_total": sum(1 for s in steps if s["token"] == TRAILING_TOKEN),
            }

    helper.normalize_mvtoken_media_paths(samples)
    helper.write_json(OUT_JSON, samples)
    helper.write_stats(
        OUT_JSON,
        {
            "total_samples": len(samples),
            "rollouts": len(rollout_dirs),
            "source_root": str(CHAZUO_ROOT),
            "task_prompt": TASK_PROMPT,
            "prompt": {"kind": "agentrobot_v3_mvtoken_generator_lite", **helper.prompt_fingerprint(PROMPT_V3)},
            "postprocess": f"dropped the trailing run of {TRAILING_TOKEN} in every rollout (说明.txt)",
            "output_token_counts": dict(sorted(token_counts.items())),
            "per_rollout": per_rollout,
        },
    )

    dropped_total = sum(v["trailing_mv_up_dropped"] for v in per_rollout.values())
    mid_mv_up = sum(v["mv_up_total"] for v in per_rollout.values()) - dropped_total
    print(f"wrote {len(samples)} samples from {len(rollout_dirs)} rollouts -> {OUT_JSON}")
    print(f"  dropped {dropped_total} trailing {TRAILING_TOKEN}; {mid_mv_up} mid-trajectory {TRAILING_TOKEN} kept")
    print(f"  output tokens: {dict(sorted(token_counts.items()))}")


if __name__ == "__main__":
    main()
