#!/usr/bin/env python3
"""Convert AgentRobot rollout directories to LLaMA Factory Alpaca multimodal format.

Each action step becomes one training sample. The two camera images are placed at the TOP
of the user turn -- the prompt template itself owns the camera-description header, and the
converter only prepends the "<image><image>" markers -- so every sample looks like:

  instruction : "<image><image>" + the rendered prompt (camera header + per-step context)
  input       : "" (everything lives in instruction)
  output      : the action token (MV_FWD/BACK/LEFT/RIGHT/UP/DOWN, GRASP, RELEASE) -- plus one
                synthesized DONE sample per episode on the final frame (all modes)
  images      : [agentview_abs_path, wrist_abs_path]

Three prompt templates (AgentRobot/prompts/), all starting with the camera header:
  mvtoken_generator_lite.txt        — default (lite). STAGE-FREE: only {task} /
                                {gripper_state} / {recent_moves}; no subgoal/affordance info.
  mvtoken_generator.txt             — used with --use-subgoal. Adds per-step
                                {stage}/{target}/{affordance}/{description}/{completion}
                                (+ {gripper_color}), taken VERBATIM from the VLM subgoals in
                                task_config.json (generate_subgoals.py). Each recorded step
                                is aligned to the subgoal active at that step: the recorded
                                GRASP/RELEASE anchor the approach/transport/retract phases
                                and the planner's subgoals advance within each phase, exactly
                                as AgentRobot's runtime steps through them.
  mvtoken_generator_affordance.txt  — used with --use-affordance. Lite + a single grasp-point
                                hint ({target}/{affordance}) reused on every step, from
                                affordance_config.json (generate_affordance.py). Lightweight
                                fix for "where to grasp" without the full subgoal machinery.

Recent moves are the last RECENT_WINDOW MV_* tokens, newest first (GRASP/RELEASE excluded),
matching inference.

Usage:
    # Lite (stage-free) — single rollout
    python data/agentrobot/rollout_to_llamafactory.py \\
        data/agentrobot/MVTOKEN/0622/banana/rollout_030 \\
        --task "pick up the banana and place it on the blue plate"

    # Lite — parent dir auto-expands into rollout_* subdirs; multiple tasks via --task-map
    python data/agentrobot/rollout_to_llamafactory.py \\
        data/agentrobot/MVTOKEN/0622/banana \\
        data/agentrobot/MVTOKEN/0622/mango \\
        --task-map "banana=pick up the banana and place it on the blue plate" \\
                   "mango=pick up the mango and place it on the blue plate" \\
        --output data/agentrobot/MVTOKEN/0622/rollout.json

    # Subgoal mode — full prompt with per-step VLM subgoal info (task_config.json auto-gen)
    python data/agentrobot/rollout_to_llamafactory.py \\
        data/agentrobot/MVTOKEN/0622/banana \\
        --use-subgoal \\
        --task "pick up the banana and place it on the blue plate" \\
        --vlm-backend mvtoken_0622_v0 \\
        --vlm-url http://localhost:8101/v1

    # Affordance mode — lite + a single grasp-point hint (affordance_config.json auto-gen)
    python data/agentrobot/rollout_to_llamafactory.py \\
        data/agentrobot/MVTOKEN/0622/banana \\
        --use-affordance \\
        --task "pick up the banana and place it on the blue plate" \\
        --vlm-backend mvtoken_0622_v0 \\
        --vlm-url http://localhost:8101/v1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_LLAMAFACTORY_ROOT = _HERE.parents[2]
_AGENTROBOT_ROOT = _LLAMAFACTORY_ROOT.parent / "AgentRobot"
_PROMPTS_DIR = _AGENTROBOT_ROOT / "prompts"
_GENERATE_SUBGOALS = _HERE.parent / "generate_subgoals.py"
_GENERATE_AFFORDANCE = _HERE.parent / "generate_affordance.py"

# ── Constants ─────────────────────────────────────────────────────────────────
RECENT_WINDOW = 5

MV_TOKENS = {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"}
GRIPPER_TOKENS = {"GRASP", "RELEASE"}
ACTION_TOKENS = MV_TOKENS | GRIPPER_TOKENS

# Terminal token. The recordings contain no DONE, so one DONE sample is synthesized per
# episode on the final observed frame to teach "task complete" (all modes).
DONE_TOKEN = "DONE"

# The two image markers (agentview, wrist) are prepended so the images sit at the TOP of
# the prompt. The camera-description header now lives inside each prompt template (lite and
# full), so there is no separate instruction header here.
IMAGE_TOKENS = "<image><image>"

# --use-subgoal motion keywords: used ONLY to locate the grasp and release subgoals in the
# VLM plan (so the recorded GRASP/RELEASE tokens can anchor the phase boundaries). Every
# field shown to the model is taken verbatim from the VLM subgoals -- no placeholders.
GRASP_MOTION_KW = ("grasp", "pick", "grip", "grab", "clamp", "secure", "close")
RELEASE_MOTION_KW = ("release", "open", "drop", "ungrip", "let_go", "letgo")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _load_config(search_dirs: list[Path], filename: str) -> dict:
    for d in search_dirs:
        cfg_path = d / filename
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f)
    return {}


def _load_subgoals(task_cfg: dict) -> list[dict]:
    """The ordered VLM subgoal list from task_config.json (generate_subgoals.py).

    Prefers the current ``subgoals`` key (validated + pre-grasp-merged, AgentRobot-
    consistent); falls back to the legacy ``subgoals_raw`` so older configs still convert.
    """
    sgs = task_cfg.get("subgoals")
    if isinstance(sgs, list) and sgs:
        return sgs
    raw = task_cfg.get("subgoals_raw")
    if isinstance(raw, list) and raw:
        return raw
    return []


def _load_affordance(task_cfg: dict) -> dict[str, str]:
    """The single grasp-point hint from affordance_config.json (generate_affordance.py).

    Returns ``{"target": ..., "affordance": ...}`` or ``{}`` when absent/incomplete.
    """
    target = str(task_cfg.get("target", "")).strip()
    affordance = str(task_cfg.get("affordance", "")).strip()
    if target and affordance:
        return {"target": target, "affordance": affordance}
    return {}


def _find_motion(subgoals: list[dict], keywords: tuple[str, ...], start: int = 0) -> int | None:
    for j in range(start, len(subgoals)):
        motion = str(subgoals[j].get("motion", "")).lower()
        if any(kw in motion for kw in keywords):
            return j
    return None


def _distribute(step_idxs: list[int], sgs: list[dict]) -> dict[int, dict]:
    """Assign each step (in order) to a subgoal (in plan order) by an even split."""
    n_steps, n_sg = len(step_idxs), len(sgs)
    if n_steps == 0 or n_sg == 0:
        return {}
    return {si: sgs[min(n_sg - 1, (k * n_sg) // n_steps)] for k, si in enumerate(step_idxs)}


def _assign_subgoals_to_steps(all_tokens: list[str], subgoals: list[dict]) -> dict[int, dict]:
    """Map each step index -> the VLM subgoal active at that step.

    The recorded GRASP / RELEASE tokens are the only observable phase boundaries, so they
    anchor the three macro-phases while the planner's subgoals advance within each phase,
    mirroring AgentRobot's runtime (which steps through every subgoal in order):

      approach+grasp : steps [0 .. grasp]        -> subgoals [0 .. grasp_sg]
      transport      : steps (grasp .. release)  -> subgoals (grasp_sg .. release_sg)
      retract        : steps [release .. end]    -> subgoals [release_sg .. end]

    Within each phase the steps are split evenly across that phase's subgoals.
    """
    n = len(all_tokens)
    if not subgoals:
        return {}

    grasp_step = next((i for i, t in enumerate(all_tokens) if t == "GRASP"), None)
    release_step = next(
        (
            i for i, t in enumerate(all_tokens)
            if t == "RELEASE" and (grasp_step is None or i > grasp_step)
        ),
        None,
    )

    # No grasp event recorded -> nothing to anchor on; spread the whole plan over all steps.
    if grasp_step is None:
        return _distribute(list(range(n)), subgoals)

    grasp_sg = _find_motion(subgoals, GRASP_MOTION_KW) or 0
    release_sg = _find_motion(subgoals, RELEASE_MOTION_KW, start=grasp_sg + 1)
    if release_sg is None:
        release_sg = len(subgoals) - 1

    mapping: dict[int, dict] = {}

    # Phase A: approach + grasp.
    a_sgs = subgoals[: grasp_sg + 1] or subgoals[:1]
    mapping.update(_distribute(list(range(0, grasp_step + 1)), a_sgs))

    if release_step is None:
        # No release recorded: treat everything after the grasp as the post-grasp plan.
        b_sgs = subgoals[grasp_sg + 1:] or [subgoals[grasp_sg]]
        mapping.update(_distribute(list(range(grasp_step + 1, n)), b_sgs))
        return mapping

    # Phase B: transport (strictly between grasp and release).
    b_sgs = subgoals[grasp_sg + 1: release_sg] or [subgoals[min(release_sg, len(subgoals) - 1)]]
    mapping.update(_distribute(list(range(grasp_step + 1, release_step)), b_sgs))

    # Phase C: release + retract.
    c_sgs = subgoals[release_sg:] or [subgoals[-1]]
    mapping.update(_distribute(list(range(release_step, n)), c_sgs))

    return mapping


def _ensure_task_config(rollout_dir: Path, task: str | None, vlm_args: list[str]) -> None:
    """Ensure a reusable task_config.json exists for this rollout's task folder.

    One task per folder, so the plan is generated ONCE at the task-folder (parent) level and
    every rollout under it reuses it (convert_rollout searches [rollout_dir, parent]).
    """
    task_dir = rollout_dir.parent
    if (rollout_dir / "task_config.json").exists() or (task_dir / "task_config.json").exists():
        return
    if not task:
        raise ValueError(
            f"No task_config.json found in {rollout_dir} (or its parent) "
            "and --task was not provided. Cannot auto-generate subgoals."
        )
    print(f"[subgoal] task_config.json missing for {task_dir.name}, generating once ...")
    cmd = [sys.executable, str(_GENERATE_SUBGOALS), str(task_dir), "--task", task] + vlm_args
    subprocess.run(cmd, check=True)


def _ensure_affordance_config(rollout_dir: Path, task: str | None, vlm_args: list[str]) -> None:
    """Ensure a reusable affordance_config.json exists for this rollout's task folder.

    Generated ONCE at the task-folder (parent) level and reused by every rollout under it.
    """
    task_dir = rollout_dir.parent
    if (rollout_dir / "affordance_config.json").exists() or (
        task_dir / "affordance_config.json"
    ).exists():
        return
    if not task:
        raise ValueError(
            f"No affordance_config.json found in {rollout_dir} (or its parent) "
            "and --task was not provided. Cannot auto-generate the affordance hint."
        )
    print(f"[affordance] affordance_config.json missing for {task_dir.name}, generating once ...")
    cmd = [sys.executable, str(_GENERATE_AFFORDANCE), str(task_dir), "--task", task] + vlm_args
    subprocess.run(cmd, check=True)


def _expand_dirs(paths: list[Path]) -> list[Path]:
    """Expand parent directories into their rollout subdirectories."""
    result: list[Path] = []
    for p in paths:
        if (p / "actions.jsonl").exists():
            result.append(p)
        else:
            children = sorted(c for c in p.iterdir() if c.is_dir() and (c / "actions.jsonl").exists())
            if children:
                print(f"[expand] {p.name}: found {len(children)} rollout(s)")
                result.extend(children)
            else:
                print(f"[skip] {p}: no actions.jsonl and no rollout subdirs", file=sys.stderr)
    return result


# ── Core conversion ───────────────────────────────────────────────────────────

def convert_rollout(
    rollout_dir: Path,
    prompt_template: str,
    mode: str = "lite",
    task_override: str | None = None,
    gripper_color: str = "black",
) -> list[dict]:
    """Convert one rollout to LLaMA Factory samples.

    ``mode`` selects the prompt body: ``lite`` (stage-free), ``subgoal`` (per-step VLM
    subgoal), or ``affordance`` (a single grasp-point hint reused for every step).
    """
    actions_path = rollout_dir / "actions.jsonl"
    if not actions_path.exists():
        print(f"[skip] no actions.jsonl in {rollout_dir}", file=sys.stderr)
        return []

    cfg_name = {
        "subgoal": "task_config.json",
        "affordance": "affordance_config.json",
    }.get(mode)
    cfg = _load_config([rollout_dir, rollout_dir.parent], cfg_name) if cfg_name else {}
    task_name = task_override or cfg.get("task", rollout_dir.parent.name)

    steps = []
    with open(actions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    all_tokens = [s["token"] for s in steps]

    # Resolve the per-mode context that does not vary inside the loop.
    step_subgoal: dict[int, dict] = {}
    afford: dict[str, str] = {}
    if mode == "subgoal":
        subgoals = _load_subgoals(cfg)
        if not subgoals:
            raise ValueError(
                f"No subgoals in task_config.json for {rollout_dir}. "
                "Run generate_subgoals.py first (or use --vlm-* to auto-generate)."
            )
        step_subgoal = _assign_subgoals_to_steps(all_tokens, subgoals)
        fallback_sg = subgoals[0]
    elif mode == "affordance":
        afford = _load_affordance(cfg)
        if not afford:
            raise ValueError(
                f"No affordance in affordance_config.json for {rollout_dir}. "
                "Run generate_affordance.py first (or use --vlm-* to auto-generate)."
            )

    def _render(step: dict, recent_str: str, sg: dict) -> str:
        gripper_state = "closed" if step.get("gripper_closed") else "open"
        if mode == "subgoal":
            return prompt_template.format(
                task=task_name,
                stage=sg.get("motion", ""),
                target=sg.get("target", ""),
                affordance=sg.get("affordance", ""),
                description=sg.get("description", ""),
                completion=sg.get("completion", ""),
                gripper_state=gripper_state,
                recent_moves=recent_str,
                gripper_color=gripper_color,
            )
        if mode == "affordance":
            return prompt_template.format(
                task=task_name,
                target=afford["target"],
                affordance=afford["affordance"],
                gripper_state=gripper_state,
                recent_moves=recent_str,
            )
        return prompt_template.format(
            task=task_name, gripper_state=gripper_state, recent_moves=recent_str
        )

    def _sample(step: dict, input_text: str, token: str) -> dict:
        return {
            "instruction": IMAGE_TOKENS + input_text,
            "input": "",
            "output": token,
            "images": [
                str((rollout_dir / step["agentview"]).resolve()),
                str((rollout_dir / step["wrist"]).resolve()),
            ],
        }

    samples = []
    mv_history: list[str] = []
    last_action_idx = -1

    for i, step in enumerate(steps):
        token = step["token"]
        if token not in ACTION_TOKENS:
            continue
        last_action_idx = i

        recent_str = ", ".join(mv_history[-RECENT_WINDOW:][::-1]) or "none"
        sg = step_subgoal.get(i, fallback_sg) if mode == "subgoal" else {}
        samples.append(_sample(step, _render(step, recent_str, sg), token))

        if token in MV_TOKENS:
            mv_history.append(token)

    # Terminal DONE: one synthesized sample per episode (no DONE in the recordings). It reuses
    # the last observed frame and the final context (last subgoal in subgoal mode); recent
    # moves now include every move, so the newest is the episode's last MV_*.
    if last_action_idx >= 0:
        last_step = steps[last_action_idx]
        recent_str = ", ".join(mv_history[-RECENT_WINDOW:][::-1]) or "none"
        sg = step_subgoal.get(last_action_idx, fallback_sg) if mode == "subgoal" else {}
        samples.append(_sample(last_step, _render(last_step, recent_str, sg), DONE_TOKEN))

    return samples


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert rollouts to LLaMA Factory Alpaca format")
    parser.add_argument(
        "rollout_dirs", nargs="+", type=Path,
        help="Rollout directories (or parent directories containing rollout subdirs)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path. Defaults to rollout[_subgoal].json inside the rollout dir "
             "(single rollout) or the rollouts' parent dir (multiple rollouts).",
    )
    parser.add_argument(
        "--task", default=None,
        help="Task description string applied to all rollouts (overridden per-folder by --task-map).",
    )
    parser.add_argument(
        "--task-map", nargs="+", metavar="DIRNAME=TASK", default=None,
        help="Map folder names to task descriptions, e.g. "
             "'banana=pick up the banana' 'mango=pick up the mango'. "
             "Takes priority over --task for matching folders.",
    )
    parser.add_argument(
        "--jsonl", action="store_true", default=False,
        help="Write output as JSON Lines (.jsonl) instead of a JSON array.",
    )
    parser.add_argument(
        "--use-subgoal", action="store_true", default=False,
        help="Use the full prompt (mvtoken_generator.txt) with per-step VLM subgoal info. "
             "Reads task_config.json; auto-generates it via generate_subgoals.py if absent. "
             "Default (off) uses the stage-free mvtoken_generator_lite.txt prompt.",
    )
    parser.add_argument(
        "--use-affordance", action="store_true", default=False,
        help="Use the lite+affordance prompt (mvtoken_generator_affordance.txt) with a single "
             "grasp-point hint (target + affordance) reused for every step. Reads "
             "affordance_config.json; auto-generates it via generate_affordance.py if absent. "
             "Mutually exclusive with --use-subgoal.",
    )
    parser.add_argument(
        "--gripper-color", default="black",
        help="Gripper color hint used by the full prompt (--use-subgoal only; default: black).",
    )
    # VLM forwarding args — only needed when auto-generating task/affordance config
    parser.add_argument("--vlm-backend", default=None, help="VLM backend for subgoal generation")
    parser.add_argument("--vlm-url", default=None, help="Override VLM base URL")
    parser.add_argument("--model", default=None, help="Override model name or path")
    parser.add_argument("--api-key", default=None, help="API key for VLM server")
    args = parser.parse_args()

    if args.use_subgoal and args.use_affordance:
        parser.error("--use-subgoal and --use-affordance are mutually exclusive.")
    mode = "subgoal" if args.use_subgoal else "affordance" if args.use_affordance else "lite"

    rollout_dirs = _expand_dirs(args.rollout_dirs)
    if not rollout_dirs:
        print("No valid rollout directories found.", file=sys.stderr)
        sys.exit(1)

    ext = ".jsonl" if args.jsonl else ".json"
    stem = {"subgoal": "rollout_subgoal", "affordance": "rollout_affordance"}.get(
        mode, "rollout"
    )
    default_filename = stem + ext
    if args.output is not None:
        output_path = args.output
    elif len(rollout_dirs) == 1:
        output_path = rollout_dirs[0] / default_filename
    else:
        output_path = rollout_dirs[0].parent / default_filename

    # Build task map: dirname -> task description
    task_map: dict[str, str] = {}
    for entry in (args.task_map or []):
        if "=" not in entry:
            parser.error(f"--task-map entry must be 'dirname=task description', got: {entry!r}")
        dirname, _, task_desc = entry.partition("=")
        task_map[dirname.strip()] = task_desc.strip()

    prompt_filename = {
        "subgoal": "mvtoken_generator.txt",
        "affordance": "mvtoken_generator_affordance.txt",
    }.get(mode, "mvtoken_generator_lite.txt")
    prompt_template = _load_prompt(prompt_filename)

    # Build VLM forwarding args for the subgoal-generation subprocess.
    vlm_args: list[str] = []
    for flag, value in (
        ("--vlm-backend", args.vlm_backend),
        ("--vlm-url", args.vlm_url),
        ("--model", args.model),
        ("--api-key", args.api_key),
    ):
        if value:
            vlm_args += [flag, value]

    def _resolve_task(rollout_dir: Path) -> str | None:
        """Return task string for this rollout: task_map > --task > None."""
        return task_map.get(rollout_dir.parent.name) or args.task or None

    # Ensure the per-rollout config exists (auto-generates via the matching VLM script).
    if mode == "subgoal":
        for d in rollout_dirs:
            _ensure_task_config(d, _resolve_task(d), vlm_args)
    elif mode == "affordance":
        for d in rollout_dirs:
            _ensure_affordance_config(d, _resolve_task(d), vlm_args)

    all_samples: list[dict] = []
    for d in rollout_dirs:
        samples = convert_rollout(
            d,
            prompt_template=prompt_template,
            mode=mode,
            task_override=_resolve_task(d),
            gripper_color=args.gripper_color,
        )
        print(f"{d.name}: {len(samples)} action steps")
        all_samples.extend(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if args.jsonl:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(all_samples)} samples -> {output_path}")


if __name__ == "__main__":
    main()
