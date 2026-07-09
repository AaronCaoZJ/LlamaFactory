#!/usr/bin/env python3
"""Back up a qwen3.5-2b mikomiko full_v0 checkpoint to a PRIVATE HF repo — FOR RESUMING TRAINING.

Unlike an inference-only export, this uploads the WHOLE checkpoint directory so training can be
RESUMED on another machine: model weights + the DeepSpeed optimizer state (global_step*/),
LR scheduler (scheduler.pt), per-rank RNG (rng_state_*.pth), trainer_state.json, training_args.bin,
the `latest` DeepSpeed pointer, and all tokenizer/processor/config files.

Auth: HF_TOKEN env var or the cached huggingface token (no token hardcoded here).

Usage:
    python scripts/qwen3_5/mikomiko_tag/hf_upload_mikomiko_full_v0.py                 # step 11530 (default)
    STEP=8000 python scripts/qwen3_5/mikomiko_tag/hf_upload_mikomiko_full_v0.py       # a different checkpoint
    python scripts/qwen3_5/mikomiko_tag/hf_upload_mikomiko_full_v0.py --step 8000     # same, via flag

Resume note: DeepSpeed ZeRO-0 stores the optimizer state inside global_step*/mp_rank_00_model_states.pt.
After downloading on the target machine, resume by pointing the trainer at the checkpoint dir
(e.g. `resume_from_checkpoint: <dir>/checkpoint-<STEP>`). Do a short resume smoke-test first to
confirm the optimizer state is complete and step/lr continue correctly.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "aaroncaozj/Mikomiko_pornpic_tagger"
PRIVATE = True
# machine-agnostic: this script lives at <repo>/scripts/qwen3_5/mikomiko_tag/, so parents[3] is the repo root.
LF_ROOT = Path(__file__).resolve().parents[3]
CKPT_ROOT = str(LF_ROOT / "saves/qwen3.5-2b/mikomiko/full_v0")

# resume-critical files we assert are present (warn loudly if any is missing).
RESUME_CRITICAL = ["model.safetensors", "scheduler.pt", "trainer_state.json",
                   "training_args.bin", "latest"]


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def manifest(src):
    """List every file under src (recursively) with sizes; return (files, total_bytes)."""
    files, total = [], 0
    for root, _dirs, names in os.walk(src):
        for name in sorted(names):
            p = os.path.join(root, name)
            sz = os.path.getsize(p)
            files.append((os.path.relpath(p, src), sz))
            total += sz
    return files, total


def main():
    ap = argparse.ArgumentParser(description="Upload a FULL resumable mikomiko checkpoint to HF.")
    ap.add_argument("--step", default=os.environ.get("STEP", "11530"),
                    help="checkpoint step to upload (default 11530 / $STEP)")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--dry-run", action="store_true", help="show the manifest and exit, no upload")
    args = ap.parse_args()

    src = os.path.join(CKPT_ROOT, f"checkpoint-{args.step}")
    dst_prefix = f"checkpoint-{args.step}"      # keep the step visible as a repo subfolder
    if not os.path.isdir(src):
        raise SystemExit(f"!! source checkpoint missing: {src}")

    files, total = manifest(src)
    has = {rel for rel, _ in files}
    print(f"source : {src}")
    print(f"target : https://huggingface.co/{args.repo}/tree/main/{dst_prefix}  (private={PRIVATE})")
    print(f"total  : {len(files)} files, {human(total)}  (uploading EVERYTHING for resume)\n")

    # highlight the resume-critical pieces so it's obvious the optimizer state is included
    opt_files = [(rel, sz) for rel, sz in files if rel.startswith("global_step")]
    print("  resume state being uploaded:")
    for rel in RESUME_CRITICAL:
        mark = "OK " if rel in has else "!! MISSING"
        sz = next((s for r, s in files if r == rel), 0)
        print(f"    [{mark}] {rel:<28} {human(sz) if sz else ''}")
    if opt_files:
        for rel, sz in opt_files:
            print(f"    [OK ] {rel:<28} {human(sz)}   <- DeepSpeed optimizer state")
    else:
        print("    [!! MISSING] global_step*/  <- NO optimizer state found; resume will NOT be lossless!")
    missing = [f for f in RESUME_CRITICAL if f not in has]
    if missing or not opt_files:
        print(f"\n  WARNING: resume state incomplete (missing: {missing}"
              f"{' + optimizer' if not opt_files else ''}). Uploading anyway.\n")

    if args.dry_run:
        print("[dry-run] no upload performed.")
        return

    api = HfApi()
    print("HF user:", api.whoami().get("name"))
    create_repo(args.repo, private=PRIVATE, repo_type="model", exist_ok=True)

    # upload_folder recurses (incl. global_step*/), handles large safetensors via multipart, and
    # is resumable if interrupted. path_in_repo puts it under the checkpoint-<STEP>/ subfolder.
    print(f"uploading {human(total)} -> {dst_prefix}/ ...", flush=True)
    api.upload_folder(
        folder_path=src,
        path_in_repo=dst_prefix,
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"full resumable checkpoint-{args.step} (weights + optimizer + scheduler + RNG)",
    )
    print("DONE ->", f"https://huggingface.co/{args.repo}/tree/main/{dst_prefix}")


if __name__ == "__main__":
    main()
