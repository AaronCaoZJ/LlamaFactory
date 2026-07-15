#!/usr/bin/env python3
"""Re-slot an existing LLaMA-Factory sample file from the two-image layout to the video slot.

Input is a converter output from ``rollout_to_llamafactory.py`` -- samples shaped
``{instruction: "<image><image>...", input, output, images: [agentview, wrist]}``.
Output is the same samples with both views moved into the video slot:

    {instruction: "<video>...", input, output, videos: [[agentview, wrist]]}

Why re-slot instead of re-running the converter with ``--video-slot``: this keeps the sample
set, their order and every prompt byte-identical to the image-layout dataset, so the two can
be trained head-to-head as a controlled experiment. Use ``--video-slot`` when converting fresh
rollouts; use this when you want the video twin of a dataset you already trained on.

What changes for the model (Qwen3.5, 256x256 views):

    <image><image>  ->  two vision blocks, 64 + 64 = 128 visual tokens
    <video>         ->  ONE vision block, 64 visual tokens, prefixed with a "<0.2 seconds>"
                        timestamp -- the 3D patch embed (temporal_patch_size=2) fuses the two
                        frames, so the two viewpoints are mixed at every spatial position.

The timestamp text is a function of ``video_fps`` in the training yaml; eval must generate its
mp4 at the same fps or the prompt will not match. See .ai/espresso/ShowRobot-VLM_HANDOFF.md.

Usage:
    python data/agentrobot/to_video_slot.py \\
        data/agentrobot/MVTOKEN/mix_22_27/v3/rollout_lite.json \\
        --output data/agentrobot/MVTOKEN/mix_22_27/v3/rollout_lite_video.json
"""

import argparse
import json
import sys
from pathlib import Path


IMAGE_TOKENS = "<image><image>"
VIDEO_TOKEN = "<video>"


def _load(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    return json.loads(text)


def reslot(sample: dict, idx: int) -> dict:
    """Move the two views of one sample from the image slot into the video slot."""
    instruction = sample["instruction"]
    images = sample.get("images") or []

    if not instruction.startswith(IMAGE_TOKENS):
        raise ValueError(f"sample {idx}: instruction does not start with {IMAGE_TOKENS!r}")

    if len(images) != 2:
        raise ValueError(f"sample {idx}: expected exactly 2 images, got {len(images)}")

    out = {k: v for k, v in sample.items() if k != "images"}
    out["instruction"] = VIDEO_TOKEN + instruction[len(IMAGE_TOKENS) :]
    out["videos"] = [list(images)]  # nested: one frame list per <video> placeholder
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Sample file in the two-image layout (.json or .jsonl).")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path. Defaults to '<input stem>_video<ext>' next to the input.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[error] no such file: {args.input}", file=sys.stderr)
        sys.exit(1)

    samples = _load(args.input)
    output_path = args.output or args.input.with_name(f"{args.input.stem}_video{args.input.suffix}")

    try:
        reslotted = [reslot(s, i) for i, s in enumerate(samples)]
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if output_path.suffix == ".jsonl":
            for sample in reslotted:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            json.dump(reslotted, f, ensure_ascii=False, indent=2)

    print(f"{len(reslotted)} samples re-slotted to <video>: {output_path}")


if __name__ == "__main__":
    main()
