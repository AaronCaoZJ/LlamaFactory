#!/usr/bin/env python3
"""Create a contact sheet for grounding the unique task descriptions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT = Path("data/agentrobot/MVTOKEN/mix_22-06_fk-pp/02_exchange_token/rollout_lite.json")
TASK_RE = re.compile(r"^Task:\s*(.+)$", re.MULTILINE)


def extract_task(instruction: str) -> str:
    match = TASK_RE.search(instruction)
    return match.group(1).strip() if match else "<NO_TASK>"


def wrap_text(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=Path("/tmp/task_template_aug_contact_sheet.jpg"))
    parser.add_argument("--thumb-width", type=int, default=180)
    parser.add_argument("--thumb-height", type=int, default=144)
    args = parser.parse_args()

    samples = json.loads(args.input.read_text(encoding="utf-8"))
    examples = {}
    for sample in samples:
        task = extract_task(sample.get("instruction", ""))
        examples.setdefault(task, sample)

    items = sorted(examples.items())
    label_h = 62
    canvas = Image.new("RGB", (args.thumb_width * 2, len(items) * (args.thumb_height + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font = None

    for row, (task, sample) in enumerate(items):
        y0 = row * (args.thumb_height + label_h)
        for col, image_path in enumerate((sample.get("images") or [])[:2]):
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((args.thumb_width, args.thumb_height))
            x = col * args.thumb_width + (args.thumb_width - image.width) // 2
            canvas.paste(image, (x, y0))

        label = f"{row + 1}. {task}"
        for line_idx, line in enumerate(wrap_text(label, 46)[:4]):
            draw.text((4, y0 + args.thumb_height + 2 + line_idx * 13), line, fill="black", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
    print(args.output)


if __name__ == "__main__":
    main()
