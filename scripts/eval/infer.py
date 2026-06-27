"""VLM 推理客户端 — 单条推理 + 批量评估

配置通过环境变量传入（用 sh 脚本封装）：
  API_URL      vllm server 地址（默认 http://localhost:8101）
  MODEL_NAME   --lora-modules 里的 key（默认 MVTOKEN）

模式:
  python infer.py "描述图中场景" --image /path/to/img.png   # 单条推理
  python infer.py --eval [-n N] [--raw] [--seed S]          # 批量评估

先启动 server:
  bash scripts/eval/start_vllm_server.sh
"""
import argparse
import base64
import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ── 环境变量配置 ────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8101")
MODEL_NAME = os.environ.get("MODEL_NAME", "MVTOKEN")

# ── 评估数据集配置 ──────────────────────────────────────────────────────────
# DATASET = "/workspace1/zhijun/LlamaFactory/data/agentrobot/overfit_test/rollout_000.json"
DATASET = "/workspace1/zhijun/LlamaFactory/data/robot_rollout.json"
VALID_TOKENS = {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN", "GRASP", "RELEASE"}
_IMAGE_TOKEN = "<image>"


# ── 消融：去掉 input 里的 "Stage:" 行 ────────────────────────────────────────
def strip_stage(text: str) -> str:
    """删除 input 中以 'Stage:' 开头的整行（消融实验：测试模型是否真的依赖 stage）。"""
    return "\n".join(
        line for line in text.split("\n") if not line.strip().lower().startswith("stage:")
    )


# ── 图片编码 ────────────────────────────────────────────────────────────────
def encode_image(path: str) -> str:
    suffix = Path(path).suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{data}"


# ── HTTP 推理 ───────────────────────────────────────────────────────────────
def chat(
    messages: list[dict],
    image_paths: list[str],
    max_tokens: int,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> str:
    formatted: list[dict] = []
    for msg in messages:
        if msg["role"] == "user" and image_paths:
            clean_text = msg["content"].replace(_IMAGE_TOKEN, "").strip()
            content: list[dict] = [{"type": "text", "text": clean_text}]
            for p in image_paths:
                content.append({"type": "image_url", "image_url": {"url": encode_image(p)}})
            formatted.append({"role": "user", "content": content})
        else:
            formatted.append(msg)

    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": formatted,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }).encode()

    req = urllib.request.Request(
        f"{API_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"[ERROR] HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError:
        print(f"[ERROR] Cannot reach server at {API_URL}. Run: bash scripts/eval/start_vllm_server.sh", file=sys.stderr)
        sys.exit(1)

    return result["choices"][0]["message"]["content"]


# ── 单条推理模式 ────────────────────────────────────────────────────────────
def run_single(args: argparse.Namespace) -> None:
    image_paths = args.images or []
    messages = [{"role": "user", "content": args.prompt}]

    print(f"Server  : {API_URL}  model={MODEL_NAME}")
    print(f"Prompt  : {args.prompt}")
    if image_paths:
        print(f"Images  : {image_paths}")
    print("─" * 72)

    response = chat(messages, image_paths, args.max_tokens, args.temperature, args.think)
    print(response)


# ── 批量评估模式 ────────────────────────────────────────────────────────────
def run_eval(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    dataset_path = args.evalset or DATASET
    with open(dataset_path) as f:
        all_samples = json.load(f)

    samples = random.sample(all_samples, min(args.n_samples, len(all_samples)))
    max_tokens = 128 if args.raw else 8

    print(f"Server  : {API_URL}  model={MODEL_NAME}")
    print(f"Dataset : {dataset_path}  ({len(samples)}/{len(all_samples)} samples, seed={args.seed})")
    if args.no_stage:
        print("Ablation: --no-stage 已开启（input 中的 'Stage:' 行已删除）")
    print("─" * 88)
    if args.raw:
        print(f"{'#':>3}  {'Label':>10}  {'Pred':>14}  {'Match':>5}  Full Response")
    else:
        print(f"{'#':>3}  {'Label':>10}  {'Pred':>14}  {'Match':>5}  Input context")
    print("─" * 88)

    correct = 0
    per_token_total: dict[str, int] = {}
    per_token_correct: dict[str, int] = {}

    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        sample_input = sample.get("input") or ""
        if args.no_stage and sample_input:
            sample_input = strip_stage(sample_input)
        user_text = f"{instruction}\n\n{sample_input}" if sample_input else instruction
        messages = [{"role": "user", "content": user_text}]
        image_paths: list[str] = sample["images"]

        pred_text = chat(messages, image_paths, max_tokens).strip()
        label = sample["output"]
        pred_token = next((w for w in pred_text.split() if w in VALID_TOKENS), pred_text[:20])
        match = pred_token == label

        if match:
            correct += 1
        per_token_total[label] = per_token_total.get(label, 0) + 1
        per_token_correct[label] = per_token_correct.get(label, 0) + (1 if match else 0)

        if args.raw:
            print(f"{i+1:>3}  {label:>10}  {pred_token:>14}  {'✓' if match else '✗':>5}  {pred_text!r}")
        else:
            context = sample["input"].replace("\n", " | ")[:38]
            print(f"{i+1:>3}  {label:>10}  {pred_token:>14}  {'✓' if match else '✗':>5}  {context}")

    print("─" * 88)
    print("Per-token accuracy:")
    for token in sorted(per_token_total):
        n = per_token_total[token]
        c = per_token_correct.get(token, 0)
        print(f"  {token:<12}  {c:>2}/{n:<2} = {c/n*100:5.1f}%")
    print("─" * 88)
    print(f"Overall : {correct}/{len(samples)} = {correct/len(samples)*100:.1f}%")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM 推理客户端（配置通过 API_URL / MODEL_NAME 环境变量传入）"
    )
    subparsers = parser.add_subparsers(dest="mode")

    # --- single: infer.py single "prompt" --image ...
    sp = subparsers.add_parser("single", help="单条推理（默认模式，可省略 'single'）")
    sp.add_argument("prompt", help="文本 prompt")
    sp.add_argument("--image", action="append", dest="images", default=[], help="图片路径（可重复）")
    sp.add_argument("--think", action="store_true", help="开启 thinking 模式")
    sp.add_argument("--max-tokens", type=int, default=512)
    sp.add_argument("--temperature", type=float, default=0.0)

    # --- eval: infer.py eval [-n N] [--raw] [--seed S] [--evalset PATH]
    ep = subparsers.add_parser("eval", help="批量评估")
    ep.add_argument("-n", "--n-samples", type=int, default=10)
    ep.add_argument("--seed", type=int, default=42)
    ep.add_argument("--raw", action="store_true", help="显示完整原始回复")
    ep.add_argument("--evalset", "-e", default=None, metavar="PATH", help="测试集 JSON 路径（默认用 infer.py 里的 DATASET）")
    ep.add_argument("--no-stage", action="store_true", help="消融：删除 input 里的 'Stage:' 行后再推理")

    # 兼容旧用法：第一个参数不是子命令时当作 single prompt
    argv = sys.argv[1:]
    if argv and argv[0] not in ("single", "eval", "-h", "--help"):
        argv = ["single"] + argv

    args = parser.parse_args(argv)
    if args.mode is None:
        parser.print_help()
        sys.exit(0)

    if args.mode == "single":
        run_single(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
