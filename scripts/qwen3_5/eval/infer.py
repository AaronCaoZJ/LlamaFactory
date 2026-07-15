"""VLM 推理客户端 — 单条推理 + 批量评估（Qwen3.5）

按样本的字段自动选布局，A/B 两条路线用同一个脚本评测：
  样本有 "images": [agentview, wrist]     -> image 布局（<image><image>，128 visual tokens）
  样本有 "videos": [[agentview, wrist]]   -> video 槽位（<video>，64 visual tokens，方案 B）

video 布局会把两张图现场编码成一个 2 帧的无损 mp4（base64 data URI）再发出去。

⚠️ 两条铁律（HANDOFF §4.2），错一处模型就退化：
  1. 图片 / 视频排在文本之前（OpenAI content 数组顺序 = 占位符顺序）。
  2. mp4 的 fps 必须等于训练 yaml 的 video_fps —— 它决定 prompt 里的 "<0.2 seconds>" 时间戳文本。
     默认 2.0；改了 yaml 就用 VIDEO_FPS 环境变量同步改这里。
  另外 server 必须挂 --chat-template scripts/qwen3_5/eval/chat_template_qwen3_5_lf.jinja：
  Qwen3.5 官方模板即使 enable_thinking=false 也会插一个空 think 块（<think>\\n\\n</think>\\n\\n），
  与 LF 的 qwen3_5_nothink 训练分布差 4 个 token。

配置通过环境变量传入（用 run_eval.sh 封装）：
  API_URL      server 地址（默认 http://localhost:8109）
  MODEL_NAME   OpenAI 请求里的 model 字段 / vllm --lora-modules 的 key（不对会 404）
  VIDEO_FPS    生成 mp4 的帧率，必须与训练 yaml 的 video_fps 一致

模式:
  python infer.py "描述图中场景" --image a.png --image b.png   # 单条推理（image 布局）
  python infer.py "描述图中场景" --video a.png --video b.png   # 单条推理（video 槽位）
  python infer.py eval -e <evalset.json> [-n N] [--raw]        # 批量评估（布局按样本字段自动选）

先启动 server:
  bash scripts/qwen3_5/eval/start_vllm_server_9.sh
"""
import argparse
import base64
import io
import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ── 环境变量配置 ────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8109")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3_5_9b_mix_22_27_v3_video")

# ⚠️ 必须等于训练 yaml 的 video_fps：LF 用它算 "<0.2 seconds>" 时间戳，vLLM 从 mp4 元数据算，
# 两边不一致时 prompt 里的时间戳文本就对不上。
VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "2.0"))

# ── 评估数据集配置 ──────────────────────────────────────────────────────────
DATASET = os.environ.get(
    "EVALSET", "data/agentrobot/ood_sample/v3/rollout_lite_video.json"
)
VALID_TOKENS = {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN", "GRASP", "RELEASE", "DONE"}
_IMAGE_TOKEN = "<image>"
_VIDEO_TOKEN = "<video>"
THINK_OPEN_ID = 248068  # '<think>' 在 qwen3.5 词表里的 id（check_prompt_parity 用）


# ── 媒体编码 ────────────────────────────────────────────────────────────────
def encode_image(path: str) -> str:
    suffix = Path(path).suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{data}"


def encode_frames_as_video(paths: list[str], fps: float = VIDEO_FPS) -> str:
    """把若干张图编码成一个 mp4（每图一帧），返回 base64 data URI。

    无损 h264（crf=0）：训练侧喂的是无损 PNG，评测侧再被 h264 压一道会引入像素偏移。
    fps 必须与训练 yaml 的 video_fps 一致 —— 它决定 prompt 里的时间戳文本。
    """
    import av
    from PIL import Image

    frames = [Image.open(p).convert("RGB") for p in paths]
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=int(fps) if float(fps).is_integer() else fps)
    stream.width, stream.height = frames[0].size
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "0"}  # lossless
    for frame in frames:
        for packet in stream.encode(av.VideoFrame.from_image(frame)):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return f"data:video/mp4;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── HTTP 推理 ───────────────────────────────────────────────────────────────
def check_prompt_parity() -> None:
    """漏挂 --chat-template 时直接中止（训推失配是静默的，跑完才发现就晚了）。

    LF 的 qwen3_5_nothink 在 'assistant\\n' 之后什么都不加；Qwen 出厂的 jinja 无论传什么参数都会
    加 think —— enable_thinking=false 也只是把「开着的 think」换成「闭合的空 think 块」，仍然多
    4 个 token。输出看着完全正常，只是悄悄掉点（mikomiko tagger 实测 microF1 -1.2pt）。
    详见 scripts/qwen3_5/QWEN35_DEBUG.md。

    这里用 /tokenize 探一条纯文本，看渲染结果里有没有混进 <think>。
    """
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "PROBE"}],
        "add_generation_prompt": True,
    }).encode()
    req = urllib.request.Request(
        f"{API_URL}/tokenize",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            tokens = json.loads(resp.read().decode()).get("tokens") or []
    except Exception as e:  # /tokenize 不存在（非 vLLM 后端）-> 跳过，不挡正常推理
        print(f"[warn] prompt-parity 检查跳过（{e!r}）", file=sys.stderr)
        return

    if THINK_OPEN_ID in tokens:
        sys.exit(
            f"[fatal] server ({API_URL}) 的 chat template 会注入 <think> —— 训推失配，会静默掉点。\n"
            f"        重启时挂上 LF 对齐的模板：bash scripts/qwen3_5/eval/start_vllm_server_9.sh\n"
            f"        （它会传 --chat-template scripts/qwen3_5/eval/chat_template_qwen3_5_lf.jinja）\n"
            f"        背景见 scripts/qwen3_5/QWEN35_DEBUG.md"
        )
    print(f"[vllm] prompt parity OK（{MODEL_NAME} @ {API_URL}，无 think token）")


def chat(
    messages: list[dict],
    image_paths: list[str] | None = None,
    video_frames: list[str] | None = None,
    max_tokens: int = 8,
    temperature: float = 0.0,
) -> str:
    image_paths = image_paths or []
    video_frames = video_frames or []
    formatted: list[dict] = []

    for msg in messages:
        if msg["role"] == "user" and (image_paths or video_frames):
            clean_text = msg["content"].replace(_IMAGE_TOKEN, "").replace(_VIDEO_TOKEN, "").strip()
            # 媒体排在文本之前 —— content 数组顺序就是占位符顺序，放反了与训练分布失配。
            content: list[dict] = []
            if video_frames:
                content.append({
                    "type": "video_url",
                    "video_url": {"url": encode_frames_as_video(video_frames)},
                })
            for p in image_paths:
                content.append({"type": "image_url", "image_url": {"url": encode_image(p)}})
            content.append({"type": "text", "text": clean_text})
            formatted.append({"role": "user", "content": content})
        else:
            formatted.append(msg)

    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": formatted,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        f"{API_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"[ERROR] HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError:
        print(f"[ERROR] Cannot reach server at {API_URL}. "
              f"Run: bash scripts/qwen3_5/eval/start_vllm_server_9.sh", file=sys.stderr)
        sys.exit(1)

    return result["choices"][0]["message"]["content"]


def _media_of(sample: dict) -> tuple[list[str], list[str]]:
    """Return (image_paths, video_frames) for one sample -- exactly one of them is non-empty."""
    if sample.get("videos"):
        frames = sample["videos"][0]  # nested: one frame list per <video>
        return [], list(frames)

    return list(sample.get("images") or []), []


# ── 单条推理模式 ────────────────────────────────────────────────────────────
def run_single(args: argparse.Namespace) -> None:
    print(f"Server  : {API_URL}  model={MODEL_NAME}")
    print(f"Prompt  : {args.prompt}")
    if args.videos:
        print(f"Video   : {args.videos}  (fps={VIDEO_FPS})")
    if args.images:
        print(f"Images  : {args.images}")
    print("─" * 72)

    response = chat(
        [{"role": "user", "content": args.prompt}],
        image_paths=args.images, video_frames=args.videos,
        max_tokens=args.max_tokens, temperature=args.temperature,
    )
    print(response)


# ── 批量评估模式 ────────────────────────────────────────────────────────────
def run_eval(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    dataset_path = args.evalset or DATASET
    with open(dataset_path) as f:
        all_samples = json.load(f)

    samples = random.sample(all_samples, min(args.n_samples, len(all_samples)))
    max_tokens = 128 if args.raw else 8
    layout = "video slot (<video>)" if all_samples[0].get("videos") else "image layout (<image><image>)"

    print(f"Server  : {API_URL}  model={MODEL_NAME}")
    print(f"Dataset : {dataset_path}  ({len(samples)}/{len(all_samples)} samples, seed={args.seed})")
    print(f"Layout  : {layout}" + (f"   video_fps={VIDEO_FPS}" if all_samples[0].get("videos") else ""))
    print("─" * 88)
    print(f"{'#':>3}  {'Label':>10}  {'Pred':>14}  {'Match':>5}  {'Full Response' if args.raw else 'Input context'}")
    print("─" * 88)

    correct = 0
    per_token_total: dict[str, int] = {}
    per_token_correct: dict[str, int] = {}

    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        sample_input = sample.get("input") or ""
        # 训练侧 alpaca converter 用 "\n".join([instruction, input])，此处必须同样用单个 "\n"
        user_text = f"{instruction}\n{sample_input}" if sample_input else instruction
        image_paths, video_frames = _media_of(sample)

        pred_text = chat(
            [{"role": "user", "content": user_text}],
            image_paths=image_paths, video_frames=video_frames, max_tokens=max_tokens,
        ).strip()
        label = sample["output"]
        pred_token = next((w for w in pred_text.split() if w in VALID_TOKENS), pred_text[:20])
        match = pred_token == label

        if match:
            correct += 1
        per_token_total[label] = per_token_total.get(label, 0) + 1
        per_token_correct[label] = per_token_correct.get(label, 0) + (1 if match else 0)

        tail = repr(pred_text) if args.raw else sample_input.replace("\n", " | ")[:38]
        print(f"{i+1:>3}  {label:>10}  {pred_token:>14}  {'✓' if match else '✗':>5}  {tail}")

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
        description="Qwen3.5 推理客户端（配置通过 API_URL / MODEL_NAME / VIDEO_FPS 环境变量传入）"
    )
    subparsers = parser.add_subparsers(dest="mode")

    sp = subparsers.add_parser("single", help="单条推理（默认模式，可省略 'single'）")
    sp.add_argument("prompt", help="文本 prompt")
    sp.add_argument("--image", action="append", dest="images", default=[], help="图片路径（可重复，走 image 布局）")
    sp.add_argument("--video", action="append", dest="videos", default=[],
                    help="视频帧图片路径（可重复，按顺序编码成一个 mp4，走 video 槽位）")
    sp.add_argument("--max-tokens", type=int, default=512)
    sp.add_argument("--temperature", type=float, default=0.0)

    ep = subparsers.add_parser("eval", help="批量评估（布局按样本的 images / videos 字段自动选）")
    ep.add_argument("-n", "--n-samples", type=int, default=10)
    ep.add_argument("--seed", type=int, default=42)
    ep.add_argument("--raw", action="store_true", help="显示完整原始回复")
    ep.add_argument("--evalset", "-e", default=None, metavar="PATH", help="测试集 JSON 路径")

    # 兼容旧用法：第一个参数不是子命令时当作 single prompt
    argv = sys.argv[1:]
    if argv and argv[0] not in ("single", "eval", "-h", "--help"):
        argv = ["single"] + argv

    args = parser.parse_args(argv)
    if args.mode is None:
        parser.print_help()
        sys.exit(0)

    # 每次访问 server 前先验一次训推 prompt 一致性；失配直接退出，别让人跑完一整轮才发现。
    check_prompt_parity()

    if args.mode == "single":
        run_single(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
