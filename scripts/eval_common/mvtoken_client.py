# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""MVTOKEN 评测客户端的**共享核心** —— Qwen3.5 / gemma4 / InternVL3.5 三家族共用.

设计：一个核心 + 每家族一个薄入口。
  * 通用的东西（HTTP、图片/视频编码、prompt 拼接、评测循环、logprobs 分析、统计）都在这里；
  * 每个家族只在 `scripts/<family>/eval/infer.py` 里声明一个 :class:`FamilySpec`，写清楚它的**硬约束**
    （chat template / system / 归一化 / video 布局），然后调 :func:`main`。
  逻辑写三遍必然各自漂移 —— 所以任何"三家都一样"的改动只应该改这个文件。

三家族通用的训推一致铁律（错一处就静默掉点，都是踩坑换来的）：
  1. **媒体排在文本之前** —— OpenAI content 数组的顺序就是占位符顺序，训练样本是 `<image><image>TEXT`。
  2. **instruction 与 input 用单个 `\\n` 连接** —— LF 的 alpaca converter 是 `"\\n".join(...)`，不是 `\\n\\n`。
  3. **prompt 里字面量的 `<image>` / `<video>` 必须剥掉** —— 训练时 LF 是就地替换占位符的；
     推理侧图片走独立的 content part，字面量留着就多出一份文本。

logprobs（`--logprobs`）：
  9 个 action token 会被 tokenizer 切成多个 sub-token（Qwen/InternVL: `MV_FWD` -> `MV` + `_FWD`；
  gemma4: `MV` + `_` + `FW` + `D`），**首 token 只能区分 MV / GR / RELEASE / DONE 四类**，
  六个方向要看第二（gemma4 是第三）个 token。所以这里不用"首 token 概率"冒充 action 概率，
  而是**逐 token top-k 前缀累乘**，见 :func:`action_probability` 的 docstring。
"""

import argparse
import base64
import io
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path


# ── 常量 ────────────────────────────────────────────────────────────────────
#: 9 个合法 action token（顺序固定，logprobs 输出按这个顺序排）。
ACTIONS: tuple[str, ...] = (
    "MV_FWD",
    "MV_BACK",
    "MV_LEFT",
    "MV_RIGHT",
    "MV_UP",
    "MV_DOWN",
    "GRASP",
    "RELEASE",
    "DONE",
)
ACTION_SET = set(ACTIONS)

IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

#: 仓库根（scripts/eval_common/mvtoken_client.py -> parents[2]）。
LF_ROOT = Path(__file__).resolve().parents[2]


# ── 家族声明 ────────────────────────────────────────────────────────────────
@dataclass
class FamilySpec:
    """一个模型家族的硬约束。薄入口只需要填这个结构.

    Attributes:
        name: 家族名（打印 / 输出文件名用）。
        default_api_url: 默认 server 地址（可被 API_URL 环境变量或 --api-url 覆盖）。
        default_model: 默认 model 字段 / vLLM `--lora-modules` 的 key（不对会 404）。
        system_prompt: 客户端要不要显式发 system turn。None = 不发（模板自己注入）。
        video_layout: `<video>` 槽位的编码方式。"mp4" = 把若干帧编成一个无损 mp4；
            None = 该家族没有可用的 video 契约，遇到带 videos 的样本直接报错。
        video_fps: mp4 帧率，必须等于训练 yaml 的 `video_fps`。
        server_hint: server 挂了时打印的启动命令。
        preflight: 训推一致性自检（拿 /tokenize 探服务端真正渲染出来的 prompt）。失配应直接
            `sys.exit`，别让人跑完一整轮才发现。
        notes: 打印在 header 里的家族约束提醒。
    """

    name: str
    default_api_url: str
    default_model: str
    system_prompt: str | None = None
    video_layout: str | None = None
    video_fps: float = 2.0
    server_hint: str = ""
    preflight: Callable[["MvTokenClient"], None] | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


# ── 媒体编码 ────────────────────────────────────────────────────────────────
def encode_image(path: str) -> str:
    """把图片读成 base64 data URI."""
    suffix = Path(path).suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    return f"data:image/{mime};base64,{data}"


def encode_frames_as_video(paths: Sequence[str], fps: float) -> str:
    """把若干张图编码成一个 mp4（每图一帧），返回 base64 data URI.

    无损 h264（crf=0）：训练侧喂的是无损 PNG，评测侧再被 h264 压一道会引入像素偏移。
    fps 必须与训练 yaml 的 `video_fps` 一致 —— LF 用它算 prompt 里的 "<0.2 seconds>" 时间戳文本，
    vLLM 则从 mp4 元数据反推，两边不一致 prompt 就对不上。
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


# ── 消融 ────────────────────────────────────────────────────────────────────
def strip_stage(text: str) -> str:
    """删除以 'Stage:' 开头的整行（消融实验：测试模型是否真的依赖 stage）.

    只有 ood_sample v0 带 Stage 行（在 `input` 里），v1/v2/v3 没有；这里对 instruction 和
    input 都扫一遍，换数据集也不会漏。
    """
    return "\n".join(line for line in text.split("\n") if not line.strip().lower().startswith("stage:"))


def fatal(*lines: str) -> None:
    """训推失配 -> 直接退出（失配是静默的，跑完才发现就晚了）."""
    sys.exit("\n".join(["[fatal] " + lines[0], *[" " * 8 + x for x in lines[1:]]]))


# ── HTTP 客户端 ─────────────────────────────────────────────────────────────
class MvTokenClient:
    """对着一个 OpenAI 兼容 server（vLLM）的极薄客户端."""

    def __init__(self, spec: FamilySpec, api_url: str, model: str, video_fps: float) -> None:
        self.spec = spec
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.video_fps = video_fps

    # -- 底层 ---------------------------------------------------------------
    def _post(self, path: str, payload: dict, timeout: int = 180) -> dict:
        req = urllib.request.Request(
            f"{self.api_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            sys.exit(f"[ERROR] HTTP {e.code} on {path}: {body}")
        except urllib.error.URLError as e:
            sys.exit(f"[ERROR] server 不可达 ({self.api_url}): {e}\n        启动: {self.spec.server_hint}")

    # -- /tokenize ----------------------------------------------------------
    def tokenize_text(self, text: str) -> list[str]:
        """把一段纯文本切成 token 串（不套 chat template）."""
        payload = {"model": self.model, "prompt": text, "add_special_tokens": False, "return_token_strs": True}
        return self._post("/tokenize", payload, timeout=60).get("token_strs") or []

    def tokenize_messages(self, messages: list[dict]) -> list[str]:
        """让服务端按它当前的 chat template 渲染 messages，返回 token 串.

        这是唯一能看见 **vLLM 真正喂给模型那串 token** 的办法（图像展开成多少 token 是运行时
        算的，读 jinja 看不出来）—— 所有 preflight 自检都建立在它上面。
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "add_generation_prompt": True,
            "return_token_strs": True,
        }
        return self._post("/tokenize", payload, timeout=90).get("token_strs") or []

    # -- prompt 组装 --------------------------------------------------------
    def build_messages(
        self,
        user_text: str,
        image_paths: Sequence[str] = (),
        video_frames: Sequence[str] = (),
    ) -> list[dict]:
        """按训练分布拼 OpenAI messages（媒体在前、文本在后；剥掉字面量占位符）."""
        messages: list[dict] = []
        if self.spec.system_prompt is not None:
            messages.append({"role": "system", "content": self.spec.system_prompt})

        if not image_paths and not video_frames:
            messages.append({"role": "user", "content": user_text})
            return messages

        if video_frames and self.spec.video_layout is None:
            fatal(
                f"{self.spec.name} 没有可用的 <video> 契约，但样本带了 videos 字段。",
                "换 image 布局的评测集，或先补上该家族的 video 编码方式。",
            )

        clean_text = user_text.replace(IMAGE_TOKEN, "").replace(VIDEO_TOKEN, "").strip()
        # ⚠️ 媒体排在文本之前 —— content 数组顺序 = 占位符顺序，放反了与训练分布失配。
        content: list[dict] = []
        if video_frames:
            content.append(
                {"type": "video_url", "video_url": {"url": encode_frames_as_video(video_frames, self.video_fps)}}
            )

        for p in image_paths:
            content.append({"type": "image_url", "image_url": {"url": encode_image(p)}})

        content.append({"type": "text", "text": clean_text})
        messages.append({"role": "user", "content": content})
        return messages

    # -- /v1/chat/completions ----------------------------------------------
    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 8,
        temperature: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 20,
    ) -> dict:
        """返回 `choices[0]`（含 message / logprobs / finish_reason）."""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs

        return self._post("/v1/chat/completions", payload)["choices"][0]


# ── logprobs 分析 ───────────────────────────────────────────────────────────
def parse_logprob_steps(choice: dict) -> list[dict]:
    """把 OpenAI 的 logprobs 结构转成逐步的 `{token, prob, top: {token: prob}}`."""
    entries = ((choice.get("logprobs") or {}).get("content")) or []
    steps: list[dict] = []
    for entry in entries:
        top = {t["token"]: math.exp(t["logprob"]) for t in (entry.get("top_logprobs") or [])}
        token = entry["token"]
        top.setdefault(token, math.exp(entry["logprob"]))
        steps.append({"token": token, "prob": math.exp(entry["logprob"]), "top": top})

    return steps


def action_probability(steps: list[dict], action: str) -> tuple[float | None, float, str]:
    """算 P(模型生成整串 `action`)，返回 `(精确值 or None, 上界, 状态)`.

    **为什么不能用首 token 概率**：`MV_FWD` 在 Qwen/InternVL 词表里是 `MV` + `_FWD`，在 gemma4
    里是 `MV` + `_` + `FW` + `D`。首 token 只能区分 MV / GR / RELEASE / DONE 四类，六个方向全在
    后面的 token 上 —— 拿首 token 当 action 概率会把 6 个方向混成一个数。

    **算法（逐 token 前缀累乘）**：一次前向只能拿到「在**实际生成前缀**条件下」每步的 top-k。
    所以沿实际生成路径走，每步在 top-k 里找能匹配 action 剩余部分的 token 并累乘：

      * 一路匹配到 action 走完 -> `exact`，值是精确概率。注意最后一步即使选了与实际生成不同的
        token（比如实际吐 `MV_DOWN`，问 `MV_FWD`），用的仍是同一个条件分布，**结果依然精确**。
      * 中途岔开实际路径且 action 还没走完（比如实际吐 `MV_*`，问 `GRASP`：`GR` 在 top-k 里，
        但 `ASP` 的条件分布要在 `GR` 之后重新前向才知道）-> `branch_upper`，只能给上界
        （= 已知前缀概率 × 1）。实测 `GR` 后面接 `ASP` 的概率 ≈ 0.9999，所以这个上界很紧。
      * 某步需要的 token 掉出 top-k -> `topk_upper`，值 None，上界 = 已知前缀概率 × 该步 top-k 最小值。
      * 生成步数不够（max_tokens 太小）-> `len_upper`。

    Args:
        steps: :func:`parse_logprob_steps` 的输出。
        action: 9 个合法 action 之一。

    Returns:
        `(prob, upper_bound, status)`；`prob` 只有 status=="exact" 时非 None，
        `upper_bound` 永远是个数（prob 存在时就等于 prob）。
    """
    prob = 1.0
    pos = 0
    for step in steps:
        rest = action[pos:]
        if not rest:
            break

        cands = [(tok, p) for tok, p in step["top"].items() if tok and rest.startswith(tok)]
        if not cands:
            return None, prob * min(step["top"].values()), "topk_upper"

        # 取最长匹配（同一步里 'MV' 和 'MV_' 可能都在 top-k）。
        tok, p = max(cands, key=lambda kv: (len(kv[0]), kv[1]))
        prob *= p
        pos += len(tok)
        if pos == len(action):
            return prob, prob, "exact"

        if tok != step["token"]:  # 岔开实际路径，后续条件分布未知
            return None, prob, "branch_upper"

    return None, prob, "len_upper"


def analyze_choice(choice: dict, label: str, pred_token: str, token_split: dict[str, list[str]] | None = None) -> dict:
    """把一条回复的 logprobs 变成 JSONL 一行的分析字段.

    Args:
        choice: `/v1/chat/completions` 的 `choices[0]`（必须带 logprobs）。
        label: 真值 action。
        pred_token: 从回复文本里解析出来的预测 action。
        token_split: :func:`print_action_token_split` 的结果。给了就用它取每个 action 的**首 token**
            来分组（`first_token_probs`）；不给就退化成「在 top-k 里找最长前缀」。
    """
    steps = parse_logprob_steps(choice)
    if not steps:
        return {}

    first = steps[0]
    top_sorted = sorted(first["top"].items(), key=lambda kv: -kv[1])
    top1 = top_sorted[0] if top_sorted else ("", 0.0)
    top2 = top_sorted[1] if len(top_sorted) > 1 else ("", 0.0)
    entropy = -sum(p * math.log(p) for _, p in top_sorted if p > 0.0)

    probs: dict[str, float | None] = {}
    bounds: dict[str, float] = {}
    status: dict[str, str] = {}
    for action in ACTIONS:
        p, ub, st = action_probability(steps, action)
        probs[action], bounds[action], status[action] = p, ub, st

    # 首 token 层面的四分类（MV / GR / RELEASE / DONE）——它本身就是个有用的「事件 vs 移动」判别信号。
    # 必须用 action 真正的首 token 分组：光看"是不是 action 的前缀"会把 'M'、'MV' 同时算进去，
    # 概率质量重复计数（first_token_legal_mass 会 > 1）。
    if token_split:
        first_tokens = sorted({toks[0] for toks in token_split.values() if toks})
    else:
        first_tokens = sorted(
            {
                max(cands, key=len)
                for action in ACTIONS
                if (cands := [t for t in first["top"] if t and action.startswith(t)])
            }
        )

    first_groups = {tok: first["top"].get(tok, 0.0) for tok in first_tokens}
    ft_mass = sum(first_groups.values())
    ranked = sorted(bounds.items(), key=lambda kv: -kv[1])

    return {
        "pred_prob": probs.get(pred_token),
        "pred_prob_upper_bound": bounds.get(pred_token),
        "pred_first_token_prob": first["prob"],
        "top1_token": top1[0],
        "top1_prob": top1[1],
        "top2_token": top2[0],
        "top2_prob": top2[1],
        "margin": top1[1] - top2[1],
        "entropy": entropy,
        "top20": [[t, p] for t, p in top_sorted],
        "gen_tokens": [[s["token"], s["prob"]] for s in steps],
        "action_probs": probs,
        "action_probs_upper_bound": bounds,
        "action_prob_status": status,
        "action_top1": ranked[0][0],
        "action_top1_prob": ranked[0][1],
        "action_top2": ranked[1][0],
        "action_top2_prob": ranked[1][1],
        "action_margin": ranked[0][1] - ranked[1][1],
        "label_prob": probs.get(label),
        "label_prob_upper_bound": bounds.get(label),
        "label_prob_exact": status.get(label) == "exact",
        "first_token_probs": first_groups,
        "first_token_probs_norm": {k: (v / ft_mass if ft_mass else 0.0) for k, v in first_groups.items()},
        "first_token_legal_mass": ft_mass,
        "legal_mass": sum(bounds.values()),
    }


def summarize(rows: list[dict], meta: dict) -> dict:
    """整体 / per-label 统计."""

    def _mean(values: list) -> float | None:
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else None

    per_label: dict[str, dict] = {}
    confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        label = row["label"]
        bucket = per_label.setdefault(label, {"n": 0, "correct": 0, "_pred": [], "_label": [], "_ent": []})
        bucket["n"] += 1
        bucket["correct"] += int(row["correct"])
        bucket["_pred"].append(row.get("pred_prob_upper_bound"))
        bucket["_label"].append(row.get("label_prob_upper_bound"))
        bucket["_ent"].append(row.get("entropy"))
        confusion.setdefault(label, {})
        confusion[label][row["pred_token"]] = confusion[label].get(row["pred_token"], 0) + 1

    for bucket in per_label.values():
        bucket["accuracy"] = bucket["correct"] / bucket["n"]
        bucket["mean_pred_prob"] = _mean(bucket.pop("_pred"))
        bucket["mean_label_prob"] = _mean(bucket.pop("_label"))
        bucket["mean_entropy"] = _mean(bucket.pop("_ent"))

    n = len(rows)
    correct = sum(int(r["correct"]) for r in rows)
    return {
        "meta": meta,
        "overall": {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else None,
            "mean_pred_prob": _mean([r.get("pred_prob_upper_bound") for r in rows]),
            "mean_label_prob": _mean([r.get("label_prob_upper_bound") for r in rows]),
            "mean_entropy": _mean([r.get("entropy") for r in rows]),
            "mean_margin": _mean([r.get("margin") for r in rows]),
            "mean_action_margin": _mean([r.get("action_margin") for r in rows]),
            "mean_legal_mass": _mean([r.get("legal_mass") for r in rows]),
            "mean_first_token_legal_mass": _mean([r.get("first_token_legal_mass") for r in rows]),
        },
        "per_label": dict(sorted(per_label.items())),
        "confusion": dict(sorted(confusion.items())),
    }


# ── preflight 帮手（给薄入口用）─────────────────────────────────────────────
def probe_render(client: MvTokenClient, text: str = "PROBE") -> list[str]:
    """拿一条纯文本探服务端渲染出的 token 串（含 client 自己会发的 system turn）."""
    return client.tokenize_messages(client.build_messages(text))


def print_action_token_split(client: MvTokenClient) -> dict[str, list[str]]:
    """打印 9 个 action 在该家族 tokenizer 下的切分（logprobs 前缀累乘的依据）."""
    split = {a: client.tokenize_text(a) for a in ACTIONS}
    width = max(len(a) for a in ACTIONS)
    print(f"Tokens  : action 切分（{client.spec.name} @ {client.api_url}, model={client.model}）")
    for action, toks in split.items():
        print(f"          {action:<{width}}  n={len(toks)}  {toks}")

    firsts = sorted({toks[0] for toks in split.values() if toks})
    print(f"          首 token 只能区分 {len(firsts)} 类 {firsts} -> action 概率必须逐 token 前缀累乘")
    return split


# ── 评测循环 ────────────────────────────────────────────────────────────────
def _media_of(sample: dict) -> tuple[list[str], list[str]]:
    """返回 `(image_paths, video_frames)`，两者恰有一个非空."""
    if sample.get("videos"):
        return [], list(sample["videos"][0])  # 嵌套：每个 <video> 一个帧列表

    return list(sample.get("images") or []), []


def _user_text(sample: dict, no_stage: bool) -> str:
    instruction = sample["instruction"]
    sample_input = sample.get("input") or ""
    # 训练侧 alpaca converter 是 "\n".join([instruction, input])，此处必须同样用单个 "\n"。
    text = f"{instruction}\n{sample_input}" if sample_input else instruction
    return strip_stage(text) if no_stage else text


def _out_paths(spec: FamilySpec, model: str, evalset: str, explicit: str | None) -> tuple[Path, Path]:
    if explicit:
        jsonl = Path(explicit)
    else:
        tag = Path(evalset).parent.name + "_" + Path(evalset).stem
        safe_model = model.strip("/").replace("/", "_")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        jsonl = LF_ROOT / "results" / "logprobs" / f"{spec.name}__{safe_model}__{tag}__{stamp}.jsonl"

    jsonl.parent.mkdir(parents=True, exist_ok=True)
    return jsonl, jsonl.with_name(jsonl.stem + "_summary.json")


def _fmt(value: float | None, width: int, digits: int = 4) -> str:
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def run_eval(client: MvTokenClient, args: argparse.Namespace) -> None:
    """批量评测（布局按样本的 images / videos 字段自动选）."""
    random.seed(args.seed)
    with open(args.evalset) as f:
        all_samples = json.load(f)

    picked = random.sample(range(len(all_samples)), min(args.n_samples, len(all_samples)))
    is_video = bool(all_samples[0].get("videos"))
    layout = "video slot (<video>)" if is_video else "image layout (<image><image>)"
    max_tokens = 128 if args.raw else args.max_tokens

    print(f"Server  : {client.api_url}  model={client.model}")
    print(f"Dataset : {args.evalset}  ({len(picked)}/{len(all_samples)} samples, seed={args.seed})")
    print(f"Layout  : {layout}" + (f"   video_fps={client.video_fps}" if is_video else ""))
    if args.no_stage:
        print("Ablation: --no-stage 已开启（'Stage:' 行已删除）")

    split: dict[str, list[str]] = {}
    jsonl_path = summary_path = None
    if args.logprobs:
        split = print_action_token_split(client)
        jsonl_path, summary_path = _out_paths(client.spec, client.model, args.evalset, args.logprobs_out)
        print(f"Logprobs: {jsonl_path}")

    head = "P(pred)   P(label)      H  " if args.logprobs else ""
    print("─" * 104)
    print(f"{'#':>3}  {'Label':>10}  {'Pred':>10}  {'ok':>3}  {head}" + ("Full response" if args.raw else "Context"))
    print("─" * 104)

    rows: list[dict] = []
    for ord_idx, sample_idx in enumerate(picked):
        sample = all_samples[sample_idx]
        image_paths, video_frames = _media_of(sample)
        messages = client.build_messages(_user_text(sample, args.no_stage), image_paths, video_frames)
        choice = client.complete(messages, max_tokens=max_tokens, logprobs=args.logprobs)

        pred_text = (choice["message"]["content"] or "").strip()
        label = sample["output"]
        pred_token = next((w for w in pred_text.split() if w in ACTION_SET), pred_text[:20])
        row = {
            "sample_idx": sample_idx,
            "sample_ord": ord_idx,
            "label": label,
            "pred_token": pred_token,
            "pred_text": pred_text,
            "correct": pred_token == label,
            "finish_reason": choice.get("finish_reason"),
        }
        if args.logprobs:
            row.update(analyze_choice(choice, label, pred_token, split))

        rows.append(row)

        extra = ""
        if args.logprobs:
            extra = (
                f"{_fmt(row.get('pred_prob_upper_bound'), 7)}  "
                f"{_fmt(row.get('label_prob_upper_bound'), 8)}  {_fmt(row.get('entropy'), 5, 3)}  "
            )

        tail = repr(pred_text) if args.raw else (sample.get("input") or "").replace("\n", " | ")[:30]
        mark = "✓" if row["correct"] else "✗"
        print(f"{ord_idx + 1:>3}  {label:>10}  {pred_token:>10}  {mark:>3}  {extra}{tail}")

    meta = {
        "family": client.spec.name,
        "api_url": client.api_url,
        "model": client.model,
        "evalset": str(args.evalset),
        "n_samples": len(rows),
        "n_total": len(all_samples),
        "seed": args.seed,
        "layout": "video" if is_video else "image",
        "video_fps": client.video_fps if is_video else None,
        "max_tokens": max_tokens,
        "no_stage": args.no_stage,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action_token_split": split,
    }
    summary = summarize(rows, meta)

    print("─" * 104)
    print("Per-label:")
    for label, bucket in summary["per_label"].items():
        print(
            f"  {label:<10}  {bucket['correct']:>2}/{bucket['n']:<2} = {bucket['accuracy'] * 100:5.1f}%"
            f"   meanP(pred)={_fmt(bucket['mean_pred_prob'], 6)}  meanP(label)={_fmt(bucket['mean_label_prob'], 6)}"
        )

    overall = summary["overall"]
    print("─" * 104)
    print(f"Overall : {overall['correct']}/{overall['n']} = {overall['accuracy'] * 100:.1f}%")
    if args.logprobs:
        print(
            f"          meanP(pred)={_fmt(overall['mean_pred_prob'], 6)}"
            f"  meanH={_fmt(overall['mean_entropy'], 5, 3)}"
            f"  mean legal_mass={_fmt(overall['mean_legal_mass'], 6)}"
            f"  mean first-token legal mass={_fmt(overall['mean_first_token_legal_mass'], 6)}"
        )
        with open(jsonl_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"          logprobs -> {jsonl_path}")
        print(f"          summary  -> {summary_path}")

    # InternVL 的 untie-lm_head 坑是静默的：模型永远吐不出 <|im_end|> -> finish_reason 恒为 length。
    n_len = sum(1 for r in rows if r.get("finish_reason") == "length")
    if rows and n_len == len(rows):
        print(
            f"[warn] {n_len}/{len(rows)} 条 finish_reason=length —— 模型停不下来。"
            f"若是 InternVL，检查 server 有没有带 --hf-overrides 强制 untie lm_head。"
        )


def run_single(client: MvTokenClient, args: argparse.Namespace) -> None:
    """单条推理."""
    print(f"Server  : {client.api_url}  model={client.model}")
    print(f"Prompt  : {args.prompt}")
    if args.videos:
        print(f"Video   : {args.videos}  (fps={client.video_fps})")

    if args.images:
        print(f"Images  : {args.images}")

    print("─" * 72)
    messages = client.build_messages(args.prompt, args.images, args.videos)
    choice = client.complete(messages, max_tokens=args.max_tokens, temperature=args.temperature)
    print(choice["message"]["content"])


# ── CLI ─────────────────────────────────────────────────────────────────────
def build_parser(spec: FamilySpec) -> argparse.ArgumentParser:
    """三家族共用的 CLI（家族差异只体现在默认值上）.

    连接参数（--api-url / --model / …）在子命令前后都能写：子命令那份用 SUPPRESS 做默认值，
    没显式给就不会覆盖顶层解析出来的值。
    """
    # 子命令共享的连接参数；SUPPRESS = 没写就不进 namespace，于是顶层的默认值/显式值保留。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--api-url", default=argparse.SUPPRESS, help="server 地址（默认 $API_URL）")
    common.add_argument("--model", default=argparse.SUPPRESS, help="model 字段 / LoRA 名（默认 $MODEL_NAME）")
    common.add_argument("--video-fps", type=float, default=argparse.SUPPRESS, help="mp4 帧率（默认 $VIDEO_FPS）")
    common.add_argument("--no-preflight", action="store_true", default=argparse.SUPPRESS, help="跳过训推一致性自检")

    parser = argparse.ArgumentParser(description=f"MVTOKEN 推理 / 评测客户端（{spec.name}）")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", spec.default_api_url))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", spec.default_model))
    parser.add_argument("--video-fps", type=float, default=float(os.environ.get("VIDEO_FPS", spec.video_fps)))
    parser.add_argument("--no-preflight", action="store_true", help="跳过训推一致性自检（不建议）")
    sub = parser.add_subparsers(dest="mode")

    sp = sub.add_parser("single", parents=[common], help="单条推理（默认模式，可省略 'single'）")
    sp.add_argument("prompt")
    sp.add_argument("--image", action="append", dest="images", default=[], help="图片路径（可重复）")
    sp.add_argument("--video", action="append", dest="videos", default=[], help="视频帧路径（可重复，编成一个 mp4）")
    sp.add_argument("--max-tokens", type=int, default=512)
    sp.add_argument("--temperature", type=float, default=0.0)

    ep = sub.add_parser("eval", parents=[common], help="批量评测（布局按样本的 images / videos 字段自动选）")
    ep.add_argument("-e", "--evalset", required=True, metavar="PATH", help="测试集 JSON 路径")
    ep.add_argument("-n", "--n-samples", type=int, default=10)
    ep.add_argument("--seed", type=int, default=42)
    ep.add_argument("--raw", action="store_true", help="显示完整原始回复（max_tokens=128）")
    ep.add_argument("--no-stage", action="store_true", help="消融：删掉 prompt 里的 'Stage:' 行")
    ep.add_argument("--max-tokens", type=int, default=8, help="生成上限；gemma4 的 MV_FWD 要 4 个 token")
    ep.add_argument("--logprobs", action="store_true", help="逐 token 取 top-20 logprobs，写 JSONL + summary")
    ep.add_argument("--logprobs-out", default=None, metavar="PATH", help="JSONL 输出路径（默认 results/logprobs/…）")

    sub.add_parser("tokens", parents=[common], help="只打印 9 个 action 在该家族 tokenizer 下的切分")
    return parser


def main(spec: FamilySpec) -> None:
    """薄入口调这个：`main(SPEC)`."""
    parser = build_parser(spec)
    argv = sys.argv[1:]
    # 兼容旧用法：第一个位置参数不是子命令时当作 single 的 prompt。
    if argv and not argv[0].startswith("-") and argv[0] not in ("single", "eval", "tokens"):
        argv = ["single"] + argv

    args = parser.parse_args(argv)
    if args.mode is None:
        parser.print_help()
        sys.exit(0)

    client = MvTokenClient(spec, args.api_url, args.model, args.video_fps)
    for note in spec.notes:
        print(f"[{spec.name}] {note}")

    if not args.no_preflight and spec.preflight is not None:
        spec.preflight(client)

    if args.mode == "single":
        run_single(client, args)
    elif args.mode == "tokens":
        print_action_token_split(client)
    else:
        run_eval(client, args)
