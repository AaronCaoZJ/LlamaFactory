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

r"""MVTOKEN 推理 / 评测入口 —— InternVL3.5-HF（1B / 2B / 4B / 8B，LF `template: intern_vl`）.

通用逻辑全在 `scripts/eval_common/mvtoken_client.py`，这里只声明 InternVL 的**硬约束**。

⚠️ **服务端**的三条铁律都写在 `scripts/internvl/eval/start_vllm_server.sh` 的注释里（别改那个文件），
客户端这边只能自检其中能从 /tokenize 看出来的两条：

  1. `--chat-template scripts/internvl/eval/chat_template_internvl_lf.jinja`
     + `--chat-template-content-format openai`
     官方 jinja 与 LF 训练分布差 35 token：不注入 default_system（书生·万象那句 ~31 token）；
     每个图像占位符后多一个 `\\n`。content-format 判成 string 时 vLLM 会自己用 `\\n` 拼占位符，
     绕过模板把第二条又加回来。   -> preflight 的 (a)(b) 两步能查出来。
  2. `--hf-overrides '{"tie_word_embeddings": false, "text_config": {...}}'`
     config.json 里没有该字段，vLLM 默认 True -> 丢弃 lm_head 拿 embed_tokens 当输出层，
     输出语义崩坏且**永远吐不出 `<|im_end|>`**。 -> HTTP 查不到，只能靠 finish_reason 恒为
     length 的行为特征；评测循环结束时会告警。
  3. `--mm-processor-kwargs '{"image_mean": [...ImageNet...], "image_std": [...]}'`
     LF 训 `<video>` 时是拿 image_processor 处理每一帧的（ImageNet 归一化），vLLM 的 video
     分支默认 CLIP 归一化。**HTTP 完全查不到**，只能靠「server 是 start_vllm_server.sh 起的」保证。

用法（先起 server：`bash scripts/internvl/eval/start_vllm_server.sh`）：

```bash
cd /workspace1/zhijun/LlamaFactory && source .venv/bin/activate

python scripts/internvl/eval/infer.py eval \
  --api-url http://localhost:8202 --model internvl3.5-2b \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 10 --logprobs

python scripts/internvl/eval/infer.py tokens --model internvl3.5-2b
```

**只有 2 图的 `internvl3.5-2b` / `internvl3.5-1b` 能直接跑 ood_sample。**
`-History2*` 系列训的是 6 张图（history2 = 3 个时刻 × agentview/wrist）且各有各的
prompt / 图序契约（见 AgentRobot/vlm/mvtoken_roles.py 与 run_real_mvtoken.sh 的注释），
`-ms0717_*` 是 ManiSkill 仿真数据 —— 拿 ood_sample 评它们没有意义。
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> <repo>/scripts

from eval_common.mvtoken_client import (  # noqa: E402
    FamilySpec,
    MvTokenClient,
    encode_image,
    fatal,
    main,
    probe_render,
)


_PROBE_IMAGE = Path(__file__).resolve().parents[3] / "data" / "agentrobot" / "ood_sample" / "agentview" / "0000.png"


def preflight(client: MvTokenClient) -> None:
    """两级自检：(a) system turn 在不在；(b) 图像占位符后有没有多余的换行."""
    tokens = probe_render(client)
    if not tokens:
        print("[warn] /tokenize 无返回，prompt-parity 检查跳过", file=sys.stderr)
        return

    # (a) LF 在数据集没有 system 列时总会补 default_system（书生·万象那句）；官方 jinja 一句不发。
    if tokens[:2] != ["<|im_start|>", "system"]:
        fatal(
            f"server ({client.api_url}) 渲染出的 prompt 没有 system turn（开头是 {tokens[:3]}）。",
            "官方 chat_template 不注入 default_system，与 LF intern_vl 训练分布差 ~31 token。",
            "重启并挂上：--chat-template scripts/internvl/eval/chat_template_internvl_lf.jinja",
            "见 bash scripts/internvl/eval/start_vllm_server.sh",
        )

    # (b) LF 是就地替换 <image>，'<image><image>You are...' 紧贴无换行。官方模板 / content-format
    #     判成 string 都会在每个图像块后插一个 '\n'。这一步同时把两种失配都盖住了。
    if not _PROBE_IMAGE.exists():
        print(f"[warn] 探针图不存在（{_PROBE_IMAGE}），跳过图像占位符检查", file=sys.stderr)
    else:
        url = encode_image(str(_PROBE_IMAGE))
        content = [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "You are controlling"},
        ]
        img_tokens = client.tokenize_messages([{"role": "user", "content": content}])
        idx = max((i for i, t in enumerate(img_tokens) if t == "</img>"), default=-1)
        if idx < 0:
            fatal(f"server ({client.api_url}) 的图像占位符没有展开成 <img>...</img>，检查 content-format。")

        if img_tokens[idx + 1] in ("Ċ", "\n", "ĊĊ"):
            fatal(
                f"server ({client.api_url}) 在图像块之后多插了一个换行（</img> 后是 {img_tokens[idx + 1]!r}）。",
                "官方 jinja 会在每个图像占位符后加 '\\n'；content-format 判成 string 时 vLLM 也会自己拼。",
                "重启时同时带上 --chat-template …/chat_template_internvl_lf.jinja"
                " 和 --chat-template-content-format openai。",
            )

    print(f"[internvl] prompt parity OK（{client.model} @ {client.api_url}，system turn 在、图像块后无多余换行）")


SPEC = FamilySpec(
    name="internvl",
    default_api_url="http://localhost:8202",
    default_model="internvl3.5-2b",
    # LF 对齐的 jinja 自己会注入 default_system（书生·万象那句），客户端**不要**再发一份。
    system_prompt=None,
    # VideoSlot LoRA 的契约是「两个 <video> part，各 3 帧、data:video/jpeg」，与本仓库
    # ood_sample 的单 <video> 双帧布局不是一回事，所以这里不提供 video 编码，遇到就报错。
    video_layout=None,
    server_hint="bash scripts/internvl/eval/start_vllm_server.sh",
    preflight=preflight,
    notes=(
        "server 必须同时带 --chat-template …/chat_template_internvl_lf.jinja、"
        "--chat-template-content-format openai、--hf-overrides 强制 untie lm_head、"
        "--mm-processor-kwargs 把归一化拉回 ImageNet（后两条 HTTP 查不到，靠 start_vllm_server.sh 保证）。",
        "ood_sample 只适用于 2 图的 internvl3.5-1b / internvl3.5-2b；"
        "-History2* 是 6 图契约、-ms0717_* 是 ManiSkill 数据。",
    ),
)


if __name__ == "__main__":
    main(SPEC)
