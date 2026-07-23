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

"""MVTOKEN 推理 / 评测入口 —— Qwen3.5（0.8B / 2B / 9B / 27B）.

通用逻辑全在 `scripts/eval_common/mvtoken_client.py`，这里只声明 Qwen3.5 的**硬约束**。

用法（先起 server：`bash scripts/qwen3_5/eval/start_vllm_server_9.sh`）：

```bash
cd /workspace1/zhijun/LlamaFactory && source .venv/bin/activate

# 批量评测（v3 image 布局）+ logprobs
python scripts/qwen3_5/eval/infer.py eval \
  --api-url http://localhost:8109 --model mix_22_27_v3_9 \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 10 --logprobs

# video 槽位（只有 mix_22_27_v3_9_video 这个 LoRA 是这么训的）
python scripts/qwen3_5/eval/infer.py eval \
  --api-url http://localhost:8109 --model mix_22_27_v3_9_video \
  -e data/agentrobot/ood_sample/v3/rollout_lite_video.json -n 10 --logprobs

# 单条推理 / 只看 action 切分
python scripts/qwen3_5/eval/infer.py single "描述图中场景" --image a.png --image b.png
python scripts/qwen3_5/eval/infer.py tokens --model mix_22_27_v3_9
```

`--api-url` / `--model` / `--video-fps` 也可以用 API_URL / MODEL_NAME / VIDEO_FPS 环境变量给（兼容旧脚本）。
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> <repo>/scripts

from eval_common.mvtoken_client import FamilySpec, MvTokenClient, fatal, main, probe_render  # noqa: E402


def preflight(client: MvTokenClient) -> None:
    r"""漏挂 --chat-template 时直接中止（训推失配是静默的，跑完才发现就晚了）.

    LF 的 `qwen3_5_nothink` 在 'assistant\\n' 之后什么都不加；Qwen 出厂的 jinja 无论传什么参数都会
    加 think —— `enable_thinking=false` 也只是把「开着的 think」换成「闭合的空 think 块」
    （`<think>\\n\\n</think>\\n\\n`），仍然多 4 个 token。输出看着完全正常，只是悄悄掉点
    （mikomiko tagger 实测 microF1 -1.2pt）。详见 scripts/qwen3_5/QWEN35_DEBUG.md。

    这里拿 /tokenize 探一条纯文本，看渲染结果里有没有混进 think。
    """
    tokens = probe_render(client)
    if not tokens:
        print("[warn] /tokenize 无返回，prompt-parity 检查跳过", file=sys.stderr)
        return

    if any("think" in t for t in tokens):
        fatal(
            f"server ({client.api_url}) 的 chat template 会注入 <think> —— 训推失配，会静默掉点。",
            "重启时挂上 LF 对齐的模板：bash scripts/qwen3_5/eval/start_vllm_server_9.sh",
            "（它会传 --chat-template scripts/qwen3_5/eval/chat_template_qwen3_5_lf.jinja）",
            "背景见 scripts/qwen3_5/QWEN35_DEBUG.md",
        )

    print(f"[qwen3_5] prompt parity OK（{client.model} @ {client.api_url}，{len(tokens)} tok，无 think token）")


SPEC = FamilySpec(
    name="qwen3_5",
    default_api_url="http://localhost:8109",
    default_model="mix_22_27_v3_9",
    # LF 的 qwen3_5_nothink 不注入 system，客户端也不要发。
    system_prompt=None,
    # ⚠️ video 布局：把两张图现场编成一个无损 mp4（方案 B，64 visual tokens）。
    #    mp4 的 fps 必须等于训练 yaml 的 video_fps —— 它决定 prompt 里的 "<0.2 seconds>" 时间戳文本，
    #    LF 用 yaml 的值算、vLLM 从 mp4 元数据反推，不一致 prompt 就对不上。
    #    改了 examples/train_lora/qwen3_5_9b/qwen3_5_9b_mix_22_27_v3_video.yaml 就同步改这里。
    video_layout="mp4",
    video_fps=2.0,
    server_hint="bash scripts/qwen3_5/eval/start_vllm_server_9.sh",
    preflight=preflight,
    notes=(
        "server 必须挂 --chat-template scripts/qwen3_5/eval/chat_template_qwen3_5_lf.jinja"
        "（官方模板即使 enable_thinking=false 也会插空 think 块，差 4 token）。",
    ),
)


if __name__ == "__main__":
    main(SPEC)
