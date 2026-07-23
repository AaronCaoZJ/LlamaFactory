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

"""MVTOKEN 推理 / 评测入口 —— Gemma-4（E4B / 12B，LF `template: gemma4n`）.

通用逻辑全在 `scripts/eval_common/mvtoken_client.py`，这里只声明 gemma4 的**硬约束**（详见
`scripts/gemma4/GEMMA4_DEBUG.md` §4）。

用法（先起 server：`bash scripts/gemma4/eval/start_vllm_server.sh`）：

```bash
cd /workspace1/zhijun/LlamaFactory && source .venv-gemma4/bin/activate
export DISABLE_VERSION_CHECK=1   # gemma4 需 transformers>=5.10，绕过 LF 硬编码上限

python scripts/gemma4/eval/infer.py eval \
  --api-url http://localhost:8104 --model gemma4_e4b_mix_22_27_v3 \
  -e data/agentrobot/ood_sample/v3/rollout_lite.json -n 10 --logprobs

python scripts/gemma4/eval/infer.py single "Describe the image in detail" --image a.png
python scripts/gemma4/eval/infer.py tokens --model gemma4_e4b_mix_22_27_v3
```

注意 gemma4 的 action 切分比别家碎（`MV_FWD` = MV/_/FW/D，4 个 token），所以 `--max-tokens`
默认 8 不能再往下调。
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> <repo>/scripts

from eval_common.mvtoken_client import FamilySpec, MvTokenClient, fatal, main, probe_render  # noqa: E402


def preflight(client: MvTokenClient) -> None:
    r"""确认 server 挂的是 LF 对齐的模板，而不是 gemma4 出厂 jinja.

    官方 jinja 与训练渲染的差距是**结构性的**（GEMMA4_DEBUG.md §4）：
      1. 它只在 enable_thinking=true / 有 tools / 数据显式带 system 时才发 system turn，
         而 LF 无条件注入 default_system（还带 `<|think|>` 标记）；
      2. add_generation_prompt 只给 `<|turn>model\\n`，缺训练时结尾那段**空 thought**
         `<|channel>thought\\n<channel|>`（它是「关闭思考」的写法，不是在思考）。
    实测 E4B 单轮：训练 26 tok vs 官方 10 tok，公共前缀只有 2 tok。
    失配后果是复读 / 吞 `MV_` 前缀 / 乱答，不是掉几个点。
    """
    tokens = probe_render(client)
    if not tokens:
        print("[warn] /tokenize 无返回，prompt-parity 检查跳过", file=sys.stderr)
        return

    if "<|think|>" not in tokens:
        fatal(
            f"server ({client.api_url}) 渲染出的 prompt 没有 <|think|> system turn —— 用的是官方模板。",
            "重启并挂上：--chat-template scripts/gemma4/eval/chat_template_gemma4n_lf.jinja",
            "见 bash scripts/gemma4/eval/start_vllm_server.sh / scripts/gemma4/GEMMA4_DEBUG.md §4",
        )

    if tokens[-1] != "<channel|>":
        fatal(
            f"server ({client.api_url}) 的 generation prompt 结尾是 {tokens[-3:]}，缺训练时的空 thought 段。",
            "同上：必须挂 scripts/gemma4/eval/chat_template_gemma4n_lf.jinja。",
        )

    print(f"[gemma4] prompt parity OK（{client.model} @ {client.api_url}，{len(tokens)} tok，system+空 thought 齐全）")


SPEC = FamilySpec(
    name="gemma4",
    default_api_url="http://localhost:8104",
    default_model="gemma4_e4b_mix_22_27_v3",
    # ⚠️ 必须有 system。训练侧 LF 的 gemma4n 模板对每条样本都注入 default_system（数据集没有
    #    system 列），HF backend 会自动补，**vLLM 不会** —— 所以客户端显式发一份。
    #    挂了 chat_template_gemma4n_lf.jinja 时它与模板内置的默认值一字不差，是幂等的；
    #    想改这句只能重训（GEMMA4_DEBUG.md §4.2，`default_system: ""` 会把 <|think|> 一起丢掉）。
    system_prompt="You are a helpful assistant.",
    # gemma4 没有训过 <video> 槽位的 LoRA，遇到 videos 样本直接报错而不是悄悄退化。
    video_layout=None,
    server_hint="bash scripts/gemma4/eval/start_vllm_server.sh",
    preflight=preflight,
    notes=(
        "server 必须挂 --chat-template scripts/gemma4/eval/chat_template_gemma4n_lf.jinja"
        "（官方模板缺 system turn + 空 thought 段，会复读 / 吞 MV_ 前缀）。",
        "action 切分较碎（MV_FWD = 4 token），--max-tokens 不要低于 8。",
    ),
)


if __name__ == "__main__":
    main(SPEC)
