#!/usr/bin/env python3
"""Back up the final mix_22_27_v3 LoRA adapters (9B + 27B) to a PRIVATE HF repo.

Uploads only the files needed to load/use the LoRA (adapter + tokenizer/template/processor),
not the intermediate checkpoints or optimizer states. Auth comes from the HF_TOKEN env var
or the cached huggingface token (no token is hardcoded here).
"""
import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "aaroncaozj/SoundsGood-VLM"
PRIVATE = True
ROOT = "/workspace1/zhijun/LlamaFactory/saves"

# (local final-adapter dir, folder inside the repo)
UPLOADS = [
    (f"{ROOT}/qwen3.5-9b/robot/mix_22_27_v3",  "qwen3.5-9b/mix_22_27_v3"),
    (f"{ROOT}/qwen3.5-9b/robot/mix_22_27_04_v3", "qwen3.5-9b/mix_22_27_04_v3"),
    (f"{ROOT}/qwen3.5-9b/robot/piper_0705_v4", "qwen3.5-9b/piper_0705_v4"),
    (f"{ROOT}/qwen3.5-27b/robot/mix_22_27_v3", "qwen3.5-27b/mix_22_27_v3"),
]

# files that make the LoRA self-contained & loadable (skip missing ones silently)
FILES = ["adapter_config.json", "adapter_model.safetensors", "chat_template.jinja",
         "tokenizer.json", "tokenizer_config.json", "processor_config.json",
         "special_tokens_map.json", "vocab.json", "merges.txt", "README.md"]


def main():
    api = HfApi()
    print("HF user:", api.whoami().get("name"))
    create_repo(REPO_ID, private=PRIVATE, repo_type="model", exist_ok=True)
    print(f"repo ready: https://huggingface.co/{REPO_ID}  (private={PRIVATE})")

    for src, dst in UPLOADS:
        if not os.path.isdir(src):
            print(f"!! SKIP {dst}: source dir missing ({src})")
            continue
        for fn in FILES:
            p = os.path.join(src, fn)
            if not os.path.exists(p):
                continue
            mb = os.path.getsize(p) / 1e6
            print(f"  uploading {dst}/{fn}  ({mb:.1f} MB) ...", flush=True)
            api.upload_file(path_or_fileobj=p, path_in_repo=f"{dst}/{fn}",
                            repo_id=REPO_ID, repo_type="model")
    print("DONE ->", f"https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
