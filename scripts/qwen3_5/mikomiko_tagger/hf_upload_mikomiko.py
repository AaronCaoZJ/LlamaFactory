#!/usr/bin/env python3
"""把一个 checkpoint 目录传到 HF(默认私有 repo)。传什么由 --mode 决定。

三种 mode,差别在**要不要带 optimizer 状态**,这直接决定体积和用途:

    full        整个目录,含 DeepSpeed optimizer 状态(global_step*/)、scheduler、每 rank 的
                RNG。别的机器上能**接着训**。2B 一个 ckpt ~29 GB,9B ~100 GB。
    weights     只留推理要的:权重 + tokenizer/processor/config。**不能续训**,体积约 1/3。
    lora        LoRA adapter 白名单(adapter_config.json + adapter_model.safetensors + 分词器)。

不确定要传什么就先 --dry-run:它按 mode 过滤后列出每个文件和体积,并检查续训必需的几样在不在。
过滤规则和真正上传时用的是同一套,所以 dry-run 看到什么就会传什么。

认证:HF_TOKEN 环境变量,或本机缓存的 huggingface token(脚本里不写死 token)。

用法(常用组合都在同目录的 hf_upload.sh 里):
    python hf_upload_mikomiko.py --step 17296 --dry-run
    python hf_upload_mikomiko.py --step 17296 --mode weights
    python hf_upload_mikomiko.py --src /path/to/any/dir --repo me/my-repo --mode full
    python hf_upload_mikomiko.py --src DIR --include '*.safetensors' --include '*.json'

续训提示:DeepSpeed ZeRO-0 把 optimizer 状态放在 global_step*/mp_rank_00_model_states.pt。
在目标机器下载后,把 trainer 指向该 checkpoint 目录即可(resume_from_checkpoint: <dir>)。
先跑一小段确认 optimizer 状态完整、step/lr 接得上,再放开跑。
"""
import argparse
import fnmatch
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# 这个文件在 <repo>/scripts/qwen3_5/mikomiko_tagger/,parents[3] 就是仓库根,不依赖机器路径
LF_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_REPO = "aaroncaozj/Mikomiko_pornpic_tagger"
DEFAULT_CKPT_ROOT = LF_ROOT / "saves/qwen3.5-2b/mikomiko/full_v0"

# 续训缺一不可的几样,dry-run 会逐个报在不在
RESUME_CRITICAL = ["model.safetensors", "scheduler.pt", "trainer_state.json",
                   "training_args.bin", "latest"]

# mode -> (allow 白名单, ignore 黑名单, 一句话说明)。两者都是相对 src 的 glob,空表示不限制。
MODES = {
    "full": (
        [], [],
        "整个目录(含 optimizer/scheduler/RNG)—— 可在别的机器上续训",
    ),
    "weights": (
        [],
        ["global_step*/*", "global_step*", "*.pth", "scheduler.pt", "optimizer.pt",
         "training_args.bin", "trainer_state.json", "latest", "rng_state*"],
        "只留推理要的权重与分词器 —— 不能续训,体积约 1/3",
    ),
    "lora": (
        ["adapter_config.json", "adapter_model.safetensors", "chat_template.jinja",
         "tokenizer.json", "tokenizer_config.json", "processor_config.json",
         "preprocessor_config.json", "special_tokens_map.json", "vocab.json",
         "merges.txt", "README.md"],
        [],
        "LoRA adapter 白名单 —— 只传能把 adapter 加载起来的那几个文件",
    ),
}


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def keep(rel, allow, ignore):
    """这个相对路径要不要传。语义与 huggingface_hub 的 allow/ignore_patterns 一致。"""
    if allow and not any(fnmatch.fnmatch(rel, p) for p in allow):
        return False
    if any(fnmatch.fnmatch(rel, p) for p in ignore):
        return False
    return True


def manifest(src, allow, ignore):
    """递归列出 src 下要传的文件;返回 (选中的 [(相对路径, 字节)], 总字节, 被过滤掉的字节)。"""
    picked, total, skipped = [], 0, 0
    for root, _dirs, names in os.walk(src):
        for name in sorted(names):
            p = os.path.join(root, name)
            rel = os.path.relpath(p, src)
            sz = os.path.getsize(p)
            if keep(rel, allow, ignore):
                picked.append((rel, sz)); total += sz
            else:
                skipped += sz
    return sorted(picked), total, skipped


def main():
    ap = argparse.ArgumentParser(
        description="把一个 checkpoint 目录传到 HF。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src_grp = ap.add_argument_group("传哪个目录(二选一)")
    src_grp.add_argument("--src", help="直接给目录路径;给了就忽略 --ckpt-root/--step")
    src_grp.add_argument("--ckpt-root", default=os.environ.get("CKPT_ROOT", str(DEFAULT_CKPT_ROOT)),
                         help=f"checkpoint 上级目录(默认 {DEFAULT_CKPT_ROOT})")
    src_grp.add_argument("--step", default=os.environ.get("STEP", "17296"),
                         help="配合 --ckpt-root 拼出 checkpoint-<STEP>(默认 17296 / $STEP)")

    what_grp = ap.add_argument_group("传什么")
    what_grp.add_argument("--mode", choices=list(MODES), default=os.environ.get("MODE", "full"),
                          help="预设过滤规则(默认 full);各 mode 含义见 --help 顶部说明")
    what_grp.add_argument("--include", action="append", default=[], metavar="GLOB",
                          help="只传匹配的文件,可重复;给了就覆盖 mode 的白名单")
    what_grp.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                          help="排除匹配的文件,可重复;在 mode 的黑名单基础上追加")

    dst_grp = ap.add_argument_group("传到哪")
    dst_grp.add_argument("--repo", default=os.environ.get("REPO", DEFAULT_REPO))
    dst_grp.add_argument("--path-in-repo", default=os.environ.get("PATH_IN_REPO"),
                         help="repo 内子目录(默认用源目录名,例如 checkpoint-17296)")
    dst_grp.add_argument("--public", action="store_true", help="建成公开 repo(默认私有)")

    ap.add_argument("--dry-run", action="store_true", help="只列清单不上传")
    args = ap.parse_args()

    src = args.src or os.path.join(args.ckpt_root, f"checkpoint-{args.step}")
    if not os.path.isdir(src):
        sys.exit(f"!! 源目录不存在: {src}")
    dst_prefix = args.path_in_repo or os.path.basename(os.path.normpath(src))

    allow, ignore, mode_desc = MODES[args.mode]
    allow = args.include or allow          # --include 覆盖 mode 白名单
    ignore = ignore + args.exclude         # --exclude 追加到 mode 黑名单

    files, total, skipped = manifest(src, allow, ignore)
    if not files:
        sys.exit(f"!! 过滤后一个文件都不剩(mode={args.mode} include={allow} exclude={ignore})")
    has = {rel for rel, _ in files}

    print(f"源目录 : {src}")
    print(f"目标   : https://huggingface.co/{args.repo}/tree/main/{dst_prefix}"
          f"  (private={not args.public})")
    print(f"mode   : {args.mode} —— {mode_desc}")
    if args.include:
        print(f"         --include {allow}")
    if args.exclude:
        print(f"         --exclude {args.exclude}")
    print(f"要传   : {len(files)} 个文件,{human(total)}"
          + (f"(已过滤掉 {human(skipped)})" if skipped else ""))

    # 续训必需的几样在不在。weights/lora 本来就不带这些,所以只在 full 下当问题报。
    print("\n  续训状态:")
    opt_files = [(rel, sz) for rel, sz in files if rel.startswith("global_step")]
    for rel in RESUME_CRITICAL:
        sz = next((s for r, s in files if r == rel), 0)
        print(f"    [{'OK ' if rel in has else '-- '}] {rel:<28} {human(sz) if sz else ''}")
    for rel, sz in opt_files:
        print(f"    [OK ] {rel:<28} {human(sz)}   <- DeepSpeed optimizer 状态")
    if not opt_files:
        print(f"    [-- ] global_step*/{'':<11} <- 无 optimizer 状态")

    missing = [f for f in RESUME_CRITICAL if f not in has]
    if args.mode == "full" and (missing or not opt_files):
        print(f"\n  警告:mode=full 但续训状态不全(缺 {missing}"
              f"{' + optimizer' if not opt_files else ''}),传上去也接不回来。仍继续。")
    elif args.mode != "full":
        print(f"\n  (mode={args.mode} 本就不带这些,上面的 -- 是预期的)")

    print(f"\n  前 10 个文件:")
    for rel, sz in files[:10]:
        print(f"    {rel:<52} {human(sz)}")
    if len(files) > 10:
        print(f"    ... 另外 {len(files) - 10} 个")

    if args.dry_run:
        print("\n[dry-run] 没有上传。")
        return

    api = HfApi()
    print("\nHF 账号:", api.whoami().get("name"))
    create_repo(args.repo, private=not args.public, repo_type="model", exist_ok=True)
    print(f"上传 {human(total)} -> {dst_prefix}/ ...", flush=True)
    # upload_folder 会递归、大文件走分片、中断后可重跑续传
    api.upload_folder(
        folder_path=src,
        path_in_repo=dst_prefix,
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=allow or None,
        ignore_patterns=ignore or None,
        commit_message=f"{dst_prefix} ({args.mode}, {len(files)} files, {human(total)})",
    )
    print("DONE ->", f"https://huggingface.co/{args.repo}/tree/main/{dst_prefix}")


if __name__ == "__main__":
    main()
