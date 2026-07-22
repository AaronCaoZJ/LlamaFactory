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

上传器有两个,--uploader 默认 auto 按体积选(>5 GB 走 large):

    large    upload_large_folder。**断点续传**:每个文件的进度记在 <父目录>/.cache/huggingface/,
             分多次 commit,传完一批提交一批。中断后重跑同一条命令就接着传。多线程并发,
             每 30 秒打印一次进度。大目录(full 模式)必须用这个。
    folder   upload_folder。整个目录**一次 commit**,所有文件传完才提交 —— 中途断掉这个
             commit 根本不会产生,已传的字节全部作废,重跑从 0 开始。只适合小目录(lora)。
             好处是支持 --path-in-repo 任意命名。

large 的续传粒度是**文件级**:已传完的文件不再重传,但单个文件传到一半中断仍要整个重来
(DeepSpeed 的 global_step*/mp_rank_00_model_states.pt 可能单个就几十 GB,注意这点)。
网络不稳就把 --num-workers 调小,并发越低,中断时丢掉的半传文件越少。

用法(常用组合都在同目录的 hf_upload.sh 里):
    python hf_upload_mikomiko.py --step 17296 --dry-run
    python hf_upload_mikomiko.py --step 17296 --mode weights
    python hf_upload_mikomiko.py --src /path/to/any/dir --repo me/my-repo --mode full
    python hf_upload_mikomiko.py --src DIR --include '*.safetensors' --include '*.json'
    python hf_upload_mikomiko.py --src DIR --uploader large --num-workers 4   # 慢网络

续训提示:DeepSpeed ZeRO-0 把 optimizer 状态放在 global_step*/mp_rank_00_model_states.pt。
在目标机器下载后,把 trainer 指向该 checkpoint 目录即可(resume_from_checkpoint: <dir>)。
先跑一小段确认 optimizer 状态完整、step/lr 接得上,再放开跑。
"""
import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# 这个文件在 <repo>/scripts/qwen3_5/mikomiko_tagger/,parents[3] 就是仓库根,不依赖机器路径
LF_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_REPO = "aaroncaozj/Mikomiko_pornpic_tagger"
DEFAULT_CKPT_ROOT = LF_ROOT / "saves/qwen3.5-2b/mikomiko/full_v0"

# --uploader auto 的分界:超过这个体积就走可断点续传的 upload_large_folder
LARGE_THRESHOLD = 5 * 1024**3

# upload_large_folder 把断点状态写在 <folder_path>/.cache/huggingface/upload/ 下,但那份 metadata
# **不记录 repo_id**,同一个父目录先后传去两个 repo 会互相污染。我们自己在旁边放个标记来拦。
STATE_SUBDIR = os.path.join(".cache", "huggingface")
TARGET_MARKER = "lf_upload_target.json"

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


def check_resume_target(parent, repo):
    """确认 parent 下的断点状态属于 repo;不属于就停下,别把两个 repo 的进度搅在一起。"""
    marker = os.path.join(parent, STATE_SUBDIR, TARGET_MARKER)
    if os.path.exists(marker):
        try:
            prev = json.load(open(marker)).get("repo")
        except (OSError, ValueError):
            prev = None
        if prev and prev != repo:
            sys.exit(
                f"!! {parent} 下已有指向 {prev} 的断点状态,现在却要传 {repo}。\n"
                f"   两者共用同一份 metadata 会串。确认不再续传 {prev} 后,删掉状态再跑:\n"
                f"     rm -rf {os.path.join(parent, STATE_SUBDIR)}"
            )
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        json.dump({"repo": repo}, f)


def upload_resumable(api, repo, src, dst_prefix, allow, ignore, num_workers):
    """走 upload_large_folder:文件级断点续传 + 分批 commit + 多线程并发。

    它没有 path_in_repo 参数,所以改用「父目录当根 + 给每条 pattern 加 dst_prefix 前缀」等价实现:
    repo 内路径就是相对 folder_path 的路径,父目录下别的 checkpoint 被 pattern 挡在外面。
    这要求 dst_prefix 就是源目录名,调用方已经校验过。

    repo 的 private 由调用方的 create_repo 定,这里不再传 —— 少一个老版本没有的参数。
    """
    if not hasattr(api, "upload_large_folder"):
        import huggingface_hub
        sys.exit(f"!! huggingface_hub {huggingface_hub.__version__} 太老,没有 upload_large_folder"
                 f"(0.24.0 才引入)。\n   升级: pip install -U huggingface_hub\n"
                 f"   或退回不可续传的上传: --uploader folder")
    parent = os.path.dirname(os.path.normpath(os.path.abspath(src)))
    check_resume_target(parent, repo)
    # allow 为空表示不限制,但这里必须限制成只传本 checkpoint,否则会把父目录下别的 ckpt 一起传上去
    allow_p = [f"{dst_prefix}/{p}" for p in allow] if allow else [f"{dst_prefix}/*"]
    ignore_p = [f"{dst_prefix}/{p}" for p in ignore] or None
    api.upload_large_folder(
        repo_id=repo,
        folder_path=parent,
        repo_type="model",
        allow_patterns=allow_p,
        ignore_patterns=ignore_p,
        num_workers=num_workers,
        print_report_every=30,
    )


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

    how_grp = ap.add_argument_group("怎么传")
    how_grp.add_argument("--uploader", choices=["auto", "large", "folder"],
                         default=os.environ.get("UPLOADER", "auto"),
                         help=f"auto(默认):>{human(LARGE_THRESHOLD)} 走 large;"
                              "large:可断点续传;folder:单次 commit,中断即全部作废")
    how_grp.add_argument("--num-workers", type=int, default=None, metavar="N",
                         help="large 的并发线程数(默认 CPU 核数一半);网络不稳就调小")

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

    # 选上传器。large 靠「父目录当根」映射 repo 内路径,所以只在 dst_prefix == 源目录名时可用。
    use_large = args.uploader == "large" or (args.uploader == "auto" and total >= LARGE_THRESHOLD)
    if use_large and dst_prefix != os.path.basename(os.path.normpath(src)):
        if args.uploader == "large":
            sys.exit(f"!! --uploader large 要求 --path-in-repo 与源目录同名"
                     f"(现在 {dst_prefix} != {os.path.basename(os.path.normpath(src))})")
        use_large = False
        print("         注意:--path-in-repo 与源目录名不同,只能退回 folder")
    if use_large:
        print("上传器 : large —— 可断点续传,中断后重跑同一条命令接着传")
    else:
        print("上传器 : folder —— 单次 commit,中断则已传字节全部作废"
              + ("(体积不小,建议 --uploader large)" if total >= LARGE_THRESHOLD else ""))

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
    # whoami 是第一次发网络请求。token 无效或出不去网都卡在这,单独报清楚,别让人对着黑屏猜。
    try:
        print("\nHF 账号:", api.whoami().get("name"))
    except Exception as e:
        sys.exit(
            f"!! HF 认证失败: {type(e).__name__}: {e}\n"
            f"   1) token 有没有进到进程里: python -c \"import os;print(os.environ.get('HF_TOKEN','(未设置)')[:6])\"\n"
            f"      没有就检查 .env.paths 里是否写成了 export HF_TOKEN=hf_xxx,以及是否 source 过\n"
            f"   2) token 是否有效、够不够权限:\n"
            f"      curl -sS -m 20 -o /dev/null -w '%{{http_code}}\\n' \\\n"
            f"        -H \"Authorization: Bearer $HF_TOKEN\" https://huggingface.co/api/whoami-v2\n"
            f"      200=正常(还要确认是 write 权限) 401=token 无效 000/超时=网络不通"
        )
    create_repo(args.repo, private=not args.public, repo_type="model", exist_ok=True)
    print(f"上传 {human(total)} -> {dst_prefix}/ ...", flush=True)
    if use_large:
        # 分批 commit,进度记在 <父目录>/.cache/huggingface/,中断后重跑同一条命令接着传
        upload_resumable(api, args.repo, src, dst_prefix, allow, ignore, args.num_workers)
    else:
        # 一次 commit:所有文件传完才提交,中途中断则这个 commit 不会产生,已传字节作废
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
