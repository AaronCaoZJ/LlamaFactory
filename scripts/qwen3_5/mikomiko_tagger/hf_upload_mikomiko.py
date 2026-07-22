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

上传走 huggingface_hub 的 upload_folder,大目录它自己会分多次 commit(避开服务端单次提交上限),
**中断后重跑同一条命令就接着传**:已经 commit 的文件直接跳过,已上传的数据块服务端去重。
--path-in-repo 支持任意层级,例如 grok_descriptor_v0/checkpoint-13963。

(历史:早期版本用的是 upload_large_folder + 一套「父目录当根 + 硬链接暂存」的绕法,因为那个 API
没有 path_in_repo。hub 1.22 起 upload_large_folder 已标记废弃,upload_folder 同时具备分批 commit、
断点续传和 path_in_repo,绕法不再需要,已删除。)

目标 repo 必须**已经存在**,否则直接报错。要新建得显式给 --create-repo —— 名字打错或以为在网页上
改过名其实没生效时,静默新建一个空 repo 再把几十 GB 传进去是最难发现的错误。
脚本也**不会改动已存在 repo 的可见性**,只会把当前是公开还是私有打出来。

用法(常用组合都在同目录的 hf_upload.sh 里):
    python hf_upload_mikomiko.py --step 17296 --dry-run
    python hf_upload_mikomiko.py --step 17296 --mode weights
    python hf_upload_mikomiko.py --src /path/to/any/dir --repo me/my-repo --mode full
    python hf_upload_mikomiko.py --src DIR --include '*.safetensors' --include '*.json'
    python hf_upload_mikomiko.py --src DIR --repo me/new-repo --create-repo   # 新建私有 repo

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

DEFAULT_REPO = "aaroncaozj/Mikomiko_Pornpics_Annota"   # 原 Mikomiko_pornpic_tagger,2026-07 改名
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


def probe_repo(api, repo):
    """返回 (存在?, 私有?)。查不到就 (None, None) —— dry-run 没网也该能跑,不为这个报错。"""
    try:
        if not api.repo_exists(repo, repo_type="model"):
            return False, None
        return True, api.repo_info(repo, repo_type="model").private
    except Exception:
        return None, None


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
                         help="repo 内路径,可多级(如 grok_descriptor_v0/checkpoint-13963);默认用源目录名")
    dst_grp.add_argument("--create-repo", action="store_true",
                         help="repo 不存在时新建;不给就直接报错(防止名字打错把几十 GB 传到新 repo)")
    dst_grp.add_argument("--public", action="store_true",
                         help="配合 --create-repo:建成公开 repo(默认私有)。对已存在的 repo 无效")

    ap.add_argument("--dry-run", action="store_true", help="只列清单不上传")
    args = ap.parse_args()

    src = args.src or os.path.join(args.ckpt_root, f"checkpoint-{args.step}")
    if not os.path.isdir(src):
        sys.exit(f"!! 源目录不存在: {src}")
    dst_prefix = args.path_in_repo or os.path.basename(os.path.normpath(src))
    dst_prefix = dst_prefix.strip("/")
    if os.path.isabs(dst_prefix) or ".." in dst_prefix.split("/") or not dst_prefix:
        sys.exit(f"!! --path-in-repo 得是 repo 内的相对路径(不能是绝对路径、不能含 ..): {dst_prefix}")

    allow, ignore, mode_desc = MODES[args.mode]
    allow = args.include or allow          # --include 覆盖 mode 白名单
    ignore = ignore + args.exclude         # --exclude 追加到 mode 黑名单

    files, total, skipped = manifest(src, allow, ignore)
    if not files:
        sys.exit(f"!! 过滤后一个文件都不剩(mode={args.mode} include={allow} exclude={ignore})")
    has = {rel for rel, _ in files}

    api = HfApi()
    # 可见性必须查**实际**状态。这里曾经打的是 private={not args.public},即「假如新建会是私有」——
    # 对已存在的 repo 完全是误导:明明是公开 repo,却显示 private=True。
    exists, private = probe_repo(api, args.repo)

    print(f"源目录 : {src}")
    print(f"目标   : https://huggingface.co/{args.repo}/tree/main/{dst_prefix}")
    if exists is None:
        print("         repo 状态未知(查询失败:没网或 token 无效)")
    elif not exists:
        print(f"         repo 尚不存在 —— 真上传需要 --create-repo(默认建成私有)")
    else:
        print(f"         repo 已存在,当前{'私有' if private else '公开'}"
              + ("" if private else "  <- 传上去就是公开可下载的") + ";脚本不改动可见性")
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

    # token 无效或出不去网都卡在 whoami,单独报清楚,别让人对着黑屏猜。
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
    # 目标 repo 必须已经存在。以前这里是 create_repo(exist_ok=True):名字打错、或以为在 HF 上
    # 改过名其实没生效时,它会**静默新建**一个空 repo,把几十 GB 传进一个错地方,而且看不出异常。
    # 现在默认报错,要新建必须显式 --create-repo。上面 probe_repo 查过一次,这里用那个结果。
    if exists is None:                       # 刚才查询失败,但 whoami 过了,再查一次拿准确答案
        exists, private = probe_repo(api, args.repo)
    if exists:
        print(f"目标 repo: 已存在,当前{'私有' if private else '公开'} —— 脚本不改动可见性")
    elif args.create_repo:
        create_repo(args.repo, private=not args.public, repo_type="model")
        print(f"目标 repo: 新建完成(private={not args.public})")
    else:
        sys.exit(f"!! repo 不存在: {args.repo}\n"
                 f"   名字打错、或以为改过名其实没生效时,继续下去会新建一个空 repo 并把\n"
                 f"   {human(total)} 传进去。先确认名字;确实要新建就加 --create-repo。")

    # upload_folder:大目录自动分多次 commit,中断后重跑同一条命令接着传(已 commit 的文件跳过,
    # 已上传的数据块服务端去重)。hub 1.22 起 upload_large_folder 已废弃,这个是官方替代。
    print(f"上传 {human(total)} -> {dst_prefix}/ ...", flush=True)
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
