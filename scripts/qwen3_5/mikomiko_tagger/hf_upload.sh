# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"
source ${LF_VENV}/bin/activate
cd ${LF_ROOT}

UP=scripts/qwen3_5/mikomiko_tagger/hf_upload_mikomiko.py

# 所有模型共用一个 repo,靠 --path-in-repo 的第一级区分版本,repo 内结构:
#   Mikomiko_Pornpics_Annota/            <- 2026-07 由 Mikomiko_pornpic_tagger 改名而来
#     gemini_tagger_v0/checkpoint-11530/   (2B tag, full)
#     gemini_tagger_v1/checkpoint-41766/   (2B tag, full)
#     grok_descriptor_v0/checkpoint-13963/ (9B desc, weights)
# 注意这个 repo 是**公开**的,传什么都是公开可下载。脚本不会改动已存在 repo 的可见性。
REPO=aaroncaozj/Mikomiko_Pornpics_Annota

# repo 内的版本目录名(--path-in-repo 的第一级)。注意与本地 saves 目录名无关,不要混。
DESC_NAME=grok_descriptor_v0

TAG_CKPT_ROOT=${LF_ROOT}/saves/qwen3.5-2b/mikomiko/full_v0
DESC_CKPT_ROOT=${LF_ROOT}/saves/qwen3.5-9b/mikomiko/grok_desc_v0

# --mode 决定传什么,区别在带不带 optimizer 状态:
#   full     整个目录(含 global_step*/、scheduler、每 rank RNG)-> 换台机器能接着训
#   weights  只留权重 + tokenizer/processor/config -> 只能推理,体积约 1/3
#   lora     adapter 白名单
# 加 --dry-run 就只列清单不上传;过滤规则与真上传同一套,看到什么就会传什么。
# dry-run 也会去查目标 repo 存不存在、是公开还是私有,传大东西之前值得看一眼。
# 认证:HF_TOKEN 环境变量,或本机 huggingface-cli login 过的缓存 token。
#
# 上传走 upload_folder:大目录自动分多次 commit,**断了重跑同一条命令就接着传**
# (已 commit 的文件跳过,已上传的数据块服务端去重)。--path-in-repo 支持多级路径。
#
# 目标 repo 必须已存在,否则报错退出;确实要新建才加 --create-repo(默认建私有)。
# 脚本不会改动已存在 repo 的可见性 —— 这个 repo 是公开的,传什么都是公开可下载。


: <<'EOF'
# ========================================
# 传之前先看清单。不上传,只按 mode 过滤后列出文件、体积,并检查续训必需的几样在不在。
# 9B 的 full 是 ~100 GB 级别,先看一眼再决定。
# ========================================
EOF
# python $UP --dry-run --mode full \
#     --src "$DESC_CKPT_ROOT"/checkpoint-13963


: <<'EOF'
# ========================================
# desc 9B checkpoint-13963 -> grok_descriptor_v0/checkpoint-13963/
# mode=weights:只传推理要的权重与分词器(17.5 GB),滤掉 105 GB 的 optimizer 状态。
# 想让它能换机器续训就把 mode 改成 full —— 体积变 122.7 GB。
#
# 【已完成 2026-07-22】8 个文件 17.5 GB 已传上去,远端字节数与本地一致。默认注释掉:
# 重跑不会重复上传(已存在的文件会跳过),但要先在本地重新 hash 18.8 GB,约 14 分钟。
# ========================================
EOF
# python $UP --mode weights \
#     --src "$DESC_CKPT_ROOT"/checkpoint-13963 \
#     --repo "$REPO" \
#     --path-in-repo "$DESC_NAME"/checkpoint-13963

# python $UP --mode full \
#     --src "$DESC_CKPT_ROOT"/checkpoint-13963 \
#     --repo "$REPO" \
#     --path-in-repo "$DESC_NAME"/checkpoint-13963


: <<'EOF'
# ========================================
# tag 2B 发布版 checkpoint-17296。
# full = 可续训的完整备份(~29 GB);weights = 只够推理的那份(~1/3 体积)。
# 注意:repo 里已有 gemini_tagger_v0(checkpoint-11530)和 v1(checkpoint-41766),这个
# 17296 是另一次跑的,传之前先定好版本号,别覆盖已发布的两个。
# ========================================
EOF
# python $UP --mode full \
#     --ckpt-root "$TAG_CKPT_ROOT" --step 17296 \
#     --repo "$REPO" \
#     --path-in-repo gemini_tagger_vN/checkpoint-17296

# python $UP --mode weights \
#     --ckpt-root "$TAG_CKPT_ROOT" --step 17296 \
#     --repo "$REPO" \
#     --path-in-repo gemini_tagger_vN/checkpoint-17296


: <<'EOF'
# ========================================
# 自定义传哪些文件:--include 覆盖 mode 的白名单,--exclude 在 mode 黑名单上追加。
# 两者都是相对源目录的 glob,可以重复给。先带 --dry-run 确认过滤对了再去掉。
# ========================================
EOF
# python $UP --dry-run \
#     --src "$TAG_CKPT_ROOT"/checkpoint-17296 \
#     --repo "$REPO" \
#     --include '*.safetensors' \
#     --include '*.json'

# python $UP --dry-run --mode full \
#     --src "$DESC_CKPT_ROOT"/checkpoint-13963 \
#     --repo "$REPO" \
#     --exclude 'rng_state*'
