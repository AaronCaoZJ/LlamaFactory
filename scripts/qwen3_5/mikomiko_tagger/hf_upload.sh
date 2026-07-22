# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"
source ${LF_VENV}/bin/activate
cd ${LF_ROOT}

UP=scripts/qwen3_5/mikomiko_tagger/hf_upload_mikomiko.py

# 两个模型是两个 repo:tag 是已发布的那个,desc 单独一个,别混着传进同一个 repo。
REPO_TAG=aaroncaozj/Mikomiko_pornpic_tagger
REPO_DESC=aaroncaozj/Mikomiko_grok_desc

TAG_CKPT_ROOT=${LF_ROOT}/saves/qwen3.5-2b/mikomiko/full_v0
DESC_CKPT_ROOT=${LF_ROOT}/saves/qwen3.5-9b/mikomiko/grok_desc_v0

# --mode 决定传什么,区别在带不带 optimizer 状态:
#   full     整个目录(含 global_step*/、scheduler、每 rank RNG)-> 换台机器能接着训
#   weights  只留权重 + tokenizer/processor/config -> 只能推理,体积约 1/3
#   lora     adapter 白名单
# 加 --dry-run 就只列清单不上传;过滤规则与真上传同一套,看到什么就会传什么。
# 认证:HF_TOKEN 环境变量,或本机 huggingface-cli login 过的缓存 token。
#
# --uploader 决定怎么传,默认 auto(超过 5 GB 自动走 large):
#   large    断点续传。进度记在 <ckpt 的父目录>/.cache/huggingface/,断了重跑**同一条命令**
#            就接着传;分多次 commit,传完一批提交一批。full 模式必须用这个。
#   folder   整个目录一次 commit,全部传完才提交 —— 中途断掉已传的字节全部作废,从 0 重来。
# 续传粒度是文件级:传完的文件不再重传,但单个文件传一半断了仍要整个重来。网络不稳就把
# --num-workers 调小(默认 CPU 核数一半),并发越低,中断时丢掉的半传文件越少。


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
# desc 9B checkpoint-13963 -> REPO_DESC 的 checkpoint-13963/ 子目录。
# mode=full:带 DeepSpeed optimizer 状态,目的是换机器接着训。只想拿去推理就改成 weights,
# 省掉 optimizer 那部分体积。
# ========================================
EOF
python $UP --mode full \
    --src "$DESC_CKPT_ROOT"/checkpoint-13963 \
    --repo "$REPO_DESC" \
    --path-in-repo checkpoint-13963

# python $UP --mode weight \
#     --src "$DESC_CKPT_ROOT"/checkpoint-13963 \
#     --repo "$REPO_DESC" \
#     --path-in-repo checkpoint-13963


: <<'EOF'
# ========================================
# tag 2B 发布版 checkpoint-17296。
# full = 可续训的完整备份(~29 GB);weights = 只够推理的那份(~1/3 体积)。
# ========================================
EOF
# python $UP --mode full \
#     --ckpt-root "$TAG_CKPT_ROOT" --step 17296 \
#     --repo "$REPO_TAG"

# python $UP --mode weights \
#     --ckpt-root "$TAG_CKPT_ROOT" --step 17296 \
#     --repo "$REPO_TAG"


: <<'EOF'
# ========================================
# 自定义传哪些文件:--include 覆盖 mode 的白名单,--exclude 在 mode 黑名单上追加。
# 两者都是相对源目录的 glob,可以重复给。先带 --dry-run 确认过滤对了再去掉。
# ========================================
EOF
# python $UP --dry-run \
#     --src "$TAG_CKPT_ROOT"/checkpoint-17296 \
#     --repo "$REPO_TAG" \
#     --include '*.safetensors' \
#     --include '*.json'

# python $UP --dry-run --mode full \
#     --src "$DESC_CKPT_ROOT"/checkpoint-13963 \
#     --repo "$REPO_DESC" \
#     --exclude 'rng_state*'
