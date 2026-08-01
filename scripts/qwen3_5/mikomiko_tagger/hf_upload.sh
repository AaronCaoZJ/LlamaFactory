# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"
source ${LF_VENV}/bin/activate
cd ${LF_ROOT}

UP=scripts/qwen3_5/mikomiko_tagger/hf_upload_mikomiko.py

# 所有模型共用一个 repo,靠 --path-in-repo 的第一级区分版本,repo 内结构:
#   Miko_annota_collection/            <- 2026-07 由 Mikomiko_pornpic_tagger 改名而来
#     gemini_tagger_v0/checkpoint-11530/   (2B tag, full)
#     gemini_tagger_v1/checkpoint-41766/   (2B tag, full)
#     grok_descriptor_v0/checkpoint-13963/ (9B desc, weights)
#     grok_descriptor_v1/checkpoint-34502/ (9B desc 20260730, weights)
#     grok_descriptor_v1/{*.png, all_results.json}  (说明材料,放在版本目录这一级)
# 注意这个 repo 是**公开**的,传什么都是公开可下载。脚本不会改动已存在 repo 的可见性。
REPO=aaroncaozj/Miko_annota_collection

# repo 内的版本目录名(--path-in-repo 的第一级),下面再套一级 checkpoint-<步数>,
# 与 grok_descriptor_v0/checkpoint-13963/ 的结构和命名一致(不用 desc 简写)。
# 加载时 subfolder="grok_descriptor_v1/checkpoint-34502"。
DESC_NAME=grok_descriptor_v1

# TAG_CKPT_ROOT=${LF_ROOT}/saves/qwen3.5-2b/mikomiko/full_v0
# 本地 saves 目录名,是训练 yaml 的 output_dir,和上面 repo 里的 DESC_NAME 不是一回事:
# 本地叫 grok_desc_v1,repo 里叫 grok_descriptor_v1。全局改名时容易一起误改,改完跑一次
# dry-run 就能发现(源目录不存在会直接报错)。
DESC_CKPT_ROOT=${LF_ROOT}/saves/qwen3.5-9b/mikomiko/grok_desc_v1

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
# desc 9B 20260730 checkpoint-34502 -> grok_descriptor_v1/checkpoint-34502/
#
# 34502 = 34,502 步 = 完整 1 epoch(3,312,103 行 / eff_batch 96),就是这次训练的最终权重。
# 最终 eval loss:pornpics 0.4157 / of 0.5168 / av 0.6636 / oneione 0.9138,train_loss 0.5253。
#
# --src 必须指到 checkpoint-34502,**不能指本地 grok_desc_v1/ 根目录**:uploader 是 os.walk 递归,
# 而 weights 模式的黑名单(global_step*/*、rng_state* 等)是相对 --src 的 glob,在
# checkpoint-34502/global_step34502/ 这种更深一层匹配不上 —— 指根目录会把 106 GB 的
# optimizer 状态连同另一个 checkpoint-34000 一起卷进去。
#
# 根目录和 checkpoint-34502 里的 model.safetensors 是同一份权重(都是 34502 步),
# 但只有 checkpoint 目录能被 weights 模式干净地过滤,所以传它。
#
# mode=weights:model.safetensors + config/generation_config/processor_config +
# tokenizer(.json/_config) + chat_template.jinja,约 18 GB;滤掉 106 GB optimizer 状态、
# 8 个 rng_state、scheduler、trainer_state、training_args。
# 想让它能换机器续训就把 mode 改成 full —— 体积变 ~123 GB。
# ========================================
EOF
# python $UP --dry-run --mode weights \
#     --src "$DESC_CKPT_ROOT"/checkpoint-34502 \
#     --repo "$REPO" \
#     --path-in-repo "$DESC_NAME"/checkpoint-34502

python $UP --mode weights \
    --src "$DESC_CKPT_ROOT"/checkpoint-34502 \
    --repo "$REPO" \
    --path-in-repo "$DESC_NAME"/checkpoint-34502


: <<'EOF'
# ========================================
# 顺带把训练产出的说明材料放到版本目录这一级(权重传完再跑,~2 MB):
# all_results.json(四路最终 eval loss)、四张 per-dataset eval loss 曲线、training_loss.png、
# README.md(LlamaFactory 自动生成的 model card)。
# 这些在 checkpoint-34502/ 里没有,只在本地 grok_desc_v1/ 根目录,所以 --src 指它 +
# --include 白名单精确点名,不会碰到 checkpoint 子目录。放在版本目录一级(不进 checkpoint),
# 因为它们描述的是整次训练,不是某一步的权重。
# ========================================
EOF
# python $UP --mode weights \
#     --src "$DESC_CKPT_ROOT" \
#     --repo "$REPO" \
#     --path-in-repo "$DESC_NAME" \
#     --include 'all_results.json' \
#     --include 'train_results.json' \
#     --include 'eval_results.json' \
#     --include '*.png' \
#     --include 'README.md'


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
