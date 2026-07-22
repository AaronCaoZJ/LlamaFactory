#!/usr/bin/env bash
# 数据处理 — mikomiko 三批交付,一个入口。三批**互不覆盖**,各写各的 jsonl 目录。
#
#   bash process_data.sh csv       [--limit N]     124w tag(CSV 交付)  -> jsonl/
#   bash process_data.sh tag0716   [--plan|--build] 0716 tag           -> jsonl_0716/
#   bash process_data.sh desc0721  [--plan|--build] 0721 四段描述        -> jsonl_desc_0721/
#
#   bash process_data.sh download  [--dry-run]     取图 -> img_0716/(~250 GB,可续传)
#   bash process_data.sh clean     [--apply]       去掉整条复读的行 -> <dir>/cleaned/
#   bash process_data.sh mix                       124w + 0716 交错 -> mix_train.jsonl
#   bash process_data.sh verify                    0721 构建门禁(失败非零退出)
#
# builder 默认 --plan(空跑,打印丢弃漏斗),要写盘必须显式 --build。**先看漏斗是设计,不是形式。**
#
# ── 三批数据到训练的对应 ───────────────────────────────────────────────────────────────────────
#   csv      -> jsonl/cleaned/train.jsonl      ─┐ mix ─> jsonl_0716/mix/mix_train.jsonl
#   tag0716  -> jsonl_0716/train.jsonl         ─┘        └─> scripts/.../train_tag_2b.sh(2B)
#   desc0721 -> jsonl_desc_0721/train.jsonl    ────────────> scripts/.../train_desc_9b.sh(9B)
#
# 完整构建顺序(0716 与 0721 都要先有图):
#   bash process_data.sh csv                              # 124w,自带下载
#   bash process_data.sh download                         # img_0716/,后两批共用
#   bash process_data.sh tag0716 --build
#   bash process_data.sh clean --apply                     # 可选,去复读行
#   bash process_data.sh mix                               # 拼出 2B 训练用的 mix
#   bash process_data.sh desc0721 --build && bash process_data.sh verify
#
# 数据集注册名见 data/dataset_info.json。
set -euo pipefail

# machine paths: find & source scripts/workspace_dir.sh -> .env.paths (see that file)
source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-}"; shift || true

usage() {
  cat <<'EOF'
usage: bash data/mikomiko_tag/process_data.sh <command> [args]

  csv      [--limit N]         第一批:124w tag(CSV)-> jsonl/,自带下载
  download [--dry-run]         取图 -> img_0716/(~250 GB,后两批共用)
  tag0716  [--plan|--build]    第二批:0716 tag      -> jsonl_0716/
  desc0721 [--plan|--build]    第三批:0721 四段描述  -> jsonl_desc_0721/
  clean    [--apply]           去掉整条复读的行 -> <dir>/cleaned/
  mix                          124w + 0716 交错 -> mikomiko_tag_mix_train
  verify                       0721 构建门禁(失败非零退出)
EOF
  exit 1
}

case "${CMD}" in
  csv|download|tag0716|desc0721|clean|mix|verify) ;;
  *) usage ;;
esac

source "${LF_VENV}/bin/activate"
export DISABLE_VERSION_CHECK=1     # transformers 5.6.1 > LF 硬编码上限 5.6.0;Qwen3.5 需要新版
cd "${HERE}"

case "${CMD}" in

# ═══ 第一批:CSV 124w tag ═══════════════════════════════════════════════════════════════════════
# 把 Gemini 的逐图 tag CSV 与 catalog 的 URL CSV join 起来,切成 train / test_unseen /
# test_stratified,再多线程把图下到 img/。这批自带下载,不用 `download`。
#
# --plan 和 --download **必须一个进程跑完**:--limit 只在 --plan 生效,分开跑 --download 会去读
# 那份没截断的候选文件。
#   bash process_data.sh csv                  完整构建
#   bash process_data.sh csv --limit 20000    2w 试水
# Env: MIKOMIKO_CONCURRENCY (16)
csv)
  exec python3 dataset_builder.py --plan --download "$@"
  ;;

# ═══ 取图(0716 与 0721 共用)═══════════════════════════════════════════════════════════════════
# 0716 交付的图,~250 GB。可续传、原子写(tmp + rename),中断了直接重跑;失败清单落在
# download_0716_failures.tsv。后两批数据都从这个目录读图。
#   bash process_data.sh download --dry-run    只数一下要取多少
#   bash process_data.sh download --verify     复核已经落盘的
# Env: MIKOMIKO_CONCURRENCY (32)
download)
  exec python3 download_0716.py "$@"
  ;;

# ═══ 第二批:0716 tag ══════════════════════════════════════════════════════════════════════════
# 从 parquet 读 category + post_tag,按 tag_vocab.txt 做 Title-Case 归一,整 post 留 10% 做
# holdout。**需要 img_0716/ 已经在**(先跑 download)—— 只有图真在盘上的行才会被写出来。
tag0716)
  exec python3 dataset_builder_0716.py "${@:---plan}"
  ;;

# ═══ 第三批:0721 四段描述 ══════════════════════════════════════════════════════════════════════
# 按语言配 prompt,过 4 段小标题 / 长度 / 重复度三道过滤,整 post 留 2% holdout(不是 10% ——
# 137k 条 holdout 等于白扔 ~200M token),mini eval 按语言分层。图复用 img_0716/。
desc0721)
  exec python3 dataset_builder_desc_0721.py "${@:---plan}"
  ;;

# ═══ 清洗(对已构建的 jsonl 动刀)═══════════════════════════════════════════════════════════════
# 丢掉整条 label 都是复读循环的行,并剔除单个垃圾 tag。**刻意不重新划分** —— 重划会让图片在 DF
# 分档之间移动,已经报出去的每个指标都会失效。
#   bash process_data.sh clean            空跑,打印会丢什么
#   bash process_data.sh clean --apply    写 <dir>/cleaned/
# 注意:输入目录是 clean_truncated_rows.py 里的模块常量 JSONL_DIR,不是命令行参数。
clean)
  exec python3 clean_truncated_rows.py "$@"
  ;;

# ═══ 混合(124w + 0716)════════════════════════════════════════════════════════════════════════
# 轮转交错两份 train,拼成注册为 mikomiko_tag_mix_train 的那一个文件,供 2B tag 训练用。
# 输出路径这里显式传 —— 脚本自己的默认值和 data/dataset_info.json 读的路径对不上。
mix)
  exec python3 mix_train_jsonl.py --output "${HERE}/jsonl_0716/mix/mix_train.jsonl" "$@"
  ;;

# ═══ 门禁(0721)═══════════════════════════════════════════════════════════════════════════════
# 检查 prompt 语种 == 输出语种、开头有且仅有一个 <image>、图片存在且能解码、train/test 无 post
# 泄漏。非零退出,可以直接串在 desc0721 --build 后面。
verify)
  exec python3 verify_desc_0721.py "$@"
  ;;
esac
