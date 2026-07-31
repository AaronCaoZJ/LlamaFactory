#!/usr/bin/env bash
# DIAG — 为什么 0730 的吞吐只有 0721 的 1/3?
#
# 背景:0721 实测 170w 行 / 15h ≈ 31.5 samples/s;0730 在 node0 上量到 8.7-9.2 samples/s,
# 每样本慢 3.4 倍,而序列只长了约 1.13 倍(0721 中位 1,258 tok -> 0730 加权中位 1,426)。
# 这个差距对不上,必须查清再开正式训练(100h vs 40h 差 2.5 天机时)。
#
# 已排除:
#   - padding:collator 走 HF DataCollatorForSeq2Seq,按 batch 内最长动态补齐,不是补到 cutoff_len
#   - ZeRO-3 通信:z2 @ bs=6 实测 86,358MiB / 9.4 samp/s,只快 2% 却多吃 18% 显存
#   - 配置漂移:0721/0730 两份 yaml 除数据/cutoff/batch 外逐行相同
#
# 未排除(本脚本要回答的):
#   A. 是不是 node0 那台机器的问题?-> 在 node1 重跑同样的 bs 曲线
#   B. GPU 是不是被 dataloader 饿着?-> 0.2s 粒度采利用率 + dataloader_num_workers 扫描
#   C. 是不是环境变了?-> venv 里 torch/transformers 的安装时间都在 0721 之后
#      (torch 2.8.0+cu129 @ 07-28,transformers 5.6.1 @ 07-29,后者还超过 LF 硬编码上限 5.6.0)
#      本脚本只记录版本事实,不动 venv —— 降级要在人看着的时候做。
#
# 用法:  bash diag_speed_0730.sh            # 全跑,约 25-35 分钟
#        tmux new -d -s diag 'bash diag_speed_0730.sh'
# 结论:  logs/diag_speed_0730/REPORT.md
set -uo pipefail

source "$(d="$(dirname "${BASH_SOURCE[0]}")"; until [ -e "$d/scripts/workspace_dir.sh" ] || [ "$d" = / ]; do d="$(dirname "$d")"; done; echo "$d")/scripts/workspace_dir.sh"
cd "${LF_ROOT}"
source "${LF_VENV}/bin/activate"
export DISABLE_VERSION_CHECK=1 WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"

_SHIM="${LF_ROOT}/.cc-shim"
if [ -x "${_SHIM}/g++" ] && echo 'int main(){return 0;}' | "${_SHIM}/g++" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${_SHIM}:${PATH}" CC="${_SHIM}/gcc" CXX="${_SHIM}/g++" CUDAHOSTCXX="${_SHIM}/g++"
fi

YAML="examples/train_full/qwen3_5_9b_mikomiko_grok_desc_0730.yaml"
SAVE_DIR="${LF_ROOT}/saves/qwen3.5-9b/mikomiko"
OUT="${LF_ROOT}/logs/diag_speed_0730"
REPORT="${OUT}/REPORT.md"
STEPS="${STEPS:-15}"
mkdir -p "${OUT}"

# 每档跑完等显存还给驱动。不等,采样器会把上一轮的残留当成本轮峰值
# (node0 上实测见过 bs=8 报 141,815MiB 而 bs=12 只有 136,240MiB 这种不可能的结果)。
drain() {
  for _ in $(seq 60); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${U:-99999}" -lt 3000 ] && return 0
    sleep 5
  done
}

# $1 标签  $2 per_device bs  $3 dataloader_num_workers  $4.. 额外覆盖
run_case() {
  local tag="$1" bs="$2" dw="$3"; shift 3
  local log="${OUT}/${tag}.log" mem="${OUT}/${tag}.mem" util="${OUT}/${tag}.util"
  drain
  echo "[$(date +%H:%M:%S)] === ${tag}: bs=${bs} dataloader_workers=${dw} $* ===" | tee -a "${OUT}/run.log"

  # 显存 4s 一采(峰值够用);利用率 0.2s 一采(步长才 5s,1Hz 会混叠成看不出饥饿)
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 4; done ) > "${mem}" 2>/dev/null &
  local MP=$!
  ( while true; do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 0.2; done ) > "${util}" 2>/dev/null &
  local UP=$!

  llamafactory-cli train "${YAML}" \
    dataset=mikomiko_desc_0730_smoke \
    tokenized_path="${SAVE_DIR}/tokenized_smoke_0730" \
    output_dir="${SAVE_DIR}/diag_${tag}" overwrite_output_dir=true \
    per_device_train_batch_size="${bs}" gradient_accumulation_steps=1 \
    dataloader_num_workers="${dw}" max_steps="${STEPS}" \
    save_strategy='"no"' eval_strategy='"no"' logging_steps=1 \
    plot_loss=false report_to=none num_train_epochs=1.0 "$@" > "${log}" 2>&1
  local rc=$?
  kill ${MP} ${UP} 2>/dev/null; wait ${MP} ${UP} 2>/dev/null
  rm -rf "${SAVE_DIR}/diag_${tag}"
  echo "[$(date +%H:%M:%S)]     rc=${rc}" | tee -a "${OUT}/run.log"
  return 0
}

# ── 环境事实(先记下来,回来对照) ────────────────────────────────────────────────
{
  echo "# 0730 吞吐诊断报告"
  echo
  echo "生成时间: $(date '+%F %T')  主机: $(hostname)"
  echo
  echo "## 环境"
  echo '```'
  python - <<'PY'
import importlib, os, time
for m in ("torch","transformers","deepspeed","accelerate","datasets","flash_attn"):
    try:
        mod=importlib.import_module(m); v=getattr(mod,'__version__','?')
        d=os.path.dirname(mod.__file__)
        print(f"{m:14s} {v:22s} 安装于 {time.strftime('%F %T', time.localtime(os.path.getmtime(d)))}")
    except Exception as e:
        print(f"{m:14s} FAIL {type(e).__name__}")
import torch
print(f"cuda {torch.version.cuda}  gpus {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
PY
  echo '```'
  echo
  echo "> 0721 那次 31.5 samples/s 跑的**不是**这套依赖 —— torch/transformers 的安装时间都在 0721 之后。"
  echo
} > "${REPORT}"

# ── 跑各档 ──────────────────────────────────────────────────────────────────────
# A. bs 曲线(和 node0 对照:node0 上 bs6/8/12 = 5.2/7.3/11.0 s/step,9.2/8.8/8.7 samp/s)
run_case bs6_dw16  6  16
run_case bs8_dw16  8  16
run_case bs12_dw16 12 16
# B. dataloader 扫描 —— 若 dw=2 与 dw=16 同速,基本可排除 dataloader 饥饿;
#    若 dw=32 明显更快,说明确实卡在数据侧
run_case bs6_dw2   6  2
run_case bs6_dw32  6  32

# ── 汇总 ────────────────────────────────────────────────────────────────────────
python - "${OUT}" >> "${REPORT}" <<'PY'
import os, re, sys, statistics
out = sys.argv[1]
cases = ["bs6_dw16","bs8_dw16","bs12_dw16","bs6_dw2","bs6_dw32"]
NODE0 = {"bs6_dw16": (5.2, 9.2), "bs8_dw16": (7.3, 8.8), "bs12_dw16": (11.0, 8.7)}

def step_times(log):
    """tqdm 里的累计耗时 -> 逐步增量。取第 6 步之后,避开首步 JIT 编译。"""
    txt = open(log, encoding="utf-8", errors="replace").read().replace("\r", "\n")
    seen = {}
    for m in re.finditer(r"\| (\d+)/\d+ \[(\d+):(\d+)<", txt):
        i = int(m.group(1)); seen[i] = int(m.group(2))*60 + int(m.group(3))
    ks = sorted(seen)
    d = [seen[b]-seen[a] for a, b in zip(ks, ks[1:]) if b > 5]
    return d

def peak(mem):
    try:
        return max(int(x) for x in open(mem) if x.strip())
    except Exception:
        return None

def util_steady(util, tail_frac=0.5):
    """只看后半段(避开模型加载那段全 0),给出均值和低利用率占比。"""
    rows = []
    for ln in open(util):
        p = ln.strip().split(", ")
        if len(p) == 2:
            try: rows.append(int(p[1]))
            except ValueError: pass
    if not rows: return None
    tail = rows[int(len(rows)*(1-tail_frac)):]
    tail = [u for u in tail]
    if not tail: return None
    below50 = sum(1 for u in tail if u < 50)/len(tail)
    return statistics.mean(tail), below50

print("## 结果\n")
print("| 档位 | 稳态 s/step | samples/s | 峰值显存 | GPU 均值利用率 | util<50% 占比 | node0 对照 |")
print("|---|---:|---:|---:|---:|---:|---|")
rows = {}
for c in cases:
    log, mem, util = (os.path.join(out, c+e) for e in (".log", ".mem", ".util"))
    if not os.path.exists(log):
        print(f"| {c} | 缺日志 | | | | | |"); continue
    bs = int(re.search(r"bs(\d+)", c).group(1))
    d = step_times(log)
    if not d:
        err = "OOM" if re.search(r"out of memory", open(log,errors='replace').read(), re.I) else "未产出步时"
        print(f"| {c} | **{err}** | | {peak(mem) or '?'} MiB | | | |"); continue
    sps_t = statistics.median(d); sps = bs*8/sps_t
    u = util_steady(util)
    rows[c] = (sps_t, sps)
    ref = f"{NODE0[c][0]} s / {NODE0[c][1]} samp/s" if c in NODE0 else "—"
    um = f"{u[0]:.0f}%" if u else "?"
    ub = f"{u[1]:.0%}" if u else "?"
    print(f"| {c} | {sps_t:.1f} | **{sps:.1f}** | {peak(mem) or '?'} MiB | {um} | {ub} | {ref} |")

print()
print("## 判读\n")
base = rows.get("bs6_dw16")
if base:
    sps = base[1]
    print(f"- **node1 的 bs=6 吞吐 = {sps:.1f} samples/s**,node0 是 9.2。", end=" ")
    if sps > 15:
        print("→ 明显更快,问题出在 **node0 那台机器**,不是配置/环境。")
    elif sps < 12:
        print("→ 复现了,**不是 node0 的问题**,是环境或配置层面的共性问题。")
    else:
        print("→ 居中,需要人工判断。")
    d2, d32 = rows.get("bs6_dw2"), rows.get("bs6_dw32")
    if d2 and d32:
        print(f"- dataloader_workers 2/16/32 = {d2[1]:.1f} / {sps:.1f} / {d32[1]:.1f} samples/s。", end=" ")
        if d32[1] > sps*1.15:
            print("→ 加 worker 明显变快,**瓶颈在数据侧**(图像解码/collator),调 dataloader_num_workers 即可。")
        elif abs(d2[1]-sps)/sps < 0.10:
            print("→ 2 个 worker 和 16 个同速,**排除 dataloader 饥饿**,瓶颈在 GPU 侧(内核/通信)。")
        else:
            print("→ 有影响但不决定性。")
print("""
- 若上面确认「不是 node0、也不是 dataloader」,下一步就是**环境**:0721 用的是旧
  torch/transformers,现在是 torch 2.8.0+cu129 / transformers 5.6.1(还超过 LF 硬编码上限
  5.6.0,靠 DISABLE_VERSION_CHECK 压着)。验证办法是把 transformers 降到 0721 那个版本再跑
  一次 bs=6 —— 但这会动 venv,**留给人在场时做**,本脚本不碰。
- 另外注意 `util<50% 占比`:若它在 30% 以上而 dataloader 扫描又排除了数据侧,说明 GPU 在等
  通信或在跑低效内核(collator 里 `_compute_rope_position_ids` 是逐样本 Python 循环,
  和「每样本耗时恒定、随 batch 完全线性」的现象吻合,是重点怀疑对象)。
""")
print("\n## 没有做的事\n")
print("- **没有预先 tokenize 全量 3.31M 行**(那要 ~1.8h)。因为如果结论是降级 transformers,")
print("  分词器版本变了缓存可能静默失效 —— 缓存按原文本键、不校验依赖版本。等环境定下来再建。")
print("- **没有改 venv**,没有降级任何包。")
PY

echo "[$(date +%H:%M:%S)] DIAG DONE -> ${REPORT}" | tee -a "${OUT}/run.log"
