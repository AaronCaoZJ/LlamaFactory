#!/usr/bin/env bash
# One RUIBench alter-reference training run, from a LlamaFactory checkout (copied there as
# scripts/ruibench_alter/train_ruibench_alter.sh).  On any number of cards: 32 rows a step, a
# checkpoint every second epoch.
#
#   RUN=alter_joint_10 GPU=0,1 MODEL_PATH=/path/to/Qwen3.5-4B bash scripts/ruibench_alter/train_ruibench_alter.sh
#
#   RUN         alter_{joint,delta}_{10,5,1} (required): a config in examples/train_lora/qwen3_5_4b/ruibench_alter/
#   GPU         CUDA_VISIBLE_DEVICES (default 0); more than one: torchrun data parallel
#   MODEL_PATH  Qwen3.5-4B (default: the hub id in the config)
#   CC_SHIM     a folder whose gcc and g++ both compile C++ (default: .cc-shim here, if it has one)
#   PYTHON      the LlamaFactory environment's python (default .venv/bin/python here)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."           # the LlamaFactory checkout
RUN=${RUN:?RUN=alter_joint_10 or another config name}
GPU=${GPU:-0}
CFG=examples/train_lora/qwen3_5_4b/ruibench_alter/$RUN.yaml
[ -f "$CFG" ] || { echo "no config $CFG" >&2; exit 1; }
DATA=data/ruibench_alter
[ -f "$DATA/subsets.json" ] || { echo "no data bundle in $DATA (subsets.json)" >&2; exit 1; }
PY=${PYTHON:-.venv/bin/python}

# Qwen3.5's linear attention (flash-linear-attention's gated delta rule) compiles its backward kernel
# with TileLang, through nvcc, whose host compiler is the gcc on PATH: that gcc must compile C++ (have
# its cc1plus).  On showlab15 gcc is 12 without cc1plus (g++ is 11 and works, so testing g++ alone is
# not enough); .cc-shim points gcc and g++ at gcc-11, as Show-Harness's train.sh does.  Triton is no
# way round it: fla refuses Triton for that kernel on Hopper (wrong results).
SHIM=${CC_SHIM:-$PWD/.cc-shim}
cxx_ok() {  # $1: "" or a folder with a trailing slash
    local c
    for c in gcc g++; do
        echo 'int main(){return 0;}' | "$1$c" -x c++ -c - -o /dev/null > /dev/null 2>&1 || return 1
    done
}
if ! cxx_ok ""; then
    if [ -x "$SHIM/gcc" ] && cxx_ok "$SHIM/"; then
        export PATH="$SHIM:$PATH"
        echo "using the compilers in $SHIM"
    else
        echo "gcc cannot compile C++ (no cc1plus) and no working CC_SHIM: TileLang's kernels will not build" >&2
        exit 1
    fi
fi

# 32 rows a step on any number of cards; a checkpoint every second epoch, counted as transformers
# counts an epoch's steps: ceil(ceil(rows / cards) / accumulation)
N=$(echo "$GPU" | tr ',' '\n' | grep -c .)
[ $((32 % N)) -eq 0 ] || { echo "32 rows a step need a card count that divides 32, not $N" >&2; exit 1; }
ACC=$((32 / N))
ROWS=$("$PY" -c "import json; print(json.load(open('$DATA/subsets.json'))['subsets']['$RUN']['rows'])")
PER_CARD=$(( (ROWS + N - 1) / N ))
STEPS=$(( (PER_CARD + ACC - 1) / ACC ))
SAVE=$(( 2 * STEPS ))
echo "$RUN: $ROWS rows on $N card(s), accumulation $ACC, $STEPS steps an epoch, a checkpoint every $SAVE steps"

ARGS=("gradient_accumulation_steps=$ACC" "save_steps=$SAVE")
[ -n "${MODEL_PATH:-}" ] && ARGS+=("model_name_or_path=$MODEL_PATH")
[ "$N" -gt 1 ] && export FORCE_TORCHRUN=1
mkdir -p "saves/qwen3.5-4b/ruibench_alter/$RUN"
CUDA_VISIBLE_DEVICES=$GPU exec "$(dirname "$PY")/llamafactory-cli" train "$CFG" "${ARGS[@]}"
