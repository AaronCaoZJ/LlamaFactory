# RUIBench alter-reference training (Qwen3.5-4B, LoRA)

Six runs: an interface (`code-joint`, `code-delta`) at 10, 5 and 1 episodes a task, 30 tasks. Each
training row is one request of a RUIBench episode as RUIKit sent it (system prompt, tools, the whole
history, the latest six pictures), rendered by Qwen3.5's own chat template with thinking off; only the
last reply of a row is trained (`mask_history`). The episodes are Qwen3.8-Max's successes first, then
rule-based ones (the task's reference and alternative ways, half each). How the data was made:
RUIBench `CVPR27/zhijun/AlterReference/SCRIPTED.md`.

## Files

| Path | What |
|---|---|
| `examples/train_lora/qwen3_5_4b/ruibench_alter/alter_{joint,delta}_{10,5,1}.yaml` | the six configs |
| `scripts/ruibench_alter/train_ruibench_alter.sh` | runs one config on any number of cards |
| `data/ruibench_alter/` | the data bundle, not in git: copy it here |

The bundle holds `alter_*.jsonl` (the rows), `dataset_info.json`, `subsets.json` (episodes and row
counts per run) and `images/` (every picture, the rows' paths relative to the bundle).

## Run

```bash
RUN=alter_joint_10 GPU=0,1 MODEL_PATH=/path/to/Qwen3.5-4B bash scripts/ruibench_alter/train_ruibench_alter.sh
```

- 32 rows an optimizer step on any number of cards (`GPU` divides 32): the script sets the
  accumulation, and the checkpoint steps so a checkpoint falls at the end of every second epoch
  (ten epochs, five checkpoints, under `saves/qwen3.5-4b/ruibench_alter/<run>/`).
- One card holds a run: about 59 GB at the longest rows (64,580 tokens; `cutoff_len` 65536 cuts none).
- More than one card: plain data parallel through torchrun (`FORCE_TORCHRUN=1`); LoRA on a 4B fits a
  card, so no DeepSpeed is needed.

## Environment

Tested on showlab15 (H200) with: torch 2.8.0+cu129, transformers 5.7.0, peft 0.18.1, flash-attn 2.8.3,
flash-linear-attention (`fla`) 0.5.1 with TileLang, liger-kernel, CUDA 12.9. LlamaFactory at
`b922066b` (`qwen3_5_nothink` template), where this branch starts: the rows were checked token for
token there against Qwen3.5's own processor. The fork's `main` has since merged 34 upstream commits
(`bd6992a3`, with changes to `data/template.py` and `data/mm_plugin.py`); on a newer checkout, check
the rows again first (RUIBench `CVPR27/zhijun/AlterReference/scripted_check.py`).

- **A C++ compiler for TileLang.** Qwen3.5's linear-attention layers run fla's gated delta rule; on
  Hopper its backward kernel is built by TileLang through nvcc, whose host compiler is the `gcc` on
  `PATH`: that `gcc` must compile C++ (have its `cc1plus`), or nvcc stops with "Failed to preprocess
  host compiler properties". Triton is no way round: fla refuses it for that kernel on Hopper (wrong
  results). If `gcc` cannot, point `CC_SHIM` at a folder whose `gcc` and `g++` can (showlab15: gcc is
  12 without `cc1plus`, g++ is 11; `.cc-shim` points both at gcc-11). The script checks both and stops
  if neither the default nor the shim works.
- `causal_conv1d` is optional: without it transformers warns that the "fast path" is not available
  and runs the short convolution in torch; fla still does the linear attention.

## Speed (to measure again on the new machine)

A 4-step trial on showlab15 took about 15 s a row (rows average about 18k tokens), on a card other
jobs were also using; that is far slower than a 4B should be, and was not measured on a free card.
Measure a few steps before planning: rows per run are in `subsets.json` (about 7.5k, 3.8k and 0.8k
per interface at 10, 5 and 1), and every run is ten epochs.
