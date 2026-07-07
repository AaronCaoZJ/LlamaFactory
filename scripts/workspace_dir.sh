#!/usr/bin/env bash
# Minimal loader: pull machine-specific paths from LlamaFactory/.env.paths.
#
# All actual paths (models, venvs, HF cache) live in .env.paths — a gitignored file you keep ONE of
# per machine (machine A: /workspace1/zhijun/... , machine B: /highspeedstorage/Kokoro2/...).
# (We use .env.paths, NOT .env.local — the latter is an upstream LlamaFactory tracked file.)
# This loader only locates the repo (LF_ROOT) and sources that file. First-time setup:
#     cp .env.paths.example .env.paths   # then edit the paths for this machine
#
# Sourced by scripts via this depth-independent bootstrap:
#     _d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     while [ "$_d" != "/" ] && [ ! -f "$_d/scripts/workspace_dir.sh" ]; do _d="$(dirname "$_d")"; done
#     source "$_d/scripts/workspace_dir.sh"

_WSD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LF_ROOT="${LF_ROOT:-$(dirname "$_WSD")}"     # this repo (only value that must be auto — it holds .env.paths)

if [ -f "$LF_ROOT/.env.paths" ]; then
  source "$LF_ROOT/.env.paths"
else
  echo "ERROR: $LF_ROOT/.env.paths not found." >&2
  echo "       cp $LF_ROOT/.env.paths.example $LF_ROOT/.env.paths  and set this machine's paths." >&2
  return 1 2>/dev/null || exit 1
fi
