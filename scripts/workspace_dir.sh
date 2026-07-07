#!/usr/bin/env bash
# Minimal loader: pull machine-specific paths from LlamaFactory/.env.paths, then sanity-check them.
#
# All actual paths (models, venvs, HF cache) live in .env.paths — a gitignored file you keep ONE of
# per machine (machine A: /workspace1/zhijun/... , machine B: /highspeedstorage/Kokoro2/...).
# (We use .env.paths, NOT .env.local — the latter is an upstream LlamaFactory tracked file.)
# This loader locates the repo (LF_ROOT), sources that file, and validates the result so a bad path
# fails HERE with a clear message instead of deep inside model loading. First-time setup:
#     cp .env.paths.example .env.paths   # then edit the paths for this machine
#
# Sourced by scripts via this depth-independent bootstrap:
#     _wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
#     source "$_wsd/scripts/workspace_dir.sh"
#
# Debug:  WORKSPACE_DIR_DEBUG=1 <script>   -> print every resolved path
# Bypass: WORKSPACE_DIR_SKIP_CHECK=1       -> skip the sanity checks (e.g. odd one-off setups)

_WSD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LF_ROOT="${LF_ROOT:-$(dirname "$_WSD")}"     # this repo (the only value that must be auto — it holds .env.paths)

if [ ! -f "$LF_ROOT/.env.paths" ]; then
  echo "ERROR[workspace_dir]: $LF_ROOT/.env.paths not found." >&2
  echo "  fix: cp $LF_ROOT/.env.paths.example $LF_ROOT/.env.paths   # then edit this machine's paths" >&2
  return 1 2>/dev/null || exit 1
fi
source "$LF_ROOT/.env.paths"

# ── sanity checks: catch config mistakes at the source, not 200 lines into a run ────────────────
if [ -z "${WORKSPACE_DIR_SKIP_CHECK:-}" ]; then
  _wsd_bad=0
  # (a) every path we hand to scripts must be set by .env.paths
  for _v in MODELS_DIR LF_VENV VLLM_VENV HF_HOME; do
    if [ -z "${!_v:-}" ]; then
      echo "ERROR[workspace_dir]: $_v is empty — add it to $LF_ROOT/.env.paths" >&2; _wsd_bad=1
    fi
  done
  # (b) MODELS_DIR must actually exist (models can't be built on the fly). Venvs may not exist yet
  #     (env_setup.sh creates LF_VENV), so those are only warned, never fatal.
  if [ -n "${MODELS_DIR:-}" ] && [ ! -d "${MODELS_DIR}" ]; then
    echo "ERROR[workspace_dir]: MODELS_DIR=${MODELS_DIR} does not exist — wrong path in .env.paths?" >&2; _wsd_bad=1
  fi
  for _v in LF_VENV VLLM_VENV; do
    _p="${!_v:-}"
    [ -n "$_p" ] && [ ! -d "$_p" ] && echo "WARN[workspace_dir]: $_v=$_p not present yet (build it, or fix .env.paths)" >&2
  done
  if [ "$_wsd_bad" = 1 ]; then
    echo "  -> fix $LF_ROOT/.env.paths (template: .env.paths.example)" >&2
    return 1 2>/dev/null || exit 1
  fi
fi

if [ -n "${WORKSPACE_DIR_DEBUG:-}" ]; then
  echo "[workspace_dir] LF_ROOT=$LF_ROOT" >&2
  echo "[workspace_dir] MODELS_DIR=$MODELS_DIR  LF_VENV=$LF_VENV  VLLM_VENV=$VLLM_VENV  HF_HOME=$HF_HOME" >&2
fi
