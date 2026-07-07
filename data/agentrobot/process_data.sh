# resolve machine paths: locate & source scripts/workspace_dir.sh (sets LF_ROOT, MODELS_DIR, LF_VENV, VLLM_VENV, AGENTROBOT_ROOT, HF_HOME)
_wsd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$_wsd" != "/" ] && [ ! -f "$_wsd/scripts/workspace_dir.sh" ]; do _wsd="$(dirname "$_wsd")"; done
source "$_wsd/scripts/workspace_dir.sh"
source ${AGENTROBOT_ROOT}/.venv/bin/activate
cd ${LF_ROOT}

DATA_DIR=data/agentrobot/MVTOKEN/0622
DATA_DIR_0627=data/agentrobot/MVTOKEN/0627_cleaned
DATA_DIR_0704=data/agentrobot/MVTOKEN/0704_cleaned
MIX_DIR=data/agentrobot/MVTOKEN/mix_22_27_04
PIPER_DIR_0705=data/agentrobot/MVTOKEN/0705_piper

TASK_MAP_0622=(
    "pap_banana=pick up the banana and place it on the blue plate"
    "pap_yellow_cup=pick up the yellow cup and place it on the green coaster"
    "pap_mango=pick up the mango and place it on the blue plate"
    "stack_white_bowl=pick up the white bowl and stack it on the pink bowl"
)

TASK_MAP_0627=(
    "pap_banana=pick up the banana and place it on the blue plate"
    "pap_mango=pick up the mango and place it on the blue plate"
    "stack_pink_cup=pick up the pink cup and stack it on the blue cup"
    "pap_gray_mug=pick up the gray mug and place it on the green coaster"
)

TASK_MAP_0704=(
    "pap_orange_block=pick up the orange block and place it on the plate"
    "pap_mango=pick up the mango and place it in the bowl"
    "pap_pink_cube=pick up the pink cube and place it on the plate"
    "rearrange_show=rearrange the letters to spell \"SHOW\""
)

TASK_MAP_0705_PIPER=(
    "pap_banana=pick up the banana and place it on the plate"
    "pap_blue_block=pick up the blue block and place it on the plate"
    "pap_orange_block=pick up the orange block and place it on the plate"
    "pap_tennis=pick up the tennis ball and place it on the green coaster"
    "pap_wrench=pick up the wrench and insert it into the socket"
    "pap_mango=pick up the mango and place it in the bowl"
    "stack_gray_cup=pick up the gray cup and stack it on the yellow cup"
    "rearrange_show=rearrange the letters to spell \"SHOW\""
)
# --version <vX> selects the prompt folder AgentRobot/prompts/<vX>/ (fixed per-mode filenames:
# lite=mvtoken_generator_lite.txt, affordance=mvtoken_generator_affordance.txt,
# subgoal=mvtoken_generator.txt). Keep --version aligned with the output subdir below.

# VLM for subgoal/affordance planning: drive the :8101 vLLM server but override --model to
# its BASE model (the strong general VLM), NOT the mvtoken_0622_v0 action LoRA. The
# mvtoken_0622_v0 backend only supplies the connection (provider=vllm / base_url / no-think
# template); --model swaps the served model to the base for planning.
BASE_MODEL=${MODELS_DIR}/Qwen3.5-27B
VLM_ARGS=(--vlm-backend mvtoken_0622_v0 --model "$BASE_MODEL")


: <<'EOF'
# ========================================
# Clean the 0627 grasp/release raw rollouts -> MVTOKEN/0627_cleaned.
# grasp/ drops every RELEASE step (reset), release/ drops every GRASP step (reset); kept
# steps are re-indexed contiguously. grasp ids 000-007 + release ids 008-012 merge cleanly.
# ========================================
EOF
# RAW_0704=${HF_HOME}/datasets/MVTOKEN_RAW/0704
# python data/agentrobot/clean_grasp_release.py \
#     --grasp-dir   "$RAW_0704"/left_right \
#     --out-dir     data/agentrobot/MVTOKEN/0704_cleaned


: <<'EOF'
# ========================================
# Lite mode (no subgoal): merge all 4 tasks into one rollout.json
# ========================================
EOF
# python data/agentrobot/rollout_to_llamafactory.py \
#     "$DATA_DIR"/pap_banana \
#     "$DATA_DIR"/pap_yellow_cup \
#     "$DATA_DIR"/pap_mango \
#     "$DATA_DIR"/stack_white_bowl \
#     --version v3 \
#     --task-map "${TASK_MAP_0622[@]}" \
#     --output "$DATA_DIR"/v3/rollout_lite.json

# python data/agentrobot/rollout_to_llamafactory.py \
#     "$DATA_DIR_0627"/pap_banana \
#     "$DATA_DIR_0627"/stack_pink_cup \
#     "$DATA_DIR_0627"/pap_mango \
#     "$DATA_DIR_0627"/pap_gray_mug \
#     --version v3 \
#     --task-map "${TASK_MAP_0627[@]}" \
#     --output "$DATA_DIR_0627"/v3/rollout_lite.json

# python data/agentrobot/rollout_to_llamafactory.py \
#     "$DATA_DIR_0704"/pap_orange_block \
#     "$DATA_DIR_0704"/pap_mango \
#     "$DATA_DIR_0704"/pap_pink_cube \
#     "$DATA_DIR_0704"/rearrange_show \
#     --version v3 \
#     --task-map "${TASK_MAP_0704[@]}" \
#     --output "$DATA_DIR_0704"/v3/rollout_lite.json


: <<'EOF'
# ========================================
# Mix the two v3 lite sets (0622 + 0627_cleaned) -> MVTOKEN/mix_22_27/rollout_lite.json.
# Plain concatenation (samples carry absolute image paths); run the two lite commands above
# first so both v3/rollout_lite.json exist.
# ========================================
EOF
# python data/agentrobot/merge_rollouts.py \
#     "${LF_ROOT}/data/agentrobot/MVTOKEN/mix_22_27/v3/rollout_lite.json" \
#     "$DATA_DIR_0704"/v3/rollout_lite.json \
#     --output "$MIX_DIR"/v3/rollout_lite.json


: <<'EOF'
# ========================================
# Piper (ego) 0705 — v3 lite, PIPER-ONLY training set.
# Straight from the sorted raw task folders (no clean_grasp_release: continuous pick-place).
# Images stay referenced in-place under hf_download; only the json is written under MVTOKEN.
# Output: MVTOKEN/0705_piper/v3/rollout_lite.json (registered as mvtoken_0705_piper_v3_lite).
# ========================================
EOF
python data/agentrobot/rollout_to_llamafactory.py \
    "$PIPER_DIR_0705"/pap_banana \
    "$PIPER_DIR_0705"/pap_blue_block \
    "$PIPER_DIR_0705"/pap_orange_block \
    "$PIPER_DIR_0705"/pap_tennis \
    "$PIPER_DIR_0705"/pap_wrench \
    "$PIPER_DIR_0705"/pap_mango \
    "$PIPER_DIR_0705"/stack_gray_cup \
    "$PIPER_DIR_0705"/rearrange_show \
    --version v4 \
    --piper \
    --task-map "${TASK_MAP_0705_PIPER[@]}" \
    --output "$PIPER_DIR_0705"/v4/rollout_lite.json


: <<'EOF'
# ========================================
# Affordance mode: lite + a single grasp-point hint (target + affordance) per rollout.
# affordance_config.json is auto-generated (via generate_affordance.py) when missing.
# ========================================
EOF
# python data/agentrobot/rollout_to_llamafactory.py \
#     "$DATA_DIR"/pap_banana \
#     "$DATA_DIR"/pap_yellow_cup \
#     "$DATA_DIR"/pap_mango \
#     "$DATA_DIR"/stack_white_bowl \
#     --version v1 \
#     --task-map "${TASK_MAP_0622[@]}" \
#     --use-affordance \
#     --output "$DATA_DIR"/v1/rollout_affordance.json \
#     "${VLM_ARGS[@]}"


: <<'EOF'
# ========================================
# Subgoal mode: full prompt with per-step VLM subgoal info.
# task_config.json is auto-generated (via generate_subgoals.py) for any rollout missing it,
# using "${VLM_ARGS[@]}" below.
# ========================================
EOF
# python data/agentrobot/rollout_to_llamafactory.py \
#     "$DATA_DIR"/pap_banana \
#     "$DATA_DIR"/pap_yellow_cup \
#     "$DATA_DIR"/pap_mango \
#     "$DATA_DIR"/stack_white_bowl \
#     --version v1 \
#     --task-map "${TASK_MAP_0622[@]}" \
#     --use-subgoal \
#     --output "$DATA_DIR"/v1/rollout_subgoal.json \
#     "${VLM_ARGS[@]}"


: <<'EOF'
# ========================================
# Generate subgoals manually (one rollout per task as reference / inspection).
# Add --dry-run to print the plan without writing task_config.json.
# ========================================
EOF
# python data/agentrobot/generate_subgoals.py \
#     "$DATA_DIR"/pap_banana/rollout_030 \
#     --task "pick up the banana and place it on the blue plate" \
#     "${VLM_ARGS[@]}"


: <<'EOF'
# ========================================
# Generate affordance hints manually (one rollout per task as reference / inspection).
# Add --dry-run to print the grasp point without writing affordance_config.json.
# ========================================
EOF
# python data/agentrobot/generate_affordance.py \
#     "$DATA_DIR"/pap_banana/rollout_030 \
#     --task "pick up the banana and place it on the blue plate" \
#     "${VLM_ARGS[@]}"


: <<'EOF'
# ========================================
# Scratch: single-sample conversions (eval ID/OOD probes).
# ========================================
EOF
# python data/agentrobot/rollout_to_llamafactory.py \
#     ${LF_ROOT}/data/agentrobot/ood_sample \
#     --version v3 \
#     --task "pick up the white cup and place it on the green coaster" \
#     --output ${LF_ROOT}/data/agentrobot/ood_sample/v3/rollout_lite.json

# python data/agentrobot/rollout_to_llamafactory.py \
#     ${LF_ROOT}/scripts/eval/id_sample \
#     --version v1 \
#     --task "pick up the yellow cup and place it on the green coaster" \
#     --use-affordance \
#     --output ${LF_ROOT}/scripts/eval/id_sample/v1/rollout_affordance.json \
#     "${VLM_ARGS[@]}"
