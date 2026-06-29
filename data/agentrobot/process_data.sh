source /workspace1/zhijun/AgentRobot/.venv/bin/activate
cd /workspace1/zhijun/LlamaFactory

DATA_DIR=data/agentrobot/MVTOKEN/0622
DATA_DIR_0627=data/agentrobot/MVTOKEN/0627_cleaned
MIX_DIR=data/agentrobot/MVTOKEN/mix_22_27

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

# --version <vX> selects the prompt folder AgentRobot/prompts/<vX>/ (fixed per-mode filenames:
# lite=mvtoken_generator_lite.txt, affordance=mvtoken_generator_affordance.txt,
# subgoal=mvtoken_generator.txt). Keep --version aligned with the output subdir below.

# VLM for subgoal/affordance planning: drive the :8101 vLLM server but override --model to
# its BASE model (the strong general VLM), NOT the mvtoken_0622_v0 action LoRA. The
# mvtoken_0622_v0 backend only supplies the connection (provider=vllm / base_url / no-think
# template); --model swaps the served model to the base for planning.
BASE_MODEL=/workspace1/zhijun/hf_download/models/Qwen3.5-27B
VLM_ARGS=(--vlm-backend mvtoken_0622_v0 --model "$BASE_MODEL")


: <<'EOF'
# ========================================
# Clean the 0627 grasp/release raw rollouts -> MVTOKEN/0627_cleaned.
# grasp/ drops every RELEASE step (reset), release/ drops every GRASP step (reset); kept
# steps are re-indexed contiguously. grasp ids 000-007 + release ids 008-012 merge cleanly.
# ========================================
EOF
# RAW_0627=/workspace1/zhijun/hf_download/datasets/MVTOKEN_RAW/0627
# python data/agentrobot/clean_grasp_release.py \
#     --grasp-dir   "$RAW_0627"/grasp \
#     --release-dir "$RAW_0627"/release \
#     --out-dir     data/agentrobot/MVTOKEN/0627_cleaned


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


: <<'EOF'
# ========================================
# Mix the two v3 lite sets (0622 + 0627_cleaned) -> MVTOKEN/mix_22_27/rollout_lite.json.
# Plain concatenation (samples carry absolute image paths); run the two lite commands above
# first so both v3/rollout_lite.json exist.
# ========================================
EOF
python data/agentrobot/merge_rollouts.py \
    "$DATA_DIR"/v3/rollout_lite.json \
    "$DATA_DIR_0627"/v3/rollout_lite.json \
    --output "$MIX_DIR"/v3/rollout_lite.json


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
#     /workspace1/zhijun/LlamaFactory/scripts/eval/ood_sample \
#     --version v2 \
#     --task "pick up the white cup and place it on the green coaster" \
#     --output /workspace1/zhijun/LlamaFactory/scripts/eval/ood_sample/v2/rollout_lite.json

# python data/agentrobot/rollout_to_llamafactory.py \
#     /workspace1/zhijun/LlamaFactory/scripts/eval/id_sample \
#     --version v1 \
#     --task "pick up the yellow cup and place it on the green coaster" \
#     --use-affordance \
#     --output /workspace1/zhijun/LlamaFactory/scripts/eval/id_sample/v1/rollout_affordance.json \
#     "${VLM_ARGS[@]}"
