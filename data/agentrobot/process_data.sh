source /workspace1/zhijun/AgentRobot/.venv/bin/activate
cd /workspace1/zhijun/LlamaFactory

DATA_DIR=data/agentrobot/MVTOKEN/0622

TASK_MAP=(
    "pap_banana=pick up the banana and place it on the blue plate"
    "pap_yellow_cup=pick up the yellow cup and place it on the green coaster"
    "pap_mango=pick up the mango and place it on the blue plate"
    "stack_white_bowl=pick up the white bowl and stack it on the pink bowl"
)
# TASK_MAP=(
#     "pap_banana=pick up the banana and place it on the blue plate"
#     "pap_pink_cup=pick up the pink cup and place it on the blue cup"
#     "pap_gray_mug=pick up the gray mug and place it on the green coaster"
#     "pap_mango=pick up the mango and place it on the blue plate"
# )

# VLM for subgoal/affordance planning: drive the :8101 vLLM server but override --model to
# its BASE model (the strong general VLM), NOT the mvtoken_0622_v0 action LoRA. The
# mvtoken_0622_v0 backend only supplies the connection (provider=vllm / base_url / no-think
# template); --model swaps the served model to the base for planning.
BASE_MODEL=/workspace1/zhijun/hf_download/models/Qwen3.5-27B
VLM_ARGS=(--vlm-backend mvtoken_0622_v0 --model "$BASE_MODEL")


: <<'EOF'
# ========================================
# Lite mode (no subgoal): merge all 4 tasks into one rollout.json
# ========================================
EOF
python data/agentrobot/rollout_to_llamafactory.py \
    "$DATA_DIR"/pap_banana \
    "$DATA_DIR"/pap_yellow_cup \
    "$DATA_DIR"/pap_mango \
    "$DATA_DIR"/stack_white_bowl \
    --task-map "${TASK_MAP[@]}" \
    --output "$DATA_DIR"/rollout_v1.json


: <<'EOF'
# ========================================
# Subgoal mode: full prompt with per-step VLM subgoal info.
# task_config.json is auto-generated (via generate_subgoals.py) for any rollout missing it,
# using "${VLM_ARGS[@]}" below.
# ========================================
EOF
python data/agentrobot/rollout_to_llamafactory.py \
    "$DATA_DIR"/pap_banana \
    "$DATA_DIR"/pap_yellow_cup \
    "$DATA_DIR"/pap_mango \
    "$DATA_DIR"/stack_white_bowl \
    --task-map "${TASK_MAP[@]}" \
    --use-subgoal \
    --output "$DATA_DIR"/rollout_subgoal.json \
    "${VLM_ARGS[@]}"


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
#     --task-map "${TASK_MAP[@]}" \
#     --use-affordance \
#     --output "$DATA_DIR"/rollout_affordance.json \
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

# python data/agentrobot/generate_subgoals.py \
#     "$DATA_DIR"/pap_yellow_cup/rollout_053 \
#     --task "pick up the yellow cup and place it on the green coaster" \
#     "${VLM_ARGS[@]}"

# python data/agentrobot/generate_subgoals.py \
#     "$DATA_DIR"/pap_mango/rollout_038 \
#     --task "pick up the mango and place it on the blue plate" \
#     "${VLM_ARGS[@]}"

# python data/agentrobot/generate_subgoals.py \
#     "$DATA_DIR"/stack_white_bowl/rollout_000 \
#     --task "pick up the white bowl and stack it on the pink bowl" \
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

# python data/agentrobot/generate_affordance.py \
#     "$DATA_DIR"/pap_yellow_cup/rollout_053 \
#     --task "pick up the yellow cup and place it on the green coaster" \
#     "${VLM_ARGS[@]}"

# python data/agentrobot/generate_affordance.py \
#     "$DATA_DIR"/pap_mango/rollout_038 \
#     --task "pick up the mango and place it on the blue plate" \
#     "${VLM_ARGS[@]}"

# python data/agentrobot/generate_affordance.py \
#     "$DATA_DIR"/stack_white_bowl/rollout_000 \
#     --task "pick up the white bowl and stack it on the pink bowl" \
#     "${VLM_ARGS[@]}"


: <<'EOF'
# ========================================
# Scratch: single-sample conversions (eval ID/OOD probes).
# ========================================
EOF
# python data/agentrobot/rollout_to_llamafactory.py \
#     /workspace1/zhijun/LlamaFactory/scripts/eval/ood_sample \
#     --task "pick up the white cup and place it on the green coaster"

# python data/agentrobot/rollout_to_llamafactory.py \
#     /workspace1/zhijun/LlamaFactory/scripts/eval/id_sample \
#     --task "pick up the yellow cup and place it on the green coaster"

# python data/agentrobot/rollout_to_llamafactory.py \
#     /workspace1/zhijun/LlamaFactory/scripts/eval/id_sample \
#     --task "pick up the yellow cup and place it on the green coaster" \
#     --use-subgoal \
#     "${VLM_ARGS[@]}"