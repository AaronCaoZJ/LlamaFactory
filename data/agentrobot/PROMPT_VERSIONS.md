# MVTOKEN 0622 训练 Prompt 对照（v0 / v1 / v2）

| | v0 | v1 (affordance) | v2 (lite) |
|---|---|---|---|
| Task | ✅ | ✅ | ✅ |
| Gripper 状态 | ✅ | ✅ | ❌ |
| Recent moves | ✅ | ✅ | ❌ |
| Affordance 提示 | ❌ | ✅ | ❌ |
| 收敛 loss | **3.9e-5** | 5.9e-4 | 7.0e-3 |

> 三个版本都有 task，task 不是差异点。真正的区别：**v1 = v0 + affordance**（更长，多了依赖外部 VLM 的恒定抓取点文本）；**v2 = v0 − Gripper − Recent moves**（更短，删掉了两个关键状态信号 → 大幅变差）。

---

## v0

```
<image><image>You are controlling a robot arm with two cameras:
- Image 1 (agentview): global overhead view of the robot and workspace
- Image 2 (wrist): local view from the gripper

Task: pick up the banana and place it on the blue plate
Gripper now: open
Recent moves, newest first: none

Output exactly one action token:
MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN, GRASP, RELEASE

Look at both camera views and choose the next action:
- Use AgentView to locate the target when it is not in wrist view
- Use wrist view to fine-align when target is visible up close
- GRASP when gripper fingers are aligned around the object
- RELEASE when object is above the destination
- Avoid repeating a direction that conflicts with the most recent move

Return the single token only, no punctuation, no explanation:
```

## v1

```
<image><image>You are controlling a robot arm with two cameras:
- Image 1 (agentview): global overhead view of the robot and workspace
- Image 2 (wrist): local view from the gripper

Task: pick up the banana and place it on the blue plate
Grasp target: the banana
Grasp point: the middle section of the banana's body
Gripper now: open
Recent moves, newest first: none

Output exactly one action token:
MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN, GRASP, RELEASE, DONE

Look at both camera views and choose the next action:
- Use AgentView to locate the grasp target when it is not in wrist view
- Use wrist view to fine-align the grasp point between the gripper fingers
- Aim the fingers at the grasp point above, not just the center of the target; for hollow
  objects (bowls/cups) keep the left/right contact region between the fingers
- GRASP when the grasp point is centered between the gripper fingers in both views
- RELEASE when the held object is above the destination and lowered onto it
- DONE when the task is complete: the object is at its destination and the gripper is clear
- Avoid repeating a direction that conflicts with the most recent move

Return the single token only, no punctuation, no explanation:
```

## v2

```
<image><image>Task: pick up the banana and place it on the blue plate

BAsed on two camera views:
- Agentview: overhead view of the robot and workspace
- Wristview: close-up view from the gripper

Choose the next action token:
MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN, GRASP, RELEASE, DONE

Return the single token only, no punctuation, no explanation:
```

---

> 注：v0 真实训练 json 已删且未提交，无法逐字节恢复；以上 v0 为从 AgentRobot git `52561b6`（MVTOKEN_v0 planner-free/stage-free 推理代码，runner 确认 feeds task+gripper+recent）还原的 lite 格式，推理与训练一致。`overfit_test` 用的是无 task 版，与 v0 MVTOKEN 不同。v0 此版无 DONE token（仅到 RELEASE），v1/v2 才加了 DONE。
