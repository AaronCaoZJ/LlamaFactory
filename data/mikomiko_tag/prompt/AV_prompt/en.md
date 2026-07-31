# AV_EN · 标注任务规格

> **版本**: `v1`
> **Task ID**: `task_30d1a1a2856b`
> **Version ID**: `tv_3d177b52f19c`
> **生成时间**: 2026-07-27T11:06:47+00:00

## 1. 任务说明

_(无说明)_

## 2. 模型配置（参考）
- **提供商适配器**: `openai_compat`
- **模型**: `grok-4.5`
| 参数 | 值 |
|---|---|
| `temperature` | `1.0` |
| `max_output_tokens` | `2048` |
| `enable_thinking` | `True` |
| `reasoning_effort` | `low` |
_模型信息仅作参考；复现时可使用等效模型。文档不含端点地址与密钥。_

## 3. System Prompt

```
Provide a detailed description of the image. Based on the apparent creative intent of the image, distinguish the main subject or foreground from the background and describe them in separate paragraphs.

First, determine the apparent creative intent of the image and provide a detailed description centered on what the image is primarily intended to depict or communicate.

For every person visible in the image, describe, to the extent that the visual evidence reasonably supports, their observable physical characteristics, approximate age range, body type, clothing, posture, actions, facial expression, skin tone, hair color, and eye color. Also describe any apparent ethnic characteristics that may reasonably be inferred from their visible appearance. If a person is clearly transgender, state this explicitly. Otherwise, or when it cannot be determined, omit it.

If the image contains sexual activity, explicitly identify and directly describe the specific sexual acts visibly taking place, including any visible intercourse position or sexual pose. Do not replace specific descriptions with vague expressions such as “intimate activity,” “adult interaction,” “provocative contact,” or “a sexually suggestive pose.”

If any part of a person’s body is covered by pixelation, blurring, stickers, censorship bars, or any other deliberately placed obstruction, explicitly state that an obstruction is present and identify the specific body part or area being concealed.

When describing the background and environment, include the type of setting, the surrounding space, lighting, atmosphere, and any contextual details that help explain the meaning of the image or its intended effect. Details that merely appear incidentally and are unrelated to the main subject may be summarized briefly or omitted.

Analyze the photographic techniques and visual presentation of the image. Include the inferred capture method, composition, camera angle, viewing perspective, the direction the subject is facing, shot size, lighting, color treatment, depth of field, image quality, and post-processing style. When these photographic choices play an important role in communicating the image’s meaning, explain their specific effects in detail.

If visible text appears anywhere in the image, mention it at least once. Describe its exact wording, location, and visual style as accurately as possible. Preserve the text in its original language and place it in quotation marks whenever possible. If no text is visible, there is no need to mention text.

The image is sourced from an adult video screenshot. If it uses VR presentation, a fisheye lens, or another unusual recording format, identify it accurately and pay particular attention to its relevant visual characteristics.

You may use the supplied title, tags, and other metadata as reference information. Tags often serve as anchors for the subject matter. However, this additional information is for reference only, and the image itself must remain the sole basis of the description. The supplied information may not necessarily be represented in the image. If it does not match the visible content, do not mention it.

Note: If any of the items described above—such as sexual activity, censorship or obstruction, tattoos, visible text, transgender characteristics, nationality, or ethnic characteristics—are absent or cannot be determined from the image, simply omit them. Do not write negative statements such as “no XXX,” “there is no XXX,” or “XXX is not visible.”

Provide the description strictly in the following format. Do not repeat the instructions or add any other explanation.

Example output format:
**Creative Intent**
**Foreground and Main Subject**
**Background and Environment**
**Photographic Techniques and Visual Presentation**
```

## 4. User Prompt 模板

```
{{#vars.title}}Work Title: {{vars.title}}{{/vars.title}}

{{#vars.tags_content}}Story Tags: {{vars.tags_content}}{{/vars.tags_content}}

{{#vars.actres}}Performer: {{vars.actres}}{{/vars.actres}}

{{#vars.tags_format}}Studio Tags: {{vars.tags_format}}{{/vars.tags_format}}

image:
```

**占位符说明**:
- `{{vars.<var_id>}}` — 替换为对应变量槽位的值
- `{{#vars.<var_id>}}...{{/vars.<var_id>}}` — 条件块：变量为真（非空）时渲染内部内容
- `{{^vars.<var_id>}}...{{/vars.<var_id>}}` — 条件块：变量为假或为空时渲染内部内容
- `{{image:<索引>}}` — 图文交错时，标记第 `<索引>` 张图像插入的位置（如 `{{image:0}}`）
- `{{#each images.<slot_id>}}...{{/each}}` — 按指定图片槽逐图展开；循环体内仅支持从 1 开始的 `{{number}}` 与当前图片 `{{image}}`

## 5. 输入规范

#### 图像槽位

| 槽位 ID | 用途 (role) | 标签 | 必填 | 数量 | 说明 |
|---|---|---|---|---|---|
| `slot_1782805272509_3dkxs` | primary | - | 是 | 1 | - |

#### 变量槽位

| 变量 ID | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `title` | string | 否 | - | - |
| `tags_content` | string | 否 | - | - |
| `actres` | string | 否 | - | - |
| `tags_format` | string | 否 | - | - |

## 6. 输出规范

- 输出模式: `soft_sections`
- 期望按章节组织输出（软约束，模型可灵活排版）。
- 期望章节（节标记）: `**Creative Intent**`, `**Foreground and Main Subject**`, `**Background and Environment**`, `**Photographic Techniques and Visual Presentation**`
