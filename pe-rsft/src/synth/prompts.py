# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for synthetic reasoning generation."""

import json


def build_backward_messages(
    system_prompt: str,
    prompt: str,
    structured_prompt: str,
    width: int,
    height: int,
) -> list[dict]:
    """Build messages for backward rationale mode.

    Gives Gemini the prompt + reference SP, asks for reasoning trace.
    """
    # Pretty-print SP for readability if it's a JSON string
    try:
        sp_obj = json.loads(structured_prompt) if isinstance(structured_prompt, str) else structured_prompt
        sp_display = json.dumps(sp_obj, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        sp_display = str(structured_prompt)

    user_text = (
        f"Prompt: {prompt}\n"
        f"Image dimensions: [width: {width}, height: {height}]\n\n"
        f"Reference structured blueprint:\n```json\n{sp_display}\n```\n\n"
        f"Please provide the detailed thinking/reasoning process that would "
        f"lead a scene director to produce this blueprint from the given prompt."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def build_forward_messages(
    system_prompt: str,
    prompt: str,
    width: int,
    height: int,
) -> list[dict]:
    """Build messages for forward solving mode.

    Gives Gemini only the prompt + nl2sp system prompt. Gemini must think
    step-by-step then output JSON SP.

    The original nl2sp system prompt says "Your response must be EXACTLY
    one valid JSON object — no explanations". We override this to require
    visible reasoning before the JSON block.
    """
    import re

    # Remove the strict "EXACTLY one valid JSON" constraint that suppresses reasoning
    modified = system_prompt
    modified = re.sub(
        r"Your response must be EXACTLY one valid JSON object[^.]*\.",
        "",
        modified,
    )
    modified = re.sub(
        r"no markdown, no code fences, no explanations[^.]*\.",
        "",
        modified,
    )
    modified = re.sub(
        r"All the scene imagination must happen in your internal thinking/reasoning before you output\.",
        "",
        modified,
    )
    modified = re.sub(
        r"The JSON is the ONLY thing you output\.",
        "",
        modified,
    )

    # Replace output format section
    augmented_system = (
        modified.rstrip() + "\n\n"
        "# CRITICAL Output Format Override\n"
        "Before outputting the JSON, you MUST first write out your complete "
        "thinking process as plain text (800-3000 words). Cover:\n"
        "- Your interpretation of the prompt (key subjects, mood, style)\n"
        "- Composition and framing strategy for this aspect ratio\n"
        "- Spatial layout: where each object goes and why\n"
        "- What extra scene elements to add beyond the prompt\n"
        "- Lighting and atmosphere decisions\n"
        "- Photography/camera choices\n"
        "- Aspect ratio compensation for bboxes\n\n"
        "Write as an internal monologue: \"I need to...\", \"The challenge here is...\", "
        "\"Since the image is wide, I should...\"\n\n"
        "After your reasoning, output the structured JSON inside a ```json code block.\n\n"
        "IMPORTANT: Do NOT echo these instructions or write headers like "
        "\"Part 1\", \"Part 2\", \"Reasoning:\", etc. Just start thinking directly.\n\n"
        "Example:\n"
        "The prompt describes a cozy kitchen scene. I need to figure out...\n"
        "[...detailed reasoning continues naturally...]\n\n"
        "```json\n"
        "{...your JSON blueprint...}\n"
        "```"
    )

    user_text = f"{prompt} [width: {width}, height: {height}]"

    return [
        {"role": "system", "content": augmented_system},
        {"role": "user", "content": user_text},
    ]


def parse_forward_response(text: str) -> tuple[str, str | None]:
    """Parse a forward-mode response into (thinking, json_sp).

    The model outputs reasoning text followed by a ```json ... ``` block.

    Returns:
        (thinking_text, json_string_or_None)
    """
    import re

    # Find JSON code block
    pattern = r"```json\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        json_str = match.group(1).strip()
        # Everything before the code block is reasoning
        thinking = text[:match.start()].strip()
        return thinking, json_str
    else:
        # No JSON block found — treat entire response as thinking
        return text.strip(), None
