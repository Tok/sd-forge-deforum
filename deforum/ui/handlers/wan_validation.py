"""Wan generation validation helpers.

Extracted from ui_elements.py to reduce complexity.
"""

import json
from typing import Dict, Tuple, Optional


def _is_empty_prompts(prompts: str) -> bool:
    """Check if prompts are empty or whitespace."""
    return not prompts or prompts.strip() == ""


def _has_placeholder_text(prompts: str) -> bool:
    """Check if prompts contain placeholder text."""
    placeholders = ["required:", "load prompts", "placeholder"]
    return any(placeholder in prompts.lower() for placeholder in placeholders)


def _parse_prompts_json(prompts: str) -> Tuple[bool, Optional[Dict[str, str]], str]:
    """Parse prompts as JSON and validate.

    Args:
        prompts: JSON string of prompts

    Returns:
        Tuple of (is_valid, prompts_dict, error_message)
    """
    try:
        prompts_dict = json.loads(prompts)
        if not prompts_dict:
            return False, None, "empty"
        return True, prompts_dict, ""
    except json.JSONDecodeError:
        return False, None, "invalid_json"


def _has_default_prompts(prompts_dict: Dict[str, str]) -> bool:
    """Check if prompts dict contains default/placeholder values."""
    if not prompts_dict:
        return False

    first_prompt = list(prompts_dict.values())[0].lower()
    default_texts = ["prompt text", "beautiful landscape", "load prompts"]
    return any(default_text in first_prompt for default_text in default_texts)


def _build_validation_message(
    status: str,
    prompts_dict: Optional[Dict[str, str]],
    emojis: Dict[str, str]
) -> str:
    """Build validation message based on status.

    Args:
        status: Validation status (empty, placeholder, default, ready, invalid_json)
        prompts_dict: Parsed prompts dictionary
        emojis: Emoji symbols dict

    Returns:
        Formatted validation message
    """
    if status == "empty":
        return f"""{emojis['warning']} **Prompts Required**

{emojis['memo']} **Load prompts to get started:**
• Click "Load from Deforum Prompts" to use your animation prompts
• Or click "Load Default Wan Prompts" for examples
• Then optionally enhance with AI or add movement descriptions"""

    if status == "placeholder":
        return f"""{emojis['warning']} **Load Real Prompts**

{emojis['memo']} **Replace placeholder text:**
• Click "Load from Deforum Prompts" to copy your animation prompts
• Or click "Load Default Wan Prompts" for examples"""

    if status == "default":
        return f"""{emojis['warning']} **Default/Placeholder Prompts Detected**

{emojis['memo']} **Load your real prompts:**
• Click "Load from Deforum Prompts" to copy your animation prompts
• Or edit the prompts manually to describe your desired video"""

    if status == "invalid_json":
        return f"""{emojis['cross']} **Invalid JSON Format**

{emojis['wrench']} **Fix the format:**
• Prompts should be in JSON format like: {{"0": "prompt text", "60": "another prompt"}}
• Check for missing quotes, commas, or brackets"""

    if status == "ready" and prompts_dict:
        num_prompts = len(prompts_dict)
        return f"""{emojis['check']} **Ready to Generate!**

{emojis['movie_camera']} **Found {num_prompts} prompt{'s' if num_prompts != 1 else ''}** for Wan video generation
{emojis['fire']} **Click "Generate Flux/Wan" above** to start I2V chaining generation
{emojis['zap']} **Optional:** Add movement descriptions or AI enhancement first"""

    return f"{emojis['cross']} **Unknown validation status**"
