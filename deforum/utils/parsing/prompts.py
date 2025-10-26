"""Pure functions for prompt parsing and transformation.

This module contains pure functions for cleaning, converting, and formatting
prompts between different formats (Deforum, Wan, etc.).

Following Phase 2 of REFACTORING_STRATEGY.md:
- Pure functions only (no I/O, no state mutation)
- Full type hints
- Complexity ≤ 10
- Extracted from ui_elements.py prompt handlers
"""

import json
from typing import Tuple


def remove_negative_prompt(prompt: str) -> str:
    """Remove negative prompt section from a prompt string.

    Negative prompts are separated by '--neg' marker.

    Args:
        prompt: Full prompt text (may include negative prompt)

    Returns:
        Cleaned prompt without negative section
    """
    return prompt.split('--neg')[0].strip()


def convert_deforum_to_wan_prompts(
    deforum_prompts: dict[str, str]
) -> dict[str, str]:
    """Convert Deforum prompts to Wan format.

    Wan format uses clean prompts without negative sections.

    Args:
        deforum_prompts: Dictionary mapping frames to Deforum prompts

    Returns:
        Dictionary with cleaned prompts for Wan
    """
    wan_prompts = {}

    for frame, prompt in deforum_prompts.items():
        clean_prompt = remove_negative_prompt(prompt)
        wan_prompts[frame] = clean_prompt

    return wan_prompts


def format_prompts_as_multiline(prompts: dict[str, str]) -> str:
    """Format prompts dictionary as multiline string.

    Format: "frame: prompt text"

    Args:
        prompts: Dictionary mapping frames to prompts

    Returns:
        Multiline string with one prompt per line, sorted numerically by frame
    """
    if not prompts:
        return "0: "

    # Sort by numeric frame value, not alphabetically
    sorted_items = sorted(
        prompts.items(),
        key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')
    )

    lines = [f"{frame}: {prompt}" for frame, prompt in sorted_items]
    return "\n".join(lines)


def format_prompts_as_json(
    prompts: dict[str, str],
    indent: int = 2
) -> str:
    """Format prompts dictionary as JSON string.

    Args:
        prompts: Dictionary mapping frames to prompts
        indent: JSON indentation (default: 2)

    Returns:
        Pretty-printed JSON string
    """
    return json.dumps(prompts, ensure_ascii=False, indent=indent)


def create_error_prompt(error_message: str) -> str:
    """Create a prompt dictionary with an error message at frame 0.

    Args:
        error_message: Error message to include

    Returns:
        JSON string with error prompt
    """
    return json.dumps(
        {"0": error_message},
        ensure_ascii=False,
        indent=2
    )


def parse_prompts_json(
    prompts_json: str,
    default_on_error: dict[str, str] | None = None
) -> Tuple[dict[str, str], str | None]:
    """Parse JSON prompts string with error handling.

    Args:
        prompts_json: JSON string to parse
        default_on_error: Default dict to return on error (optional)

    Returns:
        Tuple of (parsed_dict, error_message)
        - parsed_dict: Parsed dictionary or default
        - error_message: Error description or None if successful
    """
    try:
        prompts = json.loads(prompts_json)

        if not isinstance(prompts, dict):
            return (
                default_on_error or {},
                "Prompts must be a JSON object/dictionary"
            )

        return prompts, None

    except json.JSONDecodeError as e:
        return (
            default_on_error or {},
            f"Invalid JSON: {str(e)}"
        )


def validate_prompts_not_empty(
    prompts_json: str
) -> Tuple[bool, str]:
    """Validate that prompts JSON is not empty.

    Args:
        prompts_json: JSON string to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if valid
        - error_message: Empty if valid, error description otherwise
    """
    if not prompts_json or prompts_json.strip() == "":
        return False, "No prompts found! Configure prompts first."

    return True, ""


def create_fallback_prompts() -> dict[str, str]:
    """Create fallback prompts dictionary.

    Returns:
        Default prompts dictionary with placeholder text
    """
    return {
        "0": "prompt text",
        "60": "another prompt"
    }
