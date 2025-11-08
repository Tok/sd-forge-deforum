"""Pure functions for FPS conversion of prompt frame numbers.

This module contains pure functions for converting prompt keyframe timings
between different FPS rates. All functions here are side-effect free.

Following Phase 2 of REFACTORING_STRATEGY.md:
- Pure functions only (no I/O, no state mutation)
- Full type hints
- Complexity ≤ 10
- Extracted from ui_elements.py convert_fps_handler
"""

from typing import Tuple


def calculate_fps_ratio(source_fps: float, target_fps: float) -> float:
    """Calculate the FPS conversion ratio.

    Args:
        source_fps: Original FPS
        target_fps: Desired FPS

    Returns:
        Ratio to multiply frame numbers by
    """
    return target_fps / source_fps


def convert_frame_number(frame: int, fps_ratio: float) -> int:
    """Convert a single frame number using FPS ratio.

    Uses the shakify formula: new_frame = old_frame * (target_fps / source_fps)

    Args:
        frame: Original frame number
        fps_ratio: Conversion ratio from calculate_fps_ratio()

    Returns:
        Converted frame number (rounded to int)
    """
    return int(frame * fps_ratio)


def convert_prompts_dict(
    prompts: dict[str, str], fps_ratio: float
) -> Tuple[dict[str, str], list[str]]:
    """Convert all frame numbers in a prompts dictionary.

    Args:
        prompts: Dictionary mapping frame numbers (as strings) to prompt text
        fps_ratio: Conversion ratio from calculate_fps_ratio()

    Returns:
        Tuple of (converted_prompts, conversion_log)
        - converted_prompts: New dict with converted frame numbers
        - conversion_log: List of "Frame X → Y" strings for display
    """
    converted = {}
    log = []

    for frame_str, prompt_text in prompts.items():
        try:
            old_frame = int(frame_str)
            new_frame = convert_frame_number(old_frame, fps_ratio)

            converted[str(new_frame)] = prompt_text
            log.append(f"Frame {old_frame} → {new_frame}")

        except ValueError:
            # Non-numeric key, keep as-is
            converted[frame_str] = prompt_text

    return converted, log


def validate_fps_values(source_fps: float, target_fps: float) -> Tuple[bool, str]:
    """Validate FPS values for conversion.

    Args:
        source_fps: Original FPS
        target_fps: Desired FPS

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if values are valid
        - error_message: Empty string if valid, error description otherwise
    """
    if source_fps <= 0 or target_fps <= 0:
        return False, "FPS values must be positive"

    if source_fps == target_fps:
        return False, "Source and target FPS are the same - no conversion needed"

    return True, ""


def build_conversion_status(
    source_fps: float,
    target_fps: float,
    fps_ratio: float,
    conversion_log: list[str],
    preview_only: bool,
    max_entries: int = 10,
) -> str:
    """Build HTML status message for FPS conversion.

    Args:
        source_fps: Original FPS
        target_fps: Desired FPS
        fps_ratio: Conversion ratio used
        conversion_log: List of conversion entries
        preview_only: Whether this was a preview
        max_entries: Maximum log entries to show (default: 10)

    Returns:
        HTML-formatted status message
    """
    from deforum.utils.system.logging import emoji as emoji_utils

    # Theme-aware emoji symbols
    check = emoji_utils.maybe_check()
    magnifying_glass = emoji_utils.magnifying_glass()
    pencil = emoji_utils.pencil()

    result = []
    result.append(
        f"{check} <span style='color: #4CAF50;'><strong>FPS Conversion Complete</strong></span><br>"
    )
    result.append(
        f"<strong>Source FPS:</strong> {source_fps} → <strong>Target FPS:</strong> {target_fps}<br>"
    )
    result.append(f"<strong>Conversion Ratio:</strong> {fps_ratio:.4f}<br>")
    result.append(f"<strong>Prompts Converted:</strong> {len(conversion_log)}<br><br>")

    if preview_only:
        result.append(
            f"{magnifying_glass} <strong style='color: #FF9800;'>PREVIEW MODE</strong> - Prompts not updated<br><br>"
        )
    else:
        result.append(f"{pencil} <strong style='color: #4CAF50;'>Prompts Updated</strong><br><br>")

    # Show conversion table (limited entries)
    result.append("<strong>Frame Conversion:</strong><br>")
    code_style = (
        "display: block; background: #f5f5f5; padding: 8px; "
        "margin: 8px 0; border-radius: 4px;"
    )
    result.append(f"<code style='{code_style}'>")

    for entry in conversion_log[:max_entries]:
        result.append(f"{entry}<br>")

    if len(conversion_log) > max_entries:
        result.append(f"... and {len(conversion_log) - max_entries} more")

    result.append("</code>")

    # Add formula explanation
    result.append("<br><strong>Formula Used:</strong><br>")
    formula = (
        f"new_frame = old_frame × ({target_fps} / {source_fps}) = "
        f"old_frame × {fps_ratio:.4f}"
    )
    result.append(f"<code>{formula}</code>")

    return "".join(result)
