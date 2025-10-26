"""Pure functions for keyframe analysis and type assignment.

This module contains pure functions for analyzing keyframe distributions
and automatically assigning interpolation types based on distances.

Following Phase 2 of REFACTORING_STRATEGY.md:
- Pure functions only (no I/O, no state mutation)
- Full type hints
- Complexity ≤ 10
- Extracted from ui_elements.py auto_assign_keyframe_types_handler
"""

import re
from typing import Tuple


THRESHOLD_RATIO = 0.8  # 80% of chunk_size for flf2v vs tween decision
DEFAULT_TYPE = "tween"
SHORT_SECTION_TYPE = "flf2v"


def extract_frame_numbers(prompts: dict[str, str]) -> list[int]:
    """Extract and sort numeric frame numbers from prompts dictionary.

    Handles both pure numeric keys and expressions like "max_f-2".

    Args:
        prompts: Dictionary with frame numbers as keys (strings)

    Returns:
        Sorted list of frame numbers (integers)
    """
    frame_numbers = []

    for key in prompts.keys():
        if key.isdigit():
            frame_numbers.append(int(key))
        elif re.match(r"^\d+$", str(key)):
            frame_numbers.append(int(key))

    return sorted(frame_numbers)


def calculate_distance_threshold(chunk_size: int) -> int:
    """Calculate threshold for flf2v vs tween decision.

    Args:
        chunk_size: Maximum chunk size for interpolation

    Returns:
        Threshold value (80% of chunk_size, as integer)
    """
    return int(chunk_size * THRESHOLD_RATIO)


def suggest_keyframe_type(distance: int, threshold: int) -> str:
    """Suggest keyframe interpolation type based on distance.

    Logic:
    - Short sections (<= threshold): Use "flf2v" (AI video generation)
    - Long sections (> threshold): Use "tween" (simple interpolation)

    Args:
        distance: Distance in frames to previous keyframe
        threshold: Threshold from calculate_distance_threshold()

    Returns:
        Either "flf2v" or "tween"
    """
    return SHORT_SECTION_TYPE if distance <= threshold else DEFAULT_TYPE


def build_keyframe_type_schedule(frame_numbers: list[int], threshold: int) -> list[Tuple[int, str]]:
    """Build keyframe type schedule for all frames.

    Args:
        frame_numbers: Sorted list of frame numbers
        threshold: Distance threshold for type decision

    Returns:
        List of (frame_number, type) tuples
    """
    if not frame_numbers:
        return [(0, DEFAULT_TYPE)]

    schedule = []

    for i, frame in enumerate(frame_numbers):
        if i == 0:
            # First keyframe always starts with tween
            schedule.append((frame, DEFAULT_TYPE))
        else:
            # Calculate distance to previous keyframe
            distance = frame - frame_numbers[i - 1]
            keyframe_type = suggest_keyframe_type(distance, threshold)
            schedule.append((frame, keyframe_type))

    return schedule


def format_keyframe_schedule(schedule: list[Tuple[int, str]]) -> str:
    """Format keyframe schedule as string.

    Args:
        schedule: List of (frame_number, type) tuples

    Returns:
        Formatted string like "0:(tween), 60:(flf2v), 120:(tween)"
    """
    parts = [f"{frame}:({kf_type})" for frame, kf_type in schedule]
    return ", ".join(parts)


def auto_assign_keyframe_types(
    prompts: dict[str, str], chunk_size: int
) -> Tuple[str, list[Tuple[int, str]]]:
    """Auto-assign keyframe types based on tween distances.

    Complete workflow combining all steps.

    Args:
        prompts: Dictionary mapping frame numbers to prompt text
        chunk_size: Maximum chunk size for interpolation

    Returns:
        Tuple of (formatted_schedule, schedule_list)
        - formatted_schedule: String like "0:(tween), 60:(flf2v)"
        - schedule_list: List of (frame, type) tuples for inspection
    """
    frame_numbers = extract_frame_numbers(prompts)

    if not frame_numbers:
        return "0:(tween)", [(0, DEFAULT_TYPE)]

    threshold = calculate_distance_threshold(chunk_size)
    schedule = build_keyframe_type_schedule(frame_numbers, threshold)
    formatted = format_keyframe_schedule(schedule)

    return formatted, schedule
