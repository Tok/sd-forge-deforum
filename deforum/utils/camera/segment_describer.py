"""Camera Segment Description Generator - Refactored

Refactored from generate_segment_description (75 lines, complexity D-21)
into modular pure functions using lookup tables instead of if-elif chains.

Original: deforum/utils/camera/analysis.py:118-192
"""

from typing import Dict, Tuple


# ============================================================================
# Lookup Tables (Replace if-elif chains)
# ============================================================================

# Movement type -> (increasing_motion, decreasing_motion)
MOVEMENT_DESCRIPTIONS = {
    "translation_x": ("panning right", "panning left"),
    "translation_y": ("moving up", "moving down"),
    "translation_z": ("dolly forward", "dolly backward"),
    "rotation_x": ("tilting up", "tilting down"),
    "rotation_y": ("rotating right", "rotating left"),
    "rotation_z": ("rolling clockwise", "rolling counter-clockwise"),
    "zoom": ("zooming in", "zooming out"),
}


# ============================================================================
# Frame Range Classification
# ============================================================================

def _get_segment_duration_ratio(start: int, end: int, total_frames: int) -> float:
    """Calculate segment duration as ratio of total frames."""
    return (end - start) / total_frames if total_frames > 0 else 0.0


def classify_frame_range(start: int, end: int, total_frames: int) -> str:
    """Classify frame range as brief/moderate/extended.

    Args:
        start: Starting frame
        end: Ending frame
        total_frames: Total frames in animation

    Returns:
        Description string like "frames 10-25 (brief)"
    """
    duration = end - start

    if duration < 10:
        return f"frames {start}-{end}"

    ratio = _get_segment_duration_ratio(start, end, total_frames)

    if ratio < 0.3:
        return f"frames {start}-{end} (brief)"
    elif ratio < 0.7:
        return f"frames {start}-{end} (moderate)"
    else:
        return f"frames {start}-{end} (extended)"


# ============================================================================
# Movement Intensity Classification
# ============================================================================

def classify_intensity(total_range: float) -> str:
    """Classify movement intensity based on total range.

    Args:
        total_range: Total range of movement

    Returns:
        Intensity descriptor: subtle/gentle/moderate/strong
    """
    if total_range < 1.0:
        return "subtle"
    elif total_range < 5.0:
        return "gentle"
    elif total_range < 20.0:
        return "moderate"
    else:
        return "strong"


# ============================================================================
# Movement Type Description
# ============================================================================

def get_movement_description(movement_type: str, direction: str) -> str:
    """Get human-readable movement description.

    Args:
        movement_type: Type of movement (translation_x, rotation_y, etc.)
        direction: Direction of movement (increasing/decreasing)

    Returns:
        Movement description like "panning left"
    """
    if movement_type not in MOVEMENT_DESCRIPTIONS:
        return f"{movement_type} {direction}"

    increasing_desc, decreasing_desc = MOVEMENT_DESCRIPTIONS[movement_type]
    return increasing_desc if direction == "increasing" else decreasing_desc


# ============================================================================
# Main Function
# ============================================================================

def generate_segment_description(segment: Dict, total_frames: int) -> str:
    """Generate specific description for a movement segment.

    Refactored version with complexity ≤10 using lookup tables.

    Creates a human-readable description of a camera movement segment,
    including intensity, direction, and frame range information.

    Args:
        segment: Movement segment dict with keys: start_frame, end_frame,
                movement_type, direction, total_range
        total_frames: Total number of frames in animation

    Returns:
        Descriptive string like "subtle panning left (frames 10-25)"

    Examples:
        >>> segment = {
        ...     'start_frame': 10,
        ...     'end_frame': 25,
        ...     'movement_type': 'translation_x',
        ...     'direction': 'decreasing',
        ...     'total_range': 0.5
        ... }
        >>> desc = generate_segment_description(segment, 100)
        >>> 'panning left' in desc
        True
    """
    # Extract segment data
    start = segment["start_frame"]
    end = segment["end_frame"]
    movement_type = segment["movement_type"]
    direction = segment["direction"]
    total_range = segment["total_range"]

    # Generate components
    frame_desc = classify_frame_range(start, end, total_frames)
    intensity = classify_intensity(total_range)
    motion = get_movement_description(movement_type, direction)

    return f"{intensity} {motion} ({frame_desc})"
