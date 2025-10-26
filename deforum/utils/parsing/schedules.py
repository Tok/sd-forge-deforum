"""Pure functions for parsing and interpolating Deforum schedule strings.

This module contains schedule parsing functions extracted from
scripts/deforum_helpers/wan/utils/movement_analyzer.py, following functional
programming principles with no side effects.
"""

import re
from typing import List, Tuple


def parse_schedule_string(schedule_str: str, max_frames: int = 100) -> List[Tuple[int, float]]:
    """Parse Deforum schedule string into frame-value keyframe pairs.

    Parses schedule strings in Deforum format "frame:(value), frame:(value)..."
    into a list of (frame, value) tuples for further processing. Handles empty
    strings, single values, and malformed input gracefully.

    Format: "0:(1.0), 30:(2.0), 60:(1.5)"
    Returns: [(0, 1.0), (30, 2.0), (60, 1.5)]

    Args:
        schedule_str: Schedule string in Deforum format
        max_frames: Maximum number of frames (used for validation, not enforced)

    Returns:
        List of (frame_number, value) tuples sorted by frame number.
        Returns [(0, 0.0)] for empty/invalid input.

    Examples:
        >>> parse_schedule_string("0:(1.0), 30:(2.0), 60:(1.5)")
        [(0, 1.0), (30, 2.0), (60, 1.5)]
        >>> parse_schedule_string("0:(5)")
        [(0, 5.0)]
        >>> parse_schedule_string("10")  # Single value
        [(0, 10.0)]
        >>> parse_schedule_string("")  # Empty string
        [(0, 0.0)]
        >>> parse_schedule_string("invalid")
        [(0, 0.0)]
        >>> # Out of order frames get sorted
        >>> parse_schedule_string("30:(2.0), 0:(1.0), 60:(3.0)")
        [(0, 1.0), (30, 2.0), (60, 3.0)]
    """
    if not schedule_str or schedule_str.strip() == "":
        return [(0, 0.0)]

    # Clean the string
    schedule_str = schedule_str.strip()

    # Pattern to match frame:(value) pairs with optional whitespace
    # Allows: "0:(1.0)", "0: (1.0)", "0 :(1.0)", "0 : (1.0)"
    pattern = r"(\d+)\s*:\s*\(([^)]+)\)"
    matches = re.findall(pattern, schedule_str)

    if not matches:
        # Try to parse as single value
        try:
            value = float(schedule_str)
            return [(0, value)]
        except ValueError:
            return [(0, 0.0)]

    # Convert matches to frame-value pairs
    keyframes = []
    for frame_str, value_str in matches:
        try:
            frame = int(frame_str)
            value = float(value_str)
            keyframes.append((frame, value))
        except ValueError:
            continue

    # Sort by frame number
    keyframes.sort(key=lambda x: x[0])

    return keyframes if keyframes else [(0, 0.0)]


def interpolate_schedule_values(keyframes: List[Tuple[int, float]], max_frames: int) -> List[float]:
    """Interpolate schedule values linearly across all frames.

    Takes keyframe pairs and generates interpolated values for every frame
    from 0 to max_frames-1 using linear interpolation between keyframes.
    Values before first keyframe use first value, values after last keyframe
    use last value.

    Args:
        keyframes: List of (frame, value) tuples (should be sorted by frame)
        max_frames: Number of frames to generate values for

    Returns:
        List of interpolated values, one per frame (length = max_frames)

    Examples:
        >>> # Linear interpolation between two keyframes
        >>> interpolate_schedule_values([(0, 0.0), (10, 10.0)], 11)
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        >>> # Single keyframe - constant value
        >>> interpolate_schedule_values([(0, 5.0)], 5)
        [5.0, 5.0, 5.0, 5.0, 5.0]

        >>> # Empty keyframes - returns zeros
        >>> interpolate_schedule_values([], 3)
        [0.0, 0.0, 0.0]

        >>> # Multiple keyframes with different slopes
        >>> interpolate_schedule_values([(0, 0.0), (5, 10.0), (10, 5.0)], 11)
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0]

        >>> # Keyframe beyond max_frames - holds last value
        >>> interpolate_schedule_values([(0, 1.0), (20, 5.0)], 10)
        [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
    """
    if not keyframes:
        return [0.0] * max_frames

    values = []

    for frame in range(max_frames):
        # Find surrounding keyframes
        prev_kf = None
        next_kf = None

        for kf in keyframes:
            if kf[0] <= frame:
                prev_kf = kf
            if kf[0] >= frame and next_kf is None:
                next_kf = kf

        if prev_kf is None:
            # Before first keyframe - use first value
            values.append(keyframes[0][1])
        elif next_kf is None:
            # After last keyframe - use last value
            values.append(prev_kf[1])
        elif prev_kf[0] == next_kf[0]:
            # Exactly on a keyframe
            values.append(prev_kf[1])
        else:
            # Linear interpolation between keyframes
            t = (frame - prev_kf[0]) / (next_kf[0] - prev_kf[0])
            value = prev_kf[1] + t * (next_kf[1] - prev_kf[1])
            values.append(value)

    return values
