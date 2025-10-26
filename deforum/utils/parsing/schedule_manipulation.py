"""Pure functions for schedule string manipulation and combination.

This module contains functions for combining and manipulating Deforum schedule
strings, following functional programming principles with no side effects.
"""

from typing import List

# Import schedule parsing functions
from .schedules import parse_schedule_string, interpolate_schedule_values


def apply_shakify_to_schedule(
    base_schedule: str, shake_values: List[float], max_frames: int
) -> str:
    """Apply Camera Shakify values to base movement schedule to create combined schedule.

    This mimics the _maybe_shake function from the render core by
    additively combining base movement values with shake values.

    Args:
        base_schedule: Base movement schedule string (e.g., "0:(0), 50:(10)")
        shake_values: List of shake values per frame
        max_frames: Maximum number of frames

    Returns:
        Combined schedule string with shake applied

    Examples:
        >>> apply_shakify_to_schedule("0:(0), 100:(10)", [0.1] * 100, 100)
        '0:(0.100000), 5:(0.600000), ..., 95:(9.600000)'
        >>> apply_shakify_to_schedule("0:(5)", [], 100)
        '0:(5)'
        >>> apply_shakify_to_schedule("0:(1), 50:(5)", [0.0] * 100, 100)
        '0:(1.000000), 5:(1.400000), ..., 95:(5.000000)'
    """
    if not shake_values or len(shake_values) == 0:
        return base_schedule

    # Parse base schedule
    base_keyframes = parse_schedule_string(base_schedule, max_frames)
    base_values = interpolate_schedule_values(base_keyframes, max_frames)

    # Apply shake to base values (additive)
    combined_values = []
    for frame in range(min(len(base_values), len(shake_values))):
        combined_value = base_values[frame] + shake_values[frame]
        combined_values.append(combined_value)

    # Create new schedule string from combined values
    # Sample every few frames to keep schedule reasonable
    sample_interval = max(1, max_frames // 20)  # Max 20 keyframes
    keyframes = []

    for frame in range(0, len(combined_values), sample_interval):
        value = combined_values[frame]
        keyframes.append(f"{frame}:({value:.6f})")

    # Always include the last frame
    if (len(combined_values) - 1) % sample_interval != 0:
        last_value = combined_values[-1]
        keyframes.append(f"{len(combined_values)-1}:({last_value:.6f})")

    return ", ".join(keyframes)


def combine_schedules(
    schedule1: str, schedule2: str, max_frames: int, operation: str = "add"
) -> str:
    """Combine two schedule strings using specified operation.

    Args:
        schedule1: First schedule string
        schedule2: Second schedule string
        max_frames: Maximum number of frames
        operation: Operation to perform ("add", "subtract", "multiply", "average")

    Returns:
        Combined schedule string

    Raises:
        ValueError: If operation is not supported

    Examples:
        >>> combine_schedules("0:(1)", "0:(2)", 10, "add")
        '0:(3.000000), 9:(3.000000)'
        >>> combine_schedules("0:(10)", "0:(5)", 10, "subtract")
        '0:(5.000000), 9:(5.000000)'
        >>> combine_schedules("0:(2)", "0:(3)", 10, "multiply")
        '0:(6.000000), 9:(6.000000)'
        >>> combine_schedules("0:(4)", "0:(8)", 10, "average")
        '0:(6.000000), 9:(6.000000)'
    """
    # Parse both schedules
    keyframes1 = parse_schedule_string(schedule1, max_frames)
    keyframes2 = parse_schedule_string(schedule2, max_frames)

    # Interpolate values
    values1 = interpolate_schedule_values(keyframes1, max_frames)
    values2 = interpolate_schedule_values(keyframes2, max_frames)

    # Apply operation
    if operation == "add":
        combined_values = [v1 + v2 for v1, v2 in zip(values1, values2)]
    elif operation == "subtract":
        combined_values = [v1 - v2 for v1, v2 in zip(values1, values2)]
    elif operation == "multiply":
        combined_values = [v1 * v2 for v1, v2 in zip(values1, values2)]
    elif operation == "average":
        combined_values = [(v1 + v2) / 2 for v1, v2 in zip(values1, values2)]
    else:
        raise ValueError(
            f"Unsupported operation '{operation}'. " f"Supported: add, subtract, multiply, average"
        )

    # Create schedule string
    sample_interval = max(1, max_frames // 20)
    keyframes = []

    for frame in range(0, len(combined_values), sample_interval):
        value = combined_values[frame]
        keyframes.append(f"{frame}:({value:.6f})")

    # Always include last frame
    if (len(combined_values) - 1) % sample_interval != 0:
        last_value = combined_values[-1]
        keyframes.append(f"{len(combined_values)-1}:({last_value:.6f})")

    return ", ".join(keyframes)


def scale_schedule(schedule: str, max_frames: int, scale_factor: float) -> str:
    """Scale all values in a schedule by a factor.

    Args:
        schedule: Schedule string to scale
        max_frames: Maximum number of frames
        scale_factor: Factor to multiply all values by

    Returns:
        Scaled schedule string

    Examples:
        >>> scale_schedule("0:(1), 100:(10)", 100, 2.0)
        '0:(2.000000), 5:(2.450000), ..., 95:(19.550000)'
        >>> scale_schedule("0:(10), 50:(20)", 100, 0.5)
        '0:(5.000000), 5:(5.500000), ..., 95:(10.000000)'
        >>> scale_schedule("0:(5)", 100, 0.0)
        '0:(0.000000), 5:(0.000000), ..., 95:(0.000000)'
    """
    # Parse schedule
    keyframes = parse_schedule_string(schedule, max_frames)
    values = interpolate_schedule_values(keyframes, max_frames)

    # Scale values
    scaled_values = [v * scale_factor for v in values]

    # Create schedule string
    sample_interval = max(1, max_frames // 20)
    keyframes = []

    for frame in range(0, len(scaled_values), sample_interval):
        value = scaled_values[frame]
        keyframes.append(f"{frame}:({value:.6f})")

    # Always include last frame
    if (len(scaled_values) - 1) % sample_interval != 0:
        last_value = scaled_values[-1]
        keyframes.append(f"{len(scaled_values)-1}:({last_value:.6f})")

    return ", ".join(keyframes)


def offset_schedule(schedule: str, max_frames: int, offset: float) -> str:
    """Add a constant offset to all values in a schedule.

    Args:
        schedule: Schedule string to offset
        max_frames: Maximum number of frames
        offset: Value to add to all schedule values

    Returns:
        Offset schedule string

    Examples:
        >>> offset_schedule("0:(0), 100:(10)", 100, 5.0)
        '0:(5.000000), 5:(5.500000), ..., 95:(15.000000)'
        >>> offset_schedule("0:(10), 50:(20)", 100, -5.0)
        '0:(5.000000), 5:(5.500000), ..., 95:(15.000000)'
        >>> offset_schedule("0:(0)", 100, 10.0)
        '0:(10.000000), 5:(10.000000), ..., 95:(10.000000)'
    """
    # Parse schedule
    keyframes = parse_schedule_string(schedule, max_frames)
    values = interpolate_schedule_values(keyframes, max_frames)

    # Add offset
    offset_values = [v + offset for v in values]

    # Create schedule string
    sample_interval = max(1, max_frames // 20)
    keyframes = []

    for frame in range(0, len(offset_values), sample_interval):
        value = offset_values[frame]
        keyframes.append(f"{frame}:({value:.6f})")

    # Always include last frame
    if (len(offset_values) - 1) % sample_interval != 0:
        last_value = offset_values[-1]
        keyframes.append(f"{len(offset_values)-1}:({last_value:.6f})")

    return ", ".join(keyframes)
