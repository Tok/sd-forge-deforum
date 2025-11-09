"""Pure functions for schedule string manipulation and combination.

This module contains functions for combining and manipulating Deforum schedule
strings, following functional programming principles with no side effects.
"""

from typing import List, Dict, Tuple, Optional

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


def get_shake_values_from_production(
    shake_name: str,
    shake_intensity: float,
    shake_speed: float,
    max_frames: int,
    target_fps: int = 60,
) -> Dict[str, List[float]]:
    """Get shake values using production Shaker class.

    This uses the EXACT same code path as production rendering to ensure
    visualizations match actual output perfectly ("sim as close to prod as possible").

    Args:
        shake_name: Name of shake pattern (e.g., "GENTLE_HANDHELD")
        shake_intensity: Intensity multiplier (1.0 = normal)
        shake_speed: Speed multiplier (1.0 = normal)
        max_frames: Number of frames to generate
        target_fps: Target FPS for the animation

    Returns:
        Dict with keys 'translation_x', 'translation_y', 'translation_z',
        'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z', each containing
        a list of shake values (one per frame)
    """
    from dataclasses import dataclass
    from deforum.rendering.data.shakify.shaker import Shaker

    # Return zeros if shakify disabled
    if not shake_name or shake_name == "None" or shake_name == "":
        return {
            "translation_x": [0.0] * max_frames,
            "translation_y": [0.0] * max_frames,
            "translation_z": [0.0] * max_frames,
            "rotation_3d_x": [0.0] * max_frames,
            "rotation_3d_y": [0.0] * max_frames,
            "rotation_3d_z": [0.0] * max_frames,
        }

    # Create minimal mock objects that Shaker.create() needs
    @dataclass
    class MockVideoArgs:
        fps: int = target_fps

    @dataclass
    class MockAnimArgs:
        shake_name: str = shake_name
        shake_intensity: float = shake_intensity
        shake_speed: float = shake_speed

    @dataclass
    class MockRenderInitArgs:
        video_args: MockVideoArgs
        anim_args: MockAnimArgs

    @dataclass
    class MockRenderData:
        args: MockRenderInitArgs

        def fps(self):
            return self.args.video_args.fps

    # Create mock with required attributes
    mock_data = MockRenderData(
        args=MockRenderInitArgs(
            video_args=MockVideoArgs(fps=target_fps),
            anim_args=MockAnimArgs(
                shake_name=shake_name, shake_intensity=shake_intensity, shake_speed=shake_speed
            ),
        )
    )

    # Use production Shaker class (EXACT same code as rendering)
    shaker = Shaker.create(mock_data)

    if not shaker.is_enabled:
        return {
            "translation_x": [0.0] * max_frames,
            "translation_y": [0.0] * max_frames,
            "translation_z": [0.0] * max_frames,
            "rotation_3d_x": [0.0] * max_frames,
            "rotation_3d_y": [0.0] * max_frames,
            "rotation_3d_z": [0.0] * max_frames,
        }

    # Extract shake values using production get_data() method
    return {
        "translation_x": [shaker.get_data("translation", "x", i) for i in range(max_frames)],
        "translation_y": [shaker.get_data("translation", "y", i) for i in range(max_frames)],
        "translation_z": [shaker.get_data("translation", "z", i) for i in range(max_frames)],
        "rotation_3d_x": [shaker.get_data("rotation_3d", "x", i) for i in range(max_frames)],
        "rotation_3d_y": [shaker.get_data("rotation_3d", "y", i) for i in range(max_frames)],
        "rotation_3d_z": [shaker.get_data("rotation_3d", "z", i) for i in range(max_frames)],
    }


def get_final_schedules_with_shakify(
    base_schedules: Dict[str, str],
    shake_name: str,
    shake_intensity: float,
    shake_speed: float,
    max_frames: int,
    target_fps: int = 60,
) -> Dict[str, str]:
    """Get final combined schedules (base + shakify overlay).

    This is the centralized function for combining base camera movement
    with shakify overlay, used by both visualizers.

    Uses production Shaker class to ensure visualizations match actual output exactly.

    Args:
        base_schedules: Dict with keys 'translation_x', 'translation_y', etc.
        shake_name: Shakify pattern name (or "None" to disable)
        shake_intensity: Intensity multiplier
        shake_speed: Speed multiplier
        max_frames: Number of frames
        target_fps: Target FPS

    Returns:
        Dict of final combined schedule strings
    """
    # Get shake values using production Shaker class
    shake_values = get_shake_values_from_production(
        shake_name, shake_intensity, shake_speed, max_frames, target_fps
    )

    # Apply shake to each schedule
    final_schedules = {}
    for axis in [
        "translation_x",
        "translation_y",
        "translation_z",
        "rotation_3d_x",
        "rotation_3d_y",
        "rotation_3d_z",
    ]:
        base_schedule = base_schedules.get(axis, "0:(0)")
        shake_for_axis = shake_values.get(axis, [])

        final_schedules[axis] = apply_shakify_to_schedule(base_schedule, shake_for_axis, max_frames)

    return final_schedules
