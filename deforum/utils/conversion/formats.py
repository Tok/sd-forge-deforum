"""Pure functions for formatting and conversion utilities.

This module contains formatting-related pure functions extracted from
scripts/deforum_helpers/deprecation_utils.py and other modules, following
functional programming principles with no side effects.
"""


def format_value_to_schedule(value: int | float) -> str:
    """Convert a numeric value to schedule format '0:(value)'.

    Used for converting deprecated single numeric values to the new schedule
    string format used in animation parameters.

    Args:
        value: Numeric value to format (int or float)

    Returns:
        Formatted schedule string in the format '0:(value)'

    Examples:
        >>> format_value_to_schedule(0.5)
        '0:(0.5)'
        >>> format_value_to_schedule(1)
        '0:(1)'
        >>> format_value_to_schedule(3.14159)
        '0:(3.14159)'
    """
    return f"0:({value})"


def format_frame_time(frame_idx: int, fps: float) -> str:
    """Format frame index and FPS into timestamp string.

    Args:
        frame_idx: Frame index (0-based)
        fps: Frames per second

    Returns:
        Formatted time string in format 'MM:SS.mmm'

    Examples:
        >>> format_frame_time(0, 30)
        '00:00.000'
        >>> format_frame_time(30, 30)
        '00:01.000'
        >>> format_frame_time(90, 30)
        '00:03.000'
    """
    total_seconds = frame_idx / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds % 1) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def format_bytes(bytes_value: int) -> str:
    """Format byte count into human-readable string with units.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string with appropriate unit (B, KB, MB, GB, TB)

    Examples:
        >>> format_bytes(512)
        '512.0 B'
        >>> format_bytes(1024)
        '1.0 KB'
        >>> format_bytes(1048576)
        '1.0 MB'
        >>> format_bytes(1073741824)
        '1.0 GB'
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format float value as percentage string.

    Args:
        value: Float value between 0.0 and 1.0
        decimal_places: Number of decimal places to show (default: 1)

    Returns:
        Formatted percentage string with '%' sign

    Examples:
        >>> format_percentage(0.5)
        '50.0%'
        >>> format_percentage(0.755)
        '75.5%'
        >>> format_percentage(0.755, 2)
        '75.50%'
        >>> format_percentage(1.0)
        '100.0%'
    """
    return f"{value * 100:.{decimal_places}f}%"


def format_resolution(width: int, height: int) -> str:
    """Format width and height into resolution string.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Formatted resolution string 'WIDTHxHEIGHT'

    Examples:
        >>> format_resolution(1920, 1080)
        '1920x1080'
        >>> format_resolution(512, 512)
        '512x512'
        >>> format_resolution(3840, 2160)
        '3840x2160'
    """
    return f"{width}x{height}"
