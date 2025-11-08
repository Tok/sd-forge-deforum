"""Pure functions for frame interpolation calculations.

This module contains interpolation-related pure functions extracted from
scripts/deforum_helpers/frame_interpolation.py and general_utils.py, following
functional programming principles with no side effects.
"""


def extract_rife_name(model_string: str) -> str:
    """Extract RIFE model folder name from display string.

    Converts display format (e.g., "RIFE v4.26") to folder format (e.g., "RIFE426").
    Used for locating RIFE model directories.

    Args:
        model_string: RIFE model display string in format "RIFE v{major}.{minor}"

    Returns:
        RIFE model folder name with version numbers concatenated (e.g., "RIFE426")

    Raises:
        ValueError: If input doesn't match expected "RIFE v{number}.{number}" format

    Examples:
        >>> extract_rife_name("RIFE v4.26")
        'RIFE426'
        >>> extract_rife_name("RIFE v2.0")
        'RIFE20'
        >>> extract_rife_name("RIFE v10.15")
        'RIFE1015'
    """
    parts = model_string.split()

    # Validate format: exactly 2 words
    if len(parts) != 2:
        raise ValueError(
            "Input string should contain exactly 2 words: 'RIFE' and version"
        )

    # Validate first word
    if parts[0] != "RIFE":
        raise ValueError("First word should be 'RIFE'")

    # Validate version format: starts with 'v' followed by numbers and dots
    version_part = parts[1]
    if not version_part.startswith("v"):
        raise ValueError("Version should start with 'v'")

    # Extract and validate version numbers
    version_numbers = version_part[1:]  # Remove 'v' prefix
    if not version_numbers.replace(".", "").isdigit():
        raise ValueError("Version should contain only digits and dots after 'v'")

    # Build folder name: "RIFE" + version without dots
    return "RIFE" + version_numbers.replace(".", "")


def clean_folder_name(filename: str) -> str:
    """Convert filename to legal Linux/Windows folder name.

    Replaces illegal filesystem characters with underscores to create
    a safe folder name from any input string.

    Args:
        filename: Original filename or string to clean

    Returns:
        Cleaned string safe for use as folder name on all platforms

    Examples:
        >>> clean_folder_name("my_video.mp4")
        'my_video_mp4'
        >>> clean_folder_name("test/file:name.txt")
        'test_file_name_txt'
        >>> clean_folder_name("anime<girl>")
        'anime_girl_'
    """
    # Characters illegal in Windows/Linux folder names
    illegal_chars = '/\\<>:"|?*.,\' '

    # Create translation table: each illegal char → underscore
    translation_table = str.maketrans(illegal_chars, "_" * len(illegal_chars))

    return filename.translate(translation_table)


def set_interp_out_fps(
    interp_x: str | int,
    slow_x_enabled: bool,
    slow_x: str | int,
    in_vid_fps: str | float | None,
) -> str | float:
    """Calculate output FPS after interpolation and optional slow-motion.

    Computes the final FPS by:
    1. Multiplying input FPS by interpolation factor (interp_x)
    2. Dividing by slow-motion factor if enabled

    Args:
        interp_x: Interpolation multiplier ('Disabled', '2', '3', etc., or int)
        slow_x_enabled: Whether slow-motion is enabled
        slow_x: Slow-motion divisor (string or int)
        in_vid_fps: Input video FPS (string, float, or None)

    Returns:
        Output FPS as float or int if whole number, or '---' if disabled/invalid

    Examples:
        >>> set_interp_out_fps('2', False, '1', 30.0)
        60.0
        >>> set_interp_out_fps('4', True, '2', 30.0)
        60.0
        >>> set_interp_out_fps('Disabled', False, '1', 30.0)
        '---'
        >>> set_interp_out_fps('3', False, '1', 24.0)
        72.0
    """
    # Return disabled marker if interpolation is disabled or FPS is invalid
    if interp_x == "Disabled" or in_vid_fps in ("---", None, "", "None"):
        return "---"

    # Calculate output FPS: input_fps * interpolation_factor
    fps = float(in_vid_fps) * int(interp_x)

    # Apply slow-motion divisor if enabled
    if slow_x_enabled:
        fps /= int(slow_x)

    # Return as int if whole number, otherwise as float
    return int(fps) if fps.is_integer() else fps


def calculate_frames_to_add(total_frames: int, interp_x: int) -> int:
    """Calculate number of interpolated frames to add between each frame pair.

    Used by FILM interpolation to determine how many intermediate frames
    to generate between consecutive frames to achieve the desired
    interpolation factor.

    Formula: (total_frames * interp_x - total_frames) / (total_frames - 1)

    Args:
        total_frames: Total number of input frames
        interp_x: Interpolation multiplier (2x, 3x, etc.)

    Returns:
        Number of frames to add between each consecutive frame pair

    Examples:
        >>> calculate_frames_to_add(10, 2)
        1
        >>> calculate_frames_to_add(10, 3)
        2
        >>> calculate_frames_to_add(5, 4)
        4
    """
    frames_to_add = (total_frames * interp_x - total_frames) / (total_frames - 1)
    return int(round(frames_to_add))
