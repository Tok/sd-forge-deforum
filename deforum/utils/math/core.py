"""Pure functions for mathematical calculations and 3D transformations.

This module contains math-related pure functions extracted from
scripts/deforum_helpers/auto_navigation.py and wan/utils/qwen_vl_utils.py,
following functional programming principles with no side effects.
"""

import math
import numpy as np
import torch
from typing import Tuple

# Constants from qwen_vl_utils
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def rotation_matrix(axis: np.ndarray | list, angle: float) -> np.ndarray:
    """Generate 3D rotation matrix using Rodrigues' rotation formula.

    Args:
        axis: 3D rotation axis vector (will be normalized)
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix as numpy array

    Examples:
        >>> import numpy as np
        >>> axis = [0, 0, 1]  # Z-axis
        >>> angle = np.pi / 2  # 90 degrees
        >>> mat = rotation_matrix(axis, angle)
        >>> mat.shape
        (3, 3)
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def rotate_camera_towards_depth(
    depth_tensor: torch.Tensor,
    turn_weight: float,
    width: int,
    height: int,
    h_fov: float = 60,
    target_depth: float = 1,
) -> torch.Tensor:
    """Calculate camera rotation towards maximum depth point.

    Computes a rotation matrix to turn the camera towards the point of maximum
    depth in the scene, useful for auto-navigation in 3D animations.

    Args:
        depth_tensor: Depth map tensor (height x width)
        turn_weight: Number of frames to spread rotation over (higher = slower turn)
        width: Image width in pixels
        height: Image height in pixels
        h_fov: Horizontal field of view in degrees (default: 60)
        target_depth: Target depth as fraction of depth map height (0-1, default: 1)

    Returns:
        Rotation matrix tensor of shape (1, 3, 3)

    Examples:
        >>> import torch
        >>> depth = torch.rand(480, 640)
        >>> rot = rotate_camera_towards_depth(depth, 10, 640, 480)
        >>> rot.shape
        torch.Size([1, 3, 3])
    """
    # Compute the depth at the target depth
    target_depth_index = int(target_depth * depth_tensor.shape[0])
    # Clamp to valid range (0 to height-1)
    target_depth_index = min(target_depth_index, depth_tensor.shape[0] - 1)
    target_depth_values = depth_tensor[target_depth_index]
    max_depth_index = torch.argmax(target_depth_values).item()
    max_depth_index = (max_depth_index, target_depth_index)
    max_depth = target_depth_values[max_depth_index[0]].item()

    # Compute the normalized x and y coordinates
    x, y = max_depth_index
    x_normalized = (x / (width - 1)) * 2 - 1
    y_normalized = (y / (height - 1)) * 2 - 1

    # Calculate horizontal and vertical field of view (in radians)
    h_fov_rad = np.radians(h_fov)
    aspect_ratio = width / height
    v_fov_rad = h_fov_rad / aspect_ratio

    # Calculate the world coordinates (x, y) at the target depth
    x_world = np.tan(h_fov_rad / 2) * max_depth * x_normalized
    y_world = np.tan(v_fov_rad / 2) * max_depth * y_normalized

    # Compute the target position using the world coordinates and max_depth
    target_position = np.array([x_world, y_world, max_depth])

    # Assuming the camera is initially at the origin, and looking in the negative Z direction
    cam_position = np.array([0, 0, 0])
    current_direction = np.array([0, 0, -1])

    # Compute the direction vector and normalize it
    direction = target_position - cam_position
    direction = direction / np.linalg.norm(direction)

    # Compute the rotation angle based on the turn_weight (number of frames)
    axis = np.cross(current_direction, direction)
    axis_norm = np.linalg.norm(axis)

    # Avoid division by zero and invalid arcsin values
    if axis_norm > 0:
        axis = axis / axis_norm
        # Clamp to valid arcsin range [-1, 1]
        angle = np.arcsin(np.clip(axis_norm, -1.0, 1.0))
    else:
        # Parallel or anti-parallel vectors - no rotation needed
        axis = np.array([0, 1, 0])  # Arbitrary perpendicular axis
        angle = 0.0
    max_angle = np.pi * (0.1 / turn_weight)  # Limit the maximum rotation angle
    rotation_angle = np.clip(
        np.sign(np.cross(current_direction, direction)) * angle / turn_weight,
        -max_angle,
        max_angle,
    )

    # Compute the rotation matrix
    rotation_matrix_np = np.eye(3) + np.sin(rotation_angle) * np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    ) + (1 - np.cos(rotation_angle)) * np.outer(axis, axis)

    # Convert the NumPy array to a PyTorch tensor
    rotation_matrix_tensor = torch.from_numpy(rotation_matrix_np).float()

    # Add an extra dimension to match the expected shape (1, 3, 3)
    rotation_matrix_tensor = rotation_matrix_tensor.unsqueeze(0)

    return rotation_matrix_tensor


# ============================================================================
# Factor-based rounding functions (from qwen_vl_utils.py)
# ============================================================================


def round_by_factor(number: int | float, factor: int) -> int:
    """Round number to closest integer divisible by factor.

    Returns the closest integer to 'number' that is divisible by 'factor'.

    Args:
        number: Number to round (int or float).
        factor: Factor to round by (must be non-zero).

    Returns:
        Closest integer to number that is divisible by factor.

    Raises:
        ValueError: If factor is zero.
        TypeError: If inputs are not numeric.

    Examples:
        >>> round_by_factor(17, 5)
        15
        >>> round_by_factor(18, 5)
        20
        >>> round_by_factor(100, 28)
        112
    """
    if not isinstance(number, (int, float)):
        raise TypeError(f"number must be int or float, got {type(number).__name__}")
    if not isinstance(factor, int):
        raise TypeError(f"factor must be int, got {type(factor).__name__}")
    if factor == 0:
        raise ValueError("factor cannot be zero")

    return round(number / factor) * factor


def ceil_by_factor(number: int | float, factor: int) -> int:
    """Round number up to smallest integer divisible by factor.

    Returns the smallest integer greater than or equal to 'number' that is
    divisible by 'factor'.

    Args:
        number: Number to round up (int or float).
        factor: Factor to round by (must be non-zero).

    Returns:
        Smallest integer >= number that is divisible by factor.

    Raises:
        ValueError: If factor is zero.
        TypeError: If inputs are not numeric.

    Examples:
        >>> ceil_by_factor(17, 5)
        20
        >>> ceil_by_factor(20, 5)
        20
        >>> ceil_by_factor(100, 28)
        112
    """
    if not isinstance(number, (int, float)):
        raise TypeError(f"number must be int or float, got {type(number).__name__}")
    if not isinstance(factor, int):
        raise TypeError(f"factor must be int, got {type(factor).__name__}")
    if factor == 0:
        raise ValueError("factor cannot be zero")

    return math.ceil(number / factor) * factor


def floor_by_factor(number: int | float, factor: int) -> int:
    """Round number down to largest integer divisible by factor.

    Returns the largest integer less than or equal to 'number' that is
    divisible by 'factor'.

    Args:
        number: Number to round down (int or float).
        factor: Factor to round by (must be non-zero).

    Returns:
        Largest integer <= number that is divisible by factor.

    Raises:
        ValueError: If factor is zero.
        TypeError: If inputs are not numeric.

    Examples:
        >>> floor_by_factor(17, 5)
        15
        >>> floor_by_factor(20, 5)
        20
        >>> floor_by_factor(100, 28)
        84
    """
    if not isinstance(number, (int, float)):
        raise TypeError(f"number must be int or float, got {type(number).__name__}")
    if not isinstance(factor, int):
        raise TypeError(f"factor must be int, got {type(factor).__name__}")
    if factor == 0:
        raise ValueError("factor cannot be zero")

    return math.floor(number / factor) * factor


# ============================================================================
# Smart resize/frame calculation functions (from qwen_vl_utils.py)
# ============================================================================


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """Calculate resize dimensions maintaining aspect ratio within pixel bounds.

    Rescales image dimensions so that:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. Total pixels is within range [min_pixels, max_pixels].
    3. Aspect ratio is maintained as closely as possible.

    Args:
        height: Original height in pixels.
        width: Original width in pixels.
        factor: Factor both dimensions must be divisible by (default: 28).
        min_pixels: Minimum total pixels allowed (default: 3136).
        max_pixels: Maximum total pixels allowed (default: 12845056).

    Returns:
        Tuple of (new_height, new_width) meeting all constraints.

    Raises:
        ValueError: If aspect ratio exceeds MAX_RATIO (200).
        TypeError: If inputs are not integers.

    Examples:
        >>> smart_resize(1080, 1920)  # HD video
        (1092, 1932)
        >>> smart_resize(100, 100, factor=28)  # Small square
        (112, 112)
        >>> smart_resize(5000, 5000)  # Very large, will be scaled down
        (3528, 3528)
    """
    if not isinstance(height, int):
        raise TypeError(f"height must be int, got {type(height).__name__}")
    if not isinstance(width, int):
        raise TypeError(f"width must be int, got {type(width).__name__}")
    if not isinstance(factor, int):
        raise TypeError(f"factor must be int, got {type(factor).__name__}")
    if not isinstance(min_pixels, int):
        raise TypeError(f"min_pixels must be int, got {type(min_pixels).__name__}")
    if not isinstance(max_pixels, int):
        raise TypeError(f"max_pixels must be int, got {type(max_pixels).__name__}")

    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")

    # Check aspect ratio constraint
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {aspect_ratio}"
        )

    # Initial rounding to factor
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    # Scale down if too large
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)

    # Scale up if too small
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def smart_nframes(
    total_frames: int,
    video_fps: int | float,
    target_fps: float | None = None,
    nframes: int | None = None,
    min_frames: int | None = None,
    max_frames: int | None = None,
) -> int:
    """Calculate optimal number of frames for model input.

    Calculates the number of frames to extract from video for model inputs,
    ensuring the result is divisible by FRAME_FACTOR (2).

    Can specify either target_fps OR nframes, but not both.

    Args:
        total_frames: Total number of frames in original video.
        video_fps: Original video FPS.
        target_fps: Target FPS for frame extraction (default: 2.0).
        nframes: Explicit number of frames to extract (overrides FPS calculation).
        min_frames: Minimum frames when using FPS (default: 4).
        max_frames: Maximum frames when using FPS (default: 768 or total_frames).

    Returns:
        Number of frames to extract, divisible by FRAME_FACTOR (2).

    Raises:
        ValueError: If both nframes and target_fps are specified, or if
            calculated nframes is out of valid range [FRAME_FACTOR, total_frames].
        TypeError: If inputs are not numeric.

    Examples:
        >>> smart_nframes(100, 30.0, target_fps=2.0)
        6
        >>> smart_nframes(100, 30.0, nframes=10)
        10
        >>> smart_nframes(200, 60.0, target_fps=3.0)
        10
    """
    if not isinstance(total_frames, int):
        raise TypeError(f"total_frames must be int, got {type(total_frames).__name__}")
    if not isinstance(video_fps, (int, float)):
        raise TypeError(f"video_fps must be numeric, got {type(video_fps).__name__}")

    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")

    # Cannot specify both fps and nframes
    if nframes is not None and target_fps is not None:
        raise ValueError("Only accept either `target_fps` or `nframes`, not both")

    if nframes is not None:
        # Explicit frame count mode
        result_nframes = round_by_factor(nframes, FRAME_FACTOR)
    else:
        # FPS-based calculation mode
        fps = target_fps if target_fps is not None else FPS
        min_f = ceil_by_factor(
            min_frames if min_frames is not None else FPS_MIN_FRAMES, FRAME_FACTOR
        )
        max_f = floor_by_factor(
            max_frames if max_frames is not None else min(FPS_MAX_FRAMES, total_frames),
            FRAME_FACTOR,
        )

        # Calculate frames based on FPS ratio
        calc_nframes = total_frames / video_fps * fps
        # Clamp to min/max range
        calc_nframes = min(max(calc_nframes, min_f), max_f)
        result_nframes = round_by_factor(calc_nframes, FRAME_FACTOR)

    # Validate result is in valid range
    if not (FRAME_FACTOR <= result_nframes <= total_frames):
        raise ValueError(
            f"nframes should be in interval [{FRAME_FACTOR}, {total_frames}], "
            f"but got {result_nframes}."
        )

    return result_nframes
