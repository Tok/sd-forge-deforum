"""Resolution Validation and Letterboxing Utilities

Handles resolution validation for LTX-2 and automatic letterboxing to standard formats.
"""

from typing import Tuple, Optional
from deforum.utils.system.logging import get_logger

logger = get_logger()


def calculate_letterbox_dimensions(width: int, height: int) -> Tuple[int, int, str]:
    """Calculate letterboxed dimensions to nearest valid resolution.

    LTX-2 requires both width and height to be multiples of 32.
    This function finds the nearest standard resolution and calculates padding needed.

    Args:
        width: Original width
        height: Original height

    Returns:
        Tuple of (new_width, new_height, explanation)
    """
    # Round to nearest multiple of 32
    new_width = ((width + 15) // 32) * 32  # Round up
    new_height = ((height + 15) // 32) * 32

    # Calculate aspect ratio
    aspect_ratio = width / height

    # Common standard resolutions (16:9, 4:3, 21:9)
    standard_resolutions = {
        # 16:9 (most common)
        (1280, 720): "720p 16:9",
        (1920, 1080): "1080p 16:9",
        (2560, 1440): "1440p 16:9",
        (3840, 2160): "4K 16:9",
        (1024, 576): "HD 16:9",
        (768, 432): "SD 16:9",

        # 4:3 (older standard)
        (1024, 768): "XGA 4:3",
        (1280, 960): "SXGA 4:3",

        # 21:9 (ultrawide)
        (2560, 1080): "UW 21:9",
        (3440, 1440): "UW+ 21:9",
    }

    # Find closest standard resolution that fits
    min_diff = float('inf')
    best_resolution = (new_width, new_height)
    best_name = "Custom"

    for (std_w, std_h), name in standard_resolutions.items():
        # Only consider resolutions that are larger than current
        if std_w >= width and std_h >= height:
            # Calculate difference
            diff = abs(std_w - width) + abs(std_h - height)
            if diff < min_diff:
                min_diff = diff
                best_resolution = (std_w, std_h)
                best_name = name

    # If no standard resolution fits, use rounded dimensions
    if min_diff == float('inf'):
        best_resolution = (new_width, new_height)
        best_name = f"Custom {new_width}x{new_height}"

    width_pad = best_resolution[0] - width
    height_pad = best_resolution[1] - height

    explanation = f"{best_name} ({best_resolution[0]}x{best_resolution[1]})"
    if width_pad > 0 or height_pad > 0:
        explanation += f" - adds {width_pad}px horizontal + {height_pad}px vertical padding"

    return best_resolution[0], best_resolution[1], explanation


def validate_ltx2_resolution(args, interp_method: str, auto_fix: bool = True) -> bool:
    """Validate and optionally fix resolution for LTX-2 compatibility.

    LTX-2 requires both width and height to be multiples of 32.
    If resolution is invalid, can automatically add letterboxing.

    Args:
        args: DeforumArgs (contains W, H)
        interp_method: Interpolation method ("LTX-2", "Wan", "FILM", etc.)
        auto_fix: If True, automatically apply letterboxing

    Returns:
        True if resolution is valid or was fixed, False if invalid and not fixed

    Raises:
        ValueError: If resolution is invalid and auto_fix is False
    """
    # Only validate for LTX-2
    if interp_method != "LTX-2":
        return True

    # Check if resolution is valid
    if args.W % 32 == 0 and args.H % 32 == 0:
        logger.info(f"Resolution {args.W}x{args.H} is valid for LTX-2 ✓", emoji='check')
        return True

    # Resolution needs fixing
    new_width, new_height, explanation = calculate_letterbox_dimensions(args.W, args.H)

    width_pad = new_width - args.W
    height_pad = new_height - args.H

    logger.warning(f"Resolution {args.W}x{args.H} is not compatible with LTX-2", emoji='warning')
    logger.info(f"  LTX-2 requires multiples of 32")
    logger.info(f"  Suggested: {explanation}")

    if auto_fix:
        logger.info(f"Auto-applying letterboxing: {args.W}x{args.H} → {new_width}x{new_height}", emoji='check')

        if width_pad > 0:
            logger.info(f"  Adding {width_pad}px horizontal padding ({width_pad//2}px left + {width_pad//2}px right)")

        if height_pad > 0:
            logger.info(f"  Adding {height_pad}px vertical padding ({height_pad//2}px top + {height_pad//2}px bottom)")

        # Update args
        args.W = new_width
        args.H = new_height

        # Store padding info for use during image processing
        args.ltx2_letterbox_pad_h = height_pad
        args.ltx2_letterbox_pad_w = width_pad

        logger.info(f"Keyframes will be generated at {new_width}x{new_height} with black bars", emoji='info')
        return True
    else:
        raise ValueError(
            f"LTX-2 requires resolution multiples of 32. "
            f"Current: {args.W}x{args.H}. "
            f"Use: {new_width}x{new_height} instead (adds {width_pad}px horizontal + {height_pad}px vertical padding)."
        )


def apply_letterbox_padding(image, target_width: int, target_height: int) -> 'Image.Image':
    """Apply letterbox padding to an image.

    Args:
        image: PIL Image
        target_width: Target width after padding
        target_height: Target height after padding

    Returns:
        PIL Image with letterbox padding
    """
    from PIL import Image

    current_width, current_height = image.size

    # Calculate padding
    pad_w = target_width - current_width
    pad_h = target_height - current_height

    if pad_w == 0 and pad_h == 0:
        return image

    # Create new image with black background
    padded = Image.new('RGB', (target_width, target_height), (0, 0, 0))

    # Paste original image centered
    offset_x = pad_w // 2
    offset_y = pad_h // 2
    padded.paste(image, (offset_x, offset_y))

    return padded
