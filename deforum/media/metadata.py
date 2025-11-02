"""Video metadata embedding for Deforum settings.

Embeds technical generation settings into video files for reproducibility,
similar to ComfyUI's workflow embedding in images.

The settings are:
1. Serialized to JSON
2. Base64 encoded
3. Embedded in video metadata comment field
4. Prefixed with "DEFORUM_SETTINGS:" marker for identification

Only technical settings are embedded - no user-identifying information or branding.
"""

import base64
import json
from typing import Any


# Metadata field constants
METADATA_PREFIX = "DEFORUM_SETTINGS:"
METADATA_VERSION = "1.0"


def _get_commit_id_safe() -> str:
    """Get commit ID safely, returning 'Unknown' if not available.

    This is wrapped to avoid heavy import dependencies in tests.

    Returns:
        Commit ID string or "Unknown"
    """
    try:
        from deforum.utils.general import get_deforum_version
        return get_deforum_version()
    except Exception:
        return "Unknown"


def encode_settings_for_metadata(settings_dict: dict[str, Any]) -> str:
    """Encode Deforum settings as base64 string for video metadata.

    Args:
        settings_dict: Dictionary of Deforum settings/arguments

    Returns:
        Base64-encoded string with DEFORUM_SETTINGS: prefix

    Examples:
        >>> settings = {"fps": 24, "max_frames": 100}
        >>> encoded = encode_settings_for_metadata(settings)
        >>> encoded.startswith("DEFORUM_SETTINGS:")
        True
    """
    # Add metadata version for future compatibility
    metadata = {
        "version": METADATA_VERSION,
        "settings": settings_dict
    }

    # Serialize to JSON (compact)
    settings_json = json.dumps(metadata, separators=(',', ':'))

    # Base64 encode
    encoded_bytes = base64.b64encode(settings_json.encode('utf-8'))
    encoded_str = encoded_bytes.decode('ascii')

    # Add prefix marker
    return f"{METADATA_PREFIX}{encoded_str}"


def decode_settings_from_metadata(metadata_string: str) -> dict[str, Any]:
    """Decode Deforum settings from base64 metadata string.

    Args:
        metadata_string: Base64-encoded metadata string with prefix

    Returns:
        Dictionary of decoded settings

    Raises:
        ValueError: If string doesn't have DEFORUM_SETTINGS: prefix
        json.JSONDecodeError: If decoded data is not valid JSON

    Examples:
        >>> encoded = encode_settings_for_metadata({"fps": 24})
        >>> decoded = decode_settings_from_metadata(encoded)
        >>> decoded["settings"]["fps"]
        24
    """
    if not metadata_string.startswith(METADATA_PREFIX):
        raise ValueError(f"Not a Deforum settings string (missing {METADATA_PREFIX} prefix)")

    # Remove prefix
    encoded_str = metadata_string[len(METADATA_PREFIX):]

    # Base64 decode
    decoded_bytes = base64.b64decode(encoded_str.encode('ascii'))
    settings_json = decoded_bytes.decode('utf-8')

    # Parse JSON
    metadata = json.loads(settings_json)

    # Validate structure
    if "version" not in metadata or "settings" not in metadata:
        raise ValueError("Invalid metadata structure (missing version or settings)")

    return metadata


def create_essential_metadata(
    render_mode: str,
    fps: int,
    max_frames: int,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg_scale: float,
    model_name: str = "Unknown",
    scheduler: str = "Unknown",
    prompts: dict[int, str] | None = None
) -> dict[str, Any]:
    """Create essential metadata for video embedding.

    Includes only technical generation settings for reproducibility.
    No user-identifying information or branding.

    Args:
        render_mode: Render mode (e.g., "New 3D", "Flux + Interpolation")
        fps: Frames per second
        max_frames: Total frame count
        width: Video width in pixels
        height: Video height in pixels
        seed: Generation seed
        steps: Sampling steps
        cfg_scale: CFG scale value
        model_name: Name of the model used (default: "Unknown")
        scheduler: Scheduler/sampler name (default: "Unknown")
        prompts: Optional prompt schedule dict {frame: prompt}

    Returns:
        Dictionary of essential metadata (technical settings only)
    """
    metadata = {
        "commit_id": _get_commit_id_safe(),
        "render_mode": render_mode,
        "model": model_name,
        "scheduler": scheduler,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "fps": fps,
        "total_frames": max_frames,
        "width": width,
        "height": height,
    }

    if prompts is not None:
        metadata["prompts"] = prompts

    return metadata


def create_ffmpeg_metadata_args(settings_dict: dict[str, Any]) -> list[str]:
    """Create ffmpeg command arguments for metadata embedding.

    Embeds only technical generation settings in the comment field.
    No user-identifying information or branding.

    Args:
        settings_dict: Dictionary of settings to embed

    Returns:
        List of ffmpeg arguments for -metadata options

    Examples:
        >>> settings = {"fps": 24, "seed": 12345}
        >>> args = create_ffmpeg_metadata_args(settings)
        >>> '-metadata' in args
        True
    """
    # Encode full settings for comment field
    encoded_settings = encode_settings_for_metadata(settings_dict)

    # Only embed technical settings in comment field
    metadata_args = [
        '-metadata', f'comment={encoded_settings}',
    ]

    return metadata_args
