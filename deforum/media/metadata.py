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


def create_comprehensive_metadata(settings_dict: dict[str, Any]) -> dict[str, Any]:
    """Create comprehensive metadata from all generation settings.

    Includes ALL technical generation settings for full reproducibility.
    No user-identifying information or branding - only generation parameters.

    Args:
        settings_dict: Full dictionary of all settings (args, anim_args, video_args, etc.)

    Returns:
        Dictionary with commit ID added and ready for embedding
    """
    # Add commit ID for version tracking
    metadata = {"commit_id": _get_commit_id_safe()}

    # Add all provided settings
    metadata.update(settings_dict)

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


def extract_metadata_from_video(video_path: str) -> dict[str, Any] | None:
    """Extract Deforum settings metadata from video file.

    Uses ffprobe to read the comment field and decode embedded settings.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary of extracted settings, or None if no metadata found

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If ffprobe fails
    """
    import os
    import subprocess

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Find ffprobe (usually alongside ffmpeg)
    from deforum.media.video_audio_utilities import find_ffmpeg_binary
    ffmpeg_path = find_ffmpeg_binary()
    if not ffmpeg_path:
        raise RuntimeError("ffprobe not found - cannot extract metadata")

    # ffprobe is usually in the same directory as ffmpeg
    ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')

    # Use ffprobe to extract comment metadata
    cmd = [
        ffprobe_path,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata_json = json.loads(result.stdout)

        # Extract comment field from format tags
        comment = metadata_json.get('format', {}).get('tags', {}).get('comment', '')

        if not comment:
            return None

        # Check if it's Deforum metadata
        if not comment.startswith(METADATA_PREFIX):
            return None

        # Decode and return settings
        decoded = decode_settings_from_metadata(comment)
        return decoded.get('settings', {})

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}")
    except (json.JSONDecodeError, ValueError) as e:
        # Invalid or corrupted metadata
        return None
