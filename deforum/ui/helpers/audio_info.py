"""Audio info calculation helpers for UI.

Extracted from ui_elements.py get_tab_init() to reduce complexity.
"""

from pathlib import Path
from typing import Tuple, Optional


def calculate_audio_info(audio_path: str, current_fps: int) -> str:
    """Calculate audio duration and suggested max_frames from path.

    Args:
        audio_path: Local file path or URL to audio file
        current_fps: Target FPS for calculation

    Returns:
        Info text string with duration and suggested max_frames
    """
    if not audio_path or not str(audio_path).strip():
        return ""

    try:
        import librosa
        from deforum.media.video_audio_utilities import download_audio

        # Download if URL, or pass through if local path
        local_path = download_audio(audio_path)

        # Get audio duration (faster than loading full audio)
        duration = librosa.get_duration(path=local_path)

        # Calculate suggested max_frames
        fps = current_fps if current_fps and current_fps > 0 else 24
        suggested_max_frames = int(duration * fps)

        return f"Duration: {duration:.2f}s | Suggested max_frames @ {fps} FPS: {suggested_max_frames}"
    except Exception as e:
        return f"Could not load audio: {str(e)}"


def handle_audio_upload(audio_filepath: Optional[str], current_fps: int) -> Tuple[Optional[str], str, str]:
    """Save uploaded audio to output directory and calculate suggested max_frames.

    Args:
        audio_filepath: Path to uploaded audio file
        current_fps: Current FPS setting

    Returns:
        Tuple of (absolute_path, add_soundtrack_value, info_text)
    """
    if audio_filepath is None:
        return None, "File", ""

    import shutil

    # Create output/audio directory
    output_dir = Path("output/audio")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get filename from uploaded file
    filename = Path(audio_filepath).name
    dest_path = output_dir / filename

    # Copy uploaded file to output directory
    shutil.copy2(audio_filepath, dest_path)
    abs_path = str(dest_path.absolute())

    # Calculate audio info using helper function
    info_text = calculate_audio_info(abs_path, current_fps)

    # Return absolute path, set add_soundtrack to "File", and info text
    return abs_path, "File", info_text


def auto_load_audio_info(soundtrack_path_val: str, current_fps: int) -> str:
    """Auto-load audio info when tab is opened if soundtrack path is valid.

    Args:
        soundtrack_path_val: Soundtrack path value
        current_fps: Current FPS setting

    Returns:
        Audio info text
    """
    return calculate_audio_info(soundtrack_path_val, current_fps)
