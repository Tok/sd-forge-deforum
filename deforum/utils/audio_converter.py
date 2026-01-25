"""
Audio Format Converter Utility

Helper functions to convert audio files to LTX-2-friendly formats (WAV/FLAC)
if MP3 decoding fails or for maximum reliability.
"""

import subprocess
from pathlib import Path
from typing import Optional
from deforum.utils.system.logging import get_logger

logger = get_logger()


def convert_audio_to_wav(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    channels: int = 1
) -> Optional[str]:
    """
    Convert audio file to WAV format using ffmpeg.

    Args:
        input_path: Path to input audio file (MP3, FLAC, OGG, M4A, etc.)
        output_path: Path for output WAV file (default: same name with .wav)
        sample_rate: Target sample rate in Hz (default: 16000 for LTX-2)
        channels: Number of audio channels (1=mono, 2=stereo)

    Returns:
        Path to converted WAV file, or None if conversion failed

    Example:
        >>> convert_audio_to_wav("soundtrack.mp3")
        "soundtrack.wav"
    """
    input_path = Path(input_path)

    if not input_path.exists():
        logger.error(f"Input audio file not found: {input_path}")
        return None

    # Default output path: same directory, .wav extension
    if output_path is None:
        output_path = input_path.with_suffix('.wav')
    else:
        output_path = Path(output_path)

    # Check if already WAV
    if input_path.suffix.lower() == '.wav':
        logger.info(f"Audio already in WAV format: {input_path}")
        return str(input_path)

    try:
        logger.info(f"Converting {input_path.name} → {output_path.name} (WAV, {sample_rate}Hz, {channels}ch)")

        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ar', str(sample_rate),  # Sample rate
            '-ac', str(channels),     # Channels (1=mono)
            '-y',                     # Overwrite output
            str(output_path)
        ]

        # Run conversion
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg conversion failed: {result.stderr.decode()}")
            return None

        if not output_path.exists():
            logger.error(f"Conversion completed but output file not found: {output_path}")
            return None

        logger.info(f"✅ Converted to WAV: {output_path}")
        return str(output_path)

    except FileNotFoundError:
        logger.error("ffmpeg not found - please install ffmpeg to convert audio")
        logger.info("Install: sudo apt install ffmpeg  (or brew install ffmpeg on macOS)")
        return None
    except subprocess.TimeoutExpired:
        logger.error("Audio conversion timed out (>5 minutes)")
        return None
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return None


def convert_audio_to_flac(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000
) -> Optional[str]:
    """
    Convert audio file to FLAC format (lossless compression).

    Args:
        input_path: Path to input audio file
        output_path: Path for output FLAC file (default: same name with .flac)
        sample_rate: Target sample rate in Hz (default: 16000)

    Returns:
        Path to converted FLAC file, or None if conversion failed
    """
    input_path = Path(input_path)

    if not input_path.exists():
        logger.error(f"Input audio file not found: {input_path}")
        return None

    if output_path is None:
        output_path = input_path.with_suffix('.flac')
    else:
        output_path = Path(output_path)

    # Check if already FLAC
    if input_path.suffix.lower() == '.flac':
        logger.info(f"Audio already in FLAC format: {input_path}")
        return str(input_path)

    try:
        logger.info(f"Converting {input_path.name} → {output_path.name} (FLAC lossless)")

        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ar', str(sample_rate),
            '-ac', '1',  # Mono
            '-c:a', 'flac',  # FLAC codec
            '-y',
            str(output_path)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg conversion failed: {result.stderr.decode()}")
            return None

        if not output_path.exists():
            logger.error(f"Conversion completed but output file not found: {output_path}")
            return None

        logger.info(f"✅ Converted to FLAC: {output_path}")
        return str(output_path)

    except FileNotFoundError:
        logger.error("ffmpeg not found - please install ffmpeg")
        return None
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return None


def check_audio_format(audio_path: str) -> dict:
    """
    Check audio file format and properties using ffprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Dict with audio properties (format, sample_rate, channels, duration)
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            audio_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )

        if result.returncode != 0:
            return {"error": "ffprobe failed"}

        import json
        data = json.loads(result.stdout)

        # Extract audio stream info
        audio_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'audio':
                audio_stream = stream
                break

        if not audio_stream:
            return {"error": "No audio stream found"}

        return {
            "format": data.get('format', {}).get('format_name', 'unknown'),
            "codec": audio_stream.get('codec_name', 'unknown'),
            "sample_rate": int(audio_stream.get('sample_rate', 0)),
            "channels": int(audio_stream.get('channels', 0)),
            "duration": float(data.get('format', {}).get('duration', 0)),
            "bit_rate": int(data.get('format', {}).get('bit_rate', 0))
        }

    except Exception as e:
        return {"error": str(e)}
