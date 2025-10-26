"""Audio event detection and analysis using librosa.

This module provides multiple detection methods for finding audio events
(beats, onsets, bass kicks, etc.) suitable for animation keyframe generation.
"""

from typing import List, Tuple, Optional
import numpy as np

# Optional import - will be loaded when needed
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def load_audio_file(
    file_path: str,
    sample_rate: int = 22050,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """Load audio file using librosa.

    Args:
        file_path: Path to audio file (MP3, WAV, FLAC, etc.)
        sample_rate: Target sample rate (22050Hz is good for music analysis)
        mono: Convert to mono if True

    Returns:
        Tuple of (audio_data, sample_rate)

    Raises:
        ImportError: If librosa is not installed
        FileNotFoundError: If audio file doesn't exist
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required for audio analysis. "
            "Install with: pip install librosa"
        )

    # Load audio file
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=mono)

    return audio, sr


def detect_onsets(
    audio: np.ndarray,
    sample_rate: int = 22050,
    sensitivity: float = 0.5,
    min_delta: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect onset events (transients, attacks, percussive hits).

    Best for: General percussion, kicks, snares, any sharp transient

    Args:
        audio: Input audio signal
        sample_rate: Audio sample rate in Hz
        sensitivity: Detection sensitivity (0.0-1.0, higher = more sensitive)
        min_delta: Minimum time between onsets in seconds

    Returns:
        Tuple of (onset_times_seconds, onset_strengths)

    Example:
        >>> times, strengths = detect_onsets(audio, sr, sensitivity=0.6)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required")

    # Convert sensitivity to threshold (inverse relationship)
    threshold = 1.0 - sensitivity

    # Detect onset frames
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sample_rate,
        backtrack=False,
        delta=threshold,
        wait=int(min_delta * sample_rate / 512)  # Convert to hop frames
    )

    # Convert frames to times
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

    # Get onset strengths
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    onset_strengths = onset_env[onset_frames]

    # Normalize strengths to 0-1
    if len(onset_strengths) > 0:
        onset_strengths = onset_strengths / np.max(onset_strengths)

    return onset_times, onset_strengths


def detect_beats(
    audio: np.ndarray,
    sample_rate: int = 22050
) -> Tuple[np.ndarray, float]:
    """Detect beat positions and tempo.

    Best for: Musical timing, rhythmic pulse, BPM-based sync

    Args:
        audio: Input audio signal
        sample_rate: Audio sample rate in Hz

    Returns:
        Tuple of (beat_times_seconds, tempo_bpm)

    Example:
        >>> beat_times, bpm = detect_beats(audio, sr)
        >>> print(f"Detected {len(beat_times)} beats at {bpm:.1f} BPM")
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required")

    # Detect tempo and beats
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)

    # Convert frames to times
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)

    return beat_times, float(tempo)


def extract_bass_energy(
    audio: np.ndarray,
    sample_rate: int = 22050,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract low-frequency energy envelope for bass kick detection.

    Best for: Bass kicks, sub-bass events, low-frequency transients

    Args:
        audio: Input audio signal (should be lowpass filtered for best results)
        sample_rate: Audio sample rate in Hz
        hop_length: Number of samples between frames

    Returns:
        Tuple of (time_axis, energy_envelope)

    Example:
        >>> # Use with processed audio
        >>> from .processing import process_audio_for_detection
        >>> bass_audio = process_audio_for_detection(audio, frequency_band='bass')
        >>> times, energy = extract_bass_energy(bass_audio, sr)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required")

    # Compute spectral centroid to get energy distribution
    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=np.median  # Use median for stability
    )

    # Convert frames to time
    times = librosa.frames_to_time(
        np.arange(len(onset_env)),
        sr=sample_rate,
        hop_length=hop_length
    )

    # Normalize energy
    if len(onset_env) > 0:
        max_val = np.max(onset_env)
        if max_val > 0:
            onset_env = onset_env / max_val
        else:
            # All zeros - keep as is
            pass

    return times, onset_env


def detect_events(
    audio: np.ndarray,
    sample_rate: int = 22050,
    method: str = 'onset',
    sensitivity: float = 0.5,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Unified event detection function.

    Args:
        audio: Input audio signal
        sample_rate: Audio sample rate in Hz
        method: Detection method
            - 'onset': Onset detection (general transients)
            - 'beat': Beat tracking
            - 'bass': Bass energy peaks
        sensitivity: Detection sensitivity (0.0-1.0)
        **kwargs: Additional method-specific arguments

    Returns:
        Tuple of (event_times_seconds, event_intensities)

    Example:
        >>> times, intensities = detect_events(
        ...     audio, sr, method='onset', sensitivity=0.7
        ... )
    """
    if method == 'onset':
        return detect_onsets(audio, sample_rate, sensitivity, **kwargs)

    elif method == 'beat':
        beat_times, tempo = detect_beats(audio, sample_rate)
        # Create uniform intensities for beats
        intensities = np.ones_like(beat_times)
        return beat_times, intensities

    elif method == 'bass':
        # Extract bass energy and find peaks
        times, energy = extract_bass_energy(audio, sample_rate, **kwargs)

        # Find peaks in energy envelope
        from scipy.signal import find_peaks

        # Convert sensitivity to peak prominence threshold
        prominence = (1.0 - sensitivity) * 0.5

        peak_indices, properties = find_peaks(
            energy,
            prominence=prominence,
            distance=int(0.1 * sample_rate / 512)  # Min 0.1s between peaks
        )

        event_times = times[peak_indices]
        event_intensities = energy[peak_indices]

        # Normalize intensities
        if len(event_intensities) > 0:
            event_intensities = event_intensities / np.max(event_intensities)

        return event_times, event_intensities

    else:
        raise ValueError(f"Unknown detection method: {method}")


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Get audio duration in seconds.

    Args:
        audio: Input audio signal
        sample_rate: Audio sample rate in Hz

    Returns:
        Duration in seconds
    """
    return len(audio) / float(sample_rate)


def get_audio_info(file_path: str) -> dict:
    """Get audio file information without loading full audio.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary with audio metadata
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required")

    import soundfile as sf

    try:
        info = sf.info(file_path)
        return {
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype,
        }
    except Exception as e:
        # Fallback to librosa
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        duration = len(audio) / sr if audio.ndim == 1 else audio.shape[1] / sr
        channels = 1 if audio.ndim == 1 else audio.shape[0]

        return {
            'duration': duration,
            'sample_rate': sr,
            'channels': channels,
            'format': 'unknown',
            'subtype': 'unknown',
        }
