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


def get_n_strongest_events(
    event_times: np.ndarray,
    event_intensities: np.ndarray,
    n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Select N strongest events by intensity.

    Useful for +/- adjustment buttons to increase/decrease keyframe count
    without being limited by spacing restrictions.

    Args:
        event_times: Event timestamps in seconds
        event_intensities: Event intensities (normalized 0-1)
        n: Number of events to return

    Returns:
        Tuple of (selected_times, selected_intensities) sorted by time

    Example:
        >>> # Get top 20 strongest events from 100 detected
        >>> times_top20, intensities_top20 = get_n_strongest_events(
        ...     all_times, all_intensities, n=20
        ... )
    """
    if len(event_times) == 0:
        return np.array([]), np.array([])

    # If requesting more events than available, return all
    if n >= len(event_times):
        return event_times, event_intensities

    # Pair times with intensities
    events = list(zip(event_times, event_intensities))

    # Sort by intensity (strongest first)
    events_sorted = sorted(events, key=lambda x: x[1], reverse=True)

    # Take top N
    top_n = events_sorted[:n]

    # Re-sort by time for chronological order
    top_n_sorted = sorted(top_n, key=lambda x: x[0])

    times, intensities = zip(*top_n_sorted) if top_n_sorted else ([], [])
    return np.array(times), np.array(intensities)


def detect_events_bpm_aware(
    audio: np.ndarray,
    sample_rate: int,
    method: str = "onset",
    target_bpm: Optional[float] = None,
    tolerance: float = 0.15,
    prefer_under_detection: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Detect events with BPM-aware sensitivity auto-adjustment.

    Iteratively adjusts detection sensitivity until the number of detected
    events matches the expected count based on BPM (1 event per beat ideally).

    This solves the problem where fixed sensitivity either misses events or
    detects too many. By targeting BPM-based event count, we get consistent
    results that match the actual rhythm of the audio.

    Args:
        audio: Input audio signal
        sample_rate: Audio sample rate in Hz
        method: Detection method ('onset', 'beat', 'bass')
        target_bpm: Optional target BPM (auto-detected if None)
        tolerance: Acceptable deviation from target (0.15 = ±15%)
        prefer_under_detection: If True, favor fewer events over too many
            (better to drop frames at weak events than add frames where no event exists)

    Returns:
        Tuple of (event_times, event_intensities, detected_bpm)

    Example:
        >>> # Auto-detect BPM and get matching number of events
        >>> times, intensities, bpm = detect_events_bpm_aware(
        ...     audio, sr, method='onset', tolerance=0.15
        ... )
        >>> print(f"Detected {len(times)} events at {bpm:.1f} BPM")
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required")

    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    # 1. DETECT BPM if not provided
    if target_bpm is None:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        target_bpm = float(tempo)

    duration = get_audio_duration(audio, sample_rate)

    # Expected events based on BPM (1 event per beat)
    expected_events_per_sec = target_bpm / 60.0
    expected_total_events = int(duration * expected_events_per_sec)

    # Acceptable range
    if prefer_under_detection:
        # Stricter upper bound - prefer missing weak events over false positives
        min_events = int(expected_total_events * (1 - tolerance))
        max_events = int(expected_total_events * (1 + tolerance * 0.5))
    else:
        # Symmetric tolerance
        min_events = int(expected_total_events * (1 - tolerance))
        max_events = int(expected_total_events * (1 + tolerance))

    logger.info(
        f"BPM-aware detection: {target_bpm:.1f} BPM, "
        f"target {expected_total_events} events (range: {min_events}-{max_events})"
    )

    # 2. BINARY SEARCH for optimal sensitivity
    sensitivity_low = 0.1
    sensitivity_high = 0.9
    best_events = None
    best_sensitivity = 0.5
    best_distance = float('inf')

    for attempt in range(12):  # Max 12 iterations for convergence
        sensitivity = (sensitivity_low + sensitivity_high) / 2.0

        # Detect events at current sensitivity
        event_times, event_intensities = detect_events(
            audio=audio,
            sample_rate=sample_rate,
            method=method,
            sensitivity=sensitivity
        )

        num_events = len(event_times)
        distance_from_target = abs(num_events - expected_total_events)

        logger.debug(
            f"  Attempt {attempt+1}: sensitivity={sensitivity:.3f} → "
            f"{num_events} events (target: {expected_total_events})"
        )

        # Check if in acceptable range
        if min_events <= num_events <= max_events:
            # Found acceptable solution
            best_events = (event_times, event_intensities)
            best_sensitivity = sensitivity
            logger.info(
                f"✓ Found optimal sensitivity: {sensitivity:.3f} "
                f"({num_events} events)"
            )
            break

        # Track best so far (closest to target within constraints)
        if prefer_under_detection:
            # Only update if not over-detecting
            if num_events <= max_events and distance_from_target < best_distance:
                best_events = (event_times, event_intensities)
                best_sensitivity = sensitivity
                best_distance = distance_from_target
        else:
            # Update if closer to target
            if distance_from_target < best_distance:
                best_events = (event_times, event_intensities)
                best_sensitivity = sensitivity
                best_distance = distance_from_target

        # Adjust search range for next iteration
        if num_events < min_events:
            # Too few events - decrease sensitivity threshold (more sensitive)
            sensitivity_high = sensitivity
        else:
            # Too many events - increase sensitivity threshold (less sensitive)
            sensitivity_low = sensitivity

        # Check for convergence (search range too narrow)
        if abs(sensitivity_high - sensitivity_low) < 0.01:
            logger.debug("Sensitivity search converged")
            break

    # Fallback if no solution found
    if best_events is None:
        logger.warning(
            f"Could not find optimal sensitivity, using default 0.5"
        )
        best_events = detect_events(audio, sample_rate, method, sensitivity=0.5)
        best_sensitivity = 0.5

    event_times, event_intensities = best_events
    logger.info(
        f"Final: {len(event_times)} events at sensitivity {best_sensitivity:.3f} "
        f"(BPM: {target_bpm:.1f})"
    )

    return event_times, event_intensities, target_bpm


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
