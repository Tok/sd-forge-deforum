"""Audio signal processing for event detection.

This module provides filtering and distortion to help isolate bass kicks
and other percussive events for better detection accuracy.
"""

from typing import Tuple
import numpy as np
from scipy import signal


def apply_lowpass_filter(
    audio: np.ndarray,
    cutoff_hz: float = 250.0,
    sample_rate: int = 22050,
    order: int = 4
) -> np.ndarray:
    """Apply Butterworth lowpass filter to isolate low frequencies (bass kicks).

    Args:
        audio: Input audio signal
        cutoff_hz: Cutoff frequency in Hz (20-2000Hz range, 250Hz good for bass)
        sample_rate: Audio sample rate in Hz
        order: Filter order (higher = steeper rolloff, 4 is good default)

    Returns:
        Filtered audio signal (bass frequencies only)

    Example:
        >>> bass_only = apply_lowpass_filter(audio, cutoff_hz=200, sample_rate=22050)
    """
    # Calculate normalized cutoff frequency (0 to 1, where 1 = Nyquist)
    nyquist = sample_rate / 2.0
    normalized_cutoff = cutoff_hz / nyquist

    # Design Butterworth lowpass filter
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

    # Apply filter using zero-phase filtering (filtfilt prevents phase distortion)
    filtered = signal.filtfilt(b, a, audio)

    return filtered


def apply_highpass_filter(
    audio: np.ndarray,
    cutoff_hz: float = 20.0,
    sample_rate: int = 22050,
    order: int = 4
) -> np.ndarray:
    """Apply Butterworth highpass filter to remove rumble/DC offset.

    Args:
        audio: Input audio signal
        cutoff_hz: Cutoff frequency in Hz (typically 20Hz to remove subsonic rumble)
        sample_rate: Audio sample rate in Hz
        order: Filter order

    Returns:
        Filtered audio signal
    """
    nyquist = sample_rate / 2.0
    normalized_cutoff = cutoff_hz / nyquist

    b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
    filtered = signal.filtfilt(b, a, audio)

    return filtered


def apply_bandpass_filter(
    audio: np.ndarray,
    low_hz: float = 20.0,
    high_hz: float = 250.0,
    sample_rate: int = 22050,
    order: int = 4
) -> np.ndarray:
    """Apply Butterworth bandpass filter for specific frequency range.

    Args:
        audio: Input audio signal
        low_hz: Low cutoff frequency in Hz
        high_hz: High cutoff frequency in Hz
        sample_rate: Audio sample rate in Hz
        order: Filter order

    Returns:
        Filtered audio signal (frequencies between low_hz and high_hz)

    Example:
        >>> # Isolate bass kick range (60-200Hz)
        >>> kick_range = apply_bandpass_filter(audio, low_hz=60, high_hz=200)
    """
    nyquist = sample_rate / 2.0
    low_normalized = low_hz / nyquist
    high_normalized = high_hz / nyquist

    b, a = signal.butter(order, [low_normalized, high_normalized], btype='band', analog=False)
    filtered = signal.filtfilt(b, a, audio)

    return filtered


def apply_distortion(
    audio: np.ndarray,
    gain: float = 1.0,
    distortion_type: str = 'tanh'
) -> np.ndarray:
    """Apply distortion/saturation to emphasize transients (bass kicks).

    Distortion compresses loud signals and boosts quiet ones, making
    transients (like kicks) more prominent for detection.

    Args:
        audio: Input audio signal
        gain: Distortion amount (0.0 = none, 1.0 = moderate, 2.0 = heavy)
        distortion_type: Type of distortion curve
            - 'tanh': Soft clipping (smooth, musical)
            - 'hard': Hard clipping (aggressive)
            - 'arctan': Arctangent (gentle)

    Returns:
        Distorted audio signal

    Example:
        >>> # Emphasize kicks with soft saturation
        >>> emphasized = apply_distortion(audio, gain=1.5, distortion_type='tanh')
    """
    if gain == 0.0:
        return audio

    # Apply gain
    boosted = audio * gain

    # Apply distortion curve
    if distortion_type == 'tanh':
        # Hyperbolic tangent (smooth soft clipping)
        distorted = np.tanh(boosted)
    elif distortion_type == 'hard':
        # Hard clipping at ±1
        distorted = np.clip(boosted, -1.0, 1.0)
    elif distortion_type == 'arctan':
        # Arctangent (gentle compression)
        distorted = (2.0 / np.pi) * np.arctan(boosted * np.pi / 2.0)
    else:
        raise ValueError(f"Unknown distortion type: {distortion_type}")

    return distorted


def normalize_audio(
    audio: np.ndarray,
    target_peak: float = 1.0
) -> np.ndarray:
    """Normalize audio to target peak amplitude.

    Args:
        audio: Input audio signal
        target_peak: Target peak amplitude (0.0 to 1.0, typically 1.0 or 0.99)

    Returns:
        Normalized audio signal
    """
    if len(audio) == 0:
        return audio

    # Find current peak
    current_peak = np.max(np.abs(audio))

    if current_peak == 0:
        return audio

    # Calculate normalization factor
    norm_factor = target_peak / current_peak

    # Apply normalization
    normalized = audio * norm_factor

    return normalized


def process_audio_for_detection(
    audio: np.ndarray,
    sample_rate: int = 22050,
    frequency_band: str = 'bass',
    lowpass_cutoff: float = 250.0,
    distortion_gain: float = 0.0,
    normalize: bool = True
) -> np.ndarray:
    """Complete audio processing pipeline for event detection.

    This is the main processing function that combines filtering,
    distortion, and normalization to prepare audio for event detection.

    Args:
        audio: Input audio signal
        sample_rate: Audio sample rate in Hz
        frequency_band: Frequency band to isolate
            - 'bass': 20-250Hz (bass kicks)
            - 'mid': 250-2000Hz (snares, claps)
            - 'high': 2000-8000Hz (hi-hats, cymbals)
            - 'full': No filtering (full spectrum)
        lowpass_cutoff: Custom lowpass cutoff (Hz) if frequency_band='bass'
        distortion_gain: Distortion amount (0.0 = none, 2.0 = heavy)
        normalize: Whether to normalize output

    Returns:
        Processed audio signal ready for event detection

    Example:
        >>> # Process for bass kick detection
        >>> processed = process_audio_for_detection(
        ...     audio,
        ...     sample_rate=22050,
        ...     frequency_band='bass',
        ...     lowpass_cutoff=200,
        ...     distortion_gain=1.2
        ... )
    """
    processed = audio.copy()

    # Apply frequency band filtering
    if frequency_band == 'bass':
        # Bandpass for bass kicks (20Hz to cutoff)
        processed = apply_bandpass_filter(
            processed,
            low_hz=20.0,
            high_hz=lowpass_cutoff,
            sample_rate=sample_rate
        )
    elif frequency_band == 'mid':
        # Bandpass for mid frequencies (snares, claps)
        processed = apply_bandpass_filter(
            processed,
            low_hz=250.0,
            high_hz=2000.0,
            sample_rate=sample_rate
        )
    elif frequency_band == 'high':
        # Bandpass for high frequencies (hi-hats, cymbals)
        processed = apply_bandpass_filter(
            processed,
            low_hz=2000.0,
            high_hz=8000.0,
            sample_rate=sample_rate
        )
    elif frequency_band == 'full':
        # Just apply highpass to remove rumble
        processed = apply_highpass_filter(
            processed,
            cutoff_hz=20.0,
            sample_rate=sample_rate
        )
    else:
        raise ValueError(f"Unknown frequency band: {frequency_band}")

    # Apply distortion to emphasize transients
    if distortion_gain > 0.0:
        processed = apply_distortion(processed, gain=distortion_gain)

    # Normalize
    if normalize:
        processed = normalize_audio(processed)

    return processed
