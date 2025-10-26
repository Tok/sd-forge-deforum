"""Integration test to calibrate audio detection parameters.

This test validates that default audio detection settings can approximate
the pre-synchronized amen break example from defaults.py.

Target keyframes (18 total at 60 FPS):
0, 12, 43, 74, 85, 106, 119, 126, 147, 158, 178, 210, 241, 262, 272, 293, 314, 324

This serves as:
1. Regression test - ensures detection still works after algorithm changes
2. Calibration tool - helps tune default sensitivity
3. Documentation - shows expected performance on reference audio
"""

import pytest
import numpy as np
from pathlib import Path


# Expected keyframes from deforum/config/defaults.py DeforumAnimPrompts()
EXPECTED_KEYFRAMES = [0, 12, 43, 74, 85, 106, 119, 126, 147, 158, 178, 210, 241, 262, 272, 293, 314, 324]
TARGET_FPS = 60
AMEN_BREAK_URL = "https://ia801303.us.archive.org/26/items/amen-breaks/cw_amen13_173.mp3"


@pytest.fixture
def amen_break_audio():
    """Download and load amen break audio."""
    import librosa
    from deforum.media.video_audio_utilities import download_audio

    # Download to temp location
    local_path = download_audio(AMEN_BREAK_URL)

    # Load audio
    y, sr = librosa.load(local_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    return {
        'audio': y,
        'sr': sr,
        'duration': duration,
        'path': local_path
    }


def test_audio_detection_calibration(amen_break_audio):
    """Test that default settings approximate the pre-synced example.

    This is a loose integration test - we expect to get "close" to the
    expected 18 keyframes, not an exact match. Tolerance: 12-24 events.
    """
    from deforum.audio import detect_events, generate_keyframes_from_events, process_audio_for_detection

    y = amen_break_audio['audio']
    sr = amen_break_audio['sr']
    duration = amen_break_audio['duration']

    # Process audio (same as audio_sync.py)
    y_processed = process_audio_for_detection(
        y, sr,
        frequency_band='full',
        lowpass_cutoff=4000,
        distortion_gain=10.0
    )

    # Test with default sensitivity (0.7)
    event_times, event_intensities = detect_events(
        audio=y_processed,
        sample_rate=sr,
        method='onset',
        sensitivity=0.7  # Default from args.py
    )

    # Generate keyframes
    total_frames = int(duration * TARGET_FPS)
    keyframes = generate_keyframes_from_events(
        event_times=event_times,
        event_intensities=event_intensities,
        fps=TARGET_FPS,
        max_frames=total_frames,
        min_spacing_frames=12,  # Default
        intensity_threshold=0.5
    )

    num_keyframes = len(keyframes)
    num_expected = len(EXPECTED_KEYFRAMES)

    # Loose tolerance: within 50% of expected (12-24 keyframes)
    assert 12 <= num_keyframes <= 24, (
        f"Got {num_keyframes} keyframes, expected ~{num_expected} "
        f"(tolerance: 12-24). This may indicate detection sensitivity needs tuning."
    )

    # Info for manual review
    print(f"\n📊 Calibration Results:")
    print(f"  Expected keyframes: {num_expected}")
    print(f"  Detected keyframes: {num_keyframes}")
    print(f"  Difference: {num_keyframes - num_expected:+d}")
    print(f"  Detection rate: {num_keyframes / num_expected * 100:.0f}%")


def test_sensitivity_sweep(amen_break_audio):
    """Sweep sensitivity parameter to find optimal value.

    This test helps tune the default sensitivity by testing a range
    of values and reporting which gets closest to 18 keyframes.
    """
    from deforum.audio import detect_events, generate_keyframes_from_events, process_audio_for_detection

    y = amen_break_audio['audio']
    sr = amen_break_audio['sr']
    duration = amen_break_audio['duration']

    # Process audio
    y_processed = process_audio_for_detection(
        y, sr,
        frequency_band='full',
        lowpass_cutoff=4000,
        distortion_gain=10.0
    )

    total_frames = int(duration * TARGET_FPS)
    num_expected = len(EXPECTED_KEYFRAMES)

    # Test sensitivity range
    sensitivity_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    for sensitivity in sensitivity_values:
        event_times, event_intensities = detect_events(
            audio=y_processed,
            sample_rate=sr,
            method='onset',
            sensitivity=sensitivity
        )

        keyframes = generate_keyframes_from_events(
            event_times=event_times,
            event_intensities=event_intensities,
            fps=TARGET_FPS,
            max_frames=total_frames,
            min_spacing_frames=12,
            intensity_threshold=0.5
        )

        num_kf = len(keyframes)
        error = abs(num_kf - num_expected)
        results.append((sensitivity, num_kf, error))

    # Find best sensitivity
    best = min(results, key=lambda x: x[2])
    best_sensitivity, best_num_kf, best_error = best

    print(f"\n🎯 Sensitivity Sweep Results:")
    print(f"  Target: {num_expected} keyframes")
    print(f"  Sensitivity | Keyframes | Error")
    print(f"  ------------|-----------|------")
    for sens, num_kf, err in results:
        marker = " ✓" if sens == best_sensitivity else ""
        print(f"      {sens:.1f}     |    {num_kf:2d}     |  {err:2d}{marker}")

    print(f"\n  Best sensitivity: {best_sensitivity} ({best_num_kf} keyframes, error: {best_error})")

    # This test always passes - it's informational for tuning
    assert True


@pytest.mark.skipif(
    "not config.getoption('--run-slow')",
    reason="Slow calibration test - use --run-slow to run"
)
def test_timing_accuracy(amen_break_audio):
    """Compare detected keyframe timing to expected timing.

    This validates that events are detected at approximately the right times,
    not just the right count.
    """
    from deforum.audio import detect_events, generate_keyframes_from_events, process_audio_for_detection

    y = amen_break_audio['audio']
    sr = amen_break_audio['sr']
    duration = amen_break_audio['duration']

    # Process audio
    y_processed = process_audio_for_detection(
        y, sr,
        frequency_band='full',
        lowpass_cutoff=4000,
        distortion_gain=10.0
    )

    # Detect with default settings
    event_times, event_intensities = detect_events(
        audio=y_processed,
        sample_rate=sr,
        method='onset',
        sensitivity=0.7
    )

    total_frames = int(duration * TARGET_FPS)
    keyframes = generate_keyframes_from_events(
        event_times=event_times,
        event_intensities=event_intensities,
        fps=TARGET_FPS,
        max_frames=total_frames,
        min_spacing_frames=12,
        intensity_threshold=0.5
    )

    detected_frames = [kf['frame'] for kf in keyframes]

    # For each expected keyframe, find nearest detected keyframe
    timing_errors = []
    for expected_frame in EXPECTED_KEYFRAMES:
        if detected_frames:
            nearest = min(detected_frames, key=lambda f: abs(f - expected_frame))
            error = abs(nearest - expected_frame)
            timing_errors.append(error)

    if timing_errors:
        avg_error = np.mean(timing_errors)
        max_error = np.max(timing_errors)

        print(f"\n⏱️  Timing Accuracy:")
        print(f"  Average timing error: {avg_error:.1f} frames ({avg_error/TARGET_FPS*1000:.0f}ms)")
        print(f"  Max timing error: {max_error} frames ({max_error/TARGET_FPS*1000:.0f}ms)")

        # Loose tolerance: average error < 20 frames (333ms at 60 FPS)
        assert avg_error < 20, f"Average timing error too high: {avg_error:.1f} frames"
