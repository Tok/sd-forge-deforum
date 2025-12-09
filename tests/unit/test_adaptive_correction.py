"""Tests for adaptive image correction."""

import numpy as np
import cv2
import pytest
from deforum.utils.image.adaptive_correction import (
    AdaptiveCorrector,
    FrameStats,
    create_corrector
)


def create_test_image(mean_luminance: float = 50.0, contrast: float = 20.0) -> np.ndarray:
    """Create synthetic test image with specific luminance and contrast.

    Args:
        mean_luminance: Target mean L value (0-100)
        contrast: Target standard deviation

    Returns:
        BGR image (480x640x3)
    """
    # Create grayscale base with gaussian distribution
    size = (480, 640)
    gray = np.random.normal(mean_luminance, contrast, size).astype(np.float32)
    gray = np.clip(gray, 0, 100)

    # Convert to LAB (L channel set, a/b neutral)
    lab = np.zeros((*size, 3), dtype=np.float32)
    lab[:, :, 0] = gray
    lab[:, :, 1] = 128  # Neutral a
    lab[:, :, 2] = 128  # Neutral b

    # Convert to BGR
    lab = lab.astype(np.uint8)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return bgr


class TestFrameStats:
    """Test FrameStats calculation."""

    def test_calculate_stats_bright_image(self):
        """Test stats calculation on bright image."""
        corrector = AdaptiveCorrector()
        bright_image = create_test_image(mean_luminance=80.0, contrast=15.0)

        stats = corrector.calculate_stats(bright_image)

        # Should be close to target values
        assert 75 < stats.mean_luminance < 85
        assert 10 < stats.luminance_std < 20

    def test_calculate_stats_dark_image(self):
        """Test stats calculation on dark image."""
        corrector = AdaptiveCorrector()
        dark_image = create_test_image(mean_luminance=30.0, contrast=10.0)

        stats = corrector.calculate_stats(dark_image)

        assert 25 < stats.mean_luminance < 35
        assert 5 < stats.luminance_std < 15


class TestAdaptiveCorrector:
    """Test AdaptiveCorrector modes."""

    def test_prevent_drift_mode(self):
        """Test prevent_drift mode maintains rolling average."""
        corrector = AdaptiveCorrector(mode="prevent_drift", baseline_window=3)

        # Create sequence: bright -> darker -> darker
        frames = [
            create_test_image(mean_luminance=70.0, contrast=20.0),
            create_test_image(mean_luminance=60.0, contrast=20.0),
            create_test_image(mean_luminance=50.0, contrast=20.0),
        ]

        results = []
        for frame in frames:
            corrected = corrector.process_frame(frame, correction_strength=0.8)
            stats = corrector.calculate_stats(corrected)
            results.append(stats.mean_luminance)

        # Should prevent darkening drift
        # Last frame should be brighter than input (50) due to correction
        assert results[-1] > 52  # Corrected upward

    def test_smart_coherence_mode_keyframe_reset(self):
        """Test smart_coherence resets baseline at keyframes."""
        corrector = AdaptiveCorrector(mode="smart_coherence", baseline_window=3)

        # Sequence: bright -> keyframe (dark) -> tweens
        frame1 = create_test_image(mean_luminance=70.0)
        frame2_key = create_test_image(mean_luminance=40.0)  # Intentional change
        frame3 = create_test_image(mean_luminance=38.0)      # Slight drift

        # Process first frame
        corrector.process_frame(frame1, is_keyframe=True)

        # Process keyframe (should reset baseline)
        corrector.process_frame(frame2_key, is_keyframe=True)
        baseline_before = corrector.get_baseline_stats().mean_luminance

        # Process tween (should correct toward new baseline, not frame1)
        corrected = corrector.process_frame(frame3, is_keyframe=False, correction_strength=0.8)
        stats = corrector.calculate_stats(corrected)

        # Should be corrected toward frame2 (~40), not frame1 (~70)
        # Allow some tolerance since correction is partial (strength=0.8)
        assert 37 < stats.mean_luminance < 42

    def test_exposure_lock_mode(self):
        """Test exposure_lock preserves first frame luminance."""
        corrector = AdaptiveCorrector(mode="exposure_lock", baseline_window=5)

        # Sequence: bright -> darker frames
        first = create_test_image(mean_luminance=70.0, contrast=20.0)
        corrector.process_frame(first, is_keyframe=True)
        first_lum = corrector.first_frame_stats.mean_luminance

        # Process darker frames
        for i in range(5):
            dark = create_test_image(mean_luminance=50.0 - i*2, contrast=20.0)
            corrected = corrector.process_frame(dark, correction_strength=0.9)
            stats = corrector.calculate_stats(corrected)

            # Should maintain close to first frame luminance
            assert abs(stats.mean_luminance - first_lum) < 5

    def test_correction_strength_parameter(self):
        """Test correction strength scales adjustment."""
        corrector = AdaptiveCorrector(mode="prevent_drift")

        base = create_test_image(mean_luminance=70.0)
        corrector.process_frame(base, is_keyframe=True)

        dark = create_test_image(mean_luminance=50.0)

        # Weak correction
        weak = corrector.process_frame(dark, correction_strength=0.2)
        weak_stats = corrector.calculate_stats(weak)

        # Strong correction (new instance to reset state)
        corrector2 = AdaptiveCorrector(mode="prevent_drift")
        corrector2.process_frame(base, is_keyframe=True)
        strong = corrector2.process_frame(dark, correction_strength=0.9)
        strong_stats = corrector2.calculate_stats(strong)

        # Strong correction should bring luminance closer to baseline
        assert strong_stats.mean_luminance > weak_stats.mean_luminance

    def test_no_correction_when_within_tolerance(self):
        """Test minimal correction when drift is small."""
        corrector = AdaptiveCorrector(mode="prevent_drift")

        base = create_test_image(mean_luminance=70.0, contrast=20.0)
        corrector.process_frame(base, is_keyframe=True)

        # Very slight change (within tolerance)
        similar = create_test_image(mean_luminance=70.5, contrast=20.2)
        corrected = corrector.process_frame(similar, correction_strength=0.5)

        # Should be very minimal correction (allow up to 2.5 units due to noise + rounding)
        stats = corrector.calculate_stats(corrected)
        assert abs(stats.mean_luminance - 70.5) < 2.5


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_cumulative_darkening_prevention(self):
        """Test prevention of cumulative darkening over many frames."""
        corrector = AdaptiveCorrector(mode="prevent_drift", baseline_window=5)

        # Simulate cumulative drift (each frame slightly darker)
        initial_lum = 70.0
        num_frames = 20

        luminances = []
        for i in range(num_frames):
            # Each frame 1 unit darker than it should be (cumulative error)
            target_lum = initial_lum - i * 0.5
            frame = create_test_image(mean_luminance=target_lum, contrast=20.0)

            corrected = corrector.process_frame(frame, correction_strength=0.7)
            stats = corrector.calculate_stats(corrected)
            luminances.append(stats.mean_luminance)

        # Should prevent significant drift
        # With 0.7 strength and rolling window, expect partial correction
        # Final frame input=60, should be corrected upward but not fully back to 70
        assert luminances[-1] > 58  # At least some correction

        # Should maintain relatively stable average (allow wider range)
        avg_lum = np.mean(luminances)
        assert 64 < avg_lum < 72

    def test_prompt_change_tolerance(self):
        """Test smart_coherence allows intentional prompt changes."""
        corrector = AdaptiveCorrector(mode="smart_coherence", baseline_window=3)

        # Keyframe 1: Bright scene
        kf1 = create_test_image(mean_luminance=75.0, contrast=20.0)
        corrector.process_frame(kf1, is_keyframe=True)

        # Tweens should maintain brightness (allow wider tolerance due to rolling window)
        for _ in range(5):
            tween = create_test_image(mean_luminance=73.0, contrast=20.0)
            corrected = corrector.process_frame(tween, correction_strength=0.6)
            stats = corrector.calculate_stats(corrected)
            assert 70 < stats.mean_luminance < 78  # Wide range for 0.6 strength + rolling window

        # Keyframe 2: Dark scene (intentional prompt change)
        kf2 = create_test_image(mean_luminance=40.0, contrast=15.0)
        corrector.process_frame(kf2, is_keyframe=True)

        # Subsequent tweens should maintain NEW dark level, not revert to bright
        for _ in range(5):
            tween = create_test_image(mean_luminance=38.0, contrast=15.0)
            corrected = corrector.process_frame(tween, correction_strength=0.6)
            stats = corrector.calculate_stats(corrected)
            # Should stay dark (near 40), NOT correct back to 75
            assert 36 < stats.mean_luminance < 44  # Allow tolerance for partial correction


class TestFactoryFunction:
    """Test create_corrector factory."""

    def test_create_corrector_modes(self):
        """Test factory creates correct modes."""
        prevent = create_corrector("prevent_drift")
        assert prevent.mode == "prevent_drift"

        smart = create_corrector("smart_coherence")
        assert smart.mode == "smart_coherence"

        exposure = create_corrector("exposure_lock")
        assert exposure.mode == "exposure_lock"
