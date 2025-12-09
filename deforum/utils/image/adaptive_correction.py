"""Adaptive image correction for preventing cumulative drift.

This module provides intelligent correction that measures actual frame statistics
and applies targeted adjustments, rather than blind scaling factors.

Key Features:
- Perceptual measurement (LAB color space)
- Multiple correction modes (prevent drift, smart coherence, exposure lock)
- Rolling baseline tracking
- Separates intended changes (prompts) from unintended drift
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass


@dataclass
class FrameStats:
    """Perceptual statistics for a single frame."""
    mean_luminance: float      # L channel: 0-100 (perceived brightness)
    luminance_std: float        # Standard deviation (contrast)
    mean_a: float               # Green-Red axis (-128 to 127)
    mean_b: float               # Blue-Yellow axis (-128 to 127)
    saturation: float           # Color saturation (std of a/b channels)


@dataclass
class CorrectionParams:
    """Parameters for adaptive correction."""
    target_luminance: float
    target_contrast: float
    luminance_tolerance: float = 2.0    # ±2 units acceptable drift
    contrast_tolerance: float = 5.0     # ±5 units acceptable drift
    correction_strength: float = 0.5    # How aggressively to correct (0-1)


class AdaptiveCorrector:
    """Adaptive image correction engine.

    Tracks frame statistics and applies intelligent corrections to prevent
    cumulative drift while preserving intentional prompt-driven changes.
    """

    def __init__(self, mode: str = "vibrancy_lock", baseline_window: int = 5):
        """Initialize adaptive corrector.

        Args:
            mode: Correction mode:
                - "prevent_drift": Maintain rolling average (single prompt)
                - "smart_coherence": Allow prompt changes, fix drift only
                - "vibrancy_lock": Lock brightness/saturation, allow hue changes (DEFAULT)
                - "exposure_lock": Lock luminance only, allow color changes
            baseline_window: Number of recent frames to track for rolling average
        """
        self.mode = mode
        self.baseline_window = baseline_window
        self.frame_history: deque[FrameStats] = deque(maxlen=baseline_window)
        self.first_frame_stats: Optional[FrameStats] = None
        self.is_keyframe_boundary = False  # Set externally when prompt changes

    def calculate_stats(self, image: np.ndarray) -> FrameStats:
        """Calculate perceptual statistics for image.

        Args:
            image: Input image (BGR format)

        Returns:
            FrameStats object with perceptual measurements
        """
        # Convert to LAB (perceptually uniform color space)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # L channel: luminance (0-100)
        l_channel = lab[:, :, 0]

        # a/b channels: color (-128 to 127)
        ab_channels = lab[:, :, 1:]

        return FrameStats(
            mean_luminance=float(np.mean(l_channel)),
            luminance_std=float(np.std(l_channel)),
            mean_a=float(np.mean(lab[:, :, 1])),
            mean_b=float(np.mean(lab[:, :, 2])),
            saturation=float(np.std(ab_channels))
        )

    def update_baseline(self, stats: FrameStats, is_keyframe: bool = False):
        """Update baseline tracking with new frame stats.

        Args:
            stats: Statistics from current frame
            is_keyframe: Whether this is a keyframe (prompt boundary)
        """
        # Store first frame as absolute reference
        if self.first_frame_stats is None:
            self.first_frame_stats = stats

        # For smart coherence mode, reset rolling average at keyframes
        if self.mode == "smart_coherence" and is_keyframe:
            self.frame_history.clear()

        # Add to rolling window
        self.frame_history.append(stats)

    def get_baseline_stats(self) -> FrameStats:
        """Get baseline statistics based on correction mode.

        Returns:
            Baseline FrameStats to use as target
        """
        if not self.frame_history:
            # No history yet - use first frame
            return self.first_frame_stats

        if self.mode == "prevent_drift":
            # Rolling average of recent frames
            return FrameStats(
                mean_luminance=np.mean([s.mean_luminance for s in self.frame_history]),
                luminance_std=np.mean([s.luminance_std for s in self.frame_history]),
                mean_a=np.mean([s.mean_a for s in self.frame_history]),
                mean_b=np.mean([s.mean_b for s in self.frame_history]),
                saturation=np.mean([s.saturation for s in self.frame_history])
            )

        elif self.mode == "smart_coherence":
            # Average since last keyframe (allows prompt-driven changes)
            return FrameStats(
                mean_luminance=np.mean([s.mean_luminance for s in self.frame_history]),
                luminance_std=np.mean([s.luminance_std for s in self.frame_history]),
                mean_a=np.mean([s.mean_a for s in self.frame_history]),
                mean_b=np.mean([s.mean_b for s in self.frame_history]),
                saturation=np.mean([s.saturation for s in self.frame_history])
            )

        elif self.mode == "exposure_lock":
            # Use first frame luminance, allow color to drift
            return self.first_frame_stats

        else:
            return self.first_frame_stats

    def calculate_correction(
        self,
        current: FrameStats,
        baseline: FrameStats,
        strength: float = 0.5
    ) -> Tuple[float, float]:
        """Calculate correction factors for luminance and contrast.

        Args:
            current: Current frame statistics
            baseline: Target baseline statistics
            strength: Correction strength (0=none, 1=full correction)

        Returns:
            Tuple of (luminance_adjustment, contrast_adjustment)
        """
        # Calculate deviations
        lum_diff = baseline.mean_luminance - current.mean_luminance
        contrast_diff = baseline.luminance_std - current.luminance_std

        # Apply correction strength
        lum_correction = lum_diff * strength
        contrast_correction = contrast_diff * strength

        # Convert to multiplicative factors
        # Luminance: additive adjustment (LAB L channel is 0-100)
        # Contrast: multiplicative factor
        luminance_adjustment = lum_correction
        contrast_adjustment = 1.0 + (contrast_correction / max(current.luminance_std, 1.0))

        return luminance_adjustment, contrast_adjustment

    def apply_correction(
        self,
        image: np.ndarray,
        luminance_adj: float,
        contrast_adj: float
    ) -> np.ndarray:
        """Apply calculated corrections to image.

        Args:
            image: Input image (BGR format)
            luminance_adj: Additive luminance adjustment
            contrast_adj: Multiplicative contrast adjustment

        Returns:
            Corrected image (BGR format)
        """
        # Convert to LAB for perceptual adjustments
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Adjust luminance (L channel) - additive
        lab[:, :, 0] = np.clip(lab[:, :, 0] + luminance_adj, 0, 100)

        # Adjust contrast - multiplicative around mean
        if abs(contrast_adj - 1.0) > 0.01:  # Only if meaningful change
            mean_l = np.mean(lab[:, :, 0])
            lab[:, :, 0] = mean_l + (lab[:, :, 0] - mean_l) * contrast_adj
            lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100)

        # Convert back to BGR
        lab = lab.astype(np.uint8)
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return corrected

    def process_frame(
        self,
        image: np.ndarray,
        is_keyframe: bool = False,
        correction_strength: float = 0.5
    ) -> np.ndarray:
        """Process frame with adaptive correction.

        Main entry point for per-frame correction.

        Args:
            image: Input frame (BGR format)
            is_keyframe: Whether this is a keyframe (prompt change)
            correction_strength: How aggressively to correct (0-1)

        Returns:
            Corrected frame (BGR format)
        """
        # Calculate current frame stats
        current_stats = self.calculate_stats(image)

        # Update baseline tracking
        self.update_baseline(current_stats, is_keyframe)

        # Get target baseline
        baseline_stats = self.get_baseline_stats()

        # Calculate correction needed
        lum_adj, contrast_adj = self.calculate_correction(
            current_stats,
            baseline_stats,
            strength=correction_strength
        )

        # Apply correction
        corrected = self.apply_correction(image, lum_adj, contrast_adj)

        return corrected


def create_corrector(mode: str = "vibrancy_lock") -> AdaptiveCorrector:
    """Factory function to create adaptive corrector.

    Args:
        mode: Correction mode (prevent_drift, smart_coherence, vibrancy_lock, exposure_lock)

    Returns:
        Configured AdaptiveCorrector instance
    """
    return AdaptiveCorrector(mode=mode, baseline_window=5)


def preserve_vibrancy(
    image: np.ndarray,
    reference_brightness: float,
    reference_saturation: float,
    strength: float = 0.7
) -> np.ndarray:
    """Simple vibrancy preservation - match brightness and saturation to reference.

    This is the recommended mode for most use cases: prevents brownout and maintains
    vibrant colors while allowing hue to change freely with prompts.

    Args:
        image: Input image (BGR format)
        reference_brightness: Target mean luminance (0-100)
        reference_saturation: Target color saturation
        strength: Correction strength (0-1)

    Returns:
        Corrected image with preserved vibrancy (BGR format)
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Measure current state
    current_brightness = np.mean(lab[:, :, 0])
    current_saturation = np.std(lab[:, :, 1:])

    # Calculate adjustments
    brightness_diff = reference_brightness - current_brightness
    saturation_ratio = reference_saturation / max(current_saturation, 1.0)

    # Apply brightness correction (additive)
    lab[:, :, 0] = np.clip(
        lab[:, :, 0] + (brightness_diff * strength),
        0,
        100
    )

    # Apply saturation correction (multiplicative on a/b channels)
    if abs(saturation_ratio - 1.0) > 0.05:  # Only if meaningful change
        # Center a/b channels around 128 (neutral)
        lab[:, :, 1:] = 128 + ((lab[:, :, 1:] - 128) * (1.0 + (saturation_ratio - 1.0) * strength))
        lab[:, :, 1:] = np.clip(lab[:, :, 1:], 0, 255)

    # Convert back to BGR
    lab = lab.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
