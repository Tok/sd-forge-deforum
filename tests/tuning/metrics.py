"""Quality metrics for automated parameter tuning.

This module provides empirical quality measurements for evaluating
Deforum generation parameters through automated testing.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any
from pathlib import Path
from PIL import Image


def measure_color_preservation(image: np.ndarray) -> float:
    """Measure how much color is preserved in the image.

    Uses HSV saturation to detect grayscale degradation. Colorful images
    will have high saturation, while images that have degraded to grayscale
    will have low saturation.

    Args:
        image: RGB image as numpy array (H, W, 3)

    Returns:
        Score 0-100 where 100 = full color, 0 = complete grayscale

    Example:
        >>> colorful = load_image("colorful_test.png")
        >>> score = measure_color_preservation(colorful)
        >>> print(f"Color score: {score:.1f}/100")
        Color score: 87.3/100
    """
    if image.dtype == np.uint8:
        img_uint8 = image
    else:
        img_uint8 = (image * 255).astype(np.uint8)

    # Convert to HSV
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    # Calculate mean saturation (S channel)
    saturation = hsv[:, :, 1].mean() / 255.0

    return saturation * 100


def measure_temporal_consistency(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Measure frame-to-frame stability using SSIM.

    Uses Structural Similarity Index to detect jitter and instability
    between consecutive frames. High scores indicate smooth transitions.

    Args:
        frame1: First frame as RGB numpy array (H, W, 3)
        frame2: Second frame as RGB numpy array (H, W, 3)

    Returns:
        Score 0-100 where 100 = identical, 0 = completely different

    Example:
        >>> consistency = measure_temporal_consistency(frame_n, frame_n_plus_1)
        >>> if consistency < 80:
        ...     print("Warning: High frame jitter detected!")
    """
    from skimage.metrics import structural_similarity as ssim

    # Ensure same shape
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frame shapes must match: {frame1.shape} vs {frame2.shape}")

    # Convert to grayscale for SSIM (more stable than multichannel)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    # Calculate SSIM
    score = ssim(gray1, gray2)

    return score * 100


def measure_perceptual_hash_distance(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Measure perceptual similarity using difference hashing.

    Complementary metric to SSIM - detects larger structural changes
    while being robust to minor variations.

    Args:
        frame1: First frame as RGB numpy array (H, W, 3)
        frame2: Second frame as RGB numpy array (H, W, 3)

    Returns:
        Distance 0-100 where 0 = identical, 100 = completely different

    Note:
        Lower is better (opposite of SSIM score direction)
    """
    def compute_dhash(image: np.ndarray, hash_size: int = 8) -> str:
        """Compute difference hash of image."""
        # Resize to hash_size + 1 to allow horizontal gradient
        resized = cv2.resize(image, (hash_size + 1, hash_size))

        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

        # Compute horizontal gradient
        diff = resized[:, 1:] > resized[:, :-1]

        # Convert to hex hash
        return ''.join('1' if val else '0' for val in diff.flatten())

    hash1 = compute_dhash(frame1)
    hash2 = compute_dhash(frame2)

    # Hamming distance (number of different bits)
    distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    # Normalize to 0-100 (64 bits total for 8x8 hash)
    max_distance = len(hash1)
    return (distance / max_distance) * 100


def calculate_comprehensive_quality_score(
    frames: list[np.ndarray],
    weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """Calculate comprehensive quality metrics across a sequence.

    Evaluates:
    - Color preservation across iterations
    - Temporal consistency between frames
    - Overall quality degradation rate

    Args:
        frames: List of frames as RGB numpy arrays
        weights: Optional weights for each metric (default: equal weighting)

    Returns:
        Dict containing:
        - 'color_scores': List of color preservation scores
        - 'temporal_scores': List of frame-to-frame consistency scores
        - 'overall_score': Weighted average quality score
        - 'degradation_rate': How fast quality degrades per iteration

    Example:
        >>> frames = [load_frame(i) for i in range(10)]
        >>> metrics = calculate_comprehensive_quality_score(frames)
        >>> print(f"Overall quality: {metrics['overall_score']:.1f}/100")
        >>> print(f"Degradation rate: {metrics['degradation_rate']:.2f}%/iteration")
    """
    if weights is None:
        weights = {
            'color': 0.5,
            'temporal': 0.5,
        }

    # Measure color preservation for each frame
    color_scores = [measure_color_preservation(frame) for frame in frames]

    # Measure temporal consistency between consecutive frames
    temporal_scores = []
    for i in range(len(frames) - 1):
        consistency = measure_temporal_consistency(frames[i], frames[i + 1])
        temporal_scores.append(consistency)

    # Calculate degradation rate (linear regression slope)
    if len(color_scores) > 1:
        iterations = np.arange(len(color_scores))
        # Fit line: y = mx + b
        slope, _ = np.polyfit(iterations, color_scores, 1)
        degradation_rate = abs(slope)  # Positive = degrading
    else:
        degradation_rate = 0.0

    # Calculate overall score (weighted average)
    avg_color = np.mean(color_scores) if color_scores else 0
    avg_temporal = np.mean(temporal_scores) if temporal_scores else 100

    # Special case: if no frames at all, overall score should be 0
    if not frames:
        overall_score = 0
    else:
        overall_score = (
            weights['color'] * avg_color +
            weights['temporal'] * avg_temporal
        )

    return {
        'color_scores': color_scores,
        'temporal_scores': temporal_scores,
        'overall_score': overall_score,
        'degradation_rate': degradation_rate,
        'avg_color': avg_color,
        'avg_temporal': avg_temporal,
        'num_frames': len(frames),
    }


def load_image_as_numpy(path: Path) -> np.ndarray:
    """Load image file as RGB numpy array.

    Args:
        path: Path to image file

    Returns:
        RGB numpy array (H, W, 3) in range [0, 255] as uint8
    """
    img = Image.open(path).convert('RGB')
    return np.array(img)
