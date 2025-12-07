"""Quality metrics for DA3-3DGS evaluation."""

import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import imagehash

from deforum.utils.system.logging import get_logger

logger = get_logger()


def calculate_multi_view_ssim(images: List[Image.Image]) -> float:
    """Calculate average SSIM between consecutive images.

    Measures temporal consistency in rendered sequence.
    Higher = smoother, more consistent frames.

    Args:
        images: List of PIL Images (rendered frames)

    Returns:
        Average SSIM score (0.0-1.0)
    """
    if len(images) < 2:
        return 1.0  # Single image is perfectly consistent with itself

    ssim_scores = []
    for i in range(len(images) - 1):
        # Convert to numpy arrays
        img1 = np.array(images[i].convert('L'))  # Grayscale for SSIM
        img2 = np.array(images[i + 1].convert('L'))

        # Calculate SSIM
        score = ssim(img1, img2, data_range=255)
        ssim_scores.append(score)

    avg_ssim = np.mean(ssim_scores)
    logger.debug(f"Multi-view SSIM: {avg_ssim:.4f} (over {len(ssim_scores)} frame pairs)")
    return float(avg_ssim)


def calculate_perceptual_hash_similarity(
    rendered_images: List[Image.Image],
    original_keyframes: List[Image.Image]
) -> float:
    """Calculate average perceptual hash similarity to original keyframes.

    Measures how well rendered frames match original diffusion keyframes.
    Lower distance = better fidelity to originals.

    Args:
        rendered_images: List of 3DGS rendered frames
        original_keyframes: List of original diffusion keyframes

    Returns:
        Average hamming distance (0.0 = perfect match, higher = more different)
    """
    if not rendered_images or not original_keyframes:
        return 0.0

    # Calculate perceptual hashes
    rendered_hashes = [imagehash.phash(img) for img in rendered_images]
    original_hashes = [imagehash.phash(img) for img in original_keyframes]

    # Find closest original for each rendered frame
    distances = []
    for rendered_hash in rendered_hashes:
        min_dist = min(rendered_hash - orig_hash for orig_hash in original_hashes)
        distances.append(min_dist)

    avg_distance = np.mean(distances)
    logger.debug(f"Perceptual hash distance: {avg_distance:.2f} (lower = better)")
    return float(avg_distance)


def calculate_color_consistency(images: List[Image.Image]) -> float:
    """Calculate color consistency across frame sequence.

    Measures color drift over time using standard deviation of mean RGB.
    Lower = more consistent colors.

    Args:
        images: List of PIL Images

    Returns:
        Average std deviation of RGB channels (0.0 = perfect consistency)
    """
    if len(images) < 2:
        return 0.0

    # Extract mean RGB for each frame
    mean_colors = []
    for img in images:
        rgb = np.array(img.convert('RGB'))
        mean_r = rgb[:, :, 0].mean()
        mean_g = rgb[:, :, 1].mean()
        mean_b = rgb[:, :, 2].mean()
        mean_colors.append([mean_r, mean_g, mean_b])

    mean_colors = np.array(mean_colors)

    # Calculate std deviation across time for each channel
    std_r = mean_colors[:, 0].std()
    std_g = mean_colors[:, 1].std()
    std_b = mean_colors[:, 2].std()

    avg_std = (std_r + std_g + std_b) / 3
    logger.debug(f"Color consistency std: {avg_std:.2f} (R={std_r:.2f}, G={std_g:.2f}, B={std_b:.2f})")
    return float(avg_std)


def calculate_depth_correlation(
    depth_maps_1: np.ndarray,
    depth_maps_2: np.ndarray
) -> float:
    """Calculate Pearson correlation between two sets of depth maps.

    Measures geometric accuracy of depth estimation.
    Higher = better geometric consistency.

    Args:
        depth_maps_1: First set of depth maps [N, H, W]
        depth_maps_2: Second set of depth maps [N, H, W]

    Returns:
        Average Pearson correlation coefficient (-1.0 to 1.0)
    """
    if depth_maps_1.shape != depth_maps_2.shape:
        logger.warning(f"Depth map shape mismatch: {depth_maps_1.shape} vs {depth_maps_2.shape}")
        return 0.0

    correlations = []
    for i in range(len(depth_maps_1)):
        # Flatten depth maps
        d1 = depth_maps_1[i].flatten()
        d2 = depth_maps_2[i].flatten()

        # Calculate Pearson correlation
        corr, _ = pearsonr(d1, d2)
        correlations.append(corr)

    avg_corr = np.mean(correlations)
    logger.debug(f"Depth correlation: {avg_corr:.4f} (over {len(correlations)} depth maps)")
    return float(avg_corr)


def calculate_splat_coverage(
    rendered_images: List[Image.Image],
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> float:
    """Calculate percentage of frame covered by gaussian splats.

    Measures scene coverage - too low indicates holes/gaps.

    Args:
        rendered_images: List of 3DGS rendered frames
        background_color: RGB tuple of background (default black)

    Returns:
        Average coverage percentage (0.0-100.0)
    """
    if not rendered_images:
        return 0.0

    coverages = []
    for img in rendered_images:
        rgb = np.array(img.convert('RGB'))

        # Find pixels that match background color
        background_mask = (
            (rgb[:, :, 0] == background_color[0]) &
            (rgb[:, :, 1] == background_color[1]) &
            (rgb[:, :, 2] == background_color[2])
        )

        # Coverage = non-background pixels
        total_pixels = rgb.shape[0] * rgb.shape[1]
        covered_pixels = total_pixels - background_mask.sum()
        coverage = (covered_pixels / total_pixels) * 100

        coverages.append(coverage)

    avg_coverage = np.mean(coverages)
    logger.debug(f"Splat coverage: {avg_coverage:.1f}% (over {len(coverages)} frames)")
    return float(avg_coverage)


def evaluate_3dgs_quality(
    rendered_frames: List[Image.Image],
    original_keyframes: List[Image.Image] = None,
    depth_maps_rendered: np.ndarray = None,
    depth_maps_original: np.ndarray = None
) -> Dict[str, float]:
    """Comprehensive 3DGS quality evaluation.

    Args:
        rendered_frames: List of 3DGS rendered frames
        original_keyframes: Optional list of original diffusion keyframes
        depth_maps_rendered: Optional rendered depth maps [N, H, W]
        depth_maps_original: Optional original depth maps [N, H, W]

    Returns:
        Dictionary of metric name -> score
    """
    metrics = {}

    # Visual metrics
    metrics['temporal_consistency'] = calculate_multi_view_ssim(rendered_frames)
    metrics['color_consistency'] = calculate_color_consistency(rendered_frames)
    metrics['splat_coverage'] = calculate_splat_coverage(rendered_frames)

    # Fidelity to originals
    if original_keyframes:
        metrics['perceptual_similarity'] = calculate_perceptual_hash_similarity(
            rendered_frames, original_keyframes
        )

    # Geometric accuracy
    if depth_maps_rendered is not None and depth_maps_original is not None:
        metrics['depth_correlation'] = calculate_depth_correlation(
            depth_maps_rendered, depth_maps_original
        )

    # Log summary
    logger.info("3DGS Quality Metrics:")
    for name, score in metrics.items():
        logger.info(f"  {name}: {score:.4f}")

    return metrics


def calculate_quality_score(metrics: Dict[str, float]) -> float:
    """Calculate overall quality score from individual metrics.

    Weighted combination of metrics into single score for ranking.

    Args:
        metrics: Dictionary of metric name -> score

    Returns:
        Overall quality score (0.0-100.0, higher = better)
    """
    # Weights for each metric (tunable based on importance)
    weights = {
        'temporal_consistency': 0.30,  # Most important: smooth animation
        'color_consistency': 0.15,      # Moderate: avoid drift
        'splat_coverage': 0.15,         # Moderate: avoid holes
        'perceptual_similarity': 0.25,  # Important: match originals
        'depth_correlation': 0.15,      # Moderate: geometric accuracy
    }

    # Normalize and weight metrics
    score = 0.0
    total_weight = 0.0

    for metric_name, weight in weights.items():
        if metric_name in metrics:
            value = metrics[metric_name]

            # Normalize metrics to 0-1 range
            if metric_name == 'temporal_consistency':
                normalized = value  # Already 0-1
            elif metric_name == 'color_consistency':
                normalized = 1.0 / (1.0 + value / 50.0)  # Lower is better, ~50 is typical
            elif metric_name == 'splat_coverage':
                normalized = value / 100.0  # Convert percentage to 0-1
            elif metric_name == 'perceptual_similarity':
                normalized = 1.0 / (1.0 + value / 10.0)  # Lower distance is better
            elif metric_name == 'depth_correlation':
                normalized = (value + 1.0) / 2.0  # Convert -1 to 1 → 0 to 1
            else:
                normalized = value  # Default: assume already normalized

            score += normalized * weight
            total_weight += weight

    # Normalize by total weight (in case some metrics missing)
    if total_weight > 0:
        score = (score / total_weight) * 100.0

    logger.info(f"Overall Quality Score: {score:.1f}/100.0")
    return float(score)
