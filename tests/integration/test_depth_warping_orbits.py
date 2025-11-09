"""Integration tests for depth warping with orbital camera paths.

Tests real I2I depth warping chains to validate translation/rotation factors
and measure subject position stability over many depth warp iterations.

These tests use GPU and require the Deforum API server to be running.
"""

import glob
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import pytest
import requests
import numpy as np
from PIL import Image

from .utils import (
    API_BASE_URL,
    get_test_options_overrides,
    gpu_disabled,
    wait_for_job_to_complete,
    get_test_batch_name
)
from deforum.api.models import DeforumJobStatusCategory
from tests.tuning.metrics import (
    measure_temporal_consistency,
    calculate_comprehensive_quality_score
)


# Path to testdata directory
TESTDATA_DIR = Path(__file__).parent / 'testdata'


def measure_subject_position_drift(frames: List[np.ndarray]) -> dict:
    """Measure how much the subject drifts from center over iterations.

    Uses centroid tracking of salient regions to measure drift.
    For rotate-around paths, subject should stay centered.

    Args:
        frames: List of RGB frames as numpy arrays

    Returns:
        Dict with:
        - 'centroid_positions': List of (x, y) centroids
        - 'drift_from_center': List of distances from image center
        - 'max_drift': Maximum drift in pixels
        - 'avg_drift': Average drift across all frames
        - 'drift_rate': Linear drift rate (pixels/frame)
    """
    import cv2

    centroids = []
    h, w = frames[0].shape[:2]
    center = np.array([w / 2, h / 2])

    for frame in frames:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold to find salient regions (Otsu's method)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate moments to find centroid
        moments = cv2.moments(binary)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            centroids.append(np.array([cx, cy]))
        else:
            # Fallback to center if no features found
            centroids.append(center.copy())

    # Calculate drift from center for each frame
    drift_distances = [np.linalg.norm(c - center) for c in centroids]

    # Calculate drift rate (linear regression)
    if len(drift_distances) > 1:
        iterations = np.arange(len(drift_distances))
        slope, _ = np.polyfit(iterations, drift_distances, 1)
        drift_rate = abs(slope)
    else:
        drift_rate = 0.0

    return {
        'centroid_positions': centroids,
        'drift_from_center': drift_distances,
        'max_drift': max(drift_distances) if drift_distances else 0,
        'avg_drift': np.mean(drift_distances) if drift_distances else 0,
        'drift_rate': drift_rate,
    }


def measure_depth_map_consistency(depth_maps: List[np.ndarray]) -> dict:
    """Measure consistency of depth maps over iterations.

    Depth maps should remain relatively stable for orbital paths
    where the subject doesn't move.

    Args:
        depth_maps: List of depth maps as grayscale numpy arrays

    Returns:
        Dict with:
        - 'consistency_scores': Frame-to-frame SSIM scores
        - 'avg_consistency': Average consistency
        - 'min_consistency': Worst frame transition
    """
    from skimage.metrics import structural_similarity as ssim

    consistency_scores = []
    for i in range(len(depth_maps) - 1):
        score = ssim(depth_maps[i], depth_maps[i + 1]) * 100
        consistency_scores.append(score)

    return {
        'consistency_scores': consistency_scores,
        'avg_consistency': np.mean(consistency_scores) if consistency_scores else 100,
        'min_consistency': min(consistency_scores) if consistency_scores else 100,
    }


def generate_orbit_schedules(
    num_frames: int,
    radius: float,
    aspect_ratio: float = 16/9,
    rotation_factor: float = -5.0
) -> dict:
    """Generate circular orbit camera schedules.

    Creates circular translation + counter-rotation to keep subject centered.

    Args:
        num_frames: Number of frames
        radius: Orbit radius in pixels
        aspect_ratio: Width/height ratio (affects optimal rotation factor)
        rotation_factor: Translation/rotation ratio (typically -5.0)

    Returns:
        Dict with schedule strings for translation_x, translation_y, rotation_3d_y
    """
    angles = np.linspace(0, 2 * np.pi, num_frames + 1)[:-1]  # Exclude duplicate endpoint

    # Calculate positions
    x_positions = radius * np.cos(angles)
    y_positions = radius * np.sin(angles)

    # Calculate rotation angles to look at center
    # Rotation should counter the translation direction
    rotation_angles = -np.degrees(angles) / rotation_factor

    # Create schedule strings (sample every few frames)
    sample_interval = max(1, num_frames // 20)

    def create_schedule(values: np.ndarray) -> str:
        keyframes = []
        for i in range(0, len(values), sample_interval):
            keyframes.append(f"{i}:({values[i]:.2f})")
        # Always include last frame
        if (len(values) - 1) % sample_interval != 0:
            keyframes.append(f"{len(values)-1}:({values[-1]:.2f})")
        return ", ".join(keyframes)

    return {
        'translation_x': create_schedule(x_positions),
        'translation_y': create_schedule(y_positions),
        'rotation_3d_y': create_schedule(rotation_angles),
    }


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU for depth warping")
@pytest.mark.parametrize("aspect_ratio,width,height", [
    (16/9, 512, 288),   # Landscape (16:9)
    (9/16, 288, 512),   # Portrait (9:16)
    (1.0, 512, 512),    # Square
])
def test_orbit_subject_stability_by_aspect_ratio(aspect_ratio, width, height):
    """Test that subject stays centered during orbital path at different aspect ratios.

    Different aspect ratios may require different translation/rotation factors
    for optimal subject stability.
    """
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as f:
        settings = json.load(f)

    # Test configuration
    num_frames = 20
    radius = 50.0  # Small radius for ~20 I2I iterations
    rotation_factor = -5.0  # Empirically validated from unit tests

    test_name = f"orbit_stability_{int(aspect_ratio*100)}_{width}x{height}"
    settings['batch_name'] = get_test_batch_name(test_name)

    # Configure 3D orbit
    settings['animation_mode'] = "3D"
    settings['max_frames'] = num_frames
    settings['W'] = width
    settings['H'] = height
    settings['save_depth_maps'] = True

    # Generate orbit schedules
    schedules = generate_orbit_schedules(num_frames, radius, aspect_ratio, rotation_factor)
    settings['translation_x'] = schedules['translation_x']
    settings['translation_y'] = schedules['translation_y']
    settings['rotation_3d_y'] = schedules['rotation_3d_y']
    settings['translation_z'] = "0:(0)"  # No z movement

    # Submit job
    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    job_status = wait_for_job_to_complete(job_id)

    assert job_status.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job failed: {job_status.message}"

    # Load generated frames
    frame_files = sorted(glob.glob(os.path.join(job_status.outdir, "*.png")))
    frames = [np.array(Image.open(f).convert('RGB')) for f in frame_files]

    # Load depth maps
    depth_dir = os.path.join(job_status.outdir, "depth-maps")
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    depth_maps = [np.array(Image.open(f).convert('L')) for f in depth_files]

    # Measure subject position stability
    drift_metrics = measure_subject_position_drift(frames)

    # Measure temporal consistency
    quality_metrics = calculate_comprehensive_quality_score(frames)

    # Measure depth map consistency
    depth_metrics = measure_depth_map_consistency(depth_maps)

    # Log results
    print(f"\n{'='*60}")
    print(f"Aspect Ratio: {aspect_ratio:.2f} ({width}x{height})")
    print(f"Orbit Radius: {radius}px, Rotation Factor: {rotation_factor}")
    print(f"{'='*60}")
    print(f"Subject Position Drift:")
    print(f"  Max drift: {drift_metrics['max_drift']:.1f}px")
    print(f"  Avg drift: {drift_metrics['avg_drift']:.1f}px")
    print(f"  Drift rate: {drift_metrics['drift_rate']:.2f}px/frame")
    print(f"\nTemporal Consistency:")
    print(f"  Avg: {quality_metrics['avg_temporal']:.1f}%")
    print(f"  Color preservation: {quality_metrics['avg_color']:.1f}%")
    print(f"\nDepth Map Consistency:")
    print(f"  Avg: {depth_metrics['avg_consistency']:.1f}%")
    print(f"  Min: {depth_metrics['min_consistency']:.1f}%")
    print(f"{'='*60}\n")

    # Assertions for subject stability
    # Subject should stay relatively centered (max drift < 20% of smaller dimension)
    max_allowed_drift = min(width, height) * 0.20
    assert drift_metrics['max_drift'] < max_allowed_drift, (
        f"Subject drifted {drift_metrics['max_drift']:.1f}px (max allowed: {max_allowed_drift:.1f}px)"
    )

    # Drift rate should be low (< 2px per frame)
    assert drift_metrics['drift_rate'] < 2.0, (
        f"Drift rate too high: {drift_metrics['drift_rate']:.2f}px/frame"
    )

    # Temporal consistency should be high (> 80%)
    assert quality_metrics['avg_temporal'] > 80, (
        f"Temporal consistency too low: {quality_metrics['avg_temporal']:.1f}%"
    )

    # Depth maps should be consistent (> 85%)
    assert depth_metrics['avg_consistency'] > 85, (
        f"Depth consistency too low: {depth_metrics['avg_consistency']:.1f}%"
    )


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU for depth warping")
@pytest.mark.slow
@pytest.mark.parametrize("rotation_factor", [-3.0, -4.0, -5.0, -6.0, -7.0])
def test_rotation_factor_sweep(rotation_factor):
    """Sweep translation/rotation factors to find optimal value.

    This is a comprehensive parameter sweep test to empirically determine
    the best translation/rotation ratio for subject stability.

    Mark as slow since it runs 5 full depth warping sequences.
    """
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as f:
        settings = json.load(f)

    # Test configuration
    num_frames = 20
    radius = 50.0
    width, height = 512, 288  # 16:9 landscape

    test_name = f"rotation_factor_sweep_{abs(rotation_factor):.1f}"
    settings['batch_name'] = get_test_batch_name(test_name)

    # Configure 3D orbit
    settings['animation_mode'] = "3D"
    settings['max_frames'] = num_frames
    settings['W'] = width
    settings['H'] = height
    settings['save_depth_maps'] = True

    # Generate orbit schedules with this rotation factor
    schedules = generate_orbit_schedules(num_frames, radius, 16/9, rotation_factor)
    settings['translation_x'] = schedules['translation_x']
    settings['translation_y'] = schedules['translation_y']
    settings['rotation_3d_y'] = schedules['rotation_3d_y']
    settings['translation_z'] = "0:(0)"

    # Submit job
    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    job_status = wait_for_job_to_complete(job_id)

    assert job_status.status == DeforumJobStatusCategory.SUCCEEDED

    # Load frames and measure stability
    frame_files = sorted(glob.glob(os.path.join(job_status.outdir, "*.png")))
    frames = [np.array(Image.open(f).convert('RGB')) for f in frame_files]

    drift_metrics = measure_subject_position_drift(frames)
    quality_metrics = calculate_comprehensive_quality_score(frames)

    # Log results for analysis
    print(f"\nRotation Factor: {rotation_factor}")
    print(f"  Max drift: {drift_metrics['max_drift']:.1f}px")
    print(f"  Avg drift: {drift_metrics['avg_drift']:.1f}px")
    print(f"  Drift rate: {drift_metrics['drift_rate']:.2f}px/frame")
    print(f"  Temporal consistency: {quality_metrics['avg_temporal']:.1f}%")

    # Don't assert specific values - this is a sweep to find optimal factor
    # Results will be compared manually to find best value
