"""Parameter tuning for depth warping with orbital camera paths.

This test sweeps translation/rotation factors to find optimal values for subject
stability during I2I depth warping chains. Different aspect ratios may require
different factors.

**Goal:** Empirically determine optimal rotation factors for coordinated orbital
camera paths that maintain subject position and temporal consistency.
"""

import pytest
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import glob

from .metrics import (
    measure_temporal_consistency,
    calculate_comprehensive_quality_score,
    load_image_as_numpy,
)

# Import shared test utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from integration.utils import (
    API_BASE_URL,
    get_test_options_overrides,
    wait_for_job_to_complete,
    get_test_batch_name,
)

# Use Forge's standard outputs directory
import os
FORGE_ROOT = Path(os.getcwd())  # Forge webui root directory
OUTPUT_DIR = FORGE_ROOT / "outputs" / "deforum-tuning" / "depth_warping_orbits"


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
    rotation_factor: float
) -> dict:
    """Generate circular orbit camera schedules.

    Creates circular translation + counter-rotation to keep subject centered.

    Args:
        num_frames: Number of frames
        radius: Orbit radius in pixels
        rotation_factor: Translation/rotation ratio (e.g., -5.0)

    Returns:
        Dict with schedule strings for translation_x, translation_y, rotation_3d_y
    """
    angles = np.linspace(0, 2 * np.pi, num_frames + 1)[:-1]  # Exclude duplicate endpoint

    # Calculate ABSOLUTE positions first
    x_positions_abs = radius * np.cos(angles)
    y_positions_abs = radius * np.sin(angles)

    # Convert to DELTAS (frame-to-frame changes) since Deforum expects per-frame movement
    # Frame 0 starts at origin, subsequent frames show delta from previous frame
    x_positions = np.zeros(num_frames)
    y_positions = np.zeros(num_frames)
    x_positions[0] = x_positions_abs[0]  # First frame: move to starting position
    y_positions[0] = y_positions_abs[0]
    for i in range(1, num_frames):
        x_positions[i] = x_positions_abs[i] - x_positions_abs[i-1]
        y_positions[i] = y_positions_abs[i] - y_positions_abs[i-1]

    # Calculate rotation angles to look at center (ABSOLUTE first)
    # Rotation should counter the translation direction
    # For counter-clockwise orbit (positive angles), camera rotates clockwise (negative Y rotation)
    # rotation_factor = -1 gives perfect orbit (360° travel = 360° counter-rotation)
    # rotation_factor = -5 gives under-rotation (360° travel = 72° counter-rotation)
    rotation_angles_abs = np.degrees(angles) / rotation_factor

    # Convert rotation to DELTAS too!
    rotation_angles = np.zeros(num_frames)
    rotation_angles[0] = rotation_angles_abs[0]
    for i in range(1, num_frames):
        rotation_angles[i] = rotation_angles_abs[i] - rotation_angles_abs[i-1]

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


@pytest.mark.parametrize("aspect_ratio,width,height,rotation_factor", [
    # Test empirically validated optimal range (-6.0 to -8.5)
    # Per CLAUDE.md: rotation_factor = -8.0 is empirically optimal
    # Values outside this range (-3.0, -4.0, -5.0) show excessive drift (expected)

    # Landscape (16:9) - most common use case
    (16/9, 512, 288, -6.0),  # Lower bound of optimal range
    (16/9, 512, 288, -7.0),  # Within optimal range
    (16/9, 512, 288, -8.0),  # Empirical optimal

    # Portrait (9:16) - may need different factor
    (9/16, 288, 512, -7.0),  # Within optimal range
    (9/16, 288, 512, -8.0),  # Empirical optimal

    # Square (1:1) - baseline comparison
    (1.0, 512, 512, -8.0),  # Empirical optimal
])
def test_orbit_rotation_factor_sweep(aspect_ratio, width, height, rotation_factor):
    """Sweep rotation factors for orbital camera paths with depth warping.

    This test empirically measures subject stability for different rotation factors
    across various aspect ratios. Results help tune optimal parameters for camera
    path generation and optimization.

    Metrics:
    - Subject position drift (should stay centered)
    - Temporal consistency (frame-to-frame SSIM)
    - Depth map consistency (depth should be stable)

    Args:
        aspect_ratio: Width/height ratio
        width: Frame width in pixels
        height: Frame height in pixels
        rotation_factor: Translation/rotation ratio (negative = counter-rotation)
    """
    # Create test output directory
    aspect_str = f"{int(aspect_ratio*100):03d}"
    test_name = f"aspect{aspect_str}_{width}x{height}_factor{abs(rotation_factor):.1f}"
    test_dir = OUTPUT_DIR / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Test configuration
    num_frames = 20  # 20 I2I depth warp iterations
    radius = 50.0  # Small radius for gentle orbit

    # Generate orbit schedules
    schedules = generate_orbit_schedules(num_frames, radius, rotation_factor)

    # Load base settings template
    from pathlib import Path
    testdata_dir = Path(__file__).parent.parent / 'integration' / 'testdata'
    with open(testdata_dir / 'simple.input_settings.txt', 'r') as f:
        base_settings = json.load(f)

    # Configure job - use custom output directory for this specific test
    # (not the shared test output directory)
    options_overrides = {
        "outdir_samples": str(test_dir),  # Direct to test-specific directory
        "deforum_save_gen_info_as_srt": False,
    }

    # Override specific settings for this test
    base_settings.update({
            # Basic settings
            "W": width,
            "H": height,
            "seed": 42,
            "sampler": "euler",
            "steps": 20,
            "cfg_scale": 1.0,
            "distilled_cfg_scale": 3.5,

            # Animation: 3D with depth warping
            "animation_mode": "3D",
            "render_mode": "new_3d",
            "max_frames": num_frames,
            "fps": 24,
            "save_depth_maps": True,

            # Camera movement (orbital path)
            "translation_x": schedules['translation_x'],
            "translation_y": schedules['translation_y'],
            "rotation_3d_y": schedules['rotation_3d_y'],
            "translation_z": "0:(0)",

            # Prompt
            "animation_prompts": json.dumps({
                "0": "a detailed 3D render of a colorful geometric sculpture, studio lighting"
            }),

            # Disable audio
            "audio_mode": "None",
            "audio_sync": False,

            # Output - batch_name will create subdirectory under outdir_samples
            "batch_name": get_test_batch_name(test_name),
    })

    # Construct settings dict for API
    settings = {
        "deforum_settings": base_settings,
        "options_overrides": options_overrides,
    }

    print(f"\n{'='*60}")
    print(f"Testing Rotation Factor: {rotation_factor}")
    print(f"Aspect Ratio: {aspect_ratio:.2f} ({width}x{height})")
    print(f"Orbit Radius: {radius}px, Frames: {num_frames}")
    print(f"{'='*60}")

    # Submit job
    response = requests.post(f"{API_BASE_URL}/batches/", json=settings)
    assert response.status_code in [200, 202], f"Failed to submit job: {response.text}"

    batch_info = response.json()
    job_ids = batch_info["job_ids"]

    # Wait for completion
    from deforum.api.models import DeforumJobStatusCategory
    final_status = wait_for_job_to_complete(job_ids[0])
    assert final_status.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job failed: {final_status.message}"

    # Get output directory - need to construct full batch name
    # get_test_batch_name returns "module-testname_{timestring}" pattern
    # Deforum replaces {timestring} with actual value, creating the directory name
    timestring = final_status.timestring
    batch_name_pattern = get_test_batch_name(test_name)
    batch_name_actual = batch_name_pattern.replace("{timestring}", timestring)
    output_frames_dir = test_dir / batch_name_actual

    # Load generated frames
    frame_files = sorted(glob.glob(str(output_frames_dir / "*.png")))
    frames = [np.array(Image.open(f).convert('RGB')) for f in frame_files]

    # Load depth maps
    depth_dir = output_frames_dir / "depth-maps"
    depth_files = sorted(glob.glob(str(depth_dir / "*.png")))
    depth_maps = [np.array(Image.open(f).convert('L')) for f in depth_files]

    # Measure metrics
    drift_metrics = measure_subject_position_drift(frames)
    quality_metrics = calculate_comprehensive_quality_score(frames)
    depth_metrics = measure_depth_map_consistency(depth_maps)

    # Save results
    results = {
        'parameters': {
            'aspect_ratio': aspect_ratio,
            'width': width,
            'height': height,
            'rotation_factor': rotation_factor,
            'radius': radius,
            'num_frames': num_frames,
        },
        'drift_metrics': {
            'max_drift': drift_metrics['max_drift'],
            'avg_drift': drift_metrics['avg_drift'],
            'drift_rate': drift_metrics['drift_rate'],
        },
        'quality_metrics': {
            'avg_temporal': quality_metrics['avg_temporal'],
            'avg_color': quality_metrics['avg_color'],
            'degradation_rate': quality_metrics['degradation_rate'],
        },
        'depth_metrics': {
            'avg_consistency': depth_metrics['avg_consistency'],
            'min_consistency': depth_metrics['min_consistency'],
        },
    }

    results_file = test_dir / f"metrics_{timestring}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for rotation_factor={rotation_factor}:")
    print(f"{'='*60}")
    print(f"Subject Position Drift:")
    print(f"  Max drift: {drift_metrics['max_drift']:.1f}px")
    print(f"  Avg drift: {drift_metrics['avg_drift']:.1f}px")
    print(f"  Drift rate: {drift_metrics['drift_rate']:.2f}px/frame")
    print(f"\nTemporal Consistency:")
    print(f"  Avg: {quality_metrics['avg_temporal']:.1f}%")
    print(f"  Color preservation: {quality_metrics['avg_color']:.1f}%")
    print(f"  Degradation rate: {quality_metrics['degradation_rate']:.2f}%/iteration")
    print(f"\nDepth Map Consistency:")
    print(f"  Avg: {depth_metrics['avg_consistency']:.1f}%")
    print(f"  Min: {depth_metrics['min_consistency']:.1f}%")
    print(f"{'='*60}\n")

    # Assertions for stability
    max_allowed_drift = min(width, height) * 0.25  # Allow 25% drift
    assert drift_metrics['max_drift'] < max_allowed_drift, (
        f"Subject drifted {drift_metrics['max_drift']:.1f}px "
        f"(max allowed: {max_allowed_drift:.1f}px)"
    )

    # Temporal consistency should be reasonable
    assert quality_metrics['avg_temporal'] > 75, (
        f"Temporal consistency too low: {quality_metrics['avg_temporal']:.1f}%"
    )

    # Depth should be reasonably consistent
    assert depth_metrics['avg_consistency'] > 80, (
        f"Depth consistency too low: {depth_metrics['avg_consistency']:.1f}%"
    )
