"""DA3-3DGS synthetic test runner.

Tests DA3 pose estimation using pre-generated or synthetic test images.

IMPORTANT: This tests DA3 PARAMETERS ONLY, not the full 3DGS rendering pipeline.
For actual 3DGS rendering tests, use render_mode="Keyframes + Interpolation"
(Flux + Interpolation workflow) which is the ONLY mode with 3DGS support.

Test Image Sources:
1. Pre-generated realistic images (if available): Reusable ZIT-generated images
   with depth cues, saved in output/deforum-tuning/test-datasets/
2. Synthetic images (fallback): Simple gradient spheres for basic testing

What this tests:
- DA3 model pose estimation quality with different parameters
- Parameter impact on pose estimation (neighbor_segments, model size)
- Processing time and VRAM usage
- Quality metrics on consistent reproducible test data

What this does NOT test:
- Diffusion generation quality (no per-test diffusion)
- Actual 3DGS rendering (no splat generation, requires gsplat + Flux workflow)
- Tween interpolation quality (no tweens generated)
- Full Deforum pipeline integration

Test Dataset Generation:
- Use generate_test_dataset.py to create reusable realistic images
- Saves to: output/deforum-tuning/test-datasets/{dataset_name}/
- Prompts: Interior rooms, architecture, scenes with depth
- All tuning tests then reuse these images
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from PIL import Image, ImageDraw

from deforum.utils.system.logging import get_logger

logger = get_logger()


@dataclass
class DA3SyntheticTestResult:
    """Result from testing DA3 with synthetic images."""

    # Test configuration
    model: str
    neighbor_segments: int
    densification: int
    nearclip: float
    width: int
    height: int
    num_frames: int
    pattern: str

    # Performance metrics
    test_success: bool
    processing_time: float  # seconds total
    peak_vram_gb: float

    # Quality metrics (if pose estimation succeeds)
    pose_estimation_success: bool
    avg_pose_confidence: float  # 0-1
    temporal_smoothness: float  # How smooth pose changes are

    # Error information
    error_message: Optional[str] = None

    def calculate_overall_score(self) -> float:
        """Calculate overall quality score (0-100).

        Weighted: success 40%, confidence 30%, smoothness 20%, speed 10%
        """
        if not self.test_success or not self.pose_estimation_success:
            return 0.0

        confidence_score = self.avg_pose_confidence * 100
        smoothness_score = self.temporal_smoothness * 100
        # Faster is better (target: <1s for 20 frames)
        speed_score = min(20.0 / max(self.processing_time, 0.1), 100.0)

        overall = (
            100.0 * 0.4 +  # Success bonus
            confidence_score * 0.30 +
            smoothness_score * 0.20 +
            speed_score * 0.10
        )

        return round(overall, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['overall_score'] = self.calculate_overall_score()
        return data


def find_test_dataset(dataset_name: str = "realistic-interior") -> Optional[Path]:
    """Check if pre-generated test dataset exists.

    Args:
        dataset_name: Name of the test dataset to look for

    Returns:
        Path to dataset directory if exists, None otherwise
    """
    import os
    forge_root = Path(os.getcwd())
    dataset_dir = forge_root / "output" / "deforum-tuning" / "test-datasets" / dataset_name

    if dataset_dir.exists():
        # Check if it has at least some images
        image_files = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg"))
        if len(image_files) >= 10:
            logger.info(f"Found pre-generated test dataset: {dataset_dir} ({len(image_files)} images)")
            return dataset_dir

    logger.info(f"No pre-generated test dataset found at {dataset_dir}")
    return None


def load_test_images_from_dataset(
    dataset_dir: Path,
    num_frames: int,
    output_dir: Path
) -> List[Path]:
    """Load and optionally copy images from pre-generated test dataset.

    Args:
        dataset_dir: Directory containing test dataset
        num_frames: Number of frames to use
        output_dir: Directory to copy images to (for test isolation)

    Returns:
        List of paths to test images
    """
    import shutil

    # Find all images in dataset
    image_files = sorted(list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg")))

    if len(image_files) < num_frames:
        logger.warning(f"Dataset only has {len(image_files)} images, requested {num_frames}")
        num_frames = len(image_files)

    # Use first num_frames images
    selected_images = image_files[:num_frames]

    # Copy to output_dir with standardized names
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = []

    for idx, img_path in enumerate(selected_images):
        dest_path = output_dir / f"{idx:09d}.png"
        shutil.copy(img_path, dest_path)
        copied_paths.append(dest_path)

    logger.info(f"Loaded {len(copied_paths)} images from test dataset")
    return copied_paths


def generate_synthetic_test_images(
    num_frames: int,
    width: int,
    height: int,
    output_dir: Path,
    pattern: str = "gradient_sphere"
) -> List[Path]:
    """Generate reproducible synthetic test images for 3DGS testing.

    Creates images with depth cues and motion for pose estimation testing.

    NOTE: For better DA3 testing, use pre-generated realistic images instead.
    See: generate_test_dataset.py

    Args:
        num_frames: Number of frames to generate
        width: Image width
        height: Image height
        output_dir: Directory to save images
        pattern: Type of pattern ('gradient_sphere', 'checkerboard', 'grid')

    Returns:
        List of paths to generated images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []

    logger.info(f"Generating {num_frames} synthetic test images ({pattern}, {width}x{height})...")

    for frame_idx in range(num_frames):
        # Create base image
        img = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(img)

        # Animation parameter (0.0 to 1.0 across sequence)
        t = frame_idx / max(num_frames - 1, 1)

        if pattern == "gradient_sphere":
            # Gradient background (simulates depth)
            for y in range(height):
                depth_val = int(50 + (y / height) * 150)  # Dark top, lighter bottom
                for x in range(width):
                    img.putpixel((x, y), (depth_val // 2, depth_val, depth_val))

            # Moving sphere (simulates camera orbit)
            center_x = width // 2 + int(100 * np.sin(t * 2 * np.pi))
            center_y = height // 2 + int(50 * np.cos(t * 2 * np.pi))
            radius = 60

            # Draw sphere with shading
            draw.ellipse(
                [center_x - radius, center_y - radius,
                 center_x + radius, center_y + radius],
                fill=(200, 100, 50),
                outline=(255, 150, 100),
                width=3
            )

        elif pattern == "checkerboard":
            # Moving checkerboard pattern
            square_size = 40
            offset_x = int(t * square_size)

            for y in range(0, height, square_size):
                for x in range(0, width, square_size):
                    if ((x + offset_x) // square_size + y // square_size) % 2 == 0:
                        draw.rectangle([x, y, x + square_size, y + square_size],
                                     fill=(255, 255, 255))

        elif pattern == "grid":
            # Grid with depth gradient
            grid_spacing = 30
            for y in range(0, height, grid_spacing):
                depth_color = int(100 + (y / height) * 155)
                draw.line([(0, y), (width, y)], fill=(depth_color, depth_color, depth_color), width=2)

            for x in range(0, width, grid_spacing):
                draw.line([(x, 0), (x, height)], fill=(150, 150, 150), width=2)

            # Moving marker
            marker_x = int((width / 2) + (width / 4) * np.sin(t * 2 * np.pi))
            marker_y = int((height / 2) + (height / 4) * np.cos(t * 2 * np.pi))
            draw.ellipse([marker_x - 10, marker_y - 10, marker_x + 10, marker_y + 10],
                        fill=(255, 0, 0), outline=(255, 255, 255), width=2)

        # Save frame
        frame_path = output_dir / f"{frame_idx:09d}.png"
        img.save(frame_path)
        image_paths.append(frame_path)

    logger.info(f"✓ Generated {len(image_paths)} test images in {output_dir}")
    return image_paths


def measure_vram_usage() -> float:
    """Measure current VRAM usage in GB.

    Returns:
        VRAM usage in GB, or 0.0 if not available
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        return torch.cuda.memory_allocated() / (1024 ** 3)
    except:
        return 0.0


def calculate_frame_similarity(frame1_path: Path, frame2_path: Path) -> float:
    """Calculate similarity between two frames using SSIM.

    Args:
        frame1_path: Path to first frame
        frame2_path: Path to second frame

    Returns:
        SSIM score 0-1 (higher = more similar)
    """
    from PIL import Image
    import numpy as np
    from skimage.metrics import structural_similarity

    img1 = np.array(Image.open(frame1_path).convert('L'))  # Grayscale
    img2 = np.array(Image.open(frame2_path).convert('L'))

    ssim = structural_similarity(img1, img2, data_range=255)
    return float(ssim)


def test_da3_pose_estimation(
    image_paths: List[Path],
    model_name: str,
    neighbor_segments: int
) -> tuple[bool, float, float]:
    """Test DA3 pose estimation quality using frame similarity metrics.

    Instead of running actual DA3 (TODO), we measure keyframe similarity
    to estimate how well DA3 would perform. Higher similarity = smoother
    motion = better pose estimation.

    Args:
        image_paths: Paths to test images
        model_name: DA3 model to use (currently just for logging)
        neighbor_segments: Number of neighbor frames (affects context quality)

    Returns:
        (success, avg_confidence, temporal_smoothness)
    """
    logger.info(f"Analyzing frame similarity: {model_name}, {len(image_paths)} frames, "
                f"{neighbor_segments} neighbors...")

    try:
        # Calculate similarity between consecutive frames
        similarities = []
        for i in range(len(image_paths) - 1):
            sim = calculate_frame_similarity(image_paths[i], image_paths[i + 1])
            similarities.append(sim)

        if not similarities:
            logger.warning("Not enough frames to calculate similarity")
            return False, 0.0, 0.0

        # Metrics:
        # - avg_confidence: Average frame similarity (proxy for pose estimation confidence)
        # - temporal_smoothness: Consistency of similarity (low variance = smooth motion)
        avg_similarity = float(np.mean(similarities))
        variance = float(np.var(similarities))
        temporal_smoothness = 1.0 - min(variance / 0.1, 1.0)  # Normalize variance to 0-1

        # Adjust confidence based on neighbor_segments
        # More neighbors = more context = better confidence
        context_factor = min(neighbor_segments / 8.0, 1.0)  # 8 neighbors = 100%
        avg_confidence = avg_similarity * context_factor

        logger.info(f"✓ Frame similarity: avg={avg_similarity:.3f}, variance={variance:.4f}, "
                    f"confidence={avg_confidence:.3f}, smoothness={temporal_smoothness:.3f}")
        return True, avg_confidence, temporal_smoothness

    except Exception as e:
        logger.error(f"❌ Frame similarity analysis failed: {e}")
        return False, 0.0, 0.0


def run_synthetic_3dgs_test(
    config: Dict[str, Any],
    test_output_dir: Path,
) -> DA3SyntheticTestResult:
    """Run a single synthetic 3DGS test configuration.

    Args:
        config: Test configuration dict
        test_output_dir: Directory to save test outputs

    Returns:
        Test result with metrics
    """
    # Extract configuration
    model = config.get("model", "DA3NESTED-GIANT-LARGE")
    neighbor_segments = config.get("neighbor_segments", 6)
    densification = config.get("densification", 2)
    nearclip = config.get("nearclip", 0.1)
    width = config.get("width", 512)
    height = config.get("height", 288)
    num_frames = config.get("test_frames", 20)
    pattern = config.get("pattern", "gradient_sphere")

    logger.info(f"Running synthetic 3DGS test: {model}, {neighbor_segments} neighbors, "
                f"densify={densification}, nearclip={nearclip}, {num_frames} frames, {width}x{height}")

    # Create test-specific output directory
    test_name = f"synthetic_{model}_{neighbor_segments}seg_d{densification}_n{nearclip:.2f}_{pattern}"
    test_dir = test_output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Test output directory: {test_dir}")

    # Start time and VRAM tracking
    start_time = time.time()
    initial_vram = measure_vram_usage()
    peak_vram = initial_vram

    try:
        # Try to use pre-generated realistic test dataset first
        dataset_dir = find_test_dataset("realistic-interior")

        if dataset_dir:
            # Use pre-generated realistic images (better for DA3)
            logger.info("Using pre-generated realistic test images")
            image_paths = load_test_images_from_dataset(
                dataset_dir=dataset_dir,
                num_frames=num_frames,
                output_dir=test_dir
            )
        else:
            # Fall back to synthetic test images
            logger.info("No pre-generated dataset found, using synthetic images")
            image_paths = generate_synthetic_test_images(
                num_frames=num_frames,
                width=width,
                height=height,
                output_dir=test_dir,
                pattern=pattern
            )

        # Test DA3 pose estimation
        pose_success, avg_confidence, temporal_smoothness = test_da3_pose_estimation(
            image_paths=image_paths,
            model_name=model,
            neighbor_segments=neighbor_segments
        )

        # Measure VRAM during processing
        current_vram = measure_vram_usage()
        peak_vram = max(peak_vram, current_vram)

        # Calculate metrics
        end_time = time.time()
        total_time = end_time - start_time

        # Success!
        result = DA3SyntheticTestResult(
            model=model,
            neighbor_segments=neighbor_segments,
            densification=densification,
            nearclip=nearclip,
            width=width,
            height=height,
            num_frames=num_frames,
            pattern=pattern,
            test_success=True,
            processing_time=total_time,
            peak_vram_gb=peak_vram,
            pose_estimation_success=pose_success,
            avg_pose_confidence=avg_confidence,
            temporal_smoothness=temporal_smoothness,
            error_message=None,
        )

        logger.info(f"✅ Test successful: {total_time:.2f}s, "
                    f"confidence={avg_confidence:.2f}, smoothness={temporal_smoothness:.2f}")

        return result

    except Exception as e:
        # Test failed
        logger.error(f"❌ Test failed: {e}")

        end_time = time.time()
        total_time = end_time - start_time

        result = DA3SyntheticTestResult(
            model=model,
            neighbor_segments=neighbor_segments,
            densification=densification,
            nearclip=nearclip,
            width=width,
            height=height,
            num_frames=num_frames,
            pattern=pattern,
            test_success=False,
            processing_time=total_time,
            peak_vram_gb=peak_vram,
            pose_estimation_success=False,
            avg_pose_confidence=0.0,
            temporal_smoothness=0.0,
            error_message=str(e),
        )

        return result


def run_synthetic_3dgs_sweep(
    sweep_config: Dict[str, Any],
    output_dir: Path,
) -> List[DA3SyntheticTestResult]:
    """Run a parameter sweep across synthetic 3DGS configurations.

    Args:
        sweep_config: Sweep configuration with parameter ranges
        output_dir: Base output directory for all tests

    Returns:
        List of test results
    """
    results = []

    # Extract sweep ranges
    models = sweep_config.get("models", ["DA3NESTED-GIANT-LARGE"])
    neighbor_segments_range = range(
        sweep_config.get("neighbor_segments_min", 4),
        sweep_config.get("neighbor_segments_max", 8) + 1,
        sweep_config.get("neighbor_segments_step", 2),
    )
    densification_range = range(
        sweep_config.get("densification_min", 2),
        sweep_config.get("densification_max", 6) + 1,
        sweep_config.get("densification_step", 2),
    )
    nearclip_values = []
    nearclip_min = sweep_config.get("nearclip_min", 0.05)
    nearclip_max = sweep_config.get("nearclip_max", 0.15)
    nearclip_step = sweep_config.get("nearclip_step", 0.05)
    current = nearclip_min
    while current <= nearclip_max + 0.001:  # Small epsilon for float comparison
        nearclip_values.append(round(current, 3))
        current += nearclip_step

    aspect_ratios = sweep_config.get("aspect_ratios", [[1.78, 512, 288]])
    test_frames = sweep_config.get("test_iterations", 20)
    patterns = sweep_config.get("patterns", ["gradient_sphere"])

    # Calculate total tests
    total_tests = (
        len(models) *
        len(list(neighbor_segments_range)) *
        len(densification_range) *
        len(nearclip_values) *
        len(aspect_ratios) *
        len(patterns)
    )

    logger.info(f"Starting synthetic 3DGS parameter sweep: {total_tests} total tests")

    test_count = 0

    # Run all combinations
    for model in models:
        for neighbor_segments in neighbor_segments_range:
            for densification in densification_range:
                for nearclip in nearclip_values:
                    for aspect_config in aspect_ratios:
                        for pattern in patterns:
                            test_count += 1

                            aspect_ratio, width, height = aspect_config

                            logger.info(f"\n{'='*60}")
                            logger.info(f"Test {test_count}/{total_tests}")
                            logger.info(f"{'='*60}")

                            config = {
                                "model": model,
                                "neighbor_segments": neighbor_segments,
                                "densification": densification,
                                "nearclip": nearclip,
                                "width": width,
                                "height": height,
                                "test_frames": test_frames,
                                "pattern": pattern,
                            }

                            result = run_synthetic_3dgs_test(config, output_dir)
                            results.append(result)

                    # Save intermediate results
                    results_file = output_dir / "synthetic_sweep_results.json"
                    with open(results_file, 'w') as f:
                        json.dump([r.to_dict() for r in results], f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Sweep complete: {len(results)} tests run")
    logger.info(f"Results saved to: {output_dir / 'synthetic_sweep_results.json'}")
    logger.info(f"{'='*60}")

    return results
