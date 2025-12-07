"""DA3-3DGS parameter sweep test runner (LEGACY - EXPENSIVE).

⚠️  DEPRECATED: This runner generates frames via DIFFUSION which is expensive
    and tests diffusion quality, not 3DGS parameters.

    USE INSTEAD: tuning_3dgs_synthetic.py for fast, reproducible parameter testing

    This runner is kept only for full end-to-end pipeline testing.

IMPORTANT: 3DGS rendering only works in render_mode="Keyframes + Interpolation"
           (Flux + Interpolation workflow). This runner attempts to use
           animation_mode="3D" + flux_flf2v_interpolation_method="DA3-3DGS"
           but may not trigger actual 3DGS rendering correctly.

Runs actual 3DGS renders with different parameter combinations and measures:
- Render success/failure (OOM, crashes)
- Render time (seconds per frame)
- Peak VRAM usage
- Temporal consistency (SSIM between frames)
- Output quality (valid frames generated)
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import json

from deforum.utils.system.logging import get_logger

logger = get_logger()


@dataclass
class DA33DGSTestResult:
    """Result from a single 3DGS parameter configuration test."""

    # Test configuration
    scene_strategy: str
    model: str
    neighbor_segments: int
    densification: int
    near_clip: float
    aspect_ratio: str
    width: int
    height: int

    # Performance metrics
    render_success: bool
    render_time_per_frame: float  # seconds
    total_frames_generated: int
    peak_vram_gb: float

    # Quality metrics
    temporal_consistency_ssim: float  # 0-1, higher = smoother
    avg_frame_sharpness: float  # Laplacian variance, higher = sharper

    # Error information
    error_message: Optional[str] = None

    def calculate_overall_score(self) -> float:
        """Calculate overall quality score (0-100).

        Weighted combination of:
        - Temporal consistency (SSIM): 40%
        - Frame sharpness: 30%
        - Render success: 20%
        - Performance (speed): 10%

        Returns:
            Score from 0-100 (higher is better)
        """
        if not self.render_success:
            return 0.0

        # Normalize metrics to 0-100 scale
        ssim_score = self.temporal_consistency_ssim * 100  # Already 0-1

        # Sharpness: typical range 0-500, normalize to 0-100
        sharpness_score = min(self.avg_frame_sharpness / 5.0, 100.0)

        # Speed: faster is better (inverse of render time)
        # Typical range: 0.5-5.0 seconds per frame
        # Convert to score: 2s/frame = 50, 1s/frame = 100, 4s/frame = 25
        speed_score = min(200.0 / max(self.render_time_per_frame, 0.1), 100.0)

        # Weighted combination
        overall = (
            ssim_score * 0.4 +
            sharpness_score * 0.3 +
            100.0 * 0.2 +  # Success bonus
            speed_score * 0.1
        )

        return round(overall, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['overall_score'] = self.calculate_overall_score()
        return data


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


def calculate_ssim_between_frames(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate SSIM between two frames.

    Args:
        frame1: First frame (H, W, 3) RGB uint8
        frame2: Second frame (H, W, 3) RGB uint8

    Returns:
        SSIM score 0-1 (higher = more similar)
    """
    try:
        from skimage.metrics import structural_similarity

        # Convert to grayscale for SSIM
        if len(frame1.shape) == 3:
            frame1_gray = np.mean(frame1, axis=2).astype(np.float32)
            frame2_gray = np.mean(frame2, axis=2).astype(np.float32)
        else:
            frame1_gray = frame1.astype(np.float32)
            frame2_gray = frame2.astype(np.float32)

        # Normalize to 0-1 range
        frame1_gray /= 255.0
        frame2_gray /= 255.0

        ssim_score = structural_similarity(frame1_gray, frame2_gray, data_range=1.0)
        return float(ssim_score)

    except Exception as e:
        logger.warning(f"Failed to calculate SSIM: {e}")
        return 0.0


def calculate_frame_sharpness(frame: np.ndarray) -> float:
    """Calculate frame sharpness using Laplacian variance.

    Higher values = sharper image.

    Args:
        frame: Frame (H, W, 3) RGB uint8

    Returns:
        Sharpness score (higher = sharper)
    """
    try:
        import cv2

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return float(variance)

    except Exception as e:
        logger.warning(f"Failed to calculate sharpness: {e}")
        return 0.0


def load_frames_from_directory(output_dir: Path, frame_indices: List[int]) -> List[np.ndarray]:
    """Load generated frames from output directory.

    Args:
        output_dir: Directory containing generated frames
        frame_indices: List of frame indices to load

    Returns:
        List of frames as numpy arrays (H, W, 3) RGB uint8
    """
    from PIL import Image

    frames = []
    for idx in frame_indices:
        # Try simple format first (000000001.png)
        frame_path = output_dir / f"{idx:09d}.png"

        if not frame_path.exists():
            logger.warning(f"Frame {idx} not found at {frame_path}")
            continue

        try:
            img = Image.open(frame_path)
            frame = np.array(img)
            frames.append(frame)
        except Exception as e:
            logger.warning(f"Failed to load frame {idx}: {e}")
            continue

    return frames


def run_3dgs_test_configuration(
    config: Dict[str, Any],
    test_output_dir: Path,
) -> DA33DGSTestResult:
    """Run a single 3DGS test configuration and measure quality.

    Args:
        config: Test configuration dict with 3DGS parameters
        test_output_dir: Directory to save test outputs

    Returns:
        Test result with metrics
    """
    from deforum.api.tuning_test_helpers import wait_for_job_to_complete
    import requests

    # Extract configuration
    scene_strategy = config.get("scene_strategy", "per_segment")
    model = config.get("model", "DA3NESTED-GIANT-LARGE")
    neighbor_segments = config.get("neighbor_segments", 6)
    densification = config.get("densification", 4)
    near_clip = config.get("near_clip", 0.1)
    width = config.get("width", 512)
    height = config.get("height", 288)
    aspect_ratio = config.get("aspect_ratio", "16:9")
    test_frames = config.get("test_frames", 10)  # Just generate 10 test frames

    logger.info(f"Running 3DGS test: {scene_strategy}, {model}, "
                f"{neighbor_segments} neighbors, {densification}x density, "
                f"near_clip={near_clip}, {width}x{height}")

    # Create test-specific output directory
    test_name = (f"3dgs_{scene_strategy}_{model}_{neighbor_segments}seg_"
                 f"{densification}x_{near_clip:.2f}clip")
    test_dir = test_output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Test output directory: {test_dir}")

    # Start time and VRAM tracking
    start_time = time.time()
    initial_vram = measure_vram_usage()
    peak_vram = initial_vram

    try:
        # Build Deforum API request for 3DGS test
        # This would be a minimal animation with just a few keyframes
        # to test the 3DGS rendering with the specified parameters

        deforum_settings = {
            # Minimal animation settings
            "max_frames": test_frames,
            "animation_mode": "3D",
            "W": width,
            "H": height,
            "fps": 12,

            # 3DGS-specific settings
            "flux_flf2v_interpolation_method": "DA3-3DGS",
            "da3_3dgs_scene_strategy": scene_strategy,
            "da3_3dgs_model": model,
            "da3_3dgs_neighbor_segments": neighbor_segments,
            "da3_3dgs_densification_factor": str(densification),
            "da3_3dgs_near_clip_distance": near_clip,
            "da3_3dgs_render_keyframes": True,

            # Simple test motion (small orbit)
            "translation_x": "0:(0), 5:(2), 9:(0)",  # Small circular motion
            "translation_y": "0:(0), 5:(0), 9:(0)",
            "rotation_3d_y": "0:(0), 5:(-0.25), 9:(0)",  # Matching rotation

            # Simple test prompts
            "animation_prompts": "0: test scene --neg bad quality",

            # Output settings
            "outdir": str(test_dir),
        }

        # Wrap settings in API request format
        api_request = {
            "deforum_settings": deforum_settings
        }

        # Submit job via API
        response = requests.post(
            "http://localhost:7860/deforum_api/batches",
            json=api_request,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        response.raise_for_status()
        job_data = response.json()
        batch_id = job_data["batch_id"]
        job_id = job_data["job_ids"][0]  # Get first job from list

        logger.info(f"Started 3DGS test batch: {batch_id}, job: {job_id}")

        # Wait for completion
        job_status = wait_for_job_to_complete(job_id)

        # Measure VRAM during render
        current_vram = measure_vram_usage()
        peak_vram = max(peak_vram, current_vram)

        # Calculate metrics
        end_time = time.time()
        total_time = end_time - start_time

        # Load generated frames
        frame_indices = list(range(test_frames))
        frames = load_frames_from_directory(test_dir, frame_indices)

        if len(frames) < 2:
            raise RuntimeError(f"Only {len(frames)} frames generated, need at least 2")

        # Calculate temporal consistency (SSIM between consecutive frames)
        ssim_scores = []
        for i in range(len(frames) - 1):
            ssim = calculate_ssim_between_frames(frames[i], frames[i + 1])
            ssim_scores.append(ssim)

        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0

        # Calculate average sharpness
        sharpness_scores = [calculate_frame_sharpness(f) for f in frames]
        avg_sharpness = np.mean(sharpness_scores) if sharpness_scores else 0.0

        # Success!
        result = DA33DGSTestResult(
            scene_strategy=scene_strategy,
            model=model,
            neighbor_segments=neighbor_segments,
            densification=densification,
            near_clip=near_clip,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            render_success=True,
            render_time_per_frame=total_time / len(frames) if frames else 0.0,
            total_frames_generated=len(frames),
            peak_vram_gb=peak_vram,
            temporal_consistency_ssim=float(avg_ssim),
            avg_frame_sharpness=float(avg_sharpness),
            error_message=None,
        )

        logger.info(f"✅ Test successful: {len(frames)} frames, "
                    f"{result.render_time_per_frame:.2f}s/frame, "
                    f"SSIM={avg_ssim:.3f}, sharpness={avg_sharpness:.1f}")

        return result

    except Exception as e:
        # Test failed
        logger.error(f"❌ Test failed: {e}")

        end_time = time.time()
        total_time = end_time - start_time

        result = DA33DGSTestResult(
            scene_strategy=scene_strategy,
            model=model,
            neighbor_segments=neighbor_segments,
            densification=densification,
            near_clip=near_clip,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            render_success=False,
            render_time_per_frame=0.0,
            total_frames_generated=0,
            peak_vram_gb=peak_vram,
            temporal_consistency_ssim=0.0,
            avg_frame_sharpness=0.0,
            error_message=str(e),
        )

        return result


def run_3dgs_parameter_sweep(
    sweep_config: Dict[str, Any],
    output_dir: Path,
) -> List[DA33DGSTestResult]:
    """Run a parameter sweep across 3DGS configurations.

    Args:
        sweep_config: Sweep configuration with parameter ranges
        output_dir: Base output directory for all tests

    Returns:
        List of test results
    """
    results = []

    # Extract sweep ranges
    scene_strategies = sweep_config.get("scene_strategies", ["per_segment"])
    models = sweep_config.get("models", ["DA3NESTED-GIANT-LARGE"])
    neighbor_segments_range = range(
        sweep_config.get("neighbor_segments_min", 6),
        sweep_config.get("neighbor_segments_max", 6) + 1,
        sweep_config.get("neighbor_segments_step", 1),
    )
    densification_range = range(
        sweep_config.get("densification_min", 4),
        sweep_config.get("densification_max", 4) + 1,
        sweep_config.get("densification_step", 1),
    )
    near_clip_range = np.arange(
        sweep_config.get("nearclip_min", 0.1),
        sweep_config.get("nearclip_max", 0.1) + 0.001,
        sweep_config.get("nearclip_step", 1.0),
    )

    aspect_ratios = sweep_config.get("aspect_ratios", [[1.78, 512, 288]])

    # Calculate total tests
    total_tests = (
        len(scene_strategies) *
        len(models) *
        len(list(neighbor_segments_range)) *
        len(list(densification_range)) *
        len(list(near_clip_range)) *
        len(aspect_ratios)
    )

    logger.info(f"Starting 3DGS parameter sweep: {total_tests} total tests")

    test_count = 0

    # Run all combinations
    for scene_strategy in scene_strategies:
        for model in models:
            for neighbor_segments in neighbor_segments_range:
                for densification in densification_range:
                    for near_clip in near_clip_range:
                        for aspect_config in aspect_ratios:
                            test_count += 1

                            aspect_ratio, width, height = aspect_config
                            aspect_str = f"{aspect_ratio:.2f}"

                            logger.info(f"\n{'='*60}")
                            logger.info(f"Test {test_count}/{total_tests}")
                            logger.info(f"{'='*60}")

                            config = {
                                "scene_strategy": scene_strategy,
                                "model": model,
                                "neighbor_segments": neighbor_segments,
                                "densification": densification,
                                "near_clip": near_clip,
                                "width": width,
                                "height": height,
                                "aspect_ratio": aspect_str,
                                "test_frames": sweep_config.get("test_iterations", 10),
                            }

                            result = run_3dgs_test_configuration(config, output_dir)
                            results.append(result)

                            # Save intermediate results
                            results_file = output_dir / "sweep_results.json"
                            with open(results_file, 'w') as f:
                                json.dump([r.to_dict() for r in results], f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Sweep complete: {len(results)} tests run")
    logger.info(f"Results saved to: {output_dir / 'sweep_results.json'}")
    logger.info(f"{'='*60}")

    return results
