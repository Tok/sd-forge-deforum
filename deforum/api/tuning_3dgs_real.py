"""Real DA3-3DGS test runner with actual depth estimation and splat rendering.

This module implements REAL 3DGS parameter testing that:
1. Uses existing test frames (gradient spheres, no ZIT generation needed)
2. Runs ACTUAL DA3 depth estimation on keyframes
3. Builds ACTUAL 3DGS splat scenes
4. Renders novel views with REAL splat rendering
5. Measures quality (SSIM), VRAM usage, and processing time
6. Saves all rendered frames to output directory
7. Generates markdown report for easy copy-paste

This replaces the placeholder synthetic tests that only measured frame similarity
without actually using DA3 or 3DGS rendering.
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from PIL import Image

from deforum.utils.system.logging import get_logger

logger = get_logger()


@dataclass
class DA3GS_RealTestResult:
    """Result from real DA3-3DGS testing with actual rendering."""

    # Test configuration
    model: str
    neighbor_segments: int
    densification: int
    nearclip: float
    width: int
    height: int
    num_keyframes: int  # Keyframes used for 3DGS scene
    num_tweens: int  # Tween frames rendered from splats

    # Performance metrics
    test_success: bool
    da3_load_time: float  # seconds to load DA3 model
    scene_build_time: float  # seconds to build 3DGS scene
    render_time_per_frame: float  # seconds per tween frame
    total_processing_time: float  # seconds total
    peak_vram_gb: float

    # Quality metrics (comparing rendered tweens to ground truth)
    avg_ssim: float  # 0-1, quality of rendered vs ground truth
    temporal_smoothness: float  # 0-1, consistency of rendered frames
    frames_rendered: int  # Number of successfully rendered frames

    # Error information
    error_message: Optional[str] = None

    def calculate_overall_score(self) -> float:
        """Calculate overall quality score (0-100).

        Weighted: SSIM 40%, smoothness 30%, success 20%, speed 10%
        """
        if not self.test_success:
            return 0.0

        ssim_score = self.avg_ssim * 100
        smoothness_score = self.temporal_smoothness * 100
        # Speed score: faster is better (target: 0.5s per frame)
        speed_score = min(0.5 / max(self.render_time_per_frame, 0.01), 1.0) * 100

        overall = (
            ssim_score * 0.40 +
            smoothness_score * 0.30 +
            100.0 * 0.20 +  # Success bonus
            speed_score * 0.10
        )

        return round(overall, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['overall_score'] = self.calculate_overall_score()
        return data


def measure_vram_usage() -> float:
    """Measure current VRAM usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.memory_allocated() / (1024 ** 3)
    except:
        return 0.0


def calculate_ssim(img1_path: Path, img2_path: Path) -> float:
    """Calculate SSIM between two images."""
    from skimage.metrics import structural_similarity
    img1 = np.array(Image.open(img1_path).convert('L'))
    img2 = np.array(Image.open(img2_path).convert('L'))
    return float(structural_similarity(img1, img2, data_range=255))


def load_da3_model(model_name: str, device: torch.device):
    """Load DA3 depth estimation model.

    Args:
        model_name: "DA3-GIANT" or "DA3NESTED-GIANT-LARGE"
        device: torch device

    Returns:
        DA3 model instance
    """
    from deforum.depth.depth_anything_v3 import DepthAnythingV3

    # Parse model name to get variant and model_size
    # DA3-GIANT -> variant="giant", model_size="giant"
    # DA3NESTED-GIANT-LARGE -> variant="giant", model_size="nested-giant-large"
    variant = "giant"  # Both models are 3DGS-capable giants

    if "NESTED" in model_name:
        model_size = "nested-giant-large"
    else:
        model_size = "giant"

    logger.info(f"Loading DA3 model: {model_name} (variant={variant}, model_size={model_size})...")
    model = DepthAnythingV3(
        device=device,
        model_size=model_size,
        variant=variant
    )
    return model


def run_real_3dgs_test(
    test_images_dir: Path,
    model_name: str,
    neighbor_segments: int,
    densification: int,
    nearclip: float,
    output_dir: Path,
    device: torch.device = None
) -> DA3GS_RealTestResult:
    """Run a single real DA3-3DGS test with actual rendering.

    Args:
        test_images_dir: Directory containing test frames (000000000.png, etc.)
        model_name: DA3 model to use
        neighbor_segments: Number of keyframes for 3DGS scene
        densification: Splat densification factor (1-8)
        nearclip: Near clipping distance (0.0 = disabled)
        output_dir: Directory to save rendered frames
        device: Torch device (defaults to cuda if available)

    Returns:
        Test result with metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    try:
        # Load test images
        image_paths = sorted(list(test_images_dir.glob("*.png")))
        if not image_paths:
            raise ValueError(f"No test images found in {test_images_dir}")

        num_frames = len(image_paths)
        logger.info(f"Found {num_frames} test frames in {test_images_dir}")

        # Use first neighbor_segments images as keyframes
        num_keyframes = min(neighbor_segments, num_frames)
        keyframe_paths = image_paths[:num_keyframes]
        keyframe_images = [Image.open(p) for p in keyframe_paths]

        width, height = keyframe_images[0].size

        # Load DA3 model
        da3_load_start = time.time()
        da3_model = load_da3_model(model_name, device)
        da3_load_time = time.time() - da3_load_start
        logger.info(f"✓ DA3 model loaded in {da3_load_time:.2f}s")

        vram_after_load = measure_vram_usage()
        logger.info(f"VRAM after DA3 load: {vram_after_load:.2f} GB")

        # Build 3DGS scene from keyframes
        scene_build_start = time.time()
        logger.info(f"Building 3DGS scene from {num_keyframes} keyframes...")

        # Call DA3 estimate_3d_gaussians() to get 3DGS parameters
        prediction = da3_model.estimate_3d_gaussians(keyframe_images)

        if prediction is None or not hasattr(prediction, 'gaussians'):
            raise RuntimeError(
                "DA3 model doesn't support 3DGS. Need model with trained gs_head."
            )

        gaussians = prediction.gaussians
        scene_build_time = time.time() - scene_build_start
        logger.info(f"✓ 3DGS scene built in {scene_build_time:.2f}s")

        vram_after_scene = measure_vram_usage()
        logger.info(f"VRAM after scene build: {vram_after_scene:.2f} GB")

        # Render tween frames using 3DGS (frames between keyframes)
        num_tweens = num_frames - num_keyframes
        if num_tweens <= 0:
            raise ValueError("Not enough frames to render tweens")

        logger.info(f"Rendering {num_tweens} tween frames with densification={densification}, nearclip={nearclip}...")

        from deforum.rendering.da3_3dgs_novel_view import (
            render_novel_view_from_gaussians,
            interpolate_camera_pose
        )

        # Get camera poses from DA3 prediction
        camera_poses = prediction.camera_poses  # [N, 4, 4]
        camera_intrinsics = prediction.camera_intrinsics  # [3, 3]

        rendered_paths = []
        render_times = []
        ssim_scores = []

        for tween_idx in range(num_tweens):
            render_start = time.time()

            # Interpolate camera pose between keyframes
            # Simple linear interpolation for now
            t = (tween_idx + 1) / (num_tweens + 1)  # 0 < t < 1
            pose_idx = min(int(t * (num_keyframes - 1)), num_keyframes - 2)
            local_t = (t * (num_keyframes - 1)) - pose_idx

            interpolated_pose = interpolate_camera_pose(
                camera_poses[pose_idx],
                camera_poses[pose_idx + 1],
                local_t
            )

            # Render novel view from 3DGS
            rendered_image = render_novel_view_from_gaussians(
                gaussians=gaussians,
                camera_pose=interpolated_pose,
                camera_intrinsics=camera_intrinsics,
                image_size=(width, height),
                device=device,
                densification_factor=densification,
                near_clip_distance=nearclip
            )

            # Save rendered frame
            rendered_path = output_dir / f"rendered_{tween_idx:09d}.png"
            rendered_image.save(rendered_path)
            rendered_paths.append(rendered_path)

            render_time = time.time() - render_start
            render_times.append(render_time)

            # Compare to ground truth (original test frame at this position)
            ground_truth_idx = num_keyframes + tween_idx
            if ground_truth_idx < len(image_paths):
                gt_path = image_paths[ground_truth_idx]
                ssim = calculate_ssim(rendered_path, gt_path)
                ssim_scores.append(ssim)

            logger.info(f"  Frame {tween_idx+1}/{num_tweens}: {render_time:.3f}s, SSIM={ssim_scores[-1]:.3f}")

        # Calculate metrics
        avg_render_time = np.mean(render_times)
        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0

        # Temporal smoothness: measure consistency between consecutive rendered frames
        temporal_ssims = []
        for i in range(len(rendered_paths) - 1):
            ssim = calculate_ssim(rendered_paths[i], rendered_paths[i+1])
            temporal_ssims.append(ssim)
        temporal_smoothness = np.mean(temporal_ssims) if temporal_ssims else 0.0

        peak_vram = measure_vram_usage()
        total_time = time.time() - start_time

        logger.info(f"✓ Test complete: {len(rendered_paths)} frames rendered")
        logger.info(f"  Avg SSIM: {avg_ssim:.3f}, Temporal smoothness: {temporal_smoothness:.3f}")
        logger.info(f"  Avg render time: {avg_render_time:.3f}s/frame, Peak VRAM: {peak_vram:.2f} GB")

        return DA3GS_RealTestResult(
            model=model_name,
            neighbor_segments=neighbor_segments,
            densification=densification,
            nearclip=nearclip,
            width=width,
            height=height,
            num_keyframes=num_keyframes,
            num_tweens=len(rendered_paths),
            test_success=True,
            da3_load_time=da3_load_time,
            scene_build_time=scene_build_time,
            render_time_per_frame=avg_render_time,
            total_processing_time=total_time,
            peak_vram_gb=peak_vram,
            avg_ssim=avg_ssim,
            temporal_smoothness=temporal_smoothness,
            frames_rendered=len(rendered_paths),
            error_message=None
        )

    except Exception as e:
        logger.error(f"❌ Real 3DGS test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

        return DA3GS_RealTestResult(
            model=model_name,
            neighbor_segments=neighbor_segments,
            densification=densification,
            nearclip=nearclip,
            width=0,
            height=0,
            num_keyframes=0,
            num_tweens=0,
            test_success=False,
            da3_load_time=0.0,
            scene_build_time=0.0,
            render_time_per_frame=0.0,
            total_processing_time=time.time() - start_time,
            peak_vram_gb=0.0,
            avg_ssim=0.0,
            temporal_smoothness=0.0,
            frames_rendered=0,
            error_message=str(e)
        )


def run_real_3dgs_sweep(
    sweep_config: Dict[str, Any],
    output_dir: Path,
    test_manager=None,
    test_id: str = None
) -> List[DA3GS_RealTestResult]:
    """Run parameter sweep with REAL DA3-3DGS rendering.

    Args:
        sweep_config: Test configuration with parameter ranges
        output_dir: Base output directory
        test_manager: Optional test manager for cancellation
        test_id: Optional test ID for cancellation checking

    Returns:
        List of test results
    """
    def is_cancelled() -> bool:
        """Check if test has been cancelled."""
        if test_manager and test_id:
            status = test_manager.get_status(test_id)
            return status and status.status == "cancelled"
        return False

    # Parse sweep config
    models = sweep_config.get("dgs_models", ["DA3NESTED-GIANT-LARGE"])
    neighbor_segments_range = range(
        sweep_config.get("dgs_neighbor_segments_min", 4),
        sweep_config.get("dgs_neighbor_segments_max", 8) + 1,
        sweep_config.get("dgs_neighbor_segments_step", 2)
    )
    densification_range = range(
        sweep_config.get("dgs_densification_min", 2),
        sweep_config.get("dgs_densification_max", 6) + 1,
        sweep_config.get("dgs_densification_step", 2)
    )

    # Build nearclip values with floating point handling
    nearclip_values = []
    nearclip_min = sweep_config.get("dgs_nearclip_min", 0.00)
    nearclip_max = sweep_config.get("dgs_nearclip_max", 0.15)
    nearclip_step = sweep_config.get("dgs_nearclip_step", 0.05)
    current = nearclip_min
    while current <= nearclip_max + 0.001:
        nearclip_values.append(round(current, 3))
        current += nearclip_step

    # Calculate total tests
    total_tests = (
        len(models) *
        len(list(neighbor_segments_range)) *
        len(list(densification_range)) *
        len(nearclip_values)
    )

    logger.info(f"Starting REAL DA3-3DGS parameter sweep: {total_tests} total tests")
    logger.info(f"  Models: {models}")
    logger.info(f"  Neighbor segments: {list(neighbor_segments_range)}")
    logger.info(f"  Densification: {list(densification_range)}")
    logger.info(f"  Near clip: {nearclip_values}")

    # Generate or use existing test images
    from deforum.api.tuning_3dgs_synthetic import (
        generate_synthetic_test_images,
        find_test_dataset,
        load_test_images_from_dataset
    )

    # Try to find pre-generated realistic test dataset first
    test_images_dir = output_dir / "test_images"
    dataset_dir = find_test_dataset("realistic-interior")
    if dataset_dir:
        logger.info("Using pre-generated realistic test dataset")
        load_test_images_from_dataset(dataset_dir, 20, test_images_dir)
    else:
        logger.info("Generating synthetic gradient sphere test images")
        generate_synthetic_test_images(
            num_frames=20,
            width=512,
            height=288,
            output_dir=test_images_dir,
            pattern="gradient_sphere"
        )

    # Run sweep
    results = []
    test_count = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model in models:
        if is_cancelled():
            break

        for neighbor_segments in neighbor_segments_range:
            if is_cancelled():
                break

            for densification in densification_range:
                if is_cancelled():
                    break

                for nearclip in nearclip_values:
                    if is_cancelled():
                        logger.warning(f"Test cancelled by user after {test_count}/{total_tests} tests")
                        break

                    test_count += 1
                    logger.info(f"\n[{test_count}/{total_tests}] Testing: {model}, neighbors={neighbor_segments}, "
                                f"densify={densification}, nearclip={nearclip:.3f}")

                    test_output_dir = output_dir / f"test_{test_count:03d}_M{model}_N{neighbor_segments}_D{densification}_NC{nearclip:.3f}"

                    result = run_real_3dgs_test(
                        test_images_dir=test_images_dir,
                        model_name=model,
                        neighbor_segments=neighbor_segments,
                        densification=densification,
                        nearclip=nearclip,
                        output_dir=test_output_dir,
                        device=device
                    )

                    results.append(result)

                    # Save intermediate results
                    results_file = output_dir / "results.json"
                    with open(results_file, 'w') as f:
                        json.dump([r.to_dict() for r in results], f, indent=2)

                    if is_cancelled():
                        break
                if is_cancelled():
                    break
            if is_cancelled():
                break

    logger.info(f"\n✓ Real DA3-3DGS sweep complete: {len(results)}/{total_tests} tests run")

    # Generate markdown report
    generate_markdown_report(results, output_dir)

    return results


def generate_markdown_report(results: List[DA3GS_RealTestResult], output_dir: Path):
    """Generate markdown report for easy copy-paste.

    Args:
        results: List of test results
        output_dir: Directory to save report
    """
    report_path = output_dir / "REPORT.md"

    # Find best configuration
    successful_results = [r for r in results if r.test_success]
    if not successful_results:
        logger.warning("No successful tests to generate report from")
        return

    best_result = max(successful_results, key=lambda r: r.calculate_overall_score())

    # Generate report
    report = f"""# DA3-3DGS Real Rendering Test Report

**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Total Tests:** {len(results)}
**Successful:** {len(successful_results)}
**Failed:** {len(results) - len(successful_results)}

## Best Configuration

```json
{{
  "model": "{best_result.model}",
  "neighbor_segments": {best_result.neighbor_segments},
  "densification": {best_result.densification},
  "nearclip": {best_result.nearclip:.3f},
  "overall_score": {best_result.calculate_overall_score():.2f},
  "avg_ssim": {best_result.avg_ssim:.4f},
  "temporal_smoothness": {best_result.temporal_smoothness:.4f},
  "render_time_per_frame": {best_result.render_time_per_frame:.3f},
  "peak_vram_gb": {best_result.peak_vram_gb:.2f}
}}
```

## All Results Summary

| Model | Neighbors | Densify | Nearclip | Score | SSIM | Smoothness | Time/Frame | VRAM GB |
|-------|-----------|---------|----------|-------|------|------------|------------|---------|
"""

    for r in sorted(successful_results, key=lambda x: x.calculate_overall_score(), reverse=True):
        report += f"| {r.model} | {r.neighbor_segments} | {r.densification} | {r.nearclip:.3f} | {r.calculate_overall_score():.2f} | {r.avg_ssim:.3f} | {r.temporal_smoothness:.3f} | {r.render_time_per_frame:.3f}s | {r.peak_vram_gb:.2f} |\n"

    if len(results) != len(successful_results):
        report += "\n## Failed Tests\n\n"
        for r in results:
            if not r.test_success:
                report += f"- **{r.model}** (N{r.neighbor_segments}, D{r.densification}, NC{r.nearclip:.3f}): {r.error_message}\n"

    # Write report
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"✓ Markdown report saved to: {report_path}")
    logger.info("\n" + "="*80)
    logger.info("COPY-PASTE REPORT:")
    logger.info("="*80)
    logger.info(report)
    logger.info("="*80)
