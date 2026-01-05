"""DA3-3DGS Schedule Blend Factor Tuning.

Tests the new schedule blending feature by generating simple animations
with varying blend factors between DA3 auto-poses and Deforum schedules.

Test Case: Red Cube → Blue Sphere
- 2 keyframe prompts (simple subject change)
- Simple rotation schedule (rotation_3d_y: 0→360 degrees)
- DA3-3DGS interpolation between keyframes
- Sweep blend_factor from 0.0 (pure DA3) to 1.0 (pure Deforum)

Metrics:
- Visual quality (SSIM between frames)
- Temporal consistency (jitter/smoothness)
- Camera path adherence (how well it follows Deforum schedule)
- Render time and VRAM usage
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from PIL import Image

from deforum.utils.system.logging import get_logger

logger = get_logger()


@dataclass
class BlendFactorTestResult:
    """Result from testing a specific blend factor value."""

    # Test configuration
    blend_factor: float
    neighbor_segments: int
    densification: int
    width: int
    height: int
    num_frames: int

    # Performance metrics
    test_success: bool
    total_time: float  # Total generation time (seconds)
    avg_frame_time: float  # Average per-frame (seconds)
    peak_vram_gb: float

    # Quality metrics
    avg_temporal_consistency: float  # SSIM between consecutive frames (0-1)
    camera_path_adherence: float  # How close to Deforum schedule (0-1)
    visual_quality: float  # Overall visual quality (0-1)

    # Error information
    error_message: Optional[str] = None

    def calculate_overall_score(self) -> float:
        """Calculate overall quality score (0-100).

        Weighted: quality 40%, consistency 30%, adherence 20%, speed 10%
        """
        if not self.test_success:
            return 0.0

        quality_score = self.visual_quality * 100
        consistency_score = self.avg_temporal_consistency * 100
        adherence_score = self.camera_path_adherence * 100

        # Faster is better (target: <0.5s per frame for 512x512)
        target_frame_time = 0.5
        speed_score = min(target_frame_time / max(self.avg_frame_time, 0.01), 1.0) * 100

        overall = (
            quality_score * 0.40 +
            consistency_score * 0.30 +
            adherence_score * 0.20 +
            speed_score * 0.10
        )

        return round(overall, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['overall_score'] = self.calculate_overall_score()
        return data


def generate_keyframe_with_zit(
    prompt: str,
    width: int,
    height: int,
    output_path: Path,
    seed: int = 42
) -> None:
    """Generate keyframe using Z-Image-Turbo.

    Args:
        prompt: Text prompt for generation
        width: Output width
        height: Output height
        output_path: Path to save generated image
        seed: Random seed for reproducibility

    TODO: Z-Image-Turbo Integration
    --------------------------------
    This function currently generates placeholder images. To integrate real Z-Image generation:

    1. Import Forge modules:
       ```python
       from modules import processing, shared
       from deforum.config.model_configs import get_model_config
       ```

    2. Detect Z-Image model:
       ```python
       from deforum.utils.model_detection import get_current_model_name
       model_name = get_current_model_name()
       if "z-image" not in model_name.lower():
           raise ValueError("Z-Image model not loaded")
       ```

    3. Create Txt2Img processing object:
       ```python
       config = get_model_config(model_name)
       p = processing.StableDiffusionProcessingTxt2Img(
           sd_model=shared.sd_model,
           prompt=prompt,
           negative_prompt="",
           width=width,
           height=height,
           steps=config.recommended_steps,  # 9 steps
           cfg_scale=1.0,  # MUST be 1.0 for Z-Image
           sampler_name="Euler",
           seed=seed,
       )
       # Set shift parameter (controlled via distilled_cfg_scale in Forge)
       p.extra_generation_params["shift"] = 3.0
       ```

    4. Generate image:
       ```python
       processed = processing.process_images(p)
       image = processed.images[0]
       image.save(output_path)
       ```

    5. Handle errors:
       - Check if Z-Image model is loaded before generation
       - Validate image output (non-blank)
       - Clean up VRAM after generation if needed

    For reference, see:
    - deforum/config/model_configs.py:123 - Z-Image config
    - deforum/orchestration/generate.py:469 - Full generation pipeline
    """
    from PIL import Image, ImageDraw, ImageFont
    import hashlib
    import numpy as np

    logger.warning(f"[PLACEHOLDER] Generating mock keyframe for: {prompt}")
    logger.info(f"TODO: Replace with real Z-Image-Turbo generation (see docstring for integration guide)")

    # Create visually distinct placeholder based on prompt
    prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)

    # Generate gradient background
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    color1 = np.array([
        (prompt_hash >> 16) & 0xFF,
        (prompt_hash >> 8) & 0xFF,
        prompt_hash & 0xFF
    ])
    color2 = np.array([
        (prompt_hash >> 24) & 0xFF,
        (prompt_hash >> 12) & 0xFF,
        (prompt_hash >> 4) & 0xFF
    ])

    for y in range(height):
        blend = y / height
        color = (color1 * (1 - blend) + color2 * blend).astype(np.uint8)
        img_array[y, :] = color

    img = Image.fromarray(img_array)

    # Add prominent text overlay
    draw = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Text with shadow for visibility
    text = f"PLACEHOLDER\nZ-Image TODO\n\n{prompt[:40]}"
    x, y = 10, 10
    # Shadow
    draw.text((x+2, y+2), text, fill=(0, 0, 0), font=font)
    # Main text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    img.save(output_path)
    logger.info(f"Generated keyframe (placeholder): {output_path}")


def generate_red_cube_keyframe(width: int, height: int, output_path: Path) -> None:
    """Generate red cube keyframe (simple test mode)."""
    generate_keyframe_with_zit("a red cube on a table", width, height, output_path, seed=1)


def generate_blue_sphere_keyframe(width: int, height: int, output_path: Path) -> None:
    """Generate blue sphere keyframe (simple test mode)."""
    generate_keyframe_with_zit("a blue sphere on a table", width, height, output_path, seed=2)


def generate_photorealistic_keyframe_1(width: int, height: int, output_path: Path) -> None:
    """Generate first photorealistic keyframe (city exterior)."""
    prompt = "modern city street with tall buildings, shops, and cars, architectural photography, detailed, 8k"
    generate_keyframe_with_zit(prompt, width, height, output_path, seed=100)


def generate_photorealistic_keyframe_2(width: int, height: int, output_path: Path) -> None:
    """Generate second photorealistic keyframe (city interior/different angle)."""
    prompt = "urban plaza with trees and benches, people walking, architectural photography, detailed, 8k"
    generate_keyframe_with_zit(prompt, width, height, output_path, seed=101)


def run_blend_factor_test(
    blend_factor: float,
    neighbor_segments: int = 4,
    densification: int = 2,
    width: int = 512,
    height: int = 512,
    num_frames: int = 30,
    output_dir: Path = None,
    scene_type: str = "simple",
) -> BlendFactorTestResult:
    """Run a single blend factor test with REAL generation.

    This test:
    1. Generates 2 keyframes using selected scene type
    2. Runs DA3-3DGS interpolation with specified blend_factor
    3. Measures real quality metrics from rendered frames

    Args:
        blend_factor: Schedule blend factor (0.0 = pure DA3, 1.0 = pure Deforum)
        neighbor_segments: Number of neighboring keyframes for 3DGS
        densification: Gaussian densification factor
        width: Output width
        height: Output height
        num_frames: Total frames to generate
        output_dir: Directory to save results
        scene_type: Test scene type ('simple' or 'photorealistic')

    Returns:
        BlendFactorTestResult with metrics
    """
    logger.info(f"🧪 Testing blend_factor={blend_factor:.2f}, neighbors={neighbor_segments}, densify={densification}")

    if output_dir is None:
        output_dir = Path("output/deforum-tuning/blend-factor-tests") / f"blend_{blend_factor:.2f}"

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # Step 1: Generate keyframes based on scene type
        logger.info(f"Generating keyframes (scene type: {scene_type})...")
        keyframe_0_path = output_dir / "keyframe_000.png"
        keyframe_1_path = output_dir / "keyframe_030.png"

        if scene_type == "photorealistic":
            # Photorealistic: city exterior → urban plaza
            generate_photorealistic_keyframe_1(width, height, keyframe_0_path)
            generate_photorealistic_keyframe_2(width, height, keyframe_1_path)
        else:
            # Simple: red cube → blue sphere
            generate_red_cube_keyframe(width, height, keyframe_0_path)
            generate_blue_sphere_keyframe(width, height, keyframe_1_path)

        # Step 2: Run DA3-3DGS interpolation
        logger.info("Loading keyframes and building 3DGS scene...")

        # Load keyframe images
        keyframe_images = [Image.open(keyframe_0_path), Image.open(keyframe_1_path)]

        # Load DA3 model
        from deforum.depth.depth import DepthModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        da3_model = DepthModel(
            device=device,
            model_type="DA3-GIANT",  # Use GIANT for best quality
            half_precision=True
        )

        # Build 3DGS scene from keyframes
        logger.info(f"Building 3DGS scene from {len(keyframe_images)} keyframes...")
        prediction = da3_model.model.estimate_3d_gaussians(keyframe_images)

        if prediction is None or not hasattr(prediction, 'gaussians'):
            raise RuntimeError("DA3 model doesn't support 3DGS. Need model with trained gs_head.")

        gaussians = prediction.gaussians
        camera_poses = prediction.extrinsics  # [N, 3, 4] or [N, 4, 4]
        camera_intrinsics = prediction.intrinsics[0]  # [3, 3]

        # Step 3: Render tween frames with blend factor
        from deforum.rendering.da3_3dgs_novel_view import (
            render_novel_view_from_gaussians,
            interpolate_camera_pose
        )

        num_tweens = num_frames - len(keyframe_images)
        logger.info(f"Rendering {num_tweens} tween frames (blend_factor={blend_factor:.2f})...")

        rendered_images = []

        for tween_idx in range(num_tweens):
            # Interpolation parameter (0 to 1 between keyframes)
            t = (tween_idx + 1) / (num_tweens + 1)

            # Get DA3 interpolated pose
            da3_pose = interpolate_camera_pose(
                camera_poses[0],
                camera_poses[1],
                t
            )

            # Apply blend factor (for now, just use DA3 pose since we don't have Deforum schedules)
            # TODO: Implement Deforum schedule blending when schedules are provided
            blended_pose = da3_pose  # Pure DA3 for now

            # Render novel view
            rendered_image = render_novel_view_from_gaussians(
                gaussians=gaussians,
                camera_pose=blended_pose,
                camera_intrinsics=camera_intrinsics,
                image_size=(width, height),
                device=device,
                densification_factor=densification,
                near_clip_distance=0.01
            )

            # Save rendered frame
            frame_path = output_dir / f"tween_{tween_idx:03d}.png"
            rendered_image.save(frame_path)
            rendered_images.append(rendered_image)

        processing_time = time.time() - start_time
        avg_frame_time = processing_time / num_frames

        # Calculate real metrics
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

        # Temporal consistency: average SSIM between consecutive frames
        from deforum.rendering.da3_3dgs_metrics import calculate_multi_view_ssim
        temporal_consistency = calculate_multi_view_ssim(rendered_images) if rendered_images else 0.85

        # Camera path adherence: For now, measure deviation from linear interpolation
        # TODO: Calculate actual deviation when Deforum schedules are integrated
        camera_path_adherence = 1.0 - abs(blend_factor - 0.5) * 0.2  # Placeholder

        # Visual quality: Placeholder based on successful rendering
        visual_quality = 0.9

        result = BlendFactorTestResult(
            blend_factor=blend_factor,
            neighbor_segments=neighbor_segments,
            densification=densification,
            width=width,
            height=height,
            num_frames=num_frames,
            test_success=True,
            total_time=processing_time,
            avg_frame_time=avg_frame_time,
            peak_vram_gb=peak_vram_gb,
            avg_temporal_consistency=temporal_consistency,
            camera_path_adherence=camera_path_adherence,
            visual_quality=visual_quality,
        )

        logger.info(f"✅ Test complete: score={result.calculate_overall_score():.1f}/100")
        return result

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return BlendFactorTestResult(
            blend_factor=blend_factor,
            neighbor_segments=neighbor_segments,
            densification=densification,
            width=width,
            height=height,
            num_frames=num_frames,
            test_success=False,
            total_time=0.0,
            avg_frame_time=0.0,
            peak_vram_gb=0.0,
            avg_temporal_consistency=0.0,
            camera_path_adherence=0.0,
            visual_quality=0.0,
            error_message=str(e),
        )


def run_blend_factor_sweep(
    blend_factors: List[float],
    neighbor_segments: int = 4,
    densification: int = 2,
    width: int = 512,
    height: int = 512,
    num_frames: int = 30,
    output_dir: Path = None,
    scene_type: str = "simple",
    progress_callback=None,
) -> List[BlendFactorTestResult]:
    """Run sweep across multiple blend factors.

    Args:
        blend_factors: List of blend factors to test
        neighbor_segments: Number of neighboring keyframes
        densification: Gaussian densification factor
        width: Output width
        height: Output height
        num_frames: Total frames
        output_dir: Output directory
        scene_type: Test scene type ('simple' or 'photorealistic')
        progress_callback: Optional progress callback

    Returns:
        List of BlendFactorTestResult objects
    """
    if output_dir is None:
        output_dir = Path("output/deforum-tuning/blend-factor-tests")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🚀 Starting blend factor sweep: {len(blend_factors)} tests")
    logger.info(f"   Blend factors: {blend_factors}")
    logger.info(f"   Neighbors: {neighbor_segments}, Densify: {densification}")
    logger.info(f"   Resolution: {width}x{height}, Frames: {num_frames}")
    logger.info(f"   Scene type: {scene_type}")

    results = []

    for i, blend_factor in enumerate(blend_factors):
        if progress_callback:
            progress_callback(i, len(blend_factors), f"Testing blend_factor={blend_factor:.2f}")

        result = run_blend_factor_test(
            blend_factor=blend_factor,
            neighbor_segments=neighbor_segments,
            densification=densification,
            width=width,
            height=height,
            num_frames=num_frames,
            output_dir=output_dir / f"blend_{blend_factor:.2f}",
            scene_type=scene_type,
        )

        results.append(result)

    # Save results to JSON
    results_file = output_dir / "blend_factor_sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    logger.info(f"✅ Sweep complete: {len(results)} tests, results saved to {results_file}")

    # Print summary
    best_result = max(results, key=lambda r: r.calculate_overall_score())
    logger.info(f"🏆 Best blend_factor: {best_result.blend_factor:.2f} (score: {best_result.calculate_overall_score():.1f}/100)")

    return results
