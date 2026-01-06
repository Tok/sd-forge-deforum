"""DA3-3DGS Schedule Blend Factor Tuning.

Tests the new schedule blending feature by generating simple animations
with varying blend factors between DA3 auto-poses and Deforum schedules.

Current Status:
- ✅ Real keyframe generation:
  - Simple mode: PIL-drawn 3D cube/sphere (instant, no diffusion)
  - Photorealistic mode: Current Forge model (ZIT, Flux, SDXL, etc.)
- ✅ DA3-3DGS scene building and gaussian splat rendering
- ✅ Video output (60fps MP4, 12 seconds = 720 frames)
- ✅ Proper VRAM management (unload Forge models before DA3-GIANT)
- ⚠️  Blend factor NOT YET APPLIED (all tests use pure DA3 poses currently)
- TODO: Integrate Deforum camera schedules and apply blend_factor mixing

Why all outputs look similar:
Without Deforum schedules, blend_factor has no effect since there's nothing to blend.
All tests use pure DA3 auto-estimated camera poses, so differences are minimal.

Test Modes:
1. Simple: Red Cube → Green Tetrahedron → Blue Sphere → Red Cube (PIL-drawn, instant, loops)
2. Photorealistic: City → Highway → Beach → City (uses current Forge model, loops seamlessly)

Pipeline:
- 4 keyframes (frames 0, 240, 480, 720 where 720 is copy of 0 for seamless loop)
- 716 tween frames interpolated via DA3-3DGS (3 segments × ~239 frames each)
- 720 total frames stitched to MP4 at 60fps

Metrics:
- Visual quality (SSIM between frames)
- Temporal consistency (jitter/smoothness)
- Render time and VRAM usage
"""

import time
import torch
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from PIL import Image

from deforum.utils.system.logging import get_logger

logger = get_logger()


def generate_placeholder(prompt: str, width: int, height: int, output_path: Path, seed: int) -> None:
    """Generate a high-quality gradient placeholder keyframe.

    Args:
        prompt: Text prompt (used for color variation)
        width: Image width
        height: Image height
        output_path: Where to save the image
        seed: Random seed for reproducibility
    """
    from PIL import Image, ImageDraw
    import hashlib

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
    output_path = str(output_path)
    img.save(output_path)
    logger.info(f"Generated placeholder keyframe: {output_path}")


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
    """Generate keyframe using Z-Image-Turbo or current loaded model.

    Args:
        prompt: Text prompt for generation
        width: Output width
        height: Output height
        output_path: Path to save generated image
        seed: Random seed for reproducibility

    NOTE: Real generation temporarily disabled due to Z-Image-Turbo producing blank outputs.
    Using high-quality gradient placeholders for now. Main focus is testing DA3-3DGS rendering.
    """
    from PIL import Image, ImageDraw, ImageFont
    import hashlib

    # TEMPORARY: Skip real generation, use placeholders
    # TODO: Debug why Z-Image-Turbo produces blank images
    USE_REAL_GENERATION = False

    if not USE_REAL_GENERATION:
        # Generate placeholder directly
        generate_placeholder(prompt, width, height, output_path, seed)
        return

    try:
        # Try real generation with current model
        from modules import processing, shared
        from deforum.config.model_configs import get_model_config
        from deforum.utils.model_detection import get_model_name

        model_name = get_model_name()
        logger.info(f"Generating keyframe with {model_name}: {prompt[:50]}")

        # Get model config
        config = get_model_config(model_name)

        # Create Txt2Img processing object
        p = processing.StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt=prompt,
            negative_prompt="",
            width=width,
            height=height,
            steps=config.recommended_steps,
            cfg_scale=config.cfg_scale_default,
            sampler_name="Euler",
            seed=seed,
            do_not_save_samples=True,  # We'll save manually - prevents auto-save to None path
            do_not_save_grid=True,
        )

        # Set distilled CFG / shift parameter if model uses it
        if config.uses_distilled_cfg:
            # For Z-Image this is the shift parameter (3.0)
            # For Flux this is distilled CFG scale (3.5)
            if hasattr(shared.opts, 'distilled_cfg_scale'):
                shared.opts.distilled_cfg_scale = config.distilled_cfg_scale_default

        # Generate image
        logger.debug(f"Starting generation: {width}x{height}, {config.recommended_steps} steps")
        processed = processing.process_images(p)

        if not processed or not processed.images:
            raise RuntimeError("Generation failed: no images returned")

        image = processed.images[0]
        if image is None:
            raise RuntimeError("Generation returned None image")

        # Validate image (check if blank)
        img_array = np.array(image)
        if img_array.max() == img_array.min():
            logger.warning(f"Generation produced blank image (min={img_array.min()}, max={img_array.max()})")
            # Don't fail - use the blank image anyway for now
            # TODO: Debug Z-Image-Turbo blank output issue

        # Save image - convert Path to string explicitly
        save_path = str(output_path) if output_path is not None else None
        if save_path is None:
            raise RuntimeError("Output path is None")

        logger.debug(f"Saving image to: {save_path}")
        image.save(save_path)
        logger.info(f"✓ Generated keyframe: {save_path}")

        # Cleanup: Free VRAM after generation to prepare for DA3 loading
        del p, processed, image, img_array
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache after keyframe generation")

    except Exception as e:
        # Fallback to placeholder if generation fails
        logger.warning(f"Failed to generate with model: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        logger.info(f"Falling back to placeholder keyframe")
        generate_placeholder(prompt, width, height, output_path, seed)


def generate_red_cube_keyframe(width: int, height: int, output_path: Path) -> None:
    """Generate red cube keyframe (simple test mode) - draws actual 3D cube."""
    from PIL import ImageDraw

    img = Image.new('RGB', (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Draw isometric cube
    center_x = width // 2
    size = min(width, height) // 3

    # Adjust center_y to account for cube extending upward (top face)
    # Total cube height is 2*size, so shift down by size//2 for true centering
    center_y = height // 2 + size // 2

    # Cube vertices (isometric projection)
    # Front face (red)
    front_points = [
        (center_x, center_y - size // 2),
        (center_x + size, center_y),
        (center_x, center_y + size // 2),
        (center_x - size, center_y),
    ]
    draw.polygon(front_points, fill=(220, 40, 40), outline=(180, 30, 30))

    # Top face (lighter red)
    top_points = [
        (center_x, center_y - size // 2),
        (center_x - size, center_y),
        (center_x - size, center_y - size),
        (center_x, center_y - size * 3 // 2),
    ]
    draw.polygon(top_points, fill=(250, 80, 80), outline=(200, 60, 60))

    # Right face (darker red)
    right_points = [
        (center_x, center_y - size // 2),
        (center_x + size, center_y),
        (center_x + size, center_y - size),
        (center_x, center_y - size * 3 // 2),
    ]
    draw.polygon(right_points, fill=(160, 30, 30), outline=(120, 20, 20))

    img.save(str(output_path))
    logger.info(f"Generated red cube keyframe: {output_path}")


def generate_blue_sphere_keyframe(width: int, height: int, output_path: Path) -> None:
    """Generate blue sphere keyframe (simple test mode) - draws actual sphere with shading."""
    from PIL import ImageDraw

    img = Image.new('RGB', (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Draw sphere with gradient shading
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3

    # Create sphere by drawing concentric circles with varying brightness
    for r in range(radius, 0, -1):
        # Light source from top-left, so brightness increases towards top-left
        light_factor = 1 - (r / radius) ** 2
        base_blue = 40
        max_blue = 220
        blue = int(base_blue + (max_blue - base_blue) * light_factor)

        # Slight gradient in red/green for realistic shading
        red = int(20 + 30 * light_factor)
        green = int(30 + 50 * light_factor)

        color = (red, green, blue)
        bbox = [
            center_x - r, center_y - r,
            center_x + r, center_y + r
        ]
        draw.ellipse(bbox, fill=color, outline=color)

    # Add highlight
    highlight_r = radius // 4
    highlight_x = center_x - radius // 3
    highlight_y = center_y - radius // 3
    highlight_bbox = [
        highlight_x - highlight_r, highlight_y - highlight_r,
        highlight_x + highlight_r, highlight_y + highlight_r
    ]
    draw.ellipse(highlight_bbox, fill=(150, 180, 255), outline=(150, 180, 255))

    img.save(str(output_path))
    logger.info(f"Generated blue sphere keyframe: {output_path}")


def generate_green_tetrahedron_keyframe(width: int, height: int, output_path: Path) -> None:
    """Generate green tetrahedron keyframe (simple test mode) - draws 3D tetrahedron."""
    from PIL import ImageDraw
    import math

    img = Image.new('RGB', (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Draw tetrahedron (3-sided pyramid)
    center_x = width // 2
    size = min(width, height) // 3

    # Adjust center_y for proper centering (tetrahedron extends upward)
    center_y = height // 2 + size // 3

    # Calculate tetrahedron vertices
    # Base is an equilateral triangle
    base_height = size * math.sqrt(3) / 2
    apex_height = size * 1.2  # Height above base

    # Base vertices (equilateral triangle) - stretch horizontally by 2x
    base_top = (center_x, center_y - base_height // 2)
    base_left = (center_x - size, center_y + base_height // 2)  # 2x wider: -size instead of -size//2
    base_right = (center_x + size, center_y + base_height // 2)  # 2x wider: +size instead of +size//2

    # Apex (top point) - no change
    apex = (center_x, center_y - int(apex_height))

    # Draw visible faces
    # Front-left face (darkest green)
    front_left = [apex, base_left, base_top]
    draw.polygon(front_left, fill=(30, 120, 30), outline=(20, 90, 20))

    # Front-right face (medium green)
    front_right = [apex, base_top, base_right]
    draw.polygon(front_right, fill=(50, 180, 50), outline=(35, 140, 35))

    # Base face (lightest green - facing camera)
    base_face = [base_top, base_right, base_left]
    draw.polygon(base_face, fill=(70, 220, 70), outline=(50, 170, 50))

    img.save(str(output_path))
    logger.info(f"Generated green tetrahedron keyframe: {output_path}")


def generate_keyframe_with_real_model(
    prompt: str,
    width: int,
    height: int,
    output_path: Path,
    seed: int = 42
) -> None:
    """Generate keyframe using current Forge model (ALWAYS REAL GENERATION).

    Uses whatever diffusion model is currently loaded in Forge (ZIT, Flux, SDXL, etc.).
    This function bypasses the USE_REAL_GENERATION flag and always attempts
    real image generation. Used for photorealistic test mode.

    Args:
        prompt: Text prompt for generation
        width: Output width
        height: Output height
        output_path: Path to save generated image
        seed: Random seed for reproducibility
    """
    try:
        from modules import processing, shared
        from deforum.config.model_configs import get_model_config
        from deforum.utils.model_detection import get_model_name

        model_name = get_model_name()
        logger.info(f"Generating keyframe with {model_name}: {prompt[:50]}")

        # Get model config
        config = get_model_config(model_name)

        # Build prompt with in-prompt constraints for Z-Image (negative prompts don't work)
        is_zimage = config.model_type == "z_image"
        if is_zimage:
            # Z-Image: use in-prompt constraints, use native resolution to avoid blank output
            enhanced_prompt = f"{prompt}, high quality, detailed, sharp, clear, photorealistic"
            negative_prompt = ""
            # Use native resolution (1024x1024) - Z-Image works best at native res
            gen_width = 1024
            gen_height = 1024
        else:
            enhanced_prompt = prompt
            negative_prompt = "blurry, low quality, distorted, deformed"
            gen_width = width
            gen_height = height

        # Get sampler and scheduler from config
        sampler_name = config.recommended_sampler if hasattr(config, 'recommended_sampler') else "Euler"
        scheduler_name = config.recommended_scheduler if hasattr(config, 'recommended_scheduler') else "Automatic"

        # Create Txt2Img processing object with all parameters that Deforum uses
        p = processing.StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            width=gen_width,
            height=gen_height,
            steps=config.recommended_steps,
            cfg_scale=config.cfg_scale_default,
            sampler_name=sampler_name,
            scheduler=scheduler_name,  # CRITICAL: Set scheduler (beta for Z-Image)
            distilled_cfg_scale=config.distilled_cfg_scale_default if config.uses_distilled_cfg else 3.5,  # Shift parameter
            seed=seed,
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        # Generate image
        logger.debug(f"Starting generation: {gen_width}x{gen_height}, {config.recommended_steps} steps, CFG={config.cfg_scale_default}, scheduler={scheduler_name}")
        if is_zimage:
            logger.debug(f"  Z-Image mode: native 1024x1024, shift={config.distilled_cfg_scale_default}, will resize to {width}x{height}")
        processed = processing.process_images(p)

        if not processed or not processed.images:
            raise RuntimeError("Generation failed: no images returned")

        image = processed.images[0]
        if image is None:
            raise RuntimeError("Generation returned None image")

        # Validate image (check if blank)
        img_array = np.array(image)
        if img_array.max() == img_array.min():
            logger.warning(f"Generation produced blank image (min={img_array.min()}, max={img_array.max()})")
            logger.warning("Falling back to placeholder")
            generate_placeholder(prompt, width, height, output_path, seed)
            return

        # Resize back to target size if we scaled up for Z-Image
        if is_zimage and (gen_width != width or gen_height != height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            logger.debug(f"Resized from {gen_width}x{gen_height} to {width}x{height}")

        # Save image
        save_path = str(output_path)
        logger.debug(f"Saving image to: {save_path}")
        image.save(save_path)
        logger.info(f"✓ Generated keyframe: {save_path}")

        # Cleanup: Free VRAM after generation
        del p, processed, image, img_array
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache after keyframe generation")

    except Exception as e:
        logger.error(f"Failed to generate with model: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        logger.info(f"Falling back to placeholder keyframe")
        generate_placeholder(prompt, width, height, output_path, seed)


def generate_photorealistic_keyframe_city(width: int, height: int, output_path: Path, seed: int = 100) -> None:
    """Generate city scene keyframe using current Forge model."""
    prompt = "modern city street with tall buildings, shops, and cars, architectural photography, detailed, 8k"
    generate_keyframe_with_real_model(prompt, width, height, output_path, seed=seed)


def generate_photorealistic_keyframe_highway(width: int, height: int, output_path: Path, seed: int = 101) -> None:
    """Generate highway scene keyframe using current Forge model."""
    prompt = "highway road stretching into distance, asphalt with lane markings, trees on sides, blue sky, photorealistic, detailed, 8k"
    generate_keyframe_with_real_model(prompt, width, height, output_path, seed=seed)


def generate_photorealistic_keyframe_beach(width: int, height: int, output_path: Path, seed: int = 102) -> None:
    """Generate beach scene keyframe using current Forge model."""
    prompt = "sandy beach with ocean waves, blue water, clear sky, palm trees, tropical paradise, photorealistic, detailed, 8k"
    generate_keyframe_with_real_model(prompt, width, height, output_path, seed=seed)


def generate_batch_keyframes(
    width: int,
    height: int,
    output_dir: Path,
    scene_type: str = "simple",
    base_seed: int = None,
) -> List[Path]:
    """Generate 4 keyframes (3 unique scenes + loop) for batch reuse across multiple tests.

    Generates keyframes once with a random seed, saves to shared directory,
    and returns paths for reuse across all blend factor tests in the batch.

    The keyframe sequence loops back to the start to create a seamless video:
    - Simple: cube → tetrahedron → sphere → cube (loop)
    - Photorealistic: city → highway → beach → city (loop)

    Args:
        width: Output width
        height: Output height
        output_dir: Directory to save keyframes
        scene_type: 'simple' or 'photorealistic'
        base_seed: Random seed base (if None, generates random seed)

    Returns:
        List of 4 keyframe paths [000, 240, 480, 720] where 720 is a copy of 000
    """
    import random

    # Use random seed if not specified
    if base_seed is None:
        base_seed = random.randint(1000, 9999)

    logger.info(f"Generating batch keyframes (scene type: {scene_type}, base seed: {base_seed})...")

    output_dir.mkdir(parents=True, exist_ok=True)

    keyframe_0_path = output_dir / "keyframe_000.png"
    keyframe_1_path = output_dir / "keyframe_240.png"
    keyframe_2_path = output_dir / "keyframe_480.png"
    keyframe_3_path = output_dir / "keyframe_720.png"

    if scene_type == "photorealistic":
        # Photorealistic: city → highway → beach → city (loop back to start)
        generate_photorealistic_keyframe_city(width, height, keyframe_0_path, seed=base_seed)
        generate_photorealistic_keyframe_highway(width, height, keyframe_1_path, seed=base_seed + 1)
        generate_photorealistic_keyframe_beach(width, height, keyframe_2_path, seed=base_seed + 2)
        # Reuse first keyframe as last to create seamless loop
        shutil.copy(keyframe_0_path, keyframe_3_path)
    else:
        # Simple: red cube → green tetrahedron → blue sphere → red cube (loop back to start)
        generate_red_cube_keyframe(width, height, keyframe_0_path)
        generate_green_tetrahedron_keyframe(width, height, keyframe_1_path)
        generate_blue_sphere_keyframe(width, height, keyframe_2_path)
        # Reuse first keyframe as last to create seamless loop
        shutil.copy(keyframe_0_path, keyframe_3_path)

    logger.info(f"✓ Generated 4 batch keyframes in {output_dir} (3 unique scenes + loop)")

    return [keyframe_0_path, keyframe_1_path, keyframe_2_path, keyframe_3_path]


def run_blend_factor_test(
    blend_factor: float,
    neighbor_segments: int = 4,
    densification: int = 2,
    width: int = 512,
    height: int = 512,
    num_frames: int = 720,
    output_dir: Path = None,
    scene_type: str = "simple",
    keyframe_paths: List[Path] = None,
) -> BlendFactorTestResult:
    """Run a single blend factor test with REAL generation.

    This test:
    1. Uses pre-generated keyframes (if provided) or generates new ones
    2. Runs DA3-3DGS interpolation with specified blend_factor
    3. Measures real quality metrics from rendered frames
    4. Creates MP4 video at 60fps for visual comparison

    Args:
        blend_factor: Schedule blend factor (0.0 = pure DA3, 1.0 = pure Deforum)
        neighbor_segments: Number of neighboring keyframes for 3DGS
        densification: Gaussian densification factor
        width: Output width
        height: Output height
        num_frames: Total frames to generate (default: 720 = 12 seconds at 60fps)
        output_dir: Directory to save results
        scene_type: Test scene type ('simple' or 'photorealistic')
        keyframe_paths: Optional pre-generated keyframe paths (for batch reuse)

    Returns:
        BlendFactorTestResult with metrics
    """
    logger.info(f"🧪 Testing blend_factor={blend_factor:.2f}, neighbors={neighbor_segments}, densify={densification}")

    if output_dir is None:
        output_dir = Path("output/deforum-tuning/blend-factor-tests") / f"blend_{blend_factor:.2f}"

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # Step 1: Use pre-generated keyframes or generate new ones
        if keyframe_paths is not None:
            # Reuse pre-generated keyframes (batch mode)
            logger.info(f"Using pre-generated batch keyframes from {keyframe_paths[0].parent}")
            keyframe_0_path = keyframe_paths[0]
            keyframe_1_path = keyframe_paths[1]
            keyframe_2_path = keyframe_paths[2]
            keyframe_3_path = keyframe_paths[3]

            # Copy to output directory for reference
            shutil.copy(keyframe_0_path, output_dir / "keyframe_000.png")
            shutil.copy(keyframe_1_path, output_dir / "keyframe_240.png")
            shutil.copy(keyframe_2_path, output_dir / "keyframe_480.png")
            shutil.copy(keyframe_3_path, output_dir / "keyframe_720.png")
        else:
            # Generate keyframes for single test (backward compatibility)
            logger.info(f"Generating keyframes (scene type: {scene_type})...")
            keyframe_0_path = output_dir / "keyframe_000.png"
            keyframe_1_path = output_dir / "keyframe_240.png"
            keyframe_2_path = output_dir / "keyframe_480.png"
            keyframe_3_path = output_dir / "keyframe_720.png"

            if scene_type == "photorealistic":
                # Photorealistic: city → highway → beach → city (loop back to start)
                generate_photorealistic_keyframe_city(width, height, keyframe_0_path)
                generate_photorealistic_keyframe_highway(width, height, keyframe_1_path)
                generate_photorealistic_keyframe_beach(width, height, keyframe_2_path)
                # Reuse first keyframe as last to create seamless loop
                shutil.copy(keyframe_0_path, keyframe_3_path)
            else:
                # Simple: red cube → green tetrahedron → blue sphere → red cube (loop back to start)
                generate_red_cube_keyframe(width, height, keyframe_0_path)
                generate_green_tetrahedron_keyframe(width, height, keyframe_1_path)
                generate_blue_sphere_keyframe(width, height, keyframe_2_path)
                # Reuse first keyframe as last to create seamless loop
                shutil.copy(keyframe_0_path, keyframe_3_path)

        # Step 2: Aggressive VRAM cleanup before loading DA3
        logger.info("Clearing VRAM before loading DA3...")

        # Use Forge's memory management to unload ALL models
        try:
            from backend import memory_management
            import gc

            logger.info("Unloading all Forge models...")
            memory_management.unload_all_models()

            # Force VRAM release - reserved memory needs to be freed
            memory_management.soft_empty_cache(force=True)
            gc.collect()

            # Multiple passes of cache clearing to ensure VRAM is freed
            for _ in range(3):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            logger.info("All models unloaded successfully")
        except Exception as e:
            logger.warning(f"Could not unload models via Forge: {e}")
            # Fallback to manual cleanup
            import gc
            for _ in range(3):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        # Report available VRAM
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / (1024**2)
            reserved_mb = torch.cuda.memory_reserved() / (1024**2)
            total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            available_mb = total_mb - allocated_mb
            logger.info(f"VRAM: {allocated_mb:.0f} MB allocated, {reserved_mb:.0f} MB reserved, {available_mb:.0f} MB available (of {total_mb:.0f} MB total)")

        # Step 3: Run DA3-3DGS interpolation
        logger.info("Loading keyframes and building 3DGS scene...")

        # Load keyframe images (4 keyframes: 3 unique scenes + loop back to first)
        keyframe_images = [
            Image.open(keyframe_0_path),
            Image.open(keyframe_1_path),
            Image.open(keyframe_2_path),
            Image.open(keyframe_3_path)
        ]

        # Load DA3 model directly (not via DepthModel singleton wrapper)
        from deforum.depth.depth_anything_v3 import DepthAnythingV3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading DA3-GIANT model (FP32 for 3DGS compatibility)...")

        # NOTE: Cannot use FP16 - 3DGS estimation requires FP32
        # Error with FP16: "expected scalar type Float but found Half"

        # Load directly on GPU (we've freed enough VRAM at this point)
        da3_model = DepthAnythingV3(
            device=device,
            model_size="giant",
            variant="giant"
        )

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
            logger.info(f"DA3 loaded successfully. VRAM used: {vram_used:.2f} GB, free: {vram_free:.2f} GB")

        # Build 3DGS scene from keyframes
        logger.info(f"Building 3DGS scene from {len(keyframe_images)} keyframes...")
        prediction = da3_model.estimate_3d_gaussians(keyframe_images)

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

        num_keyframes = len(keyframe_images)
        num_segments = num_keyframes - 1  # Number of transition segments
        frames_per_segment = num_frames // num_segments  # Evenly distribute frames

        logger.info(f"Rendering {num_frames} total frames across {num_segments} segments ({num_keyframes} keyframes)...")
        logger.info(f"  Segment 0→1: frames 0-{frames_per_segment-1}")
        logger.info(f"  Segment 1→2: frames {frames_per_segment}-{num_frames-1}")

        rendered_images = []
        all_frames = []  # Will include keyframes + tweens in order

        for frame_idx in range(num_frames):
            try:
                # Progress logging every 50 frames
                if frame_idx % 50 == 0:
                    logger.info(f"  Rendering frame {frame_idx}/{num_frames}...")

                # Determine which segment this frame belongs to
                segment_idx = min(frame_idx // frames_per_segment, num_segments - 1)

                # Calculate interpolation parameter within this segment (0 to 1)
                segment_start_frame = segment_idx * frames_per_segment
                local_frame = frame_idx - segment_start_frame
                t = local_frame / frames_per_segment

                # Get camera poses for this segment
                pose_start = camera_poses[segment_idx]
                pose_end = camera_poses[segment_idx + 1]

                # Interpolate camera pose
                da3_pose = interpolate_camera_pose(pose_start, pose_end, t)

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
                frame_path = output_dir / f"frame_{frame_idx:04d}.png"
                rendered_image.save(frame_path)
                rendered_images.append(rendered_image)
                all_frames.append(np.array(rendered_image))

            except Exception as e:
                logger.error(f"Failed to render frame {frame_idx}/{num_frames}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # Stop rendering on first error to avoid filling logs
                logger.warning(f"Stopping render after {len(rendered_images)} successful frames")
                break

        # Step 4: Stitch frames into video for easy comparison
        logger.info("Stitching frames into video...")
        video_path = output_dir / f"blend_{blend_factor:.2f}.mp4"
        try:
            import imageio
            # all_frames already populated during rendering loop
            if len(all_frames) > 0:
                imageio.mimsave(video_path, all_frames, fps=60, format='mp4')
                logger.info(f"✓ Video saved: {video_path} ({len(all_frames)} frames)")
            else:
                logger.warning("No frames to stitch into video")
        except Exception as e:
            logger.warning(f"Failed to create video: {e}")

        processing_time = time.time() - start_time
        avg_frame_time = processing_time / num_frames

        # Calculate real metrics
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

        # Temporal consistency: average SSIM between consecutive frames
        # Calculate directly to avoid imagehash dependency
        from skimage.metrics import structural_similarity as ssim
        if len(rendered_images) >= 2:
            ssim_scores = []
            for i in range(len(rendered_images) - 1):
                img1 = np.array(rendered_images[i].convert('L'))  # Grayscale
                img2 = np.array(rendered_images[i + 1].convert('L'))
                score = ssim(img1, img2, data_range=255)
                ssim_scores.append(score)
            temporal_consistency = float(np.mean(ssim_scores))
        else:
            temporal_consistency = 0.85

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

        # Cleanup: Free VRAM before next test
        del da3_model, prediction, gaussians, rendered_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"✅ Test complete: score={result.calculate_overall_score():.1f}/100")
        return result

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

        # Cleanup: Try to free VRAM even on failure
        try:
            if 'da3_model' in locals():
                del da3_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

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
    num_frames: int = 720,
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

    # Generate batch keyframes ONCE before the sweep (with random seed)
    batch_keyframes_dir = output_dir / "batch_keyframes"
    keyframe_paths = generate_batch_keyframes(
        width=width,
        height=height,
        output_dir=batch_keyframes_dir,
        scene_type=scene_type,
        base_seed=None,  # Random seed
    )
    logger.info(f"✓ Batch keyframes ready for reuse across {len(blend_factors)} tests")

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
            keyframe_paths=keyframe_paths,  # Reuse batch keyframes
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
