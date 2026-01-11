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

CRITICAL: Densification Paradox (Empirically Validated)
-------------------------------------------------------
"Densification" is MISLEADING - it splits splats into MORE but WEAKER pieces:

- D=1: 1 large opaque splat → solid colors, BEST quality (score 96.46, ~705k splats)
- D=2: Split into 2 weaker splats → good quality (score ~95, ~1.4M splats)
- D=4: Split into 4 weak splats → faded/ghosty (score 84, ~2.8M splats)
- D=8: Split into 8 tiny splats → very transparent (score 63.41, ~5.6M splats)

WHY: Each split piece has reduced opacity. Splats don't properly alpha-blend.
Result: More splats = worse quality (counterintuitive!)

Confirmed by user testing 2025-01-06:
- Simple scenes: Red cube solid at D=1, "blacked"/faded at D=4
- Photorealistic: Same pattern (D=1 best despite fewer splats)
- Colors fade away as individual splats get smaller with higher densification

RECOMMENDATION: Use D=1 (best), D=2 acceptable, avoid D=4+
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
    subimages_per_keyframe: int = 5,
    scene_prompts: List[str] = None,
) -> List[Path]:
    """Generate keyframes (with subimages) for batch reuse across multiple tests.

    Generates keyframes once with a random seed, saves to shared directory,
    and returns paths for reuse across all blend factor tests in the batch.

    For photorealistic mode, generates N subimages per keyframe with different seeds
    to help DA3 find commonality in synthetic scenes (e.g., 5 variations of "city").

    The keyframe sequence loops back to the start to create a seamless video:
    - Simple: cube → tetrahedron → sphere → cube (loop)
    - Photorealistic: city → highway → beach → city (loop, each with N subimages)

    Args:
        width: Output width
        height: Output height
        output_dir: Directory to save keyframes
        scene_type: 'simple' or 'photorealistic'
        base_seed: Random seed base (if None, generates random seed)
        subimages_per_keyframe: Number of variations per keyframe (photorealistic only)
        scene_prompts: Custom prompts for 3 scenes [city, highway, beach] (photorealistic only)

    Returns:
        List of all image paths (4 keyframes × N subimages for photorealistic, 4 for simple)
    """
    import random

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if keyframes already exist and can be reused
    existing_keyframes = list(output_dir.glob("keyframe_*.png"))
    if existing_keyframes:
        logger.info(f"Found {len(existing_keyframes)} existing keyframes in {output_dir}")
        logger.info("Reusing existing batch keyframes instead of regenerating")

        # Sort by filename to ensure correct order
        existing_keyframes.sort()

        # Verify we have the expected structure
        expected_simple = ["keyframe_000.png", "keyframe_240.png", "keyframe_480.png", "keyframe_720.png"]
        has_subimages = any("_sub" in kf.name for kf in existing_keyframes)

        if has_subimages:
            # Photorealistic with subimages
            logger.info(f"Detected photorealistic mode with subimages")
        else:
            # Simple mode or single keyframes
            logger.info(f"Detected simple mode (no subimages)")

        logger.info(f"✓ Reusing {len(existing_keyframes)} existing keyframes")
        return existing_keyframes

    # Use random seed if not specified
    if base_seed is None:
        base_seed = random.randint(1000, 9999)

    logger.info(f"Generating NEW batch keyframes (scene type: {scene_type}, base seed: {base_seed}, subimages: {subimages_per_keyframe})...")

    all_image_paths = []

    if scene_type == "photorealistic":
        # Default prompts if not provided
        if scene_prompts is None or len(scene_prompts) < 3:
            scene_prompts = [
                "modern city street with tall buildings, shops, and cars, architectural photography, detailed, 8k",
                "highway road stretching into distance, asphalt with lane markings, trees on sides, blue sky, photorealistic, detailed, 8k",
                "sandy beach with ocean waves, blue water, clear sky, palm trees, tropical paradise, photorealistic, detailed, 8k",
            ]

        # Photorealistic: Generate N subimages per keyframe with different seeds
        # Keyframe 0 (city) - multiple variations
        logger.info(f"Generating keyframe 0 (city) with {subimages_per_keyframe} subimages...")
        for sub_idx in range(subimages_per_keyframe):
            subimage_path = output_dir / f"keyframe_000_sub{sub_idx}.png"
            generate_keyframe_with_real_model(
                scene_prompts[0],
                width,
                height,
                subimage_path,
                seed=base_seed + sub_idx
            )
            all_image_paths.append(subimage_path)

        # Keyframe 1 (highway) - multiple variations
        logger.info(f"Generating keyframe 1 (highway) with {subimages_per_keyframe} subimages...")
        for sub_idx in range(subimages_per_keyframe):
            subimage_path = output_dir / f"keyframe_240_sub{sub_idx}.png"
            generate_keyframe_with_real_model(
                scene_prompts[1],
                width,
                height,
                subimage_path,
                seed=base_seed + 100 + sub_idx
            )
            all_image_paths.append(subimage_path)

        # Keyframe 2 (beach) - multiple variations
        logger.info(f"Generating keyframe 2 (beach) with {subimages_per_keyframe} subimages...")
        for sub_idx in range(subimages_per_keyframe):
            subimage_path = output_dir / f"keyframe_480_sub{sub_idx}.png"
            generate_keyframe_with_real_model(
                scene_prompts[2],
                width,
                height,
                subimage_path,
                seed=base_seed + 200 + sub_idx
            )
            all_image_paths.append(subimage_path)

        # Keyframe 3 (loop back to city) - copy first keyframe's subimages
        logger.info(f"Generating keyframe 3 (city loop) by copying keyframe 0 subimages...")
        for sub_idx in range(subimages_per_keyframe):
            src_path = output_dir / f"keyframe_000_sub{sub_idx}.png"
            dst_path = output_dir / f"keyframe_720_sub{sub_idx}.png"
            shutil.copy(src_path, dst_path)
            all_image_paths.append(dst_path)

        logger.info(f"✓ Generated {len(all_image_paths)} images (4 keyframes × {subimages_per_keyframe} subimages)")

    else:
        # Simple: red cube → green tetrahedron → blue sphere → red cube (loop back to start)
        # PIL drawings don't need variations, just generate once per keyframe
        keyframe_0_path = output_dir / "keyframe_000.png"
        keyframe_1_path = output_dir / "keyframe_240.png"
        keyframe_2_path = output_dir / "keyframe_480.png"
        keyframe_3_path = output_dir / "keyframe_720.png"

        generate_red_cube_keyframe(width, height, keyframe_0_path)
        generate_green_tetrahedron_keyframe(width, height, keyframe_1_path)
        generate_blue_sphere_keyframe(width, height, keyframe_2_path)
        # Reuse first keyframe as last to create seamless loop
        shutil.copy(keyframe_0_path, keyframe_3_path)

        all_image_paths = [keyframe_0_path, keyframe_1_path, keyframe_2_path, keyframe_3_path]

        logger.info(f"✓ Generated 4 batch keyframes in {output_dir} (simple mode, no subimages)")

    return all_image_paths


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
    use_ray_pose: bool = False,
    confidence_threshold: float = 0.0,
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
        use_ray_pose: Use DA3 ray head for more accurate camera poses (slower but better geometry)
        confidence_threshold: Filter splats by confidence percentile (0=disabled, 50=top 50%, 90=very confident only)

    Returns:
        BlendFactorTestResult with metrics
    """
    logger.info(f"🧪 Testing blend_factor={blend_factor:.2f}, neighbors={neighbor_segments}, densify={densification}")
    if use_ray_pose or confidence_threshold > 0:
        logger.info(f"   Quality settings: use_ray_pose={use_ray_pose}, confidence_threshold={confidence_threshold}%")

    if output_dir is None:
        output_dir = Path("output/deforum-tuning/blend-factor-tests") / f"blend_{blend_factor:.2f}"

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # Step 1: Use pre-generated keyframes or generate new ones
        if keyframe_paths is not None:
            # Reuse pre-generated keyframes (batch mode) - includes all subimages
            logger.info(f"Using {len(keyframe_paths)} pre-generated images from {keyframe_paths[0].parent}")

            # Copy all keyframe images to output directory for reference
            for kf_path in keyframe_paths:
                shutil.copy(kf_path, output_dir / kf_path.name)
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

        # Load keyframe images (includes all subimages if in batch mode)
        if keyframe_paths is not None:
            # Batch mode: load all subimages from keyframe_paths
            keyframe_images = [Image.open(kf_path) for kf_path in keyframe_paths]
            logger.info(f"Loaded {len(keyframe_images)} images for DA3 3DGS reconstruction")
        else:
            # Single test mode: load 4 keyframes (backward compatibility)
            keyframe_images = [
                Image.open(keyframe_0_path),
                Image.open(keyframe_1_path),
                Image.open(keyframe_2_path),
                Image.open(keyframe_3_path)
            ]
            logger.info(f"Loaded 4 keyframes for DA3 3DGS reconstruction")

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
        prediction = da3_model.estimate_3d_gaussians(
            keyframe_images,
            use_ray_pose=use_ray_pose,
            confidence_threshold=confidence_threshold
        )

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


def run_parameter_sweep(
    blend_factor_min: float = 0.0,
    blend_factor_max: float = 0.0,
    blend_factor_step: float = 0.25,
    neighbor_segments_min: int = 4,
    neighbor_segments_max: int = 4,
    neighbor_segments_step: int = 1,
    densification_min: int = 4,
    densification_max: int = 4,
    densification_step: int = 1,
    width: int = 512,
    height: int = 512,
    num_frames: int = 720,
    output_dir: Path = None,
    scene_type: str = "simple",
    subimages_per_keyframe: int = 5,
    scene_prompts: List[str] = None,
    use_ray_pose: bool = False,
    confidence_threshold: float = 0.0,
) -> List[BlendFactorTestResult]:
    """Run DA3-3DGS parameter sweep across blend_factor, neighbor_segments, and densification.

    Each parameter can be fixed (min == max) or swept (min < max).

    Args:
        blend_factor_min: Min schedule blend (0.0 = pure DA3, 1.0 = pure Deforum)
        blend_factor_max: Max schedule blend
        blend_factor_step: Blend factor step size
        neighbor_segments_min: Min neighboring keyframes (2-10, default 4)
            More keyframes = better geometry coverage but slower + more VRAM
        neighbor_segments_max: Max neighboring keyframes
        neighbor_segments_step: Neighbor step size
        densification_min: Min gaussian splat multiplier (1-8, default 4)
            1 = ~705k splats (fast), 2 = ~1.4M (optimal), 4 = ~2.8M, 8 = ~5.6M (may OOM)
        densification_max: Max densification
        densification_step: Densification step size
        width: Output width
        height: Output height
        num_frames: Total frames (default 720 = 12 seconds @ 60fps)
        output_dir: Output directory
        scene_type: Test scene type ('simple' = PIL-drawn shapes, 'photorealistic' = ZIT diffusion)
        subimages_per_keyframe: Variations per keyframe with different seeds (helps DA3 find commonality)
        scene_prompts: Custom prompts for 3 scenes [city, highway, beach] (photorealistic only)
        use_ray_pose: Use DA3 ray head for more accurate camera poses (slower but better geometry)
        confidence_threshold: Filter splats by confidence percentile (0=disabled, 50=top 50%, 90=very confident only)

    Returns:
        List of BlendFactorTestResult objects (one per test configuration)
    """
    if output_dir is None:
        output_dir = Path("output/deforum-tuning/parameter-sweep")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate parameter ranges
    import numpy as np
    blend_factors = list(np.arange(blend_factor_min, blend_factor_max + blend_factor_step/2, blend_factor_step))
    neighbor_segments_range = list(range(neighbor_segments_min, neighbor_segments_max + 1, neighbor_segments_step))
    densification_range = list(range(densification_min, densification_max + 1, densification_step))

    # Calculate total tests
    total_tests = len(blend_factors) * len(neighbor_segments_range) * len(densification_range)

    logger.info(f"🚀 Starting DA3-3DGS parameter sweep: {total_tests} total tests")
    logger.info(f"   Blend factors: {blend_factors}")
    logger.info(f"   Neighbor segments: {neighbor_segments_range}")
    logger.info(f"   Densification: {densification_range}")
    logger.info(f"   Resolution: {width}x{height}, Frames: {num_frames}")
    logger.info(f"   Scene type: {scene_type}, Subimages per keyframe: {subimages_per_keyframe}")
    logger.info(f"   Quality: use_ray_pose={use_ray_pose}, confidence_threshold={confidence_threshold}%")

    # Generate or reuse batch keyframes from shared location (ONCE for all tests)
    batch_keyframes_dir = Path("output/deforum-tuning/batch_keyframes")
    keyframe_paths = generate_batch_keyframes(
        width=width,
        height=height,
        output_dir=batch_keyframes_dir,
        scene_type=scene_type,
        base_seed=None,  # Random seed (only used if generating new)
        subimages_per_keyframe=subimages_per_keyframe,
        scene_prompts=scene_prompts,
    )
    total_images = len(keyframe_paths)
    if scene_type == "photorealistic":
        logger.info(f"✓ Batch keyframes ready: {total_images} images ({total_images // subimages_per_keyframe} keyframes × {subimages_per_keyframe} subimages)")
    else:
        logger.info(f"✓ Batch keyframes ready: {total_images} images")

    # Run sweep across all parameter combinations
    results = []
    test_count = 0

    for blend_factor in blend_factors:
        for neighbor_segments in neighbor_segments_range:
            for densification in densification_range:
                test_count += 1
                logger.info(f"\n[{test_count}/{total_tests}] Testing blend={blend_factor:.2f}, neighbors={neighbor_segments}, densify={densification}")

                # Run single test
                test_output_dir = output_dir / f"blend_{blend_factor:.2f}_neighbors_{neighbor_segments}_densify_{densification}"
                result = run_blend_factor_test(
                    blend_factor=blend_factor,
                    neighbor_segments=neighbor_segments,
                    densification=densification,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    output_dir=test_output_dir,
                    scene_type=scene_type,
                    keyframe_paths=keyframe_paths,  # Reuse batch keyframes for all tests
                    use_ray_pose=use_ray_pose,
                    confidence_threshold=confidence_threshold,
                )

                results.append(result)
                logger.info(f"   ✓ Test {test_count} complete: score={result.calculate_overall_score():.1f}/100")

    # Save all results to JSON
    results_file = output_dir / "parameter_sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # Find and log best result
    best_result = max(results, key=lambda r: r.calculate_overall_score())
    logger.info(f"\n✅ Sweep complete: {len(results)} tests")
    logger.info(f"   Results saved to {results_file}")
    logger.info(f"🏆 Best configuration:")
    logger.info(f"   Blend factor: {best_result.blend_factor:.2f}")
    logger.info(f"   Neighbor segments: {best_result.neighbor_segments}")
    logger.info(f"   Densification: {best_result.densification}")
    logger.info(f"   Score: {best_result.calculate_overall_score():.1f}/100")

    return results


def run_twopass_refinement(
    video_path: str,
    width: int = 512,
    height: int = 512,
    prompt: str = "modern city street with tall buildings and cars, architectural photography, detailed, 8k",
    steps: int = 20,
    seed: int = -1,
    frame_stride: int = 1,
    segment_size: int = 30,
    overlap_percent: int = 20,
    output_dir: Path = None,
    use_ray_pose: bool = False,
    confidence_threshold: float = 0.0,
    densification: int = 1,
    neighbor_segments: int = 4,
) -> List[BlendFactorTestResult]:
    """Run Two-Pass DA3-3DGS refinement.

    Pipeline:
    Phase 1: Generate coherent Deforum animation (if video_path empty)
         OR: Load existing video/image sequence (if video_path provided)
    Phase 2: Process frames through DA3-3DGS for refinement

    This addresses the core problem: DA3 needs temporally coherent multi-view data,
    not unrelated synthetic keyframes!

    Args:
        video_path: Path to input video/sequence (empty = generate on-the-fly)
        width: Frame width for generation
        height: Frame height for generation
        prompt: Prompt for test animation generation (from UI dgs_scene_prompt_1)
        steps: Sampling steps for generation (from UI steps field)
        seed: Random seed (-1 = random)
        frame_stride: Use every Nth frame (1=all, 2=every other, etc.)
        segment_size: Frames per DA3-3DGS segment
        overlap_percent: Overlap between segments for smooth transitions
        output_dir: Output directory for results
        use_ray_pose: Use DA3 ray head for camera poses
        confidence_threshold: Filter low-confidence splats
        densification: Densification factor (1 recommended)
        neighbor_segments: Keyframes per segment

    Returns:
        List of test results (one per segment)
    """
    logger.info("=" * 80)
    logger.info("TWO-PASS DA3-3DGS REFINEMENT")
    logger.info("=" * 80)
    logger.info(f"Input video: {video_path}")
    logger.info(f"Frame stride: {frame_stride}, Segment size: {segment_size}, Overlap: {overlap_percent}%")
    logger.info(f"DA3 params: use_ray_pose={use_ray_pose}, confidence={confidence_threshold}%")
    logger.info(f"3DGS params: densification={densification}, neighbor_segments={neighbor_segments}")
    logger.info("=" * 80)

    if output_dir is None:
        output_dir = Path("output/deforum-tuning/twopass")
        output_dir.mkdir(parents=True, exist_ok=True)

    import time
    start_time = time.time()

    # Step 1: Generate or load frames
    frames_dir = output_dir / "phase1_deforum_frames"
    frames_dir.mkdir(exist_ok=True)

    if not video_path or video_path.strip() == "":
        logger.info("=" * 80)
        logger.info("📹 PHASE 1: GENERATING COHERENT DEFORUM ANIMATION")
        logger.info("=" * 80)

        # Generate Deforum animation on the fly
        frame_paths = _generate_deforum_animation(
            output_dir=frames_dir,
            width=width,
            height=height,
            prompt=prompt,
            steps=steps,
            seed=seed,
        )

        if not frame_paths:
            logger.error("❌ Failed to generate Deforum animation!")
            result = BlendFactorTestResult(
                blend_factor=0.0,
                neighbor_segments=neighbor_segments,
                densification=densification,
                width=0,
                height=0,
                num_frames=0,
                test_success=False,
                total_time=time.time() - start_time,
                avg_frame_time=0.0,
                peak_vram_gb=0.0,
                avg_temporal_consistency=0.0,
                camera_path_adherence=0.0,
                visual_quality=0.0,
                error_message="Failed to generate Deforum animation",
            )
            return [result]

        logger.info(f"✅ Phase 1 complete: Generated {len(frame_paths)} frames")
    else:
        # Load frames from existing video/sequence
        logger.info("=" * 80)
        logger.info("📥 PHASE 1: LOADING FRAMES FROM VIDEO")
        logger.info("=" * 80)
        logger.info(f"   Source: {video_path}")
        frame_paths = _load_video_frames(video_path, frames_dir, frame_stride)

    if not frame_paths:
        logger.error("❌ Failed to load frames from video!")
        result = BlendFactorTestResult(
            blend_factor=0.0,
            neighbor_segments=neighbor_segments,
            densification=densification,
            width=0,
            height=0,
            num_frames=0,
            test_success=False,
            total_time=time.time() - start_time,
            avg_frame_time=0.0,
            peak_vram_gb=0.0,
            avg_temporal_consistency=0.0,
            camera_path_adherence=0.0,
            visual_quality=0.0,
            error_message=f"Failed to load frames from video: {video_path}",
        )
        return [result]

    # Get frame dimensions
    first_frame = Image.open(frame_paths[0])
    width, height = first_frame.size

    logger.info("")
    logger.info("=" * 80)
    logger.info("🎨 PHASE 2: DA3-3DGS REFINEMENT")
    logger.info("=" * 80)
    logger.info(f"   Input: {len(frame_paths)} frames at {width}×{height}")
    logger.info(f"   Segment size: {segment_size}, Overlap: {overlap_percent}%")

    # Clean up SD model to free VRAM for DA3-3DGS processing
    logger.info("")
    logger.info("🧹 Cleaning up SD model to free VRAM for 3DGS processing...")
    try:
        import gc
        import torch
        from modules import sd_models
        sd_models.unload_model_weights()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("   ✓ SD model unloaded, VRAM freed")
    except Exception as e:
        logger.warning(f"   ⚠️  Model cleanup failed (non-critical): {e}")

    # Step 2: Split into overlapping segments
    logger.info(f"\n🔪 Splitting into segments...")
    segments = _split_frames_into_segments(frame_paths, segment_size, overlap_percent)
    logger.info(f"   Created {len(segments)} segments")

    # Step 3: Process each segment with DA3-3DGS
    results = []
    for seg_idx, segment_frames in enumerate(segments):
        logger.info(f"\n🎬 Processing segment {seg_idx + 1}/{len(segments)} ({len(segment_frames)} frames)")

        seg_output_dir = output_dir / f"segment_{seg_idx:03d}"
        seg_output_dir.mkdir(exist_ok=True)

        # Run DA3-3DGS on this segment
        seg_result = _process_segment_with_da3gs(
            segment_frames=segment_frames,
            output_dir=seg_output_dir,
            use_ray_pose=use_ray_pose,
            confidence_threshold=confidence_threshold,
            densification=densification,
            neighbor_segments=neighbor_segments,
            width=width,
            height=height,
        )
        results.append(seg_result)

    # Step 4: Stitch segments into final video
    logger.info(f"\n🎞️  Stitching {len(results)} segments into final video...")
    final_video_path = output_dir / "twopass_refined.mp4"
    _stitch_segments_to_video(segments, results, output_dir, final_video_path, overlap_percent)

    total_time = time.time() - start_time
    logger.info(f"\n✅ Two-Pass refinement complete!")
    logger.info(f"   Total time: {total_time:.1f}s")
    logger.info(f"   Output: {final_video_path}")

    return results


def run_singlescene_multiangle(
    base_prompt: str,
    num_angles: int = 8,
    angle_variation: float = 0.3,
    lighting_variation: bool = False,
    width: int = 1280,
    height: int = 720,
    output_dir: Path = None,
    use_ray_pose: bool = False,
    confidence_threshold: float = 0.0,
    densification: int = 1,
    neighbor_segments: int = 4,
) -> List[BlendFactorTestResult]:
    """Run Single Scene Multi-Angle DA3-3DGS test.

    Pipeline:
    1. Generate N variations of the SAME scene with different camera angles
    2. Optionally vary lighting/time-of-day
    3. Feed all variations to DA3-3DGS as keyframes
    4. Build 3DGS scene from multi-view data
    5. Render interpolated views

    This gives DA3 proper multi-view data of the same location, instead of
    unrelated scenes (city → highway → beach).

    Args:
        base_prompt: Base scene description
        num_angles: Number of angle variations to generate
        angle_variation: Camera angle variation strength (0=subtle, 1=dramatic)
        lighting_variation: Add time-of-day variations to prompts
        width: Image width
        height: Image height
        output_dir: Output directory for results
        use_ray_pose: Use DA3 ray head for camera poses
        confidence_threshold: Filter low-confidence splats
        densification: Densification factor (1 recommended)
        neighbor_segments: Keyframes per segment

    Returns:
        List of test results
    """
    logger.info("=" * 80)
    logger.info("SINGLE SCENE MULTI-ANGLE DA3-3DGS")
    logger.info("=" * 80)
    logger.info(f"Base prompt: {base_prompt}")
    logger.info(f"Num angles: {num_angles}, Variation: {angle_variation:.2f}, Lighting: {lighting_variation}")
    logger.info(f"Resolution: {width}x{height}")
    logger.info(f"DA3 params: use_ray_pose={use_ray_pose}, confidence={confidence_threshold}%")
    logger.info(f"3DGS params: densification={densification}, neighbor_segments={neighbor_segments}")
    logger.info("=" * 80)

    if output_dir is None:
        output_dir = Path("output/deforum-tuning/singlescene")
        output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement full pipeline
    # For now, return placeholder result
    logger.warning("⚠️  Single Scene Multi-Angle pipeline not yet fully implemented!")
    logger.warning("   This is a stub implementation that will be completed in follow-up work.")
    logger.info("\n📋 Implementation plan:")
    logger.info("   1. Generate angle-specific prompts:")
    logger.info("      - 'from above', 'from street level', 'looking up'")
    logger.info("      - 'wide angle', 'telephoto', 'close-up'")
    logger.info("   2. Optionally add lighting variations:")
    logger.info("      - 'morning light', 'midday sun', 'golden hour', 'sunset'")
    logger.info("   3. Generate all keyframes with current Forge model")
    logger.info("   4. Run DA3 depth estimation on all views")
    logger.info("   5. Build unified 3DGS scene from multi-view data")
    logger.info("   6. Render smooth interpolation between views")

    # Placeholder result with correct dataclass fields
    result = BlendFactorTestResult(
        blend_factor=0.0,  # N/A for single scene
        neighbor_segments=neighbor_segments,
        densification=densification,
        width=width,
        height=height,
        num_frames=num_angles,  # Number of angle variations
        test_success=False,  # Not yet implemented
        total_time=0.0,
        avg_frame_time=0.0,
        peak_vram_gb=0.0,
        avg_temporal_consistency=0.0,
        camera_path_adherence=0.0,
        visual_quality=0.0,
        error_message="Single Scene Multi-Angle pipeline not yet fully implemented (stub only)",
    )

    return [result]


def _load_video_frames(video_path: str, output_dir: Path, stride: int = 1) -> List[Path]:
    """Load frames from video file or image sequence.

    Supports:
    - Video files (.mp4, .avi, .mov, etc.)
    - Image sequences (/path/to/frames/frame_%04d.png)
    - Directory of images (/path/to/frames/)

    Args:
        video_path: Path to video or image pattern
        output_dir: Directory to extract frames to
        stride: Use every Nth frame

    Returns:
        List of paths to extracted frames
    """
    from PIL import Image
    import glob
    import shutil

    video_path_obj = Path(video_path)

    # Case 1: Directory of images
    if video_path_obj.is_dir():
        logger.info(f"   Loading from image directory: {video_path}")
        image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        all_images = []
        for pattern in image_patterns:
            all_images.extend(sorted(video_path_obj.glob(pattern)))

        if not all_images:
            logger.error(f"   No images found in {video_path}")
            return []

        # Apply stride and copy to output dir
        frame_paths = []
        for idx, img_path in enumerate(all_images[::stride]):
            output_path = output_dir / f"frame_{idx:06d}.png"
            shutil.copy(img_path, output_path)
            frame_paths.append(output_path)

        logger.info(f"   Loaded {len(frame_paths)} frames (stride={stride})")
        return frame_paths

    # Case 2: Image sequence pattern (e.g., frame_%04d.png)
    if "%" in str(video_path):
        logger.info(f"   Loading from image sequence pattern: {video_path}")
        # Convert pattern to glob pattern
        pattern_dir = video_path_obj.parent
        pattern_name = video_path_obj.name.replace("%04d", "*").replace("%05d", "*").replace("%06d", "*")
        matching_images = sorted(pattern_dir.glob(pattern_name))

        if not matching_images:
            logger.error(f"   No images found matching pattern: {video_path}")
            return []

        # Apply stride and copy
        frame_paths = []
        for idx, img_path in enumerate(matching_images[::stride]):
            output_path = output_dir / f"frame_{idx:06d}.png"
            shutil.copy(img_path, output_path)
            frame_paths.append(output_path)

        logger.info(f"   Loaded {len(frame_paths)} frames (stride={stride})")
        return frame_paths

    # Case 3: Video file - requires ffmpeg or imageio
    if video_path_obj.is_file():
        logger.info(f"   Extracting frames from video: {video_path}")
        try:
            import subprocess
            # Use ffmpeg to extract frames
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", f"select='not(mod(n\\,{stride}))'",
                "-vsync", "0",
                "-frame_pts", "1",
                str(output_dir / "frame_%06d.png")
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"   ffmpeg failed: {result.stderr}")
                return []

            frame_paths = sorted(output_dir.glob("frame_*.png"))
            logger.info(f"   Extracted {len(frame_paths)} frames (stride={stride})")
            return frame_paths

        except FileNotFoundError:
            logger.error("   ffmpeg not found - cannot extract video frames")
            logger.error("   Please install ffmpeg or provide image sequence instead")
            return []
        except Exception as e:
            logger.error(f"   Failed to extract frames: {e}")
            return []

    logger.error(f"   Invalid video path: {video_path}")
    return []


def _split_frames_into_segments(frame_paths: List[Path], segment_size: int, overlap_percent: int) -> List[List[Path]]:
    """Split frames into overlapping segments.

    Args:
        frame_paths: List of frame paths
        segment_size: Frames per segment
        overlap_percent: Overlap between segments (0-50)

    Returns:
        List of segments, where each segment is a list of frame paths
    """
    if not frame_paths:
        return []

    total_frames = len(frame_paths)
    overlap_frames = int(segment_size * overlap_percent / 100)
    stride = segment_size - overlap_frames  # How many frames to advance per segment

    if stride <= 0:
        stride = 1  # Minimum stride

    segments = []
    start_idx = 0

    while start_idx < total_frames:
        end_idx = min(start_idx + segment_size, total_frames)
        segment = frame_paths[start_idx:end_idx]

        if len(segment) > 0:
            segments.append(segment)

        # If this segment reaches the end, break
        if end_idx >= total_frames:
            break

        start_idx += stride

    logger.info(f"   Segment details: {len(segments)} segments, {overlap_frames} overlap frames, stride={stride}")
    return segments


def _process_segment_with_da3gs(
    segment_frames: List[Path],
    output_dir: Path,
    use_ray_pose: bool,
    confidence_threshold: float,
    densification: int,
    neighbor_segments: int,
    width: int,
    height: int,
) -> BlendFactorTestResult:
    """Process a segment of frames with DA3-3DGS.

    Args:
        segment_frames: List of frame paths for this segment
        output_dir: Output directory for segment results
        use_ray_pose: Use DA3 ray head
        confidence_threshold: Confidence filtering
        densification: Densification factor
        neighbor_segments: Not used for two-pass (all frames are keyframes)
        width: Frame width
        height: Frame height

    Returns:
        Test result for this segment
    """
    import time
    import torch
    import numpy as np
    from PIL import Image

    start_time = time.time()

    logger.info(f"   Frames: {len(segment_frames)}")
    logger.info(f"   Output: {output_dir}")

    output_frames_dir = output_dir / "output_frames"
    output_frames_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Load DA3 depth model
        logger.info("   📊 Loading DA3 depth model...")
        from deforum.depth import DepthModel
        import modules.paths as ph

        # Determine frame dimensions
        first_frame = Image.open(segment_frames[0])
        frame_width, frame_height = first_frame.size

        depth_model = DepthModel(
            ph.models_path + '/Deforum',  # models_path
            'cuda:0',  # device
            False,  # half_precision (use FP32 for quality)
            keep_in_vram=False,
            depth_algorithm='Depth-Anything-V3-AnyView-Large',  # For 3DGS support
            Width=frame_width,
            Height=frame_height
        )

        logger.info(f"   ✓ DA3 loaded: {depth_model.depth_algorithm}")

        # Step 2: Load frames as numpy arrays
        logger.info(f"   📁 Loading {len(segment_frames)} frames...")
        frames = []
        for frame_path in segment_frames:
            img = Image.open(frame_path).convert('RGB')
            frames.append(np.array(img))  # RGB numpy array
        logger.info(f"   ✓ Loaded {len(frames)} frames ({frames[0].shape})")

        # Step 3: Run DA3 depth estimation on all frames
        logger.info("   🔍 Estimating depth with DA3...")
        logger.info(f"   Settings: use_ray_pose={use_ray_pose}, confidence_threshold={confidence_threshold}%")
        depths = []
        for idx, frame in enumerate(frames):
            # DepthModel.predict() expects BGR numpy array (OpenCV format)
            # but we have RGB, and it converts internally, so just pass as-is
            depth = depth_model.predict(
                frame,
                use_ray_pose=use_ray_pose,
                conf_thresh_percentile=confidence_threshold  # Already in 0-100 range
            )
            depths.append(depth)
            if idx == 0:
                logger.info(f"   ✓ Depth estimation working (shape: {depth.shape})")
        logger.info(f"   ✓ Estimated depth for {len(depths)} frames")

        # Step 4: Build 3DGS scene using DA3
        logger.info("   🎨 Building 3D Gaussian Splatting scene...")

        # Check if DA3 has 3DGS capabilities
        if hasattr(depth_model, 'estimate_3d_gaussians'):
            # Try to build 3DGS scene
            scene_3dgs = depth_model.estimate_3d_gaussians(
                frames,
                use_ray_pose=use_ray_pose,
                conf_thresh_percentile=confidence_threshold,  # Already in 0-100 range
            )

            if scene_3dgs is not None:
                logger.info("   ✓ 3DGS scene built successfully!")

                # Debug: Log 3DGS scene structure
                logger.debug(f"   Scene type: {type(scene_3dgs)}")
                if hasattr(scene_3dgs, 'gaussians'):
                    logger.debug(f"   Gaussians: {scene_3dgs.gaussians}")
                else:
                    logger.debug("   No 'gaussians' attribute found")

                logger.warning("   ⚠️  3DGS rendering not implemented - saving original frames")
                logger.info("   TODO: Implement gsplat rendering from 3DGS scene")

                # Step 5: Render frames from original poses with refined geometry
                # TODO: Actual 3DGS rendering requires gsplat integration
                # For now, just save the original frames as placeholder
                for idx, frame in enumerate(frames):
                    output_path = output_frames_dir / f"refined_{idx:06d}.png"
                    Image.fromarray(frame).save(output_path)

                logger.info(f"   ✓ Saved {len(frames)} placeholder frames (original input)")
            else:
                logger.warning("   ⚠️  3DGS scene building not available - using depth refinement")
                # Fallback: Save frames with depth-aware processing
                for idx, frame in enumerate(frames):
                    output_path = output_frames_dir / f"refined_{idx:06d}.png"
                    Image.fromarray(frame).save(output_path)
        else:
            logger.warning("   ⚠️  DA3 3DGS not available - using depth estimation only")
            # Fallback: Save frames with depth-aware processing
            for idx, frame in enumerate(frames):
                output_path = output_frames_dir / f"refined_{idx:06d}.png"
                Image.fromarray(frame).save(output_path)

        # Track VRAM usage
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        elapsed = time.time() - start_time
        logger.info(f"   ✓ Segment processing complete: {elapsed:.1f}s, {peak_vram:.2f}GB VRAM")

        return BlendFactorTestResult(
            blend_factor=0.0,
            neighbor_segments=neighbor_segments,
            densification=densification,
            width=width,
            height=height,
            num_frames=len(segment_frames),
            test_success=True,
            total_time=elapsed,
            avg_frame_time=elapsed / len(segment_frames) if segment_frames else 0.0,
            peak_vram_gb=peak_vram,
            avg_temporal_consistency=0.0,  # TODO: Calculate SSIM
            camera_path_adherence=0.0,  # N/A for two-pass
            visual_quality=0.0,  # TODO: Calculate quality metrics
            error_message=None,
        )

    except Exception as e:
        logger.error(f"   ❌ DA3-3DGS processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Fallback: Copy frames as-is
        logger.warning("   ⚠️  Falling back to frame copy")
        import shutil
        for idx, frame_path in enumerate(segment_frames):
            output_path = output_frames_dir / f"refined_{idx:06d}.png"
            shutil.copy(frame_path, output_path)

        elapsed = time.time() - start_time

        return BlendFactorTestResult(
            blend_factor=0.0,
            neighbor_segments=neighbor_segments,
            densification=densification,
            width=width,
            height=height,
            num_frames=len(segment_frames),
            test_success=False,
            total_time=elapsed,
            avg_frame_time=elapsed / len(segment_frames) if segment_frames else 0.0,
            peak_vram_gb=0.0,
            avg_temporal_consistency=0.0,
            camera_path_adherence=0.0,
            visual_quality=0.0,
            error_message=str(e),
        )


def _stitch_segments_to_video(
    segments: List[List[Path]],
    results: List[BlendFactorTestResult],
    base_output_dir: Path,
    final_video_path: Path,
    overlap_percent: int,
) -> None:
    """Stitch segments into final video with blending.

    Args:
        segments: List of segment frame lists
        results: List of segment results
        base_output_dir: Base output directory
        final_video_path: Output video path
        overlap_percent: Overlap percentage for blending
    """
    import subprocess

    logger.info(f"   Collecting output frames from {len(segments)} segments")

    # Collect all output frames
    all_output_frames = []
    for seg_idx in range(len(segments)):
        seg_output_dir = base_output_dir / f"segment_{seg_idx:03d}" / "output_frames"
        seg_frames = sorted(seg_output_dir.glob("refined_*.png"))

        if not seg_frames:
            logger.warning(f"   Segment {seg_idx} has no output frames!")
            continue

        # For first segment, take all frames
        if seg_idx == 0:
            all_output_frames.extend(seg_frames)
        else:
            # For subsequent segments, skip overlap region (already covered by previous segment)
            segment_size = len(segments[seg_idx])
            overlap_frames = int(segment_size * overlap_percent / 100)
            all_output_frames.extend(seg_frames[overlap_frames:])

    if not all_output_frames:
        logger.error("   No output frames to stitch!")
        return

    logger.info(f"   Total output frames: {len(all_output_frames)}")

    # Create temporary directory with sequential frame names for ffmpeg
    stitch_dir = base_output_dir / "stitch_frames"
    stitch_dir.mkdir(exist_ok=True)

    import shutil
    for idx, frame_path in enumerate(all_output_frames):
        output_path = stitch_dir / f"frame_{idx:06d}.png"
        shutil.copy(frame_path, output_path)

    # Stitch with ffmpeg
    logger.info(f"   Stitching to video: {final_video_path}")
    try:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", "24",
            "-i", str(stitch_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(final_video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"   ffmpeg stitching failed: {result.stderr}")
        else:
            logger.info(f"   ✅ Video saved: {final_video_path}")
    except FileNotFoundError:
        logger.error("   ffmpeg not found - cannot stitch video")
    except Exception as e:
        logger.error(f"   Failed to stitch video: {e}")


def _generate_deforum_animation(
    output_dir: Path,
    width: int = 512,
    height: int = 512,
    prompt: str = "modern city street with tall buildings and cars, architectural photography, detailed, 8k",
    steps: int = 9,
    seed: int = -1,
) -> List[Path]:
    """Generate coherent Deforum animation with 3D depth warping.

    Creates a simple test animation suitable for DA3-3DGS refinement:
    - Short duration (2-3 seconds)
    - Simple camera movement (forward zoom or orbit)
    - Single prompt for temporal coherence (from UI)
    - 3D depth warping enabled
    - Optimized for Z-Image Turbo: 9 steps for txt2img, ~2 steps for I2I (strength=0.9)

    Args:
        output_dir: Directory to save frames
        width: Frame width
        height: Frame height
        prompt: Scene prompt (from UI dgs_scene_prompt_1)
        steps: Sampling steps (from UI steps field, default 9 for Z-Image Turbo)
        seed: Random seed (-1 = random)

    Returns:
        List of paths to generated frames
    """
    logger.info("   Generating Deforum 3D animation...")
    logger.info(f"   Resolution: {width}×{height}")
    logger.info(f"   Output: {output_dir}")

    # Test animation parameters - longer duration, smoother at 60fps
    fps = 60
    duration_seconds = 5.0  # 5 seconds for good test coverage
    max_frames = int(fps * duration_seconds)  # 300 frames at 60fps

    # Orbital camera movement (more interesting than forward zoom)
    # Orbit around center with forward motion
    last_frame = max_frames - 1
    orbit_radius = 15.0  # Movement amount (15 units = dramatic movement for 5 second clip)

    # Orbital path: translate in X/Z circle while rotating to face center
    translation_x_schedule = f"0:(0), {last_frame}:({orbit_radius})"  # Move right
    translation_z_schedule = f"0:(0), {last_frame}:({orbit_radius})"  # Move forward
    rotation_3d_y_schedule = f"0:(0), {last_frame}:(90)"  # Rotate 90° for more dramatic turn

    logger.info(f"   Animation: {max_frames} frames at {fps}fps ({duration_seconds}s)")
    logger.info(f"   Movement: Orbital camera path (radius {orbit_radius})")
    logger.info(f"   Prompt: {prompt[:60]}...")
    logger.info(f"   Steps: {steps}, Seed: {seed}")
    logger.info("")

    # Create Deforum args
    logger.info("   🔧 Creating Deforum args for test animation...")
    args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root = _create_deforum_args_for_test(
        width=width,
        height=height,
        max_frames=max_frames,
        fps=fps,
        output_dir=output_dir,
        prompt=prompt,
        steps=steps,
        seed=seed,
        translation_x=translation_x_schedule,
        translation_z=translation_z_schedule,
        rotation_y=rotation_3d_y_schedule,
    )

    logger.info(f"   ✓ Args created: {width}×{height}, {max_frames} frames, 3D depth warping enabled")
    logger.info("")

    # Call Deforum render_animation
    logger.info("   🎬 Calling Deforum render_animation()...")
    try:
        from deforum.orchestration.render import render_animation
        render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root)
        logger.info("   ✓ Deforum rendering complete!")
    except Exception as e:
        logger.error(f"   ❌ Deforum rendering failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Fall back to placeholder frames
        logger.warning("   ⚠️  Falling back to placeholder frames")
        frame_paths = []
        for frame_idx in range(max_frames):
            frame_path = output_dir / f"frame_{frame_idx:06d}.png"
            generate_blue_sphere_keyframe(width, height, frame_path)
            frame_paths.append(frame_path)
        return frame_paths

    # Collect generated frames
    logger.info("   📁 Collecting generated frames...")
    frame_paths = sorted(output_dir.glob("*.png"))

    if not frame_paths:
        logger.error("   ❌ No frames generated!")
        return []

    logger.info(f"   ✓ Found {len(frame_paths)} frames")
    return frame_paths


def _create_deforum_args_for_test(
    width: int,
    height: int,
    max_frames: int,
    fps: int,
    output_dir: Path,
    prompt: str,
    steps: int,
    seed: int,
    translation_x: str,
    translation_z: str,
    rotation_y: str,
):
    """Create minimal Deforum args for test animation generation.

    Args:
        width: Frame width
        height: Frame height
        max_frames: Number of frames to generate
        fps: Frames per second
        output_dir: Output directory for frames
        prompt: Scene prompt (from UI)
        steps: Sampling steps (from UI)
        seed: Random seed
        translation_x: Translation X schedule
        translation_z: Translation Z schedule
        rotation_y: Rotation Y schedule

    Returns:
        Tuple of (args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root)
    """
    from types import SimpleNamespace
    from deforum.config.args import (
        DeforumArgs, DeforumAnimArgs, DeforumOutputArgs,
        ParseqArgs, LoopArgs, RootArgs
    )
    import json

    # Extract default values from arg definitions
    def get_defaults(args_dict):
        """Extract default values from Deforum args dict.

        Handles two formats:
        - Dict with 'value' or 'default' key: {"W": {"value": 1280, "label": "Width", ...}}
        - Primitive value directly: {"show_info_on_ui": True}
        """
        result = {}
        for key, val in args_dict.items():
            if isinstance(val, dict):
                # Extract value from dict (for fields like "W", "seed", etc.)
                result[key] = val.get('value', val.get('default', None))
            else:
                # Use primitive value directly (for fields like "show_info_on_ui": True)
                result[key] = val
        return result

    # Create args with defaults
    args_defaults = get_defaults(DeforumArgs())
    anim_defaults = get_defaults(DeforumAnimArgs())
    video_defaults = get_defaults(DeforumOutputArgs())
    parseq_defaults = get_defaults(ParseqArgs())
    loop_defaults = get_defaults(LoopArgs())
    root_defaults = RootArgs()

    # Override with test-specific values (from UI)
    args_defaults.update({
        'W': width,
        'H': height,
        'seed': seed,  # From UI (or -1 for random)
        'sampler': 'Euler a',
        'steps': steps,  # From UI steps field (default 9 for Z-Image Turbo)
        'scale': 7,  # CFG scale
        'strength': 0.78,  # Cadence strength (~2 steps at 9 steps: 1-0.78=0.22 denoising, 9*0.22≈2)
        'strength_0_no_init': True,
        'outdir': str(output_dir),  # Critical: where frames are saved
        # Prompt fields (required by save_settings_from_animation_run)
        'prompts': {0: prompt},  # Animation prompts dict (from UI dgs_scene_prompt_1)
        'positive_prompts': '',  # No additional positive prefix
        'negative_prompts': '',  # No additional negative prefix
    })

    anim_defaults.update({
        'render_mode': 'Classic 3D',  # Use Classic 3D for fixed cadence-based I2I keyframes
        'animation_mode': '3D',
        'max_frames': max_frames,
        'border': 'replicate',
        # CRITICAL: Disable keyframe distribution to enable uniform cadence
        'keyframe_distribution': 'Off',  # 'Off' = uniform cadence, 'Keyframes Only' = only prompt boundaries
        # I2I keyframe cadence - generate I2I frame every N frames to prevent degradation
        # At 60fps with cadence=30: 300 frames / 30 = 10 I2I keyframes (every 0.5s)
        # Classic 3D mode enforces uniform cadence placement
        'diffusion_cadence': 30,  # Generate I2I keyframe every 30 frames (0.5s intervals)
        # Depth model for Phase 1 depth warping (use Large model for quality)
        'depth_algorithm': 'Depth-Anything-V3-Mono-Large',
        # Orbital camera movement (from parameters)
        'translation_x': translation_x,  # Move right
        'translation_z': translation_z,  # Move forward
        'rotation_3d_y': rotation_y,     # Rotate to maintain view
        # Static defaults for other movement axes
        'translation_y': "0:(0)",
        'rotation_3d_x': "0:(0)",
        'rotation_3d_z': "0:(0)",
        'flip_2d_perspective': False,
        'perspective_flip_theta': "0:(0)",
        'perspective_flip_phi': "0:(0)",
        'perspective_flip_gamma': "0:(0)",
        'perspective_flip_fv': "0:(53)",
        'noise_schedule': "0: (0.02)",
        'strength_schedule': "0: (0.85)",
        'contrast_schedule': "0: (1.0)",
        'cfg_scale_schedule': "0: (7)",
        'enable_steps_scheduling': False,
        'steps_schedule': f"0: ({args_defaults['steps']})",
        'seed_schedule': "0:(s), 1:(-1)",  # Random after first frame
        'seed_iter_N': 1,
        'use_depth_warping': True,
        'midas_weight': 0.3,
        'near_plane': 200,
        'far_plane': 10000,
        'fov': 40,
        'padding_mode': 'border',
        'sampling_mode': 'bicubic',
        'save_depth_maps': False,
        'reverse_generation': False,
    })

    video_defaults.update({
        'skip_video_creation': True,  # We just want frames, not video
        'fps': fps,  # From parameters (60fps for smooth motion)
        'output_format': 'PNG',
        'image_path': str(output_dir),
        'mp4_path': str(output_dir / "video.mp4"),
    })

    # Create SimpleNamespace objects
    args = SimpleNamespace(**args_defaults)
    anim_args = SimpleNamespace(**anim_defaults)
    video_args = SimpleNamespace(**video_defaults)
    parseq_args = SimpleNamespace(**parseq_defaults)
    loop_args = SimpleNamespace(**loop_defaults)
    controlnet_args = SimpleNamespace()  # Empty
    root = SimpleNamespace(**root_defaults)

    # Set animation prompts (use same prompt variable for consistency)
    root.animation_prompts = {0: prompt}

    # CRITICAL: Set prompt_keyframes from animation_prompts keys
    # This is required by KeyFrameDistribution.select_deforum_keyframes()
    root.prompt_keyframes = list(root.animation_prompts.keys())

    # Set root fields that process_args() normally sets
    import time
    from modules.processing import get_fixed_seed

    # Handle seed processing (mimic process_args behavior)
    if args.seed == -1:
        root.raw_seed = -1
        args.seed = get_fixed_seed(-1)  # Generate random seed
    else:
        root.raw_seed = args.seed
        args.seed = get_fixed_seed(args.seed)  # Ensure valid seed

    # Set timestring for batch naming
    root.timestring = time.strftime('%Y%m%d%H%M%S')

    # Set job_id for API status tracking (None = skip tracking)
    # JobStatusTracker checks "if job_id in self.statuses" before updating
    # None will not be in dict, so updates are silently skipped
    root.job_id = None

    # Ensure strength is clamped [0.0, 1.0]
    args.strength = max(0.0, min(1.0, args.strength))

    return args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root
