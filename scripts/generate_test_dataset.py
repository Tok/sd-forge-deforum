#!/usr/bin/env python3
"""Generate realistic test dataset for DA3-3DGS parameter tuning.

This script generates a reusable set of realistic images with depth cues
for testing DA3 pose estimation and 3DGS parameters.

Usage:
    python scripts/generate_test_dataset.py
    python scripts/generate_test_dataset.py --num-images 30 --dataset-name custom-scene
"""

import os
import sys
import argparse
from pathlib import Path

# Add extension to path
SCRIPT_DIR = Path(__file__).parent
EXTENSION_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXTENSION_ROOT))

from deforum.utils.system.logging import get_logger

logger = get_logger()


def generate_test_dataset(
    num_images: int = 20,
    dataset_name: str = "realistic-interior",
    width: int = 512,
    height: int = 288,
    prompt: str = "interior room with furniture and depth, detailed architecture, photorealistic",
    negative_prompt: str = "blurry, low quality, distorted, flat",
    seed: int = 42,
):
    """Generate test dataset using ZIT (Z-Image-Turbo).

    Args:
        num_images: Number of test images to generate
        dataset_name: Name of the dataset (directory name)
        width: Image width
        height: Image height
        prompt: Generation prompt
        negative_prompt: Negative prompt
        seed: Starting seed (increments for each image)
    """
    # Get Forge root (3 levels up from extension)
    forge_root = EXTENSION_ROOT.parent.parent
    output_dir = forge_root / "output" / "deforum-tuning" / "test-datasets" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {num_images} test images to: {output_dir}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Size: {width}x{height}")
    logger.info("")

    # Import Forge modules
    sys.path.insert(0, str(forge_root))
    from modules import processing, shared, sd_models

    # Ensure ZIT model is loaded
    checkpoint_info = shared.opts.data.get('sd_model_checkpoint')
    if not checkpoint_info or 'z_image' not in checkpoint_info.lower():
        logger.warning("ZIT model not currently loaded. Searching for Z-Image-Turbo...")
        # Try to find ZIT model
        for ckpt in sd_models.checkpoints_list.values():
            if 'z_image' in ckpt.filename.lower():
                logger.info(f"Found ZIT model: {ckpt.filename}")
                sd_models.reload_model_weights(shared.sd_model, ckpt)
                break
        else:
            logger.error("No Z-Image-Turbo model found! Please ensure it's installed.")
            logger.error("Expected: models/Stable-diffusion/z_image_turbo_bf16.safetensors")
            return False

    # Generate images
    for i in range(num_images):
        current_seed = seed + i
        logger.info(f"Generating image {i+1}/{num_images} (seed={current_seed})...")

        # Create processing request
        p = processing.StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=20,
            cfg_scale=1.0,  # ZIT uses low CFG
            width=width,
            height=height,
            seed=current_seed,
            sampler_name="Euler",
            scheduler="Simple",
        )

        # Generate
        try:
            processed = processing.process_images(p)
            if processed and processed.images:
                # Save with standardized name
                output_path = output_dir / f"{i:09d}.png"
                processed.images[0].save(output_path)
                logger.info(f"  ✓ Saved: {output_path.name}")
            else:
                logger.error(f"  ✗ Failed to generate image {i+1}")
        except Exception as e:
            logger.error(f"  ✗ Error generating image {i+1}: {e}")
            continue

    logger.info("")
    logger.info(f"✅ Dataset generation complete!")
    logger.info(f"📁 Location: {output_dir}")
    logger.info(f"📊 Images: {len(list(output_dir.glob('*.png')))}/{num_images}")
    logger.info("")
    logger.info("All DA3-3DGS tuning tests will now use these images automatically.")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate realistic test dataset for DA3-3DGS tuning"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of test images to generate (default: 20)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="realistic-interior",
        help="Dataset name (default: realistic-interior)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width (default: 512)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=288,
        help="Image height (default: 288 for 16:9)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="interior room with furniture and depth, detailed architecture, photorealistic",
        help="Generation prompt"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, distorted, flat",
        help="Negative prompt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Starting seed (default: 42)"
    )

    args = parser.parse_args()

    success = generate_test_dataset(
        num_images=args.num_images,
        dataset_name=args.dataset_name,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
