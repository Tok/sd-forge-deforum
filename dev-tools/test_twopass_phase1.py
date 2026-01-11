#!/usr/bin/env python3
"""Test script for Two-Pass Phase 1 Deforum generation.

This script tests if the Deforum render_animation() integration works correctly
when called from the Two-Pass tuning mode.

Usage:
    python dev-tools/test_twopass_phase1.py
"""

import sys
from pathlib import Path

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

def test_phase1_generation():
    """Test Phase 1 Deforum generation."""
    from deforum.api.tuning_3dgs_blend_factor import _generate_deforum_animation
    from deforum.utils.system.logging import get_logger
    import tempfile

    logger = get_logger()

    logger.info("=" * 80)
    logger.info("Testing Two-Pass Phase 1 Deforum Generation")
    logger.info("=" * 80)

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_frames"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Generating 60 frame test animation (512x512)...")

        try:
            # Call Phase 1 generation
            frame_paths = _generate_deforum_animation(
                output_dir=output_dir,
                width=512,
                height=512,
            )

            if frame_paths:
                logger.info(f"✓ SUCCESS: Generated {len(frame_paths)} frames")
                logger.info(f"  First frame: {frame_paths[0]}")
                logger.info(f"  Last frame: {frame_paths[-1]}")

                # Verify frames exist
                missing = [f for f in frame_paths if not f.exists()]
                if missing:
                    logger.error(f"❌ MISSING FRAMES: {len(missing)} frames not found")
                    return False

                # Check frame sizes
                from PIL import Image
                img = Image.open(frame_paths[0])
                logger.info(f"  Frame size: {img.size}")

                if img.size != (512, 512):
                    logger.error(f"❌ WRONG SIZE: Expected (512, 512), got {img.size}")
                    return False

                logger.info("✓ All frames verified successfully!")
                return True
            else:
                logger.error("❌ FAILURE: No frames generated")
                return False

        except Exception as e:
            logger.error(f"❌ EXCEPTION: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    # This requires Forge to be fully initialized
    # Run from Forge root: python extensions/sd-forge-deforum/dev-tools/test_twopass_phase1.py
    success = test_phase1_generation()
    sys.exit(0 if success else 1)
