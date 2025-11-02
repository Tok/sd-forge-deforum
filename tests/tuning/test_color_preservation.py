"""Test color preservation across I2V chaining iterations.

This test measures how many iterations of image-to-video chaining it takes
before a colorful input image degrades to grayscale at different strength settings.

The test uses a brightly colored test image and repeatedly feeds the output
back as input, measuring color saturation decay.

**Goal:** Find strength values that maximize stability while preserving color.
"""

import pytest
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from .metrics import (
    measure_color_preservation,
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


# Test configuration
MAX_ITERATIONS = 20  # Stop after 20 iterations or grayscale threshold
GRAYSCALE_THRESHOLD = 20  # Color score below this = considered grayscale

# Use Forge's standard outputs directory (same location as normal generations)
# This will resolve to forge-neo/outputs/deforum-tuning/color_preservation
# instead of being buried in the extension directory
import os
FORGE_ROOT = Path(os.getcwd())  # Forge webui root directory
OUTPUT_DIR = FORGE_ROOT / "outputs" / "deforum-tuning" / "color_preservation"


def create_colorful_test_image() -> Path:
    """Generate a brightly colored test image with rainbow gradients.

    Returns:
        Path to generated test image
    """
    from PIL import Image
    import numpy as np

    # Create rainbow gradient image (512x512)
    width, height = 512, 512
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        # Hue varies from 0-360 across height
        hue = int((y / height) * 180)  # OpenCV hue is 0-180
        for x in range(width):
            # Saturation varies from 128-255 across width
            saturation = int(128 + (x / width) * 127)
            value = 255  # Full brightness

            # Convert HSV to RGB
            hsv_pixel = np.uint8([[[hue, saturation, value]]])
            import cv2
            rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0][0]

            img_array[y, x] = rgb_pixel

    # Add some high-frequency detail to make it interesting
    # (solid gradients can be too easy for the model)
    from PIL import ImageDraw, ImageFont
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # Draw colored text
    text = "COLOR\nPRESERVATION\nTEST"
    try:
        # Try to use a large font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
    except:
        font = ImageFont.load_default()

    # Draw text in white with black outline
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # Black outline
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    # White text
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    # Save
    test_img_path = OUTPUT_DIR / "test_input_rainbow.png"
    test_img_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(test_img_path)

    return test_img_path


def run_i2v_iteration(
    init_image_path: Path,
    strength: float,
    keyframe_strength: float,
    steps: int,
    output_dir: Path,
    max_frames: int = 30,  # Enough frames to test cadence/normal strength
) -> Path:
    """Run a single I2V generation iteration.

    Args:
        init_image_path: Path to input image
        strength: Normal/cadence frame strength (HIGH = 0.85 = stability)
        keyframe_strength: Keyframe strength (LOW = 0.15 = change)
        steps: Number of sampling steps
        output_dir: Where to save output
        max_frames: Total frames to generate (default 30 for good cadence coverage)

    Returns:
        Path to generated output image (last frame of animation)

    Note:
        With New 3D mode and max_frames=30:
        - Frame 0 is a keyframe (uses keyframe_strength)
        - Frames 1-29 are mostly cadence frames (use normal strength)
        - Frame 29 is also a keyframe (last frame)
        - This gives us ~27 cadence frames to test normal_strength stability
    """
    options_overrides = get_test_options_overrides()
    options_overrides.update({
        "deforum_save_gen_info_as_srt": False,  # No subtitles needed
    })

    settings = {
        "deforum_settings": {
            # Basic settings
            "W": 512,
            "H": 512,
            "seed": 42,
            "sampler": "euler",
            "steps": steps,
            "cfg_scale": 1.0,  # Flux distilled cfg
            "distilled_cfg_scale": 3.5,

            # Animation settings
            "animation_mode": "3D",
            "render_mode": "new_3d",  # New 3D with dual strength (REDISTRIBUTED)
            "max_frames": max_frames,
            "fps": 24,

            # Strength schedules (the parameters we're testing!)
            # normal_strength: HIGH (0.85+) = stability, fewer steps, used by cadence frames
            # keyframe_strength: LOW (0.15-) = change, more steps, used by keyframes
            "strength_schedule": f"0: ({strength})",
            "keyframe_strength_schedule": f"0: ({keyframe_strength})",

            # Use init image
            "use_init": True,
            "init_image": str(init_image_path),
            "strength": keyframe_strength,  # For frame 0

            # Prompts - single prompt means frame 0 is the only explicit keyframe
            # Frame 29 will also be a keyframe (last frame is always keyframe)
            # All other frames (1-28) will be cadence frames using normal_strength
            "animation_prompts": json.dumps({
                "0": "vibrant colorful rainbow gradient, highly saturated colors"
            }),

            # Disable audio for tuning tests
            "audio_mode": "None",
            "audio_sync": False,

            # Output
            "batch_name": get_test_batch_name(),
            "outdir": str(output_dir),
        },
        "options_overrides": options_overrides,
    }

    # Submit job
    response = requests.post(f"{API_BASE_URL}/batches/", json=settings)
    assert response.status_code == 200, f"Failed to submit job: {response.text}"

    batch_info = response.json()
    batch_id = batch_info["batch_id"]
    job_ids = batch_info["job_ids"]

    # Wait for completion
    final_status = wait_for_job_to_complete(job_ids[0])

    assert final_status["status"] == "SUCCEEDED", f"Job failed: {final_status.get('message')}"

    # Return path to LAST frame (which will become input for next iteration)
    timestring = final_status["timestring"]
    last_frame_idx = max_frames - 1
    output_frame = output_dir / timestring / f"{last_frame_idx:09d}.png"

    assert output_frame.exists(), f"Output frame not found: {output_frame}"

    return output_frame


@pytest.mark.parametrize("steps,normal_strength,keyframe_strength", [
    # TEST SET 1: Normal Strength Sweep (cadence frame stability)
    # Goal: Find optimal normal_strength for I2I chain stability
    # Keyframe strength fixed at 0.15, vary normal_strength
    (20, 0.80, 0.15),  # Lower - more diffusion steps on cadence frames
    (20, 0.85, 0.15),  # Current default
    (20, 0.90, 0.15),  # Higher - fewer diffusion steps, more stability
    (20, 0.95, 0.15),  # Very high - minimal diffusion, maximum stability

    # TEST SET 2: Keyframe Strength Sweep (keyframe retention)
    # Goal: How much keyframe strength affects color preservation
    # Normal strength fixed at 0.85, vary keyframe_strength
    # (Less critical for color preservation since keyframes introduce change)
    (20, 0.85, 0.10),  # Lower - more diffusion at keyframes (more change)
    (20, 0.85, 0.20),  # Higher - less diffusion at keyframes (more retention)

    # TEST SET 3: Combined optimization
    # Goal: Test if combined tuning improves results
    (20, 0.90, 0.10),  # High cadence stability + low keyframe retention
    (20, 0.95, 0.10),  # Very high cadence stability + low keyframe retention

    # TEST SET 4: Schnell viability (4 steps)
    # Goal: Check if Schnell can maintain stability with coarse resolution
    # Note: 4 steps = 0.25 strength resolution vs 20 steps = 0.05 resolution
    (4, 0.75, 0.25),   # 3/4 steps on cadence = similar to 0.85 @ 20 steps
    (4, 1.00, 0.25),   # 4/4 steps on cadence = no diffusion (pure I2I feed)
])
def test_color_preservation_sweep(steps, normal_strength, keyframe_strength):
    """Sweep strength parameters and measure color preservation through I2V chaining.

    This test runs I2V chaining iterations: each iteration generates 30 frames,
    then feeds the LAST frame back as input for the next iteration. This tests
    how well the parameters maintain color through cascading I2I diffusion.

    With New 3D mode (REDISTRIBUTED) and 30 frames:
    - Frame 0: Keyframe (uses keyframe_strength)
    - Frames 1-28: Mostly cadence frames (use normal_strength) ← THIS IS WHAT WE'RE TESTING
    - Frame 29: Keyframe (uses keyframe_strength)
    - ~27/30 frames use normal_strength, so this primarily tests cadence stability

    Metrics:
    - Iterations until grayscale (higher = better)
    - Color score trajectory (measures degradation rate)
    - Temporal consistency (not as relevant for chaining test)

    Args:
        steps: Number of sampling steps (4 for Schnell, 20 for Dev)
        normal_strength: Strength for cadence frames (HIGH = 0.85+ = stability)
        keyframe_strength: Strength for keyframes (LOW = 0.15- = change)
    """
    # Create test output directory
    test_name = f"steps{steps}_norm{normal_strength:.2f}_kf{keyframe_strength:.2f}"
    test_dir = OUTPUT_DIR / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Generate colorful test image
    test_image = create_colorful_test_image()

    # Track metrics
    color_scores = []
    frames = []
    current_image = test_image

    print(f"\n{'='*60}")
    print(f"Testing: steps={steps}, normal={normal_strength}, keyframe={keyframe_strength}")
    print(f"{'='*60}")

    for iteration in range(MAX_ITERATIONS):
        print(f"\nIteration {iteration + 1}/{MAX_ITERATIONS}...")

        # Run I2V generation
        output_frame = run_i2v_iteration(
            init_image_path=current_image,
            strength=normal_strength,
            keyframe_strength=keyframe_strength,
            steps=steps,
            output_dir=test_dir,
        )

        # Load and measure
        frame_array = load_image_as_numpy(output_frame)
        frames.append(frame_array)

        color_score = measure_color_preservation(frame_array)
        color_scores.append(color_score)

        print(f"  Color score: {color_score:.1f}/100")

        # Save copy with iteration number
        import shutil
        numbered_output = test_dir / f"iteration_{iteration:03d}_color{color_score:.0f}.png"
        shutil.copy(output_frame, numbered_output)

        # Check if grayscale threshold reached
        if color_score < GRAYSCALE_THRESHOLD:
            print(f"\n⚠️  Grayscale threshold reached at iteration {iteration + 1}")
            print(f"  Final color score: {color_score:.1f}/100")
            break

        # Use this output as input for next iteration
        current_image = output_frame

    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_quality_score(frames)

    # Save results
    results = {
        'parameters': {
            'steps': steps,
            'normal_strength': normal_strength,
            'keyframe_strength': keyframe_strength,
        },
        'iterations_completed': len(color_scores),
        'color_scores': color_scores,
        'metrics': metrics,
        'final_color_score': color_scores[-1] if color_scores else 0,
        'iterations_until_grayscale': len(color_scores),
    }

    results_file = test_dir / "metrics.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Iterations completed: {len(color_scores)}")
    print(f"  Final color score: {color_scores[-1]:.1f}/100")
    print(f"  Average color: {metrics['avg_color']:.1f}/100")
    print(f"  Average temporal consistency: {metrics['avg_temporal']:.1f}/100")
    print(f"  Degradation rate: {metrics['degradation_rate']:.2f}%/iteration")
    print(f"  Overall quality score: {metrics['overall_score']:.1f}/100")
    print(f"{'='*60}\n")

    # Test passes if we got through at least 5 iterations
    # (Shows the parameters can maintain some stability)
    assert len(color_scores) >= 5, (
        f"Color degraded too quickly ({len(color_scores)} iterations). "
        f"Parameters: steps={steps}, norm={normal_strength}, kf={keyframe_strength}"
    )
