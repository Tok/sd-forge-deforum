"""Helper functions for running tuning tests without pytest dependency.

This module extracts the core test functions from test_color_preservation.py
so they can be used by the tuning API without importing pytest or tenacity.
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, Any

from deforum.utils.system.logging import get_logger

logger = get_logger()

# Constants from integration/utils.py
SERVER_BASE_URL = "http://localhost:7860"
API_ROOT = "/deforum_api"
API_BASE_URL = SERVER_BASE_URL + API_ROOT


def simple_retry(func, max_attempts=450, wait_seconds=2, timeout_seconds=900):
    """Simple retry decorator replacement for tenacity.

    Args:
        func: Function to retry
        max_attempts: Maximum number of retry attempts
        wait_seconds: Seconds to wait between attempts
        timeout_seconds: Total timeout in seconds

    Returns:
        Function result if successful

    Raises:
        Last exception if all retries exhausted
    """
    start_time = time.time()
    attempt = 0
    last_exception = None

    while attempt < max_attempts and (time.time() - start_time) < timeout_seconds:
        try:
            return func()
        except Exception as e:
            last_exception = e
            attempt += 1
            if attempt < max_attempts and (time.time() - start_time) < timeout_seconds:
                time.sleep(wait_seconds)

    # All retries exhausted
    if last_exception:
        raise last_exception
    else:
        raise TimeoutError(f"Retry timeout after {timeout_seconds}s")


def wait_for_job_to_complete(job_id: str) -> Dict[str, Any]:
    """Wait for a Deforum job to complete.

    Args:
        job_id: Job identifier

    Returns:
        Final job status dict

    Raises:
        RuntimeError if job fails
        TimeoutError if job doesn't complete within timeout
    """
    from deforum.api.models import DeforumJobStatus, DeforumJobStatusCategory

    def check_job():
        response = requests.get(
            f"{API_BASE_URL}/jobs/{job_id}",
            headers={"accept": "application/json"}
        )
        response.raise_for_status()

        try:
            job_status = DeforumJobStatus.model_validate(response.json())
        except Exception as e:
            logger.error(f"Failed to parse job status: {e}")
            logger.error(f"Raw response: {response.text}")
            raise

        logger.info(
            f"Waiting for job {job_id}: status={job_status.status}; "
            f"phase={job_status.phase}; execution_time:{job_status.execution_time}s"
        )

        # Keep retrying if still accepted/running
        if job_status.status == DeforumJobStatusCategory.ACCEPTED:
            raise RuntimeError("Job still in ACCEPTED state, retrying...")

        # Job completed (success or failure)
        return job_status.model_dump()

    return simple_retry(check_job, max_attempts=450, wait_seconds=2, timeout_seconds=900)


def get_test_batch_name(test_name: str) -> str:
    """Get a batch name for test identification.

    Args:
        test_name: Name of the test

    Returns:
        Batch name pattern with {timestring} placeholder
    """
    return f"tuning-{test_name}_{{timestring}}"


def get_test_options_overrides(output_dir: Path = None) -> Dict[str, Any]:
    """Get options overrides for tuning tests.

    Args:
        output_dir: Specific output directory to use (optional)

    Returns:
        Dict with outdir_samples override
    """
    if output_dir:
        # Use the specific test directory
        return {
            "outdir_samples": str(output_dir),
        }
    else:
        # Use default tuning output directory
        return {
            "outdir_samples": str(Path(__file__).parent.parent.parent / "outputs" / "deforum-tuning"),
        }


def create_colorful_test_image(output_dir: Path) -> Path:
    """Generate a brightly colored test image with rainbow gradients.

    Args:
        output_dir: Directory to save test image

    Returns:
        Path to generated test image
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import cv2

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
            rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0][0]

            img_array[y, x] = rgb_pixel

    # Add some high-frequency detail to make it interesting
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
    test_img_path = output_dir / "test_input_rainbow.png"
    test_img_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(test_img_path)

    return test_img_path


def run_i2v_iteration(
    init_image_path: Path,
    strength: float,
    keyframe_strength: float,
    steps: int,
    output_dir: Path,
) -> Path:
    """Run a single I2V generation iteration.

    Args:
        init_image_path: Path to input image
        strength: Normal/tween frame strength
        keyframe_strength: Keyframe strength
        steps: Number of sampling steps
        output_dir: Where to save output

    Returns:
        Path to generated output image (frame 0 of animation)
    """
    # Use output_dir as the base for this specific test configuration
    options_overrides = get_test_options_overrides(output_dir)
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
            "render_mode": "new_3d",  # New 3D with dual strength
            "max_frames": 2,  # Just 2 frames (init + 1 generation)
            "fps": 24,

            # Strength schedules (the parameters we're testing!)
            "strength_schedule": f"0: ({strength})",
            "keyframe_strength_schedule": f"0: ({keyframe_strength})",

            # Use init image
            "use_init": True,
            "init_image": str(init_image_path),
            "strength": keyframe_strength,  # For frame 0

            # Prompts
            "animation_prompts": json.dumps({
                "0": "vibrant colorful rainbow gradient, highly saturated colors"
            }),

            # Disable audio for tuning tests
            "audio_mode": "None",
            "audio_sync": False,

            # Output - batch_name will append timestring automatically
            "batch_name": get_test_batch_name("tuning_test"),
        },
        "options_overrides": options_overrides,
    }

    # Submit job
    response = requests.post(f"{API_BASE_URL}/batches/", json=settings)
    # API returns 202 Accepted for async job submission
    if response.status_code not in [200, 202]:
        raise RuntimeError(f"Failed to submit job (status {response.status_code}): {response.text}")

    batch_info = response.json()
    batch_id = batch_info["batch_id"]
    job_ids = batch_info["job_ids"]

    # Wait for completion
    final_status = wait_for_job_to_complete(job_ids[0])

    if final_status["status"] != "SUCCEEDED":
        raise RuntimeError(f"Job failed: {final_status.get('message')}")

    # Return path to frame 0 (the generated output)
    # The actual directory name is batch_name with timestring appended: "tuning-tuning_test_{timestring}"
    timestring = final_status["timestring"]
    batch_name = get_test_batch_name("tuning_test").replace("{timestring}", timestring)

    # Frame filename is 9 digits (000000000.png), not 10
    output_frame = output_dir / batch_name / "000000000.png"

    if not output_frame.exists():
        # Debug: list what actually exists
        batch_dir = output_dir / batch_name
        if batch_dir.exists():
            import os
            files = list(os.listdir(batch_dir))
            logger.error(f"Directory exists but frame not found. Contents: {files}")
        else:
            logger.error(f"Batch directory doesn't exist: {batch_dir}")
        raise FileNotFoundError(f"Output frame not found: {output_frame}")

    return output_frame
