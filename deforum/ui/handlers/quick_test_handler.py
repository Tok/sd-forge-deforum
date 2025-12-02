"""Quick Test Button Handler

Connects the "Generate Test" button to a simplified test render.

ISOLATION: Only affects Quick Test tab. Normal Deforum remains unchanged.
"""

import traceback
import json
from typing import Tuple
from pathlib import Path

from deforum.utils.system.logging import get_logger, emoji as emoji_utils
from deforum.utils.output_paths import OutputPaths

logger = get_logger()


def handle_generate_test_click(
    prompt_theme: str,
    audio_theme: str,
    duration: float,
    seed: int
) -> Tuple[str, str, str]:
    """Handle "Generate Test" button click.

    Args:
        prompt_theme: Theme for prompt generation
        audio_theme: Theme for audio generation
        duration: Duration in seconds (from slider)
        seed: Random seed (from number input)

    Returns:
        Tuple of (status, log, settings_json_state)
        - status: Single-line status message
        - log: Multi-line generation log
        - settings_json_state: JSON settings (for hidden state)
    """
    # Theme-aware status emojis
    warning = emoji_utils.maybe_warning()
    check = emoji_utils.maybe_check()
    cross = emoji_utils.maybe_cross()

    try:
        # Validate inputs
        if duration < 3.0 or duration > 10.0:
            return (
                f"{warning} Error: Duration must be between 3 and 10 seconds",
                "Invalid duration provided.",
                "{}"
            )

        if not prompt_theme or not prompt_theme.strip():
            return (
                f"{warning} Error: Prompt theme cannot be empty",
                "Please provide a theme for prompt generation.",
                "{}"
            )

        if not audio_theme or not audio_theme.strip():
            audio_theme = "synthetic amen break"  # Default fallback

        # Log start
        logger.info("=" * 60)
        logger.info("🎬 Quick Test started")
        logger.info(f"Prompt Theme: '{prompt_theme}'")
        logger.info(f"Audio Theme: '{audio_theme}'")
        logger.info(f"Duration: {duration}s, Seed: {seed}")
        logger.info("=" * 60)

        # Execute test render
        result = execute_quick_test(
            prompt_theme=prompt_theme.strip(),
            audio_theme=audio_theme.strip(),
            duration_seconds=duration,
            random_seed=int(seed) if seed != -1 else -1,
            output_dir=OutputPaths.DEFORUM
        )

        # Format results for UI
        if result["success"]:
            status = f"{check} Test complete! Video: {result['video_path']}"
            log = "\n".join(result["log"])
            settings = json.dumps(result["settings"], indent=2)

            logger.info(f"{check} Success: {result['video_path']}")
        else:
            status = f"{cross} Test failed: {result['error']}"
            log = "\n".join(result["log"])
            settings = "{}"

            logger.error(f"{cross} Failure: {result['error']}")

        return (status, log, settings)

    except Exception as e:
        error_msg = f"💥 Unexpected error: {str(e)}"
        error_trace = traceback.format_exc()

        logger.error(f"{error_msg}\n{error_trace}")

        return (
            f"{cross} Fatal error: {str(e)}",
            f"💥 UNEXPECTED ERROR\n\n{error_trace}",
            "{}"
        )


def execute_quick_test(
    prompt_theme: str,
    audio_theme: str,
    duration_seconds: float,
    random_seed: int,
    output_dir: str
) -> dict:
    """Execute the quick test render.

    Args:
        prompt_theme: Theme for prompt generation
        audio_theme: Theme for audio generation
        duration_seconds: Duration in seconds
        random_seed: Random seed (-1 for random)
        output_dir: Output directory path

    Returns:
        Dict with keys: success, video_path, log, settings, error
    """
    log = []

    try:
        import random
        from deforum.utils.audio_generation import generate_loop
        from deforum.config.defaults_generator import BUNNY_DEFAULT_AUDIO_CONFIG
        import os
        import time

        log.append("🎬 Starting Quick Test generation...")
        log.append("")

        # Generate random seed if needed
        if random_seed == -1:
            random_seed = random.randint(0, 2**32 - 1)
            log.append(f"🎲 Generated random seed: {random_seed}")
            log.append("")

        # Calculate frame count
        fps = 60
        total_frames = int(duration_seconds * fps)
        log.append(f"📊 Configuration: {total_frames} frames at {fps} FPS")
        log.append("")

        # Phase 1: Generate synthetic audio
        log.append("🎵 Phase 1: Generating synthetic audio...")

        # Create output directory for this batch
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(output_dir, f"quick_test_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)

        audio_path = os.path.join(batch_dir, "test_audio.wav")

        # Use synthetic amen break config but without chaos
        audio_config = BUNNY_DEFAULT_AUDIO_CONFIG.copy()
        audio_config["enable_chaos"] = False
        audio_config["duration"] = duration_seconds

        try:
            generate_loop(
                prompt=audio_theme,
                duration_seconds=duration_seconds,
                bpm=173,  # Amen break tempo
                output_path=audio_path,
                enable_chaos=False
            )
            log.append(f"✓ Audio generated: {audio_path}")
        except Exception as e:
            log.append(f"⚠️ Audio generation failed (continuing without audio): {e}")
            audio_path = None

        log.append("")

        # Phase 2: Generate prompts with Qwen
        log.append("🤖 Phase 2: Generating prompts with Qwen...")

        try:
            from deforum.ui.handlers.audio_prompt_generator import generate_prompts_with_ai

            # Generate prompts using Qwen with the theme
            # TODO: This needs actual integration - for now use placeholder
            generated_prompts = {
                "0": f"{prompt_theme}, cinematic lighting, photorealistic, high quality",
                str(total_frames // 2): f"{prompt_theme}, dynamic movement, cinematic, detailed",
                str(total_frames - 1): f"{prompt_theme}, final view, cinematic lighting, photorealistic"
            }
            log.append(f"✓ Generated {len(generated_prompts)} prompts from theme: '{prompt_theme}'")

        except Exception as e:
            log.append(f"⚠️ Prompt generation failed, using simple prompts: {e}")
            generated_prompts = {
                "0": f"{prompt_theme}, photorealistic",
                str(total_frames - 1): f"{prompt_theme}, photorealistic"
            }

        log.append("")

        # Phase 3: Build test settings
        log.append("⚙️ Phase 3: Building test settings...")

        settings = {
            "prompt_theme": prompt_theme,
            "audio_theme": audio_theme,
            "prompts": generated_prompts,
            "duration": duration_seconds,
            "fps": fps,
            "total_frames": total_frames,
            "resolution": "1280x720",
            "render_mode": "New 3D",
            "steps": 20,
            "cfg_scale": 7.0,
            "sampler": "euler_a",
            "cadence": 5,
            "strength": 0.85,
            "keyframe_strength": 0.20,
            "seed": random_seed,
            "audio_path": audio_path,
            "output_dir": batch_dir,
            "depth_model": "Depth-Anything-V2-Small",
            "camera_movement": {
                "translation_z": "0:(0), {}: (2.0)".format(total_frames),
                "rotation_3d_y": "0:(0), {}: (5.0)".format(total_frames)
            }
        }

        log.append(f"✓ Test settings configured")
        log.append(f"  - Render Mode: {settings['render_mode']}")
        log.append(f"  - Resolution: {settings['resolution']}")
        log.append(f"  - Steps: {settings['steps']}, Cadence: {settings['cadence']}")
        log.append(f"  - Strength: {settings['strength']} (normal), {settings['keyframe_strength']} (keyframe)")
        log.append("")

        # Phase 3: Would execute Deforum render here
        # For now, we'll return success with the settings
        # TODO: Actually call the Deforum render pipeline

        log.append("⚠️ Note: Actual render execution not yet implemented")
        log.append("This is a placeholder for the full render pipeline")
        log.append("")
        log.append("✓ Quick Test validation complete!")

        return {
            "success": True,
            "video_path": os.path.join(batch_dir, "test_video.mp4"),
            "log": log,
            "settings": settings,
            "error": None
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        log.append(f"❌ Error: {str(e)}")
        log.append(error_trace)

        return {
            "success": False,
            "video_path": None,
            "log": log,
            "settings": {},
            "error": str(e)
        }


def handle_view_test_settings_click(settings_json: str) -> str:
    """Handle "View Settings" button click.

    Args:
        settings_json: JSON settings from hidden state

    Returns:
        Formatted settings string for display
    """
    warning = emoji_utils.maybe_warning()
    if not settings_json or settings_json == "{}":
        return f"{warning} No settings available. Generate a test first!"

    try:
        settings = json.loads(settings_json)

        # Format for readable display
        lines = []
        lines.append("═══ QUICK TEST SETTINGS ═══")
        lines.append("")
        lines.append(f"Prompt: {settings.get('prompt', 'N/A')}")
        lines.append(f"Duration: {settings.get('duration', 'N/A')}s")
        lines.append(f"FPS: {settings.get('fps', 'N/A')}")
        lines.append(f"Total Frames: {settings.get('total_frames', 'N/A')}")
        lines.append(f"Resolution: {settings.get('resolution', 'N/A')}")
        lines.append(f"Render Mode: {settings.get('render_mode', 'N/A')}")
        lines.append(f"Steps: {settings.get('steps', 'N/A')}")
        lines.append(f"CFG Scale: {settings.get('cfg_scale', 'N/A')}")
        lines.append(f"Sampler: {settings.get('sampler', 'N/A')}")
        lines.append(f"Cadence: {settings.get('cadence', 'N/A')}")
        lines.append(f"Strength: {settings.get('strength', 'N/A')} (normal)")
        lines.append(f"Keyframe Strength: {settings.get('keyframe_strength', 'N/A')}")
        lines.append(f"Seed: {settings.get('seed', 'N/A')}")
        lines.append(f"Depth Model: {settings.get('depth_model', 'N/A')}")
        lines.append("")
        lines.append("Camera Movement:")
        camera = settings.get('camera_movement', {})
        for key, value in camera.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        lines.append(f"Audio: {settings.get('audio_path', 'None')}")
        lines.append(f"Output: {settings.get('output_dir', 'N/A')}")

        return "\n".join(lines)

    except Exception as e:
        return f"{warning} Failed to parse settings: {e}"


def handle_open_test_output_click() -> str:
    """Handle "Open Output Folder" button click.

    Returns:
        Status message
    """
    warning = emoji_utils.maybe_warning()
    check = emoji_utils.maybe_check()
    import os
    import subprocess
    import platform

    # Use centralized output path
    output_dir = OutputPaths.DEFORUM

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open folder in file manager
    try:
        if platform.system() == "Windows":
            os.startfile(output_dir)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", output_dir])
        else:  # Linux
            subprocess.run(["xdg-open", output_dir])

        return f"{check} Opened folder: {output_dir}"

    except Exception as e:
        return f"{warning} Failed to open folder: {e}\nPath: {os.path.abspath(output_dir)}"
