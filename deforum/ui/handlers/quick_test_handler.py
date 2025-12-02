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
        log.append("🤖 Phase 2: Generating escalating synthwave prompts with Qwen...")

        try:
            from deforum.ui.handlers.audio_prompt_generator import generate_prompts_with_ai

            # Generate escalating synthwave prompts
            # Calculate how many prompts we need (one per beat roughly)
            beats_per_second = 173 / 60  # BPM to beats per second
            prompt_count = max(3, int(duration_seconds * beats_per_second / 4))  # One prompt every 4 beats

            log.append(f"Generating {prompt_count} escalating prompts...")

            # Call Qwen with escalating mode and synthwave style
            prompt_result = generate_prompts_with_ai(
                generation_mode="escalating",
                intensity="crazy",  # Escalating intensity
                style="synthwave",
                theme=prompt_theme,
                count=prompt_count,
                start_prompt="",  # Not used in escalating mode
                end_prompt="",
                soundtrack_path=audio_path
            )

            # Parse result (comes back as newline-separated prompts)
            if isinstance(prompt_result, dict) and 'value' in prompt_result:
                prompt_lines = prompt_result['value'].strip().split('\n')
            else:
                prompt_lines = str(prompt_result).strip().split('\n')

            # Distribute prompts evenly across frames
            generated_prompts = {}
            for i, prompt in enumerate(prompt_lines):
                if prompt.strip():
                    frame_num = int((i / len(prompt_lines)) * total_frames)
                    generated_prompts[str(frame_num)] = prompt.strip()

            log.append(f"✓ Generated {len(generated_prompts)} escalating synthwave prompts")

        except Exception as e:
            log.append(f"⚠️ Qwen generation failed, using fallback escalation: {e}")
            # Fallback escalating prompts
            generated_prompts = {
                "0": f"A {prompt_theme}, synthwave aesthetic, photorealistic",
                str(total_frames // 3): f"A {prompt_theme} with neon lights, cyberpunk synthwave, dynamic",
                str(2 * total_frames // 3): f"A synthwave {prompt_theme} with holographic effects, glowing neon, cyberpunk city",
                str(total_frames - 1): f"An epic synthwave {prompt_theme} deity, mandelbulb fractals, neon universe, transcendent"
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

        # Phase 4: Execute Deforum render
        log.append("🎬 Phase 4: Executing Deforum render...")
        log.append("")

        try:
            from deforum.config.args import (
                DeforumArgs, DeforumAnimArgs, DeforumOutputArgs,
                ParseqArgs, LoopArgs, WanArgs, AudioSyncArgs,
                process_args
            )
            from deforum.rendering.core import render_animation
            from modules.processing import StableDiffusionProcessingImg2Img
            import modules.shared as shared

            # Build args_dict with all required parameters
            args_dict = {}

            # Add all defaults from argument definition functions
            for key, config in DeforumArgs().items():
                if isinstance(config, dict) and 'value' in config:
                    args_dict[key] = config['value']
                else:
                    args_dict[key] = config

            for key, config in DeforumAnimArgs().items():
                if isinstance(config, dict) and 'value' in config:
                    args_dict[key] = config['value']
                else:
                    args_dict[key] = config

            for key, config in DeforumOutputArgs().items():
                if isinstance(config, dict) and 'value' in config:
                    args_dict[key] = config['value']
                else:
                    args_dict[key] = config

            for key, config in ParseqArgs().items():
                if isinstance(config, dict) and 'value' in config:
                    args_dict[key] = config['value']
                else:
                    args_dict[key] = config

            for key, config in LoopArgs().items():
                if isinstance(config, dict) and 'value' in config:
                    args_dict[key] = config['value']
                else:
                    args_dict[key] = config

            for key, config in WanArgs().items():
                if isinstance(config, dict) and 'value' in config:
                    args_dict[key] = config['value']
                else:
                    args_dict[key] = config

            for key, config in AudioSyncArgs().items():
                if isinstance(config, dict) and 'value' in config:
                    args_dict[key] = config['value']
                else:
                    args_dict[key] = config

            # Override with test-specific settings
            args_dict['W'] = 1280
            args_dict['H'] = 720
            args_dict['fps'] = fps
            args_dict['steps'] = settings['steps']
            args_dict['cfg_scale'] = settings['cfg_scale']
            args_dict['sampler'] = settings['sampler']
            args_dict['seed'] = random_seed
            args_dict['strength'] = settings['strength']
            args_dict['max_frames'] = total_frames
            args_dict['render_mode'] = 'New 3D'
            args_dict['animation_mode'] = '3D'
            args_dict['keyframe_distribution'] = 'redistributed'
            args_dict['diffusion_cadence'] = settings['cadence']
            args_dict['add_soundtrack'] = 'File'
            args_dict['soundtrack_path'] = audio_path
            args_dict['skip_video_creation'] = False
            args_dict['delete_imgs'] = False
            args_dict['animation_prompts'] = json.dumps(generated_prompts)
            args_dict['animation_prompts_positive'] = ""  # No global positive prompt
            args_dict['animation_prompts_negative'] = "nsfw, nude, text, watermark, blurry, low quality"
            args_dict['strength_schedule'] = f"0:({settings['strength']})"
            args_dict['keyframe_strength_schedule'] = f"0:({settings['keyframe_strength']})"

            # Camera movement
            args_dict['translation_z'] = settings['camera_movement']['translation_z']
            args_dict['rotation_3d_y'] = settings['camera_movement']['rotation_3d_y']

            # Depth settings for 3D mode
            args_dict['depth_algorithm'] = 'Depth-Anything-V2-Small'
            args_dict['midas_weight'] = 0.3
            args_dict['near_plane'] = 200
            args_dict['far_plane'] = 10000
            args_dict['fov'] = 70

            # Create fake Processing object (required by process_args)
            p = StableDiffusionProcessingImg2Img(
                sd_model=shared.sd_model,
                outpath_samples=batch_dir,
                outpath_grids=batch_dir,
                prompt="",  # Will be overridden by animation_prompts
                negative_prompt="",
                seed=random_seed,
                sampler_name=settings['sampler'],
                batch_size=1,
                n_iter=1,
                steps=settings['steps'],
                cfg_scale=settings['cfg_scale'],
                width=1280,
                height=720,
                init_images=[],
                denoising_strength=1.0 - settings['strength']  # Deforum inverted strength
            )
            args_dict['p'] = p

            # Additional required fields
            args_dict['override_settings_with_file'] = False
            args_dict['custom_settings_file'] = None

            # Generate unique run ID
            run_id = f"quick_test_{timestamp}"

            # Process args into structured namespaces
            log.append("  → Building argument namespaces...")
            (
                args_loaded_ok,
                root,
                args,
                anim_args,
                video_args,
                parseq_args,
                audio_sync_args,
                loop_args,
                controlnet_args,
                wan_args
            ) = process_args(args_dict, run_id)

            if not args_loaded_ok:
                raise Exception("Failed to load argument configuration")

            log.append("  → Starting render animation...")
            log.append("")

            # Execute render
            render_result = render_animation(
                args=args,
                anim_args=anim_args,
                video_args=video_args,
                parseq_args=parseq_args,
                loop_args=loop_args,
                controlnet_args=controlnet_args,
                root=root
            )

            log.append("")
            log.append("✓ Render complete!")

            # Find the generated video
            video_path = None
            for ext in ['.mp4', '.webm', '.mov']:
                potential_path = os.path.join(batch_dir, f"*{ext}")
                import glob
                matches = glob.glob(potential_path)
                if matches:
                    video_path = matches[0]
                    break

            if not video_path:
                video_path = os.path.join(batch_dir, "test_video.mp4")
                log.append(f"⚠️ Video file not found, expected at: {video_path}")

            return {
                "success": True,
                "video_path": video_path,
                "log": log,
                "settings": settings,
                "error": None
            }

        except Exception as render_error:
            error_trace = traceback.format_exc()
            log.append(f"❌ Render failed: {str(render_error)}")
            log.append(error_trace)

            return {
                "success": False,
                "video_path": None,
                "log": log,
                "settings": settings,
                "error": f"Render execution failed: {str(render_error)}"
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
