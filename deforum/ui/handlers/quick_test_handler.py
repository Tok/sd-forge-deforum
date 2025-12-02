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


def _get_empty_settings_components():
    """Get empty values for all settings components (for error returns)."""
    from deforum.config.settings import get_settings_component_names
    # Return empty string for each component
    return [''] * len(get_settings_component_names())


def handle_generate_test_click(
    prompt_theme: str,
    audio_theme: str,
    duration: float,
    seed: int
) -> Tuple:
    """Handle "Generate Test" button click.

    Generates audio, prompts, and schedules, then LOADS them into the UI automatically.
    User just needs to click Generate to render.

    Args:
        prompt_theme: Theme for prompt generation
        audio_theme: Theme for audio generation
        duration: Duration in seconds (from slider)
        seed: Random seed (from number input)

    Returns:
        Tuple of (status, log, settings_json) + ALL settings components
        - First 3 values: status message, generation log, settings JSON
        - Remaining values: ALL UI components (same as Load All Settings button)
    """
    # Theme-aware emojis
    warning = emoji_utils.maybe_warning()
    check = emoji_utils.maybe_check()
    cross = emoji_utils.maybe_cross()
    film = emoji_utils.maybe_film()
    music = emoji_utils.maybe_music()
    robot = emoji_utils.maybe_robot()
    gear = emoji_utils.maybe_gear()
    chart = emoji_utils.maybe_chart()

    try:
        # Validate inputs
        if duration < 3.0 or duration > 10.0:
            return (
                f"{warning} Error: Duration must be between 3 and 10 seconds",
                "Invalid duration provided.",
                "{}"  # settings_json
            ) + tuple(_get_empty_settings_components())

        if not prompt_theme or not prompt_theme.strip():
            return (
                f"{warning} Error: Prompt theme cannot be empty",
                "Please provide a theme for prompt generation.",
                "{}"  # settings_json
            ) + tuple(_get_empty_settings_components())

        if not audio_theme or not audio_theme.strip():
            audio_theme = "synthetic amen break"  # Default fallback

        # Log start
        logger.info("=" * 60)
        logger.info("Quick Test started")
        logger.info(f"Prompt Theme: '{prompt_theme}'")
        logger.info(f"Audio Theme: '{audio_theme}'")
        logger.info(f"Duration: {duration}s, Seed: {seed}")
        logger.info("=" * 60)

        # Execute test generation (returns tuple directly)
        return execute_quick_test(
            prompt_theme=prompt_theme.strip(),
            audio_theme=audio_theme.strip(),
            duration_seconds=duration,
            random_seed=int(seed) if seed != -1 else -1,
            output_dir=OutputPaths.DEFORUM
        )

    except Exception as e:
        error_msg = f"💥 Unexpected error: {str(e)}"
        error_trace = traceback.format_exc()

        logger.error(f"{error_msg}\n{error_trace}")

        # Return error with empty settings components
        return (
            f"{cross} Fatal error: {str(e)}",
            f"💥 UNEXPECTED ERROR\n\n{error_trace}",
            "{}"  # settings_json
        ) + tuple(_get_empty_settings_components())


def execute_quick_test(
    prompt_theme: str,
    audio_theme: str,
    duration_seconds: float,
    random_seed: int,
    output_dir: str
) -> Tuple[str, str, str, str, str, int, int, str, str, str, int, int]:
    """Execute the quick test generation.

    Args:
        prompt_theme: Theme for prompt generation
        audio_theme: Theme for audio generation
        duration_seconds: Duration in seconds
        random_seed: Random seed (-1 for random)
        output_dir: Output directory path

    Returns:
        Tuple of (status, log, settings_json, prompts, audio_path, fps, max_frames, translation_z, rotation_y, strength_schedule, steps, cadence)
    """
    # Theme-aware emojis
    warning = emoji_utils.maybe_warning()
    check = emoji_utils.maybe_check()
    cross = emoji_utils.maybe_cross()
    film = emoji_utils.maybe_film()
    music = emoji_utils.maybe_music()
    robot = emoji_utils.maybe_robot()
    gear = emoji_utils.maybe_gear()
    chart = emoji_utils.maybe_chart()

    log = []

    try:
        import random
        from deforum.utils.audio_generation import generate_loop
        from deforum.config.defaults_generator import BUNNY_DEFAULT_AUDIO_CONFIG
        import os
        import time

        log.append(f"{film or '🎬'} Starting Quick Test generation...")
        log.append("")

        # Generate random seed if needed
        if random_seed == -1:
            random_seed = random.randint(0, 2**32 - 1)
            log.append(f"🎲 Generated random seed: {random_seed}")
            log.append("")

        # Detect current model and set appropriate defaults
        from deforum.utils.model_detection import (
            get_model_name,
            get_recommended_steps,
            get_recommended_cfg_scale,
            is_lumina_model
        )

        model_name = get_model_name()
        recommended_steps = get_recommended_steps()
        cfg_min, cfg_max = get_recommended_cfg_scale()
        recommended_cfg = cfg_min  # Use minimum of recommended range

        # Lumina needs special scheduler
        recommended_scheduler = "Karras" if not is_lumina_model() else "Karras"  # Both use Karras

        log.append(f"{robot or '🤖'} Detected Model: {model_name}")
        log.append(f"  → Recommended Steps: {recommended_steps}")
        log.append(f"  → Recommended CFG: {recommended_cfg:.1f}")
        log.append(f"  → Scheduler: {recommended_scheduler}")
        log.append("")

        # Calculate frame count
        fps = 60
        total_frames = int(duration_seconds * fps)
        log.append(f"{chart or '📊'} Configuration: {total_frames} frames at {fps} FPS")
        log.append("")

        # Phase 1: Generate synthetic audio
        log.append(f"{music or '🎵'} Phase 1: Generating synthetic audio...")

        # Create output directory for this batch
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(output_dir, f"quick_test_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)

        audio_path = os.path.join(batch_dir, "test_audio.wav")

        # Generate percussive audio for better event detection
        # Use more percussive prompt to ensure clear transients
        percussive_prompt = f"{audio_theme}, heavy drums, clear percussion, strong beats"

        try:
            generate_loop(
                prompt=percussive_prompt,
                duration_seconds=duration_seconds,
                bpm=173,  # Amen break tempo
                output_path=audio_path,
                enable_chaos=False  # Keep it clean for reliable detection
            )
            log.append(f"✓ Audio generated: {audio_path}")
            log.append(f"   Prompt: '{percussive_prompt}'")
        except Exception as e:
            log.append(f"⚠️ Audio generation failed (continuing without audio): {e}")
            audio_path = None

        log.append("")

        # Phase 2: Detect audio events (BPM-aware)
        log.append(f"{music or '🎵'} Phase 2: Analyzing audio events...")

        try:
            import librosa
            from deforum.audio import detect_events_bpm_aware

            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            actual_duration = librosa.get_duration(y=y, sr=sr)

            # Update max_frames based on actual audio duration
            total_frames = int(actual_duration * fps)
            log.append(f"Actual audio duration: {actual_duration:.2f}s → {total_frames} frames @ {fps} FPS")

            # Detect events with VERY aggressive settings for Quick Test
            # No fallbacks - we want to test the real audio detection system
            event_times, event_intensities, detected_bpm = detect_events_bpm_aware(
                audio=y,
                sample_rate=sr,
                method="onset",  # Onset detection for transients
                target_bpm=173,  # Amen break BPM
                tolerance=0.40,   # ±40% acceptable (very relaxed for test audio)
                prefer_under_detection=False  # Allow more events
            )

            log.append(f"✓ Detected {len(event_times)} events at {detected_bpm:.1f} BPM")

            # Validate we got reasonable number of events
            min_events = max(6, int(actual_duration * 1.5))  # At least 1.5 events/second
            if len(event_times) < min_events:
                error_msg = f"❌ Audio detection found only {len(event_times)} events (expected {min_events}+)"
                log.append(error_msg)
                log.append("This indicates a problem with audio generation or detection.")
                log.append("Quick Test should generate audio with clear transients.")
                logger.error(error_msg)

        except Exception as e:
            error_msg = f"❌ Event detection failed: {e}"
            log.append(error_msg)
            log.append("Quick Test requires working audio event detection.")
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            # Don't continue - this is a test failure
            raise

        log.append("")

        # Phase 3: Generate prompts with Qwen
        log.append(f"{robot or '🤖'} Phase 3: Generating {len(event_times)} progressive prompts with Qwen...")

        try:
            from deforum.ui.handlers.audio_prompt_generator import generate_prompts_with_ai

            # Call Qwen with normal intensity for clean testing
            prompt_result = generate_prompts_with_ai(
                generation_mode="escalating",
                intensity="normal",  # Normal intensity - not crazy
                style="",  # No specific style
                theme=prompt_theme,
                count=len(event_times),  # One prompt per detected event
                start_prompt="",  # Not used in escalating mode
                end_prompt="",
                soundtrack_path=audio_path
            )

            # Parse result (comes back as newline-separated prompts)
            if isinstance(prompt_result, dict) and 'value' in prompt_result:
                prompt_lines = prompt_result['value'].strip().split('\n')
            else:
                prompt_lines = str(prompt_result).strip().split('\n')

            # Take only requested count
            prompts = [line.strip() for line in prompt_lines if line.strip()][:len(event_times)]

            log.append(f"✓ Generated {len(prompts)} progressive prompts")

        except Exception as e:
            log.append(f"⚠️ Qwen generation failed, using fallback escalation: {e}")
            # Fallback escalating prompts
            num_prompts = len(event_times)
            prompts = [
                f"{prompt_theme} {action}"
                for action in ["resting peacefully", "hopping gently", "moving actively",
                              "leaping dynamically", "racing wildly", "GOING ABSOLUTELY BONKERS"]
            ][:num_prompts]

        log.append("")

        # Phase 4: Sync prompts to audio events
        log.append(f"{film or '🎬'} Phase 4: Syncing prompts to audio events...")

        try:
            from deforum.audio import distribute_prompts_across_keyframes

            # Convert event times to keyframe dicts (required format for distribution)
            keyframes = [
                {
                    'frame': int(t * fps),
                    'intensity': intensity,
                    'time_seconds': t
                }
                for t, intensity in zip(event_times, event_intensities)
            ]

            # Distribute prompts (sequential mode - first prompt → first keyframe)
            prompt_schedule_json = distribute_prompts_across_keyframes(
                keyframes=keyframes,
                user_prompts=prompts,
                mode="sequential"
            )

            # Parse JSON string to dict
            generated_prompts = json.loads(prompt_schedule_json)

            log.append(f"✓ Synced {len(prompts)} prompts to {len(keyframes)} keyframes")

        except Exception as e:
            log.append(f"⚠️ Prompt sync failed: {e}")
            traceback.print_exc()
            # Create simple sequential schedule as fallback
            generated_prompts = {str(int(t * fps)): prompt for t, prompt in zip(event_times, prompts)}

        log.append("")

        # Phase 5: Build test settings
        log.append(f"{gear or '⚙️'} Phase 5: Building test settings...")

        # Map sampler names (Forge backend naming)
        sampler_map = {
            "Flux": "euler",  # Flux uses Euler
            "Flux Schnell": "euler",  # Schnell also uses Euler
            "Flux Dev": "euler",  # Dev also uses Euler
            "Lumina 2.0": "dpmpp_2m",  # Lumina uses DPM++ 2M
            "Unknown": "euler"  # Safe default
        }
        recommended_sampler = sampler_map.get(model_name, "euler")

        settings = {
            "prompt_theme": prompt_theme,
            "audio_theme": audio_theme,
            "prompts": generated_prompts,
            "duration": duration_seconds,
            "fps": fps,
            "total_frames": total_frames,
            "resolution": "1280x720",
            "render_mode": "New 3D",
            "steps": recommended_steps,  # Model-specific
            "cfg_scale": recommended_cfg,  # Model-specific
            "sampler": recommended_sampler,  # Model-specific
            "scheduler": recommended_scheduler,  # Model-specific
            "cadence": 5,
            "strength": 0.85,
            "keyframe_strength": 0.20,
            "seed": random_seed,
            "audio_path": audio_path,
            "output_dir": batch_dir,
            "depth_model": "Depth-Anything-V2-Small",
            "camera_movement": {
                # Slow zoom in (negative Z) with no rotation
                "translation_z": "0:(0), {}: (-1.0)".format(total_frames),
                "rotation_3d_y": "0:(0)"
            },
            "shakify": {
                "shake_name": "Investigation",  # Gentle handheld shake
                "shake_intensity": 1.0,
                "shake_speed": 1.0
            }
        }

        log.append(f"✓ Test settings configured")
        log.append(f"  - Render Mode: {settings['render_mode']}")
        log.append(f"  - Resolution: {settings['resolution']}")
        log.append(f"  - Model: {model_name}")
        log.append(f"  - Steps: {settings['steps']}, CFG: {settings['cfg_scale']:.1f}, Cadence: {settings['cadence']}")
        log.append(f"  - Sampler: {settings['sampler']}, Scheduler: {settings['scheduler']}")
        log.append(f"  - Strength: {settings['strength']} (normal), {settings['keyframe_strength']} (keyframe)")
        log.append("")

        # Phase 6: Save settings file as backup
        log.append(f"{gear or '💾'} Phase 6: Saving Quick Test settings...")
        log.append("")

        try:
            # Save settings to JSON file in batch directory
            settings_file = os.path.join(batch_dir, "quick_test_settings.json")

            # Build complete settings dict for export
            export_settings = {
                # Basic settings
                "W": 1280,
                "H": 720,
                "fps": fps,
                "steps": settings['steps'],
                "cfg_scale": settings['cfg_scale'],
                "sampler": settings['sampler'],
                "scheduler": settings['scheduler'],
                "seed": random_seed,
                "max_frames": total_frames,

                # Animation settings
                "render_mode": "New 3D",
                "animation_mode": "3D",
                "diffusion_cadence": settings['cadence'],
                "strength": settings['strength'],
                "keyframe_strength": settings['keyframe_strength'],
                "strength_schedule": f"0:({settings['strength']})",
                "keyframe_strength_schedule": f"0:({settings['keyframe_strength']})",

                # Prompts and audio
                "animation_prompts": generated_prompts,
                "animation_prompts_positive": "",
                "animation_prompts_negative": "",
                "add_soundtrack": "File",
                "soundtrack_path": audio_path,

                # Camera movement
                "translation_z": settings['camera_movement']['translation_z'],
                "rotation_3d_y": settings['camera_movement']['rotation_3d_y'],

                # Shakify settings
                "shake_name": settings['shakify']['shake_name'],
                "shake_intensity": settings['shakify']['shake_intensity'],
                "shake_speed": settings['shakify']['shake_speed'],

                # Depth settings
                "depth_algorithm": "Depth-Anything-V2-Small",
                "midas_weight": 0.3,
                "near_plane": 200,
                "far_plane": 10000,
                "fov": 70,

                # Output settings
                "skip_video_creation": False,
                "delete_imgs": False,
            }

            with open(settings_file, 'w') as f:
                json.dump(export_settings, f, indent=2)

            log.append(f"✓ Settings saved to: {settings_file}")
            log.append("")
            log.append("=" * 60)
            log.append(f"{check} QUICK TEST READY!")
            log.append("=" * 60)
            log.append("")
            log.append("Settings loaded automatically!")
            log.append("")
            log.append("Next step:")
            log.append("  → Go to Run tab and click 'Generate'!")
            log.append("")
            log.append(f"Test configuration: {total_frames} frames at {fps} FPS ({duration_seconds}s)")
            log.append(f"Audio file: {audio_path}")
            log.append(f"Settings file: {settings_file}")
            log.append(f"Output will be saved to: output/videos/")
            log.append("")

            # Load settings using the same function as "Load All Settings" button
            import gradio as gr
            from deforum.config.settings import load_all_settings, get_settings_component_names

            # Build args list: settings_path + all component values (will be overwritten)
            component_names = get_settings_component_names()
            dummy_args = [settings_file] + [''] * len(component_names)

            # Call load_all_settings - returns list of values when ui_launch=False
            loaded_values = load_all_settings(*dummy_args, ui_launch=False)

            # Return: (status, log, settings_json) + all loaded component values
            # Note: loaded_values is [settings_path, component1, component2, ..., ""]
            # We skip the first (settings_path) and last ("") elements
            return (
                gr.update(value=f"{check} Quick Test Ready! → Click Generate"),
                gr.update(value="\n".join(log)),
                json.dumps(export_settings, indent=2)
            ) + tuple(loaded_values[1:-1])  # Skip settings_path and empty string

        except Exception as save_error:
            error_trace = traceback.format_exc()
            log.append(f"❌ Failed to save settings: {str(save_error)}")
            log.append(error_trace)

            # Return error with empty settings components
            return (
                f"{cross} Settings save failed: {str(save_error)}",
                "\n".join(log),
                "{}"  # empty settings
            ) + tuple(_get_empty_settings_components())

    except Exception as e:
        error_trace = traceback.format_exc()
        log.append(f"❌ Error: {str(e)}")
        log.append(error_trace)

        # Return error with empty settings components
        return (
            f"{cross} Quick Test failed: {str(e)}",
            "\n".join(log),
            "{}"  # empty settings
        ) + tuple(_get_empty_settings_components())


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
