"""Zero-HITL Button Handler

Connects the "🔥 SLOP IT! 🔥" button to the orchestrator.

ISOLATION: Only affects Zero-HITL tab. Normal Deforum remains unchanged.
"""

import traceback
from typing import Tuple, List

from deforum.utils.zero_hitl import orchestrate_slop, OrchestrationResult
from deforum.utils.system.logging import get_logger

logger = get_logger()


def handle_slop_it_click(
    duration: float,
    theme: str,
    seed: int
) -> Tuple[str, str, str]:
    """Handle "🔥 SLOP IT! 🔥" button click.

    Args:
        duration: Duration in seconds (from slider)
        theme: Optional theme/vibe (from textbox)
        seed: Random seed (from number input)

    Returns:
        Tuple of (status, log, settings_json_state)
        - status: Single-line status message
        - log: Multi-line generation log
        - settings_json_state: JSON settings (for hidden state)
    """
    try:
        # Validate inputs
        if duration < 1.0 or duration > 10.0:
            return (
                "⚠️ Error: Duration must be between 1 and 10 seconds",
                "Invalid duration provided.",
                "{}"
            )

        # Log start
        logger.info("=" * 60)
        logger.info("🔥 SLOP IT! 🔥 Button clicked")
        logger.info(f"Duration: {duration}s, Theme: '{theme or 'PURE CHAOS'}', Seed: {seed}")
        logger.info("=" * 60)

        # Execute orchestration
        result = orchestrate_slop(
            duration_seconds=duration,
            theme=theme.strip() if theme else "",
            random_seed=int(seed) if seed != -1 else -1,
            output_dir="outputs/deforum"
        )

        # Format results for UI
        if result.success:
            status = f"✅ SLOP IT complete! Video: {result.video_path}"
            log = "\n".join(result.slop_log)
            settings = result.settings_json

            logger.info(f"✅ Success: {result.video_path}")
        else:
            status = f"❌ SLOP IT failed: {result.error_message}"
            log = "\n".join(result.slop_log)
            settings = "{}"

            logger.error(f"❌ Failure: {result.error_message}")

        return (status, log, settings)

    except Exception as e:
        error_msg = f"💥 Unexpected error: {str(e)}"
        error_trace = traceback.format_exc()

        logger.error(f"{error_msg}\n{error_trace}")

        return (
            f"❌ Fatal error: {str(e)}",
            f"💥 UNEXPECTED ERROR\n\n{error_trace}",
            "{}"
        )


def handle_view_settings_click(settings_json: str) -> str:
    """Handle "View Generated Settings" button click.

    Args:
        settings_json: JSON settings from hidden state

    Returns:
        Formatted settings string for display
    """
    if not settings_json or settings_json == "{}":
        return "⚠️ No settings available. Generate a video first!"

    import json
    try:
        settings = json.loads(settings_json)

        # Format for readable display
        lines = []
        lines.append("═══ GENERATED SETTINGS ═══")
        lines.append("")

        # Parameters
        lines.append("PARAMETERS:")
        params = settings.get("parameters", {})
        lines.append(f"  Render Mode: {params.get('render_mode')}")
        lines.append(f"  FPS: {params.get('fps')}")
        lines.append(f"  Resolution: {params.get('resolution')}")
        lines.append(f"  Steps: {params.get('steps')}")
        lines.append(f"  CFG Scale: {params.get('cfg_scale')}")
        lines.append(f"  Sampler: {params.get('sampler')}")
        lines.append(f"  Cadence: {params.get('cadence')}")
        lines.append(f"  Strength: {params.get('strength')}")
        lines.append(f"  Preset: {params.get('preset_type')}")
        lines.append(f"  Style: {params.get('style')}")
        lines.append(f"  Seed: {params.get('seed')}")
        lines.append("")

        # Prompts
        prompts = settings.get("prompts", [])
        lines.append(f"PROMPTS ({len(prompts)} keyframes):")
        for prompt in prompts[:5]:  # Show first 5
            lines.append(f"  Frame {prompt['frame']}: {prompt['prompt'][:80]}...")
        if len(prompts) > 5:
            lines.append(f"  ... and {len(prompts) - 5} more")
        lines.append("")

        # Camera chaos
        camera_chaos = settings.get("camera_chaos", {})
        if camera_chaos:
            lines.append("CAMERA CHAOS:")
            for key, value in camera_chaos.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Audio
        audio_path = settings.get("audio_path")
        lines.append(f"AUDIO: {audio_path}")

        return "\n".join(lines)

    except Exception as e:
        return f"⚠️ Failed to parse settings: {e}"


def handle_open_output_click() -> str:
    """Handle "Open Output Folder" button click.

    Returns:
        Status message
    """
    import os
    import subprocess
    import platform

    output_dir = "outputs/zero_hitl"

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

        return f"✅ Opened folder: {output_dir}"

    except Exception as e:
        return f"⚠️ Failed to open folder: {e}\nPath: {os.path.abspath(output_dir)}"
