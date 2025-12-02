"""AI-Generated Mode-Specific Defaults Generator.

This module generates complete default settings for each render mode by:
1. Loading static settings from JSON files
2. Generating audio loop with Meta MusicGen
3. Detecting audio events with BPM-aware sensitivity
4. Generating prompts with Qwen
5. Syncing prompts to audio events

This enables "smart defaults" that are AI-generated but deterministic per mode.
"""

import json
import os
from pathlib import Path
from typing import Optional, Callable, Dict

from deforum.utils.system.logging import get_logger

logger = get_logger()

# Audio generation config for bunny defaults
BUNNY_DEFAULT_AUDIO_CONFIG = {
    "prompt": "synthetic amen break",  # Simple prompt, let Meta interpret
    "target_duration_seconds": 5.0,    # Aim for ~5 seconds
    "bpm": 173,                        # Amen break BPM
    "enable_chaos": False,             # Clean audio for defaults (no glitches)
}

# Prompt generation config for bunny defaults
BUNNY_DEFAULT_PROMPT_CONFIG = {
    "generation_mode": "escalating",
    "intensity": "crazy",
    "theme": "bunny",
    "style": "",  # No style prefix
}


def detect_model_type(model_name: str) -> str:
    """Detect model type from model filename.

    Args:
        model_name: SD model filename (e.g., "Flux\\flux1-dev-bnb-nf4-v2.safetensors")

    Returns:
        Model type: "flux_dev", "flux_schnell", "lumina", or "sd15" (fallback)
    """
    model_lower = model_name.lower()

    if "schnell" in model_lower:
        return "flux_schnell"
    elif "flux" in model_lower:
        return "flux_dev"
    elif "lumina" in model_lower or "neta" in model_lower:
        return "lumina"
    else:
        return "sd15"  # Fallback for SD1.5/SDXL


def load_static_defaults(render_mode: str, model_type: str) -> Dict:
    """Load static default settings from JSON file.

    Args:
        render_mode: Render mode ("New 3D", "Classic 3D", "Keyframes Only", "Flux + Interpolation")
        model_type: Model type ("flux_dev", "flux_schnell", "lumina")

    Returns:
        Dictionary of default settings

    Raises:
        FileNotFoundError: If defaults file doesn't exist for this mode/model combo
    """
    # Map render mode names to directory names
    mode_dir_map = {
        "New 3D": "new_3d",
        "Classic 3D": "classic_3d",
        "Keyframes Only": "keyframes_only",
        "Flux + Interpolation": "flux_interpolation",
    }

    mode_dir = mode_dir_map.get(render_mode)
    if not mode_dir:
        raise ValueError(f"Unknown render mode: {render_mode}")

    # Construct path to defaults file
    # Try model-specific first, fall back to flux_dev if not found
    defaults_root = Path(__file__).parent / "defaults"
    defaults_path = defaults_root / mode_dir / f"{model_type}.json"

    if not defaults_path.exists():
        # Fallback to flux_dev
        logger.warning(
            f"No defaults found for {render_mode}/{model_type}, "
            f"falling back to flux_dev"
        )
        defaults_path = defaults_root / mode_dir / "flux_dev.json"

    if not defaults_path.exists():
        raise FileNotFoundError(
            f"No defaults file found: {defaults_path}\n"
            f"Available modes: {list(mode_dir_map.keys())}\n"
            f"Available models: flux_dev, flux_schnell, lumina"
        )

    # Load JSON
    with open(defaults_path, 'r') as f:
        defaults = json.load(f)

    logger.info(f"Loaded defaults: {defaults_path}")
    return defaults


def generate_mode_defaults(
    render_mode: str,
    current_model: str,
    batch_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict:
    """Generate AI-powered defaults for specific mode and model.

    This is the main entry point for generating complete default settings.
    It orchestrates the entire pipeline from static JSON → audio → prompts.

    Args:
        render_mode: Render mode ("New 3D", "Classic 3D", etc.)
        current_model: Current SD model name (to detect type)
        batch_dir: Optional batch directory to save audio file (instead of outputs/audio/defaults)
        progress_callback: Optional callback for progress updates

    Returns:
        Complete settings dictionary ready to populate UI

    Example:
        >>> defaults = generate_mode_defaults(
        ...     render_mode="New 3D",
        ...     current_model="Flux\\flux1-dev-bnb-nf4-v2.safetensors",
        ...     batch_dir=Path("output/Deforum_Defaults_New3D_20231105_123456"),
        ...     progress_callback=lambda msg: print(msg)
        ... )
        >>> # defaults now contains all settings including AI-generated audio + prompts
    """
    def progress(msg: str):
        """Helper to call progress callback if provided."""
        if progress_callback:
            progress_callback(msg)
        logger.info(msg)

    # 1. LOAD BASE DEFAULTS
    progress("Loading mode defaults...")

    model_type = detect_model_type(current_model)
    logger.info(f"Detected model type: {model_type}")

    defaults = load_static_defaults(render_mode, model_type)

    # 2. GENERATE AUDIO
    progress("Generating audio loop (Meta MusicGen)...")

    try:
        from deforum.utils.audio_generation import generate_loop

        # Determine audio output directory
        if batch_dir is not None:
            # Use batch directory (preferred - keeps everything together)
            audio_dir = Path(batch_dir)
        else:
            # Fallback to output/audio/defaults
            audio_dir = Path("output/audio/defaults")
            audio_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename based on mode
        audio_filename = f"bunny_default_{render_mode.replace(' ', '_').lower()}.mp3"
        audio_path = str(audio_dir / audio_filename)

        # Generate audio
        actual_audio_path, slop_log = generate_loop(
            prompt=BUNNY_DEFAULT_AUDIO_CONFIG["prompt"],
            duration_seconds=BUNNY_DEFAULT_AUDIO_CONFIG["target_duration_seconds"],
            bpm=BUNNY_DEFAULT_AUDIO_CONFIG["bpm"],
            output_path=audio_path,
            enable_chaos=BUNNY_DEFAULT_AUDIO_CONFIG["enable_chaos"]
        )

        defaults["soundtrack_path"] = actual_audio_path
        logger.info(f"Generated audio: {actual_audio_path}")

        # Log chaos decisions (should be none with chaos disabled)
        for log_entry in slop_log:
            if log_entry.triggered:
                logger.debug(f"  {log_entry.decision}")

    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        progress(f"⚠️ Audio generation failed, using fallback")
        # Don't fail entirely - just skip audio generation
        defaults["soundtrack_path"] = ""
        return defaults  # Return early without prompts

    # 3. DETECT AUDIO EVENTS (BPM-aware)
    progress("Analyzing audio events...")

    try:
        import librosa
        from deforum.audio import detect_events_bpm_aware

        # Load audio
        y, sr = librosa.load(actual_audio_path, sr=None)
        actual_duration = librosa.get_duration(y=y, sr=sr)

        # Update max_frames based on actual audio duration
        fps = defaults["fps"]
        defaults["max_frames"] = int(actual_duration * fps)
        logger.info(f"Actual audio duration: {actual_duration:.2f}s → {defaults['max_frames']} frames @ {fps} FPS")

        # Detect events with BPM-aware sensitivity
        event_times, event_intensities, detected_bpm = detect_events_bpm_aware(
            audio=y,
            sample_rate=sr,
            method="onset",  # Onset detection for transients
            target_bpm=None,  # Auto-detect
            tolerance=0.15,   # ±15% acceptable
            prefer_under_detection=True  # Prefer missing weak events over false positives
        )

        logger.info(f"Detected {len(event_times)} events at {detected_bpm:.1f} BPM")

    except Exception as e:
        logger.error(f"Event detection failed: {e}")
        progress(f"⚠️ Event detection failed")
        return defaults  # Return without prompts

    # 4. GENERATE PROMPTS
    progress(f"Generating {len(event_times)} prompts (Qwen)...")

    try:
        from deforum.integrations.wan.utils.prompt_extend import QwenPromptExpander

        # Initialize Qwen (auto-selects model based on VRAM)
        qwen = QwenPromptExpander()

        # Build prompt for Qwen based on mode
        generation_prompt = f"""Generate {len(event_times)} prompts that build in intensity for an animated sequence featuring {BUNNY_DEFAULT_PROMPT_CONFIG['theme']}.

INTENSITY: Make prompts over-the-top and extremely creative! Wild transformations and escalating intensity! Go big with each step!

Requirements:
- START CALM: Begin with simple, peaceful scene (e.g., "cute bunny in nature")
- ESCALATE DRAMATICALLY: Each prompt MORE intense than the last
- Progressive transformation: calm → active → dynamic → EXTREME → ABSOLUTELY WILD
- Final prompts should be PEAK INSANITY
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Example progression:
cute bunny sitting peacefully in grass
bunny hopping through vibrant neon forest
bunny leaping over glowing obstacles
bunny racing through laser-filled cityscape
EXTREME bunny surfing massive energy wave
INSANE bunny commanding lightning storm on motorcycle
ABSOLUTELY BONKERS bunny transcending reality in cosmic explosion

Generate {len(event_times)} prompts for bunny:"""

        system_prompt = "You are a creative AI assistant helping generate prompts for animated sequences. Return ONLY the prompts, one per line, with no numbering or extra formatting."

        result = qwen(prompt=generation_prompt, system_prompt=system_prompt, tar_lang="en")

        # Check if generation succeeded
        if not result.status:
            raise Exception(f"Qwen generation failed: {result.message}")

        result_text = result.prompt

        # Check if result is just echoing input (silent failure)
        if result_text == generation_prompt:
            raise Exception(f"Qwen returned input unchanged: {result.message}")

        # Parse prompts from result
        lines = [line.strip() for line in result_text.split('\n') if line.strip()]
        prompts = []
        for line in lines:
            # Remove numbering if present
            clean_line = line
            if len(line) > 0 and line[0].isdigit():
                import re
                clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
            if clean_line and not clean_line.startswith('#') and not clean_line.startswith('//'):
                prompts.append(clean_line)

        # Take only requested count
        prompts = prompts[:len(event_times)]

        logger.info(f"Generated {len(prompts)} prompts with Qwen")

    except Exception as e:
        logger.error(f"Prompt generation failed: {e}")
        import traceback
        traceback.print_exc()
        progress(f"⚠️ Prompt generation failed, using fallback")

        # Fallback: Simple escalating prompts
        prompts = [
            f"bunny {action}"
            for action in ["resting peacefully", "hopping gently", "moving actively",
                          "leaping dynamically", "racing wildly", "GOING ABSOLUTELY BONKERS"]
        ][:len(event_times)]

    # 5. SYNC PROMPTS TO EVENTS
    progress("Syncing prompts to audio events...")

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
        prompt_schedule = json.loads(prompt_schedule_json)
        defaults["prompts"] = prompt_schedule

        logger.info(f"Synced {len(prompts)} prompts to {len(keyframes)} keyframes")

    except Exception as e:
        logger.error(f"Prompt sync failed: {e}")
        import traceback
        traceback.print_exc()
        progress(f"⚠️ Prompt sync failed")
        # Create simple sequential schedule as fallback
        defaults["prompts"] = {str(i * 15): prompt for i, prompt in enumerate(prompts)}

    # 6. DONE
    progress("✓ Defaults generated successfully!")
    logger.info(f"Generated defaults for {render_mode} ({model_type})")
    logger.info(f"  Audio: {defaults['soundtrack_path']}")
    logger.info(f"  Prompts: {len(defaults.get('prompts', {}))} keyframes")
    logger.info(f"  FPS: {defaults['fps']}, Frames: {defaults['max_frames']}")

    return defaults
