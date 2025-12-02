"""Helper functions for Wan video generation.

Extracted from ui_elements.py to reduce complexity and improve maintainability.
"""

from typing import Dict, List, Tuple, Optional
import os


# Pure helper functions (complexity ≤ 3 each)

def _load_emoji_symbols():
    """Load all emoji symbols for Wan generation logging.

    Returns:
        Dict of emoji symbols for use in logging
    """
    from deforum.utils.system.logging import emoji as emoji_utils
    return {
        'check': emoji_utils.maybe_check(),
        'cross': emoji_utils.maybe_cross(),
        'warning': emoji_utils.maybe_warning(),
        'magnifying_glass': emoji_utils.magnifying_glass(),
        'download': emoji_utils.download(),
        'folder': emoji_utils.folder(),
        'bulb': emoji_utils.bulb(),
        'target': emoji_utils.target(),
        'memo': emoji_utils.memo(),
        'rocket': emoji_utils.rocket(),
        'palette': emoji_utils.palette(),
        'package': emoji_utils.package(),
        'ruler': emoji_utils.ruler(),
        'party': emoji_utils.party(),
    }


def _cleanup_qwen_models():
    """Ensure Qwen models are unloaded before video generation to free VRAM."""
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    try:
        from deforum.integrations.wan.utils.qwen_manager import qwen_manager
        if qwen_manager.is_model_loaded():
            logger.info("Unloading Qwen models before video generation...", emoji='refresh')
            qwen_manager.ensure_model_unloaded()
    except Exception as e:
        logger.error(f"Could not cleanup Qwen models: {e}")


def _discover_and_validate_models(integration, emojis):
    """Discover and validate Wan models are available.

    Args:
        integration: WanSimpleIntegration instance
        emojis: Dict of emoji symbols

    Returns:
        List of discovered models

    Raises:
        RuntimeError: If no models found
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    logger.info(f"{emojis['magnifying_glass']} Auto-discovering Wan models...")
    models = integration.discover_models()

    if not models:
        raise RuntimeError(f"""
{emojis['cross']} No Wan models found automatically!

{emojis['bulb']} SOLUTIONS:
1. {emojis['download']} Download a Wan model using HuggingFace CLI:
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir "models/Deforum/wan"

2. {emojis['folder']} Or place your model in one of these locations:
   • models/Deforum/wan/
   • models/Wan/

3. {emojis['check']} Restart generation after downloading

The auto-discovery will find your models automatically!
""")

    return models


def _select_model_auto_detect(integration, emojis):
    """Auto-detect best model using priority logic.

    Args:
        integration: WanSimpleIntegration instance
        emojis: Dict of emoji symbols

    Returns:
        Selected model dict or None
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    selected_model = integration.get_best_model()
    if selected_model:
        logger.info(f"{emojis['target']} Auto-detected best model: {selected_model['name']} ({selected_model['type']}, {selected_model['size']})")
    return selected_model


def _select_model_by_size(models, size_to_match: str, emojis):
    """Select model by specific size requirement.

    Args:
        models: List of available models
        size_to_match: Size identifier (5B, A14B, etc.)
        emojis: Dict of emoji symbols

    Returns:
        Selected model dict or None
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    for model in models:
        if size_to_match == model['size']:
            logger.info(f"{emojis['check']} Using user-selected model: {model['name']} ({model['size']})")
            return model
    return None


def select_wan_model(integration, wan_args, emojis):
    """Select appropriate Wan model based on user choice.

    Args:
        integration: WanSimpleIntegration instance
        wan_args: Wan arguments namespace
        emojis: Dict of emoji symbols

    Returns:
        Selected model dict

    Raises:
        RuntimeError: If no suitable model found
        NotImplementedError: If custom path requested (not yet supported)
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    user_model_choice = wan_args.wan_t2v_model.replace(" (Recommended)", "")

    # Auto-detect mode
    if user_model_choice == "Auto-Detect":
        selected_model = _select_model_auto_detect(integration, emojis)
        if not selected_model:
            raise RuntimeError("No Wan models available! Please download a model first.")
        return selected_model

    # Custom path mode
    if user_model_choice == "Custom Path":
        custom_path = wan_args.wan_model_path
        logger.info(f"{emojis['folder']} Using custom model path: {custom_path}")
        raise NotImplementedError("Custom path loading not yet implemented. Please use Auto-Detect or specific model selection.")

    # Specific size selection
    models = integration.discover_models()
    size_to_match = _parse_size_from_choice(user_model_choice)

    if size_to_match:
        selected_model = _select_model_by_size(models, size_to_match, emojis)
        if selected_model:
            return selected_model

        # Fallback to auto-detect
        logger.warning(f"{emojis['warning']} Requested {size_to_match} model not found, falling back to auto-detect")

    # Final fallback
    selected_model = integration.get_best_model()
    if not selected_model:
        raise RuntimeError("No Wan models available! Please download a model first.")

    return selected_model


def _parse_size_from_choice(user_model_choice: str) -> Optional[str]:
    """Parse model size from user choice string.

    Args:
        user_model_choice: User's model selection string

    Returns:
        Size identifier (5B, A14B) or None
    """
    if "5B" in user_model_choice:
        return "5B"
    if "A14B" in user_model_choice or "14B" in user_model_choice:
        return "A14B"
    return None


def _determine_output_from_args(args, root, emojis):
    """Try to determine output directory from args.outdir.

    Args:
        args: Arguments namespace
        root: Root object with timestring
        emojis: Dict of emoji symbols

    Returns:
        Output directory path or None if not suitable
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    if not hasattr(args, 'outdir') or not args.outdir:
        return None

    # Validate that outdir has a timestring or unique identifier
    basename = os.path.basename(args.outdir)
    has_identifier = 'Deforum_' in args.outdir or any(char.isdigit() for char in basename)

    if has_identifier:
        logger.info(f"{emojis['check']} Using args.outdir (contains identifier): {args.outdir}")
        return args.outdir

    logger.warning(f"{emojis['warning']} args.outdir lacks unique identifier: {args.outdir}")
    logger.warning(f"{emojis['warning']} Will reconstruct with batch name to avoid collisions")
    return None


def _get_batch_name(args, root, emojis):
    """Get batch name with multiple fallbacks.

    Args:
        args: Arguments namespace
        root: Root object with raw_batch_name and timestring
        emojis: Dict of emoji symbols

    Returns:
        Batch name string
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    # Try args.batch_name first
    if hasattr(args, 'batch_name') and args.batch_name:
        logger.info(f"{emojis['memo']} Using args.batch_name: {args.batch_name}")
        return args.batch_name

    # Try root.raw_batch_name
    if hasattr(root, 'raw_batch_name') and root.raw_batch_name:
        logger.info(f"{emojis['memo']} Using root.raw_batch_name: {root.raw_batch_name}")
        return root.raw_batch_name

    # Default fallback
    logger.warning(f"{emojis['warning']} No batch_name found, using default: Deforum_{{timestring}}")
    return 'Deforum_{timestring}'


def _substitute_timestring_in_batch(batch_name: str, timestring: str, emojis) -> str:
    """Substitute {timestring} placeholder in batch name.

    Args:
        batch_name: Batch name with potential placeholder
        timestring: Actual timestring value
        emojis: Dict of emoji symbols

    Returns:
        Batch name with substituted timestring
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    if '{timestring}' not in batch_name and batch_name != 'Deforum':
        return batch_name

    substituted = batch_name.replace('{timestring}', timestring)
    logger.info(f"Substituted timestring: {substituted}", emoji='refresh')
    return substituted


def _ensure_batch_has_unique_id(batch_name: str, timestring: str, emojis) -> str:
    """Ensure batch name has unique identifier.

    Args:
        batch_name: Current batch name
        timestring: Timestring to use as unique ID if needed
        emojis: Dict of emoji symbols

    Returns:
        Batch name with guaranteed unique identifier
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    if any(char.isdigit() for char in batch_name):
        return batch_name

    unique_name = f"{batch_name}_{timestring}"
    logger.warning(f"{emojis['warning']} Added timestring for uniqueness: {unique_name}")
    return unique_name


def _construct_output_directory(args, root, emojis):
    """Construct output directory from batch name.

    Args:
        args: Arguments namespace
        root: Root object with timestring
        emojis: Dict of emoji symbols

    Returns:
        Constructed output directory path
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    deforum_outpath = os.path.join(os.getcwd(), 'output', 'deforum')

    # Get batch name with fallbacks
    batch_name = _get_batch_name(args, root, emojis)

    # Substitute placeholders
    batch_name = _substitute_timestring_in_batch(batch_name, root.timestring, emojis)

    # Ensure uniqueness
    batch_name = _ensure_batch_has_unique_id(batch_name, root.timestring, emojis)

    output_directory = os.path.join(deforum_outpath, batch_name)
    logger.info(f"{emojis['check']} Constructed output directory: {output_directory}")

    return output_directory


def _validate_output_directory(output_directory: str, emojis):
    """Validate output directory has unique identifier.

    Args:
        output_directory: Directory path to validate
        emojis: Dict of emoji symbols

    Raises:
        RuntimeError: If directory lacks unique identifier
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    dir_name = os.path.basename(output_directory)
    if dir_name == 'Deforum' or dir_name == 'deforum':
        logger.info("="*80)
        logger.error(f"CRITICAL ERROR: Output directory has no unique identifier!", emoji='off')
        logger.info(f"Directory: {output_directory}", emoji='off')
        logger.info(f"This will cause files from different generations to mix!", emoji='off')
        logger.info("="*80)
        raise RuntimeError(f"Invalid output directory (no unique ID): {output_directory}")


def setup_wan_output_directory(args, root, emojis):
    """Determine and setup output directory for Wan generation.

    Args:
        args: Arguments namespace
        root: Root object with timestring and batch info
        emojis: Dict of emoji symbols

    Returns:
        Validated output directory path

    Raises:
        RuntimeError: If directory setup fails validation
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    logger.info("="*80)
    logger.debug("Output Directory Setup")
    logger.info("="*80)

    # Log all relevant attributes for debugging
    logger.info(f"{emojis['memo']} args.outdir exists: {hasattr(args, 'outdir')}")
    if hasattr(args, 'outdir'):
        logger.info(f"{emojis['memo']} args.outdir value: {args.outdir}")
    logger.info(f"{emojis['memo']} args.batch_name exists: {hasattr(args, 'batch_name')}")
    if hasattr(args, 'batch_name'):
        logger.info(f"{emojis['memo']} args.batch_name value: {args.batch_name}")
    logger.info(f"{emojis['memo']} root.timestring: {root.timestring}")
    logger.info(f"{emojis['memo']} root.raw_batch_name exists: {hasattr(root, 'raw_batch_name')}")
    if hasattr(root, 'raw_batch_name'):
        logger.info(f"{emojis['memo']} root.raw_batch_name: {root.raw_batch_name}")
    logger.info("-"*80)

    # Try args.outdir first
    output_directory = _determine_output_from_args(args, root, emojis)

    # Fallback to construction
    if not output_directory:
        output_directory = _construct_output_directory(args, root, emojis)

    # Ensure directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Validate
    _validate_output_directory(output_directory, emojis)

    # Final confirmation
    dir_name = os.path.basename(output_directory)
    logger.info("="*80)
    logger.info(f"{emojis['check']} Final output directory: {output_directory}")
    logger.info(f"{emojis['check']} Directory name: {dir_name}")
    logger.info("="*80)

    return output_directory


def parse_prompts_and_timing(animation_prompts, wan_args, video_args, emojis):
    """Calculate exact frame counts from prompt schedule for audio sync precision.

    Args:
        animation_prompts: Dict mapping frame numbers to prompt text
        wan_args: Wan arguments namespace
        video_args: Video arguments namespace
        emojis: Dict of emoji symbols

    Returns:
        List of (prompt, start_frame, frame_count) tuples
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    prompt_schedule = []

    # Sort prompts by frame number
    sorted_prompts = sorted(animation_prompts.items(), key=lambda x: int(x[0]))

    if not sorted_prompts:
        return [("a beautiful landscape", 0, 81)]  # Default: 0 start frame, 81 frames

    # Check if enhanced prompts are available and use them
    final_prompts = _get_final_prompts(animation_prompts, wan_args, emojis)

    # Add movement description if available
    movement_description = _get_movement_description(wan_args, emojis)

    # Re-sort with final prompts
    sorted_prompts = sorted(final_prompts.items(), key=lambda x: int(x[0]))

    # Calculate frame differences between prompts
    for i, (frame_str, prompt) in enumerate(sorted_prompts):
        start_frame = int(frame_str)
        clean_prompt = prompt.split('--neg')[0].strip()

        # Append movement description if available
        if movement_description:
            clean_prompt = f"{clean_prompt}. {movement_description}"

        # Calculate frame count for this clip
        frame_count = _calculate_clip_frame_count(i, sorted_prompts, video_args.fps)

        # Pad to Wan's 4n+1 requirement
        frame_count = _pad_to_wan_requirement(frame_count)

        # Add to schedule
        prompt_schedule.append((clean_prompt, start_frame, frame_count))

        # Log prompt info
        _log_prompt_info(i, clean_prompt, frame_count, wan_args, emojis)

    return prompt_schedule


def _get_final_prompts(animation_prompts, wan_args, emojis):
    """Get final prompts, using enhanced prompts if available.

    Args:
        animation_prompts: Original prompts dict
        wan_args: Wan arguments namespace
        emojis: Dict of emoji symbols

    Returns:
        Final prompts dict (either enhanced or original)
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    if not wan_args.wan_enhanced_prompts:
        return animation_prompts.copy()

    try:
        import json
        enhanced_prompts_data = json.loads(wan_args.wan_enhanced_prompts)
        if enhanced_prompts_data:
            logger.info("Using enhanced prompts from QwenPromptExpander", emoji='palette')
            return enhanced_prompts_data
    except (json.JSONDecodeError, ValueError):
        logger.error(f"{emojis['warning']} Could not parse enhanced prompts, using original prompts")

    return animation_prompts.copy()


def _get_movement_description(wan_args, emojis):
    """Get movement description from wan_args.

    Args:
        wan_args: Wan arguments namespace
        emojis: Dict of emoji symbols

    Returns:
        Movement description string or empty string
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    if not wan_args.wan_movement_description:
        return ""

    movement_description = wan_args.wan_movement_description.split('\n')[0]  # Get first line
    logger.info(f"{emojis['ruler']} Adding movement description: {movement_description}")
    return movement_description


def _calculate_clip_frame_count(clip_index: int, sorted_prompts: list, fps: int) -> int:
    """Calculate frame count for a clip.

    Args:
        clip_index: Index in sorted_prompts list
        sorted_prompts: Sorted list of (frame_num, prompt) tuples
        fps: Target FPS

    Returns:
        Frame count for this clip
    """
    current_frame = int(sorted_prompts[clip_index][0])

    # If not last prompt, calculate difference to next
    if clip_index < len(sorted_prompts) - 1:
        next_frame = int(sorted_prompts[clip_index + 1][0])
        return next_frame - current_frame

    # Last prompt - use default (minimum 2 seconds or 81 frames)
    return max(2 * fps, 81)


def _pad_to_wan_requirement(frame_count: int) -> int:
    """Pad frame count to Wan's 4n+1 requirement.

    Args:
        frame_count: Current frame count

    Returns:
        Padded frame count meeting 4n+1 requirement
    """
    # Ensure minimum frame count for Wan (at least 5 frames)
    frame_count = max(5, frame_count)

    # Check if already meets 4n+1
    if (frame_count - 1) % 4 == 0:
        return frame_count

    # Calculate closest 4n+1 value
    target_4n_plus_1 = ((frame_count - 1) // 4) * 4 + 1
    next_4n_plus_1 = target_4n_plus_1 + 4

    # Choose the closest one
    if abs(frame_count - target_4n_plus_1) <= abs(frame_count - next_4n_plus_1):
        return target_4n_plus_1
    return next_4n_plus_1


def _log_prompt_info(clip_index: int, clean_prompt: str, frame_count: int, wan_args, emojis):
    """Log information about the prompt clip.

    Args:
        clip_index: Clip index (0-based)
        clean_prompt: Cleaned prompt text
        frame_count: Number of frames
        wan_args: Wan arguments namespace
        emojis: Dict of emoji symbols
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    clip_num = clip_index + 1

    if wan_args.wan_enhanced_prompts or wan_args.wan_movement_description:
        logger.info(f"  {emojis['palette']} Enhanced Clip {clip_num}: '{clean_prompt[:80]}...' (frames: {frame_count})", emoji='palette')
    else:
        start_frame = int(list(wan_args.keys())[clip_index]) if hasattr(wan_args, 'keys') else 0
        logger.info(f"  Clip {clip_num}: '{clean_prompt[:50]}...' (start: frame {start_frame}, frames: {frame_count})")


def calculate_dynamic_motion_strength(anim_args, wan_args, emojis):
    """Calculate dynamic motion strength and intensity schedule.

    Args:
        anim_args: Animation arguments namespace
        wan_args: Wan arguments namespace
        emojis: Dict of emoji symbols

    Returns:
        Tuple of (motion_strength, motion_intensity_schedule)
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    default_strength = wan_args.wan_motion_strength

    # If no movement description or manual override, return defaults
    if not wan_args.wan_movement_description or wan_args.wan_motion_strength_override:
        if wan_args.wan_motion_strength_override:
            logger.info(f"Using manual motion strength override: {default_strength}", emoji='wrench')
        else:
            logger.info(f"Using default motion strength: {default_strength}", emoji='distribution')
        return default_strength, None

    # Try to calculate dynamic strength
    try:
        from deforum.integrations.wan.utils.movement_analyzer import analyze_deforum_movement, generate_wan_motion_intensity_schedule

        logger.info("Calculating dynamic motion strength from movement schedules...", emoji='movie_camera')

        # Generate both description and average strength
        _, dynamic_motion_strength = analyze_deforum_movement(
            anim_args=anim_args,
            sensitivity=wan_args.wan_movement_sensitivity,
            max_frames=min(anim_args.max_frames, 100)
        )

        # Generate frame-by-frame motion intensity schedule
        motion_intensity_schedule = generate_wan_motion_intensity_schedule(
            anim_args=anim_args,
            max_frames=min(anim_args.max_frames, 100),
            sensitivity=wan_args.wan_movement_sensitivity
        )

        logger.info(f"{emojis['check']} Dynamic motion strength: {dynamic_motion_strength:.2f} (average)")
        logger.info(f"{emojis['ruler']} Generated motion intensity schedule with frame-by-frame control")

        return dynamic_motion_strength, motion_intensity_schedule

    except Exception as e:
        logger.error(f"{emojis['warning']} Dynamic motion strength calculation failed: {e}, using default: {default_strength}")
        return default_strength, None


def _log_5b_720p_match(model_name: str, width: int, height: int, emojis):
    """Log perfect 5B + 720p match."""
    from deforum.utils.system.logging import get_logger
    logger = get_logger()
    logger.info(f"\n{emojis['check']} Perfect Match: TI2V-5B + 720p")
    logger.info(f"   {emojis['package']} Model: {model_name} (optimized for 720p@24fps)")
    logger.info(f"   {emojis['ruler']} Resolution: {width}x{height} (720p)")
    logger.info(f"   {emojis['target']} Optimal configuration for TI2V-5B!")


def _log_5b_480p_match(model_name: str, width: int, height: int, emojis):
    """Log 5B + 480p suboptimal match."""
    from deforum.utils.system.logging import get_logger
    logger = get_logger()
    logger.info(f"\n{emojis['bulb']} INFO: TI2V-5B + 480p Resolution", emoji='bulb')
    logger.info(f"   {emojis['package']} Model: {model_name} (optimized for 720p)")
    logger.info(f"   {emojis['ruler']} Resolution: {width}x{height} (480p)")
    logger.info(f"   {emojis['check']} This works, but you could use 1280x720 for better quality")


def _log_a14b_720p_match(model_name: str, width: int, height: int, emojis):
    """Log perfect A14B + 720p match."""
    from deforum.utils.system.logging import get_logger
    logger = get_logger()
    logger.info(f"\n{emojis['check']} Perfect Match: TI2V-A14B + 720p")
    logger.info(f"   {emojis['package']} Model: {model_name} (MoE architecture, highest quality)")
    logger.info(f"   {emojis['ruler']} Resolution: {width}x{height} (720p)")
    logger.info(f"   {emojis['target']} Maximum quality configuration!")


def _get_resolution_type(width: int, height: int) -> str:
    """Determine resolution type (720p, 480p, or Custom)."""
    is_720p = (width >= 1280 and height >= 720) or (width >= 720 and height >= 1280)
    is_480p = (width <= 864 and height <= 480) or (width <= 480 and height >= 864)
    return '720p' if is_720p else '480p' if is_480p else 'Custom'


def _log_model_resolution_feedback(model_size: str, model_name: str, width: int, height: int, res_type: str, emojis):
    """Log model/resolution match feedback based on combination."""
    # Map (model_size_key, res_type) → logging function
    match_handlers = {
        ('5B', '720p'): lambda: _log_5b_720p_match(model_name, width, height, emojis),
        ('5B', '480p'): lambda: _log_5b_480p_match(model_name, width, height, emojis),
        ('A14B', '720p'): lambda: _log_a14b_720p_match(model_name, width, height, emojis),
    }

    # Determine model size key
    model_key = '5B' if '5B' in model_size else 'A14B' if 'A14B' in model_size else None

    # Call appropriate handler if match exists
    handler = match_handlers.get((model_key, res_type))
    if handler:
        handler()


def validate_model_resolution_match(selected_model, width: int, height: int, emojis):
    """Validate and log model/resolution match quality.

    Args:
        selected_model: Selected model info dict
        width: Target width in pixels
        height: Target height in pixels
        emojis: Dict of emoji symbols
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    model_size = selected_model['size']
    model_name = selected_model['name']
    res_type = _get_resolution_type(width, height)

    logger.info(f"\n{emojis['magnifying_glass']} Model/Resolution Validation:")
    logger.info(f"   {emojis['package']} Model: {model_name} ({model_size})")
    logger.info(f"   {emojis['ruler']} Resolution: {width}x{height} ({res_type})")

    # Log model/resolution specific feedback
    _log_model_resolution_feedback(model_size, model_name, width, height, res_type, emojis)
