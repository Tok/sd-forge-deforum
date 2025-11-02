"""Mode-aware subtitle generation.

Generates clean, relevant subtitles based on render mode and active schedules.
Filters out irrelevant parameters to reduce clutter and file size.
"""

from typing import List
from deforum.rendering.data import RenderData


def get_relevant_params_for_mode(data: RenderData) -> List[str]:
    """Get list of relevant subtitle parameters based on render mode and schedules.

    Returns only parameters that are:
    1. Relevant to the current animation mode
    2. Actually being scheduled (not static)
    3. Useful for debugging/understanding the animation

    Args:
        data: Render data containing mode and schedule information

    Returns:
        List of parameter names to include in subtitles (user-friendly names)

    Examples:
        >>> # 3D mode with basic schedules
        >>> params = get_relevant_params_for_mode(data_3d)
        >>> params
        ['Trans X', 'Trans Y', 'Trans Z', 'Rot 3D X', 'Rot 3D Y', 'Rot 3D Z', 'Prompt']

        >>> # Interpolation mode (no transforms)
        >>> params = get_relevant_params_for_mode(data_interp)
        >>> params
        ['Prompt']
    """
    anim_args = data.args.anim_args
    mode = anim_args.animation_mode

    # Start with always-relevant params
    params = []

    # Mode-specific transform parameters
    if mode not in ['Video Input', 'Interpolation']:
        # 2D specific
        if mode == '2D':
            params.extend(['Angle', 'Zoom', 'Trans Center X', 'Trans Center Y'])

        # Common X/Y translation
        params.extend(['Trans X', 'Trans Y'])

        # 3D specific (removed Aspect Ratio - rarely scheduled)
        if mode == '3D':
            params.extend(['Trans Z', 'Rot 3D X', 'Rot 3D Y', 'Rot 3D Z'])

        # Perspective flip (only if enabled)
        if anim_args.enable_perspective_flip:
            params.extend(['Per Fl Theta', 'Per Fl Phi', 'Per Fl Gamma', 'Per Fl FV'])

    # Strength schedules (only if not default/static)
    # Check if strength actually varies across frames
    if _is_schedule_active(data, 'strength_schedule'):
        params.append('Str Sch')

    if _is_schedule_active(data, 'keyframe_strength_schedule'):
        params.append('Kfr Str Sch')

    # CFG scale (only if scheduled)
    if _is_schedule_active(data, 'cfg_scale_schedule'):
        params.append('CFG Sch')

    if _is_schedule_active(data, 'distilled_cfg_scale_schedule'):
        params.append('Dist. CFG Sch')

    # Other optional schedules (only if actually varying)
    if anim_args.enable_subseed_scheduling:
        params.extend(['Subseed Sch', 'Subseed Str Sch'])

    if _is_schedule_active(data, 'contrast_schedule'):
        params.append('Contrast Sch')

    if _is_schedule_active(data, 'noise_schedule'):
        params.append('Noise Sch')

    # Always include prompt last
    params.append('Prompt')

    return params


def _is_schedule_active(data: RenderData, schedule_name: str) -> bool:
    """Check if a schedule actually varies (not just static value).

    A schedule is considered "active" if it has more than one unique value
    across the animation, meaning it's actually changing over time.

    Args:
        data: Render data
        schedule_name: Name of schedule attribute (e.g., 'strength_schedule')

    Returns:
        True if schedule varies, False if static

    Examples:
        >>> # Static schedule: "0:(0.85)"
        >>> _is_schedule_active(data, 'strength_schedule')
        False

        >>> # Varying schedule: "0:(0.85), 100:(0.15)"
        >>> _is_schedule_active(data, 'strength_schedule')
        True
    """
    try:
        keys = data.animation_keys.deform_keys
        series_name = f"{schedule_name}_series"

        if not hasattr(keys, series_name):
            return False

        series = getattr(keys, series_name)
        if not series or len(series) == 0:
            return False

        # Check if all values are the same (static schedule)
        first_value = series[0]
        return not all(v == first_value for v in series)

    except (AttributeError, IndexError):
        return False


def format_subtitle_text_mode_aware(
    data: RenderData,
    frame_i: int,
    is_cadence: bool,
    seed: int,
    subseed: int,
    actual_steps: int,
    total_steps: int
) -> str:
    """Format subtitle text with mode-aware parameter selection.

    Creates clean subtitle text showing only relevant information:
    - Frame number, keyframe/cadence type
    - Seed (always relevant)
    - Steps (actual/total based on denoise)
    - Only relevant scheduled parameters for this mode
    - Prompt (always last)

    Args:
        data: Render data
        frame_i: Current frame index
        is_cadence: True if cadence frame, False if keyframe
        seed: Random seed
        subseed: Subseed value
        actual_steps: Actual diffusion steps (based on denoise)
        total_steps: Total configured steps

    Returns:
        Formatted subtitle text

    Examples:
        >>> # 3D keyframe
        >>> text = format_subtitle_text_mode_aware(data, 42, False, 12345, 0, 3, 20)
        >>> text
        'F#: 00042 [KEYFRAME]; Seed: 0000012345; Steps: 3/20; TrX: 0; TrY: 0; TrZ: 1; RotX: 0; RotY: 0; RotZ: 0\\nPrompt: parked beside lake'

        >>> # Interpolation cadence (no transforms)
        >>> text = format_subtitle_text_mode_aware(data, 15, True, 12345, 0, 17, 20)
        >>> text
        'F#: 00015 [CADENCE]; Seed: 0000012345; Steps: 17/20\\nPrompt: driving through forest'
    """
    from deforum.rendering import options as opt_utils
    from deforum.media.subtitle_handler import format_animation_params

    # Get relevant parameters for this mode
    params_to_print = get_relevant_params_for_mode(data)

    # Format parameter string
    params_str = format_animation_params(
        data.animation_keys.deform_keys,
        data.prompt_series,
        frame_i,
        params_to_print
    )

    # Simple format (just prompt)
    if opt_utils.is_simple_subtitles():
        return params_str.replace("Prompt:", "").strip()

    # Complex format with metadata
    index_str = f"{frame_i:05}"
    frame_type = "[KEYFRAME]" if not is_cadence else "[CADENCE]"
    seed_str = str(seed).zfill(10)
    steps_str = f"{actual_steps}/{total_steps}"

    # Build subtitle text
    parts = [
        f"F#: {index_str} {frame_type}",
        f"Seed: {seed_str}",
        f"Steps: {steps_str}"
    ]

    # Add subseed only if actually using subseed scheduling
    if data.args.anim_args.enable_subseed_scheduling:
        subseed_str = str(subseed).zfill(10)
        parts.append(f"SubSeed: {subseed_str}")

    # Combine metadata and params
    metadata = "; ".join(parts)

    # Put prompt on its own line if configured
    if opt_utils.is_own_line_for_prompt_srt():
        params_str = params_str.replace(" Prompt: ", "\nPrompt: ")

    return f"{metadata}; {params_str}"
