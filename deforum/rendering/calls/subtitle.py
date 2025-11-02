from deforum.utils.system.logging import log as log_utils
from deforum.rendering import options as opt_utils
from deforum.media.subtitle_handler import format_animation_params, write_subtitle_from_to
from deforum.rendering.calls.subtitle_mode_aware import (
    get_relevant_params_for_mode,
    format_subtitle_text_mode_aware
)


def _call_format_animation_params(data, frame_i, params_to_print):
    params_string = format_animation_params(data.animation_keys.deform_keys, data.prompt_series,
                                            frame_i, params_to_print)
    return _prepare_prompt_for_subtitle(params_string)


def call_write_subtitle_from_to(data, sub_i, frame_i, is_cadence, seed, subseed, start_time_s, end_time_s) -> None:
    """Write subtitle with mode-aware parameter filtering.

    Uses intelligent parameter selection based on:
    - Current render mode (3D, Interpolation, etc.)
    - Active schedules (only show parameters that actually vary)
    - User configuration (simple vs complex subtitles)

    Args:
        data: Render data
        sub_i: Subtitle index
        frame_i: Frame index
        is_cadence: True if cadence frame, False if keyframe
        seed: Random seed
        subseed: Subseed value
        start_time_s: Start time in seconds
        end_time_s: End time in seconds
    """
    # Check if mode-aware subtitles are enabled (default: True)
    use_mode_aware = opt_utils._get_opts().data.get("deforum_mode_aware_subtitles", True)

    if use_mode_aware:
        # NEW: Mode-aware subtitle generation
        # For mode-aware subtitles, we calculate steps from strength schedule
        # Since we don't have access to the actual DiffusionFrame here,
        # we'll estimate from the strength schedule
        strength_schedule = data.animation_keys.deform_keys.strength_schedule_series
        keyframe_strength_schedule = data.animation_keys.deform_keys.keyframe_strength_schedule_series
        steps_schedule = data.animation_keys.deform_keys.steps_schedule_series

        # Get strength and steps for this frame
        strength = keyframe_strength_schedule[frame_i] if not is_cadence else strength_schedule[frame_i]
        total_steps = int(steps_schedule[frame_i]) if frame_i < len(steps_schedule) else 20

        # Calculate actual steps (same formula as DiffusionFrame.actual_steps)
        if frame_i == 0 and not data.args.args.use_init:
            actual_steps = total_steps
        else:
            import math
            actual_steps = int(math.ceil(total_steps * strength)) + 1

        text = format_subtitle_text_mode_aware(
            data, frame_i, is_cadence, seed, subseed,
            actual_steps, total_steps
        )
    else:
        # OLD: Original subtitle generation (user-selected params)
        params_to_print = opt_utils.generation_info_for_subtitles() if data.parseq_adapter.use_parseq else ['Prompt']
        text = _prepare_subtitle_text(data, params_to_print, frame_i, is_cadence, seed, subseed)

    write_subtitle_from_to(data.srt.filename, sub_i, start_time_s, end_time_s, text)


def _prepare_subtitle_text(data, params_to_print, frame_i, is_cadence, seed, subseed):
    """Original subtitle text generation (legacy mode)."""
    params_str = _call_format_animation_params(data, frame_i, params_to_print)
    if opt_utils.is_simple_subtitles():
        return params_str.replace("Prompt:", "").strip()

    index_str = f"{frame_i:05}"  # Pad frame index to 5 digits with leading zeros, which ought to be enough for anybody.
    cadence_str = str(is_cadence).ljust(5)  # Pad is_cadence to 5 characters so 'True' is the same width as 'False'.
    seed_str = str(seed).zfill(10)  # Convert seed to string and pad with leading zeros to 10 digits if necessary.
    subseed_str = str(subseed).zfill(10)  # TODO also provide subseed_strength
    if not data.parseq_adapter.use_parseq:
        log_utils.warning("Complex subtitles not supported without Parseq in the render core: Params removed.")
        return f"F#: {index_str}; {params_str}"
    else:
        return f"F#: {index_str}; Cadence: {cadence_str}; Seed: {seed_str}; SubSeed: {subseed_str}; {params_str}"


def _prepare_prompt_for_subtitle(params_string):
    # prompt is always the last param if present,
    # so there's no need to add a newline after it.
    clean_params_string = (params_string.replace("--neg", "")
                           if params_string.endswith("--neg")
                           else params_string)  # TODO this should be done elsewhere.
    trimmed_params_string = clean_params_string.rstrip()
    return (trimmed_params_string.replace(" Prompt: ", "\nPrompt:")
            if opt_utils.is_own_line_for_prompt_srt()
            else trimmed_params_string)
