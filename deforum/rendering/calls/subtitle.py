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
        # Calculate actual steps based on strength
        diffusion_frames = data.diffusion_frame_data.diffusion_frames
        actual_steps = 20  # Default fallback
        total_steps = 20

        # Find the diffusion frame for this frame_i
        for df in diffusion_frames:
            if df.i == frame_i:
                actual_steps = df.actual_steps(data)
                total_steps = df.schedule.steps
                break
            # Check if it's a tween of this diffusion frame
            for tween in df.tweens:
                if tween.i == frame_i:
                    # Tween uses the diffusion frame's steps
                    actual_steps = df.actual_steps(data)
                    total_steps = df.schedule.steps
                    break

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
