# noinspection PyUnresolvedReferences
from modules.shared import opts

from .. import log_utils, opt_utils
from ....subtitle_handler import format_animation_params, write_frame_subtitle


def _call_format_animation_params(data, i, params_to_print):
    params_string = format_animation_params(data.animation_keys.deform_keys, data.prompt_series, i, params_to_print)
    return _prepare_prompt_for_subtitle(params_string)


def call_write_frame_subtitle(data, i, is_cadence, seed, subseed) -> None:
    is_simple_subtitles = opts.data.get("deforum_simple_subtitles", False)
    params_to_print = opt_utils.generation_info_for_subtitles() if data.parseq_adapter.use_parseq else ['Prompt']
    text = _prepare_subtitle_text(data, is_simple_subtitles, params_to_print, i, is_cadence, seed, subseed)
    write_frame_subtitle(data.srt.filename, i, data.srt.frame_duration, text)


def _prepare_subtitle_text(data, is_simple_subtitles, params_to_print, i, is_cadence, seed, subseed):
    params_str = _call_format_animation_params(data, i, params_to_print)
    if is_simple_subtitles:
        return params_str.replace("Prompt:", "").strip()

    index_str = f"{i:05}"  # Pad frame index to 5 digits with leading zeros, which ought to be enough for anybody.
    cadence_str = str(is_cadence).ljust(5)  # Pad is_cadence to 5 characters so 'True' is the same width as 'False'.
    seed_str = str(seed).zfill(10)  # Convert seed to string and pad with leading zeros to 10 digits if necessary.
    subseed_str = str(subseed).zfill(10)  # TODO also provide subseed_strength
    if not data.parseq_adapter.use_parseq:
        log_utils.warn("Complex subtitles not supported without Parseq in the experimental core: Params removed.")
        return f"F#: {index_str}; {params_str}"
    else:
        return f"F#: {index_str}; Cadence: {cadence_str}; Seed: {seed_str}; SubSeed: {subseed_str}; {params_str}"


def _prepare_prompt_for_subtitle(params_string):
    # prompt is always the last param if present,
    # so there's no need to add a newline after it.
    clean_params_string = params_string.replace("--neg", "") \
        if params_string.endswith("--neg") else params_string  # TODO this should be done elsewhere.
    trimmed_params_string = clean_params_string.rstrip()
    return trimmed_params_string.replace(" Prompt: ", "\nPrompt:") \
        if opts.data.get("deforum_own_line_for_prompt_srt", True) \
        else trimmed_params_string
