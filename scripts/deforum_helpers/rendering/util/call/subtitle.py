# noinspection PyUnresolvedReferences
from modules.shared import opts
from ....subtitle_handler import format_animation_params, write_frame_subtitle


def call_format_animation_params(render_data, i, params_to_print):
    return format_animation_params(render_data.animation_keys.deform_keys,
                                   render_data.prompt_series, i, params_to_print)


def call_write_frame_subtitle(render_data, i, params_string, is_cadence, seed) -> None:
    cadence_str = str(is_cadence).ljust(5)  # Pad is_cadence to 5 characters so 'True' is the same width as 'False'.
    index_str = f"{i:05}"  # Pad frame index to 5 digits with leading zeros, which ought to be enough for anybody.
    seed_str = str(seed).zfill(10)  # Convert seed to string and pad with leading zeros to 10 digits if necessary.
    clean_params_string = prepare_prompt_for_subtitle(params_string)
    text = f"F#: {index_str}; Cadence: {cadence_str}; Seed: {seed_str}; {clean_params_string}"
    write_frame_subtitle(render_data.srt.filename, i, render_data.srt.frame_duration, text)


def prepare_prompt_for_subtitle(params_string):
    # prompt is always the last param if present,
    # so there's no need to add a newline after it.
    clean_params_string = params_string.replace("--neg", "") \
        if params_string.endswith("--neg") else params_string
    trimmed_params_string = clean_params_string.rstrip()
    return trimmed_params_string.replace(" Prompt: ", "\nPrompt:") \
        if opts.data.get("deforum_own_line_for_prompt_srt", True) \
        else trimmed_params_string
