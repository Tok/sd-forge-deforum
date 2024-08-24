from ....subtitle_handler import format_animation_params, write_frame_subtitle


def call_format_animation_params(render_data, i, params_to_print):
    return format_animation_params(render_data.animation_keys.deform_keys,
                                   render_data.prompt_series, i, params_to_print)


def call_write_frame_subtitle(render_data, i, params_string, is_cadence: bool = False) -> None:
    text = f"F#: {i}; Cadence: {is_cadence}; Seed: {render_data.args.args.seed}; {params_string}"
    write_frame_subtitle(render_data.srt.filename, i, render_data.srt.frame_duration, text)
