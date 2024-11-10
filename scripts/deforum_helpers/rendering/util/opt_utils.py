from .utils import put_if_present

# noinspection PyUnresolvedReferences
from modules.shared import opts


def is_subtitle_generation_active():
    return opts.data.get("deforum_save_gen_info_as_srt", False)


def is_verbose():
    """Checks if extra console output is enabled in deforum settings."""
    return opts.data.get("deforum_debug_mode_enabled", False)


def is_nonessential_emojis_disabled():
    return opts.data.get("deforum_disable_nonessential_emojis", False)


def has_img2img_fix_steps():
    return 'img2img_fix_steps' in opts.data and opts.data["img2img_fix_steps"]


def keep_3d_models_in_vram():
    return opts.data.get("deforum_keep_3d_models_in_vram", False)


def setup(schedule):
    if has_img2img_fix_steps():
        # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
        opts.data["img2img_fix_steps"] = False
    put_if_present(opts.data, "CLIP_stop_at_last_layers", schedule.clipskip)
    put_if_present(opts.data, "initial_noise_multiplier", schedule.noise_multiplier)
    put_if_present(opts.data, "eta_ddim", schedule.eta_ddim)
    put_if_present(opts.data, "eta_ancestral", schedule.eta_ancestral)


def generation_info_for_subtitles():
    return opts.data.get("deforum_save_gen_info_as_srt_params", ['Prompt'])


def is_generate_subtitles():
    return opts.data.get("deforum_save_gen_info_as_srt")


def is_always_write_keyframe_subs():
    return opts.data.get("deforum_always_write_keyframe_subtitle", True)


def desired_subtitles_per_second():
    return int(opts.data.get("deforum_subtitles_per_second", '10'))


def always_write_keyframe_subtitle():
    return int(opts.data.get("deforum_always_write_keyframe_subtitle", True))


def is_subtitles_per_second_same_as_animation_fps(data):
    return desired_subtitles_per_second() == data.fps()


def is_simple_subtitles():
    return opts.data.get("deforum_simple_subtitles", False)


def is_own_line_for_prompt_srt():
    return opts.data.get("deforum_own_line_for_prompt_srt", True)


def is_emojis_disabled():
    return opts.data.get("deforum_disable_nonessential_emojis", False)
