from .utils import put_if_present

# noinspection PyUnresolvedReferences
from modules.shared import opts


def is_subtitle_generation_active():
    return opts.data.get("deforum_save_gen_info_as_srt", False)


def is_verbose():
    """Checks if extra console output is enabled in deforum settings."""
    return opts.data.get("deforum_debug_mode_enabled", False)


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
    return opts.data.get("deforum_save_gen_info_as_srt_params", ['Seed'])


def is_generate_subtitles():
    return opts.data.get("deforum_save_gen_info_as_srt")
