from deforum.utils.functional import put_if_present


def _get_opts():
    """Get opts instance - deferred import to avoid module-level import issues.

    A1111OptionsOverrider modifies opts.data at runtime, so we need to import
    opts fresh in each function rather than at module level.
    """
    # noinspection PyUnresolvedReferences
    try:
        from modules.shared import opts
        return opts
    except ImportError:
        # Mock opts for testing environment
        class MockOpts:
            data = {}
        return MockOpts()


def is_subtitle_generation_active():
    return _get_opts().data.get("deforum_save_gen_info_as_srt", False)


def is_verbose():
    """Checks if extra console output is enabled in deforum settings."""
    return _get_opts().data.get("deforum_debug_mode_enabled", False)


def is_dashboard_enabled():
    """Check if terminal dashboard is enabled."""
    return _get_opts().data.get("deforum_enable_dashboard", True)  # Enabled by default


def is_emojis_enabled():
    """Check if emojis are enabled in UI and console output."""
    return _get_opts().data.get("deforum_enable_emojis", False)  # Disabled by default


def is_nonessential_emojis_disabled():
    """Legacy function name - redirects to is_emojis_enabled()."""
    return not is_emojis_enabled()


def has_img2img_fix_steps():
    opts = _get_opts()
    return 'img2img_fix_steps' in opts.data and opts.data["img2img_fix_steps"]


def keep_3d_models_in_vram():
    return _get_opts().data.get("deforum_keep_3d_models_in_vram", False)


def setup(schedule):
    opts = _get_opts()
    if has_img2img_fix_steps():
        # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
        opts.data["img2img_fix_steps"] = False
    put_if_present(opts.data, "CLIP_stop_at_last_layers", schedule.clipskip)
    put_if_present(opts.data, "initial_noise_multiplier", schedule.noise_multiplier)
    put_if_present(opts.data, "eta_ddim", schedule.eta_ddim)
    put_if_present(opts.data, "eta_ancestral", schedule.eta_ancestral)


def generation_info_for_subtitles():
    return _get_opts().data.get("deforum_save_gen_info_as_srt_params", ['Prompt'])


def is_generate_subtitles():
    return _get_opts().data.get("deforum_save_gen_info_as_srt")


def is_always_write_keyframe_subs():
    return _get_opts().data.get("deforum_always_write_keyframe_subtitle", True)


def desired_subtitles_per_second():
    return int(_get_opts().data.get("deforum_subtitles_per_second", '10'))


def always_write_keyframe_subtitle():
    return int(_get_opts().data.get("deforum_always_write_keyframe_subtitle", True))


def is_subtitles_per_second_same_as_animation_fps(data):
    return desired_subtitles_per_second() == data.fps()


def is_simple_subtitles():
    return _get_opts().data.get("deforum_simple_subtitles", False)


def is_own_line_for_prompt_srt():
    return _get_opts().data.get("deforum_own_line_for_prompt_srt", True)


def is_emojis_disabled():
    """Legacy function name - redirects to is_emojis_enabled()."""
    return not is_emojis_enabled()


def get_log_theme():
    """Get console output theme (slopcore/classic/simple)."""
    return _get_opts().data.get("deforum_log_theme", "slopcore")


def get_log_level():
    """Get minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)."""
    return _get_opts().data.get("deforum_log_level", "INFO")
