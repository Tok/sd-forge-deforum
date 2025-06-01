import uuid

# noinspection PyUnresolvedReferences
import gradio as gr
# noinspection PyUnresolvedReferences
from modules import ui_components
# noinspection PyUnresolvedReferences
from modules.shared import opts, cmd_opts, OptionInfo, options_section

from deforum.utils.subtitle_handler import get_user_values
from deforum.media.video_audio_pipeline import find_ffmpeg_binary


def on_ui_settings():
    section = ('deforum', "Deforum")

    def _def(is_visible=True, choices=None, minimum=None, maximum=None):
        return {"interactive": True, "visible": is_visible, "choices": choices, "minimum": minimum, "maximum": maximum}

    def add(name, description, default_value, component=None, definition=None):
        opt_info = OptionInfo(default_value, description, component, definition, section=section)
        opts.add_option(name, opt_info)

    def add_subsection(title):
        # Forge doesn't technically support subsections, but we can separate settings by providing our own HTML element
        # that's technically another setting meant to be ignored. TODO there's probably a better way to do this...
        opt_info = OptionInfo("<br><br><strong>" + title + "</strong>", "", gr.HTML, {"visible": True}, section=section)
        opt_info.do_not_save = True
        opt_info.restrict_api = True
        # The uuid4 is just a random bs string, signaling that there's nothing relevant to be set or read here.
        opts.add_option("deforum_" + str(uuid.uuid4()), opt_info)

    def add_cb(name, description, default_value=False):
        add(name, description, default_value, gr.Checkbox)

    def add_dd(name, description, default_value, choices):
        add(name, description, default_value, gr.Dropdown, _def(choices=choices))

    def add_sl(name, description, default_value, minimum, maximum):
        add(name, description, default_value, gr.Slider, _def(minimum=minimum, maximum=maximum))

    add_subsection("General Deforum Settings")
    add_cb("deforum_keep_3d_models_in_vram", "Keep 3D models in VRAM between runs",
           not (cmd_opts.lowvram or cmd_opts.medvram))
    add_cb("deforum_enable_persistent_settings", "Keep settings persistent upon relaunch of webUI.")
    add("deforum_persistent_settings_path", "Path for saving your persistent settings file:",
        "models/Deforum/deforum_persistent_settings.txt")
    add_cb("deforum_debug_mode_enabled", "Enable Dev mode - adds extra reporting in console.")
    add_cb("deforum_disable_nonessential_emojis", "Disables non-essential emojis.")

    add_subsection("Deforum FFmpeg Settings")
    add("deforum_ffmpeg_location", "FFmpeg path/ location", find_ffmpeg_binary())
    add_sl("deforum_ffmpeg_crf", "FFmpeg CRF value", 17, 0, 51)
    add_dd("deforum_ffmpeg_preset", "FFmpeg Preset", 'slow', _ffmpeg_preset_choices())

    add_subsection("Deforum Subtitle Settings")
    add_cb("deforum_save_gen_info_as_srt",
           "Save an .srt (subtitles) file with the generation info along with each animation. Works with VLC and with YouTube (upload file 'With timing').", True)
    add_cb("deforum_embed_srt", "If .srt file is saved, soft-embed the subtitles into the video file.")
    add_cb("deforum_simple_subtitles", "Only write prompt into subtitles.")
    add_cb("deforum_own_line_for_prompt_srt", "Put 'prompt' on its own line in subtitles if present.")
    add("deforum_save_gen_info_as_srt_params",
        "Animation parameters to be saved to the .srt file (Frame # and Seed will always be saved):",
        ['Prompt'], ui_components.DropdownMulti, lambda: {"interactive": True, "choices": get_user_values()})
    add_dd("deforum_subtitles_per_second", "Desired subtitles per second (Mostly useful at high FPS. Experimental core only)", '10',
           _subtitles_per_second_choices())
    add_cb("deforum_always_write_keyframe_subtitle",
           "Always write keyframe subtitle (makes subtitles per second fuzzy, but provides better synchronization. Experimental core only).",
           True)

    add_subsection("Deforum Preview Settings")
    add_dd("deforum_preview",
           "Generate preview video during generation? (does not include frame interpolation and up-scaling)",
           'Off', _preview_choices())
    add_sl("deforum_preview_interval_frames", "Generate preview every N frames", 100, 10, 500)


def _ffmpeg_preset_choices():
    return ['veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast', 'superfast', 'ultrafast']


def _subtitles_per_second_choices():
    return ['1', '5', '6', '10', '12', '20', '24', '30', '60']


def _preview_choices():
    return ['Off', 'On', 'On, concurrent (don\'t pause generation)']
