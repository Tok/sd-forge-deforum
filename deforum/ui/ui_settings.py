# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import uuid

# noinspection PyUnresolvedReferences
import gradio as gr
# noinspection PyUnresolvedReferences
from modules import ui_components
# noinspection PyUnresolvedReferences
from modules.shared import opts, cmd_opts, OptionInfo, options_section

from deforum.media.subtitle_handler import get_user_values
from deforum.media.video_audio_utilities import find_ffmpeg_binary


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
    # Forge Neo removed lowvram/medvram flags
    has_vram_flags = getattr(cmd_opts, 'lowvram', False) or getattr(cmd_opts, 'medvram', False)
    add_cb("deforum_keep_3d_models_in_vram", "Keep 3D models in VRAM between runs",
           not has_vram_flags)
    add_cb("deforum_enable_persistent_settings", "Keep settings persistent upon relaunch of webUI.")
    add("deforum_persistent_settings_path", "Path for saving your persistent settings file:",
        "models/Deforum/deforum_persistent_settings.txt")

    add_subsection("Console & UI Output Settings")
    add_cb("deforum_enable_dashboard", "Enable fixed dashboard (real-time progress bars and VRAM stay at bottom, logs scroll above)", default_value=True)
    add_cb("deforum_dashboard_ascii_preview", "Write ASCII art preview to scrolling log on each frame (shows progression of frames)", default_value=True)
    add_dd("deforum_dashboard_ascii_size", "ASCII Preview Size - Small (16x9), Medium (32x18), Large (64x36)",
           "medium", ["small", "medium", "large"])
    add_cb("deforum_dashboard_ascii_to_log", "Legacy setting (same as dashboard_ascii_preview) - kept for compatibility", default_value=False)
    add_dd("deforum_log_level", "Log Level - Controls console output verbosity",
           "INFO", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    add_dd("deforum_log_theme", "Console Theme - Slopcore (blue→purple), Classic (vibrant), Simple (plain text)",
           "slopcore", ["slopcore", "classic", "simple"])
    add_cb("deforum_enable_emojis", "Enable emojis in UI and console output", default_value=False)
    add_sl("deforum_max_viz_animation_frames", "Max Visualization Animation Frames - Limits animated camera path frames to prevent browser crash (100-1000)", 400, 100, 1000)
    add_sl("deforum_max_schedule_display_frames", "Max Schedule Display Frames - Limits schedule string length in UI textboxes (full schedules saved to settings.json). Range: 100-5000", 1000, 100, 5000)

    add_subsection("Deforum FFmpeg Settings")
    add("deforum_ffmpeg_location", "FFmpeg path/ location", find_ffmpeg_binary())
    add_sl("deforum_ffmpeg_crf", "FFmpeg CRF value", 17, 0, 51)
    add_dd("deforum_ffmpeg_preset", "FFmpeg Preset", 'slow', _ffmpeg_preset_choices())

    add_subsection("Deforum Subtitle Settings")
    add_cb("deforum_save_gen_info_as_srt",
           "Save an .srt (subtitles) file with the generation info along with each animation. Works with VLC and with YouTube (upload file 'With timing').", True)
    add_cb("deforum_embed_srt", "If .srt file is saved, soft-embed the subtitles into the video file.")
    add_cb("deforum_mode_aware_subtitles",
           "Use mode-aware subtitles (shows only relevant parameters based on render mode, removes Aspect Ratio, shows actual steps like '3/20'). Disable to use manual parameter selection below.",
           True)
    add_cb("deforum_simple_subtitles", "Only write prompt into subtitles.")
    add_cb("deforum_own_line_for_prompt_srt", "Put 'prompt' on its own line in subtitles if present.")
    add("deforum_save_gen_info_as_srt_params",
        "Animation parameters to be saved to the .srt file (Frame # and Seed will always be saved). Only used if mode-aware subtitles are disabled:",
        ['Prompt'], ui_components.DropdownMulti, lambda: {"interactive": True, "choices": get_user_values()})
    add_dd("deforum_subtitles_per_second", "Desired subtitles per second (Mostly useful at high FPS. Render core only)", '10',
           _subtitles_per_second_choices())
    add_cb("deforum_always_write_keyframe_subtitle",
           "Always write keyframe subtitle (makes subtitles per second fuzzy, but provides better synchronization. Render core only).",
           True)

    add_subsection("Deforum Video Metadata Settings")
    add_cb("deforum_embed_metadata",
           "Embed comprehensive generation settings into video metadata (privacy-focused: technical params only, no user-identifying info). Enables full reproducibility similar to ComfyUI's workflow embedding.",
           True)
    add_cb("deforum_embed_human_readable_metadata",
           "Also embed key settings as directly readable metadata fields (deforum_model, deforum_resolution, deforum_seed, etc.) alongside base64-encoded comprehensive data.",
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
