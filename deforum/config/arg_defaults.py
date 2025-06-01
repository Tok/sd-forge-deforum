"""
Argument Defaults
Contains default values and UI component definitions for all Deforum arguments
"""

import os
import tempfile

# Conditional imports for WebUI modules
try:
    import modules.paths as ph
    import modules.shared as sh
    WEBUI_AVAILABLE = True
except ImportError:
    # Fallback for testing/standalone environment
    WEBUI_AVAILABLE = False
    ph = None
    sh = None

# Conditional imports for Deforum modules
try:
    from .defaults import (get_guided_imgs_default_json, get_camera_shake_list, get_keyframe_distribution_list,
                           get_samplers_list, get_schedulers_list)
    from .general_utils import get_os
    DEFORUM_MODULES_AVAILABLE = True
except ImportError:
    DEFORUM_MODULES_AVAILABLE = False


def get_device():
    """Get device with fallback."""
    if WEBUI_AVAILABLE and hasattr(sh, 'device'):
        return sh.device
    return "cpu"


def get_models_path():
    """Get models path with fallback."""
    if WEBUI_AVAILABLE and hasattr(ph, 'models_path'):
        return ph.models_path + '/Deforum'
    return os.path.join(os.getcwd(), 'models', 'Deforum')


def get_half_precision():
    """Get half precision setting with fallback."""
    if WEBUI_AVAILABLE and hasattr(sh, 'cmd_opts') and hasattr(sh.cmd_opts, 'no_half'):
        return not sh.cmd_opts.no_half
    return True


def get_current_os():
    """Get current OS with fallback."""
    if DEFORUM_MODULES_AVAILABLE:
        return get_os()
    import platform
    return platform.system().lower()


def get_camera_shake_options():
    """Get camera shake options with fallback."""
    if DEFORUM_MODULES_AVAILABLE:
        return get_camera_shake_list().values()
    return ["INVESTIGATION", "CLASSIC", "HANDHELD"]


def get_sampler_options():
    """Get sampler options with fallback."""
    if DEFORUM_MODULES_AVAILABLE:
        return get_samplers_list()
    return ['Euler', 'Euler a', 'DPM++ 2M Karras', 'DDIM']


def get_scheduler_options():
    """Get scheduler options with fallback.""" 
    if DEFORUM_MODULES_AVAILABLE:
        return get_schedulers_list()
    return ['Simple', 'Karras', 'Exponential', 'Polyexponential']


def RootArgs():
    """Core root arguments with system defaults."""
    return {
        "device": get_device(),
        "models_path": get_models_path(),
        "half_precision": get_half_precision(),
        "clipseg_model": None,
        "mask_preset_names": ['everywhere', 'video_mask'],
        "frames_cache": [],
        "raw_batch_name": None,
        "raw_seed": None,
        "timestring": "",
        "subseed": -1,
        "subseed_strength": 0,
        "seed_internal": 0,
        "init_sample": None,
        "noise_mask": None,
        "initial_info": None,
        "first_frame": None,
        "animation_prompts": None,
        "prompt_keyframes": None,
        "current_user_os": get_current_os(),
        "tmp_deforum_run_duplicated_folder": os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
    }


def DeforumAnimArgs():
    """Animation argument defaults with UI component definitions."""
    return {
        "animation_mode": {
            "label": "Animation mode",
            "type": "radio",
            "choices": ['2D', '3D', 'Interpolation', 'Wan Video'],
            "value": "3D",
            "info": "control animation mode, will hide non relevant params upon change"
        },
        "max_frames": {
            "label": "Max frames",
            "type": "number",
            "precision": 0,
            "value": 333,
            "info": "end the animation at this frame number",
        },
        "border": {
            "label": "Border mode",
            "type": "radio",
            "choices": ['replicate', 'wrap'],
            "value": "replicate",
            "info": "controls pixel generation method for images smaller than the frame. hover on the options to see more info"
        },
        "angle": {
            "label": "Angle",
            "type": "textbox",
            "value": "0: (0)",
            "info": "rotate canvas clockwise/anticlockwise in degrees per frame"
        },
        "zoom": {
            "label": "Zoom",
            "type": "textbox",
            "value": "0: (1.0)",
            "info": "scale the canvas size, multiplicatively. [static = 1.0]"
        },
        "translation_x": {
            "label": "Translation X",
            "type": "textbox",
            "value": "0: (0)",
            "info": "move canvas left/right in pixels per frame"
        },
        "translation_y": {
            "label": "Translation Y",
            "type": "textbox",
            "value": "0: (0)",
            "info": "move canvas up/down in pixels per frame"
        },
        "translation_z": {
            "label": "Translation Z (zoom when animation mode is '3D')",
            "type": "textbox",
            "value": "0: (0)",
            "info": "move canvas towards/away from view [speed set by FOV]"
        },
        "transform_center_x": {
            "label": "Transform Center X",
            "type": "textbox",
            "value": "0: (0.5)",
            "info": "X center axis for 2D angle/zoom"
        },
        "transform_center_y": {
            "label": "Transform Center Y",
            "type": "textbox",
            "value": "0: (0.5)",
            "info": "Y center axis for 2D angle/zoom"
        },
        "rotation_3d_x": {
            "label": "Rotation 3D X",
            "type": "textbox",
            "value": "0: (0)",
            "info": "tilt canvas up/down in degrees per frame"
        },
        "rotation_3d_y": {
            "label": "Rotation 3D Y",
            "type": "textbox",
            "value": "0: (0)",
            "info": "pan canvas left/right in degrees per frame"
        },
        "rotation_3d_z": {
            "label": "Rotation 3D Z",
            "type": "textbox",
            "value": "0: (0)",
            "info": "roll canvas clockwise/anticlockwise"
        }
        # ... additional animation arguments will be added in subsequent iterations
    }


def DeforumArgs():
    """Core generation argument defaults with UI component definitions."""
    return {
        "W": {
            "label": "Width",
            "type": "slider",
            "minimum": 64,
            "maximum": 8192,
            "step": 64,
            "value": 1024,
            "info": "Width of the generated images"
        },
        "H": {
            "label": "Height", 
            "type": "slider",
            "minimum": 64,
            "maximum": 8192,
            "step": 64,
            "value": 1024,
            "info": "Height of the generated images"
        },
        "seed": {
            "label": "Seed",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "Set a fixed seed for reproducible results, -1 for random"
        },
        "steps": {
            "label": "Steps",
            "type": "slider",
            "minimum": 1,
            "maximum": 150,
            "step": 1,
            "value": 25,
            "info": "Number of sampling steps"
        },
        "cfg_scale": {
            "label": "CFG Scale",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 30.0,
            "step": 0.5,
            "value": 8.0,
            "info": "Classifier Free Guidance scale"
        }
        # ... additional core arguments will be added in subsequent iterations
    }


def ParseqArgs():
    """Parseq integration argument defaults."""
    return {
        "parseq_manifest": {
            "label": "Parseq manifest",
            "type": "textbox",
            "value": "",
            "info": "JSON data with keyframes and values. Parseq manifest should be copied here."
        },
        "parseq_use_deltas": {
            "label": "Use deltas for parseq",
            "type": "checkbox",
            "value": False,
            "info": "Parse the input data as deltas or as full values."
        }
    }


def WanArgs():
    """WAN AI integration argument defaults."""
    return {
        "wan_mode": {
            "label": "WAN mode",
            "type": "radio", 
            "choices": ['Video Input', 'Image Animation'],
            "value": "Video Input",
            "info": "Choose WAN generation mode"
        },
        "wan_use_upscaler": {
            "label": "Use Upscaler",
            "type": "checkbox",
            "value": False,
            "info": "Apply upscaling to WAN output"
        }
        # ... additional WAN arguments will be added in subsequent iterations
    }


def DeforumOutputArgs():
    """Output and video processing argument defaults."""
    return {
        "fps": {
            "label": "FPS",
            "type": "number",
            "precision": 1,
            "value": 15,
            "info": "choose fps for video output"
        },
        "add_soundtrack": {
            "label": "Add soundtrack",
            "type": "radio",
            "choices": ['None', 'File', 'Init Video'],
            "value": "File",
            "info": "add audio to video from file/url or init video"
        },
        "soundtrack_path": {
            "label": "Soundtrack path",
            "type": "textbox",
            "value": "https://ia801303.us.archive.org/26/items/amen-breaks/cw_amen13_173.mp3",
            "info": "abs. path or url to audio file"
        },
        "make_gif": {
            "label": "Make GIF",
            "type": "checkbox",
            "value": False,
            "info": "make GIF in addition to the video/s"
        },
        "delete_imgs": {
            "label": "Delete Imgs",
            "type": "checkbox",
            "value": False,
            "info": "auto-delete imgs when video is ready. Will break Resume from timestring!"
        }
        # ... additional output arguments will be added in subsequent iterations
    } 