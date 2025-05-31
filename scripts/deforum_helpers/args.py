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

import json
import os
import pathlib
import tempfile
import time
from types import SimpleNamespace

import modules.paths as ph
import modules.shared as sh
from PIL import Image
from modules.processing import get_fixed_seed

from .defaults import (get_guided_imgs_default_json, get_camera_shake_list, get_keyframe_distribution_list,
                       get_samplers_list, get_schedulers_list)
from .deforum_controlnet import controlnet_component_names
from .general_utils import get_os, substitute_placeholders


def RootArgs():
    return {
        "device": sh.device,
        "models_path": ph.models_path + '/Deforum',
        "half_precision": not sh.cmd_opts.no_half,
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
        "current_user_os": get_os(),
        "tmp_deforum_run_duplicated_folder": os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
    }


# 'Midas-3.1-BeitLarge' is temporarily removed until fixed. Can add it back anytime
# as it's supported in the back-end depth code
def DeforumAnimArgs():
    return {
        "animation_mode": {
            "label": "Animation mode",
            "type": "radio",
            "choices": ['2D', '3D', 'Video Input', 'Interpolation', 'Wan Video'],
            "value": "2D",
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
            "value": "0: (1.0)",  # original value: "0: (1.0025+0.002*sin(1.25*3.14*t/120))"
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
            "value": "0: (0)",  # original value: "0: (1.10)"
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
        },
        "shake_name": {
            "label": "Shake Name",
            "type": "dropdown",
            "choices": get_camera_shake_list().values(),
            "value": "INVESTIGATION",
            "info": "Name of the camera shake loop.",
        },
        "shake_intensity": {
            "label": "Shake Intensity",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 3.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Intensity of the camera shake loop."
        },
        "shake_speed": {
            "label": "Shake Speed",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 3.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Speed of the camera shake loop."
        },
        "enable_perspective_flip": {
            "label": "Enable perspective flip",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "perspective_flip_theta": {
            "label": "Perspective flip theta",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "perspective_flip_phi": {
            "label": "Perspective flip phi",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "perspective_flip_gamma": {
            "label": "Perspective flip gamma",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "perspective_flip_fv": {
            "label": "Perspective flip tv",
            "type": "textbox",
            "value": "0: (53)",
            "info": "the 2D vanishing point of perspective (rec. range 30-160)"
        },
        "noise_schedule": {
            "label": "Noise schedule",
            "type": "textbox",
            "value": "0: (0.065)",
            "info": ""
        },
        "strength_schedule": {
            "label": "Strength schedule",
            "type": "textbox",
            "value": "0: (0.85)",
            "info": "amount of presence of previous frame to influence next frame, also controls steps in the following formula [steps - (strength_schedule * steps)]"
        },
        "keyframe_strength_schedule": {
            "label": "Strength schedule for keyframes",
            "type": "textbox",
            "value": "0: (0.50)",
            "info": "like 'Strength schedule' but only for frames with an entry in 'prompts'. Meant to be set somewhat lower than the regular Strengh schedule. At 0 it generates a totally new image on every prompt change. Ignored if Parseq is used or when Keyframe distribustion is disabled."
        },
        "contrast_schedule": "0: (1.0)",
        "cfg_scale_schedule": {
            "label": "CFG scale schedule",
            "type": "textbox",
            "value": "0: (1.0)",
            "info": "how closely the image should conform to the prompt. Lower values produce more creative results. (recommended value for Flux.1: 1.0, 5-15 with other models)"
        },
        "distilled_cfg_scale_schedule": {
            "label": "Distilled CFG scale schedule",
            "type": "textbox",
            "value": "0: (3.5)",
            "info": "how closely the image should conform to the prompt. Lower values produce more creative results. (recommended value for Flux.1: 3.5, ignored for most other models)`"
        },
        "enable_steps_scheduling": {
            "label": "Enable steps scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "steps_schedule": {
            "label": "Steps schedule",
            "type": "textbox",
            "value": "0: (20)",
            "info": "mainly allows using more than 200 steps. Otherwise, it's a mirror-like param of 'strength schedule'"
        },
        "fov_schedule": {
            "label": "FOV schedule",
            "type": "textbox",
            "value": "0: (70)",
            "info": "adjusts the scale at which the canvas is moved in 3D by the translation_z value. [Range -180 to +180, with 0 being undefined. Values closer to 180 will make the image have less depth, while values closer to 0 will allow more depth]"
        },
        "aspect_ratio_schedule": {
            "label": "Aspect Ratio schedule",
            "type": "textbox",
            "value": "0: (1.0)",
            "info": "adjusts the aspect ratio for the depth calculations. HAS NOTHING TO DO WITH THE VIDEO SIZE. Meant to be left at 1.0, except for emulating legacy behaviour."
        },
        "aspect_ratio_use_old_formula": {
            "label": "Use old aspect ratio formula",
            "type": "checkbox",
            "value": False,
            "info": "for backward compatibility. Uses the formula: `width/height`"
        },
        "near_schedule": {
            "label": "Near schedule",
            "type": "textbox",
            "value": "0: (200)",
            "info": ""
        },
        "far_schedule": {
            "label": "Far schedule",
            "type": "textbox",
            "value": "0: (10000)",
            "info": ""
        },
        "seed_schedule": {
            "label": "Seed schedule",
            "type": "textbox",
            "value": '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)',
            "info": ""
        },
        "enable_subseed_scheduling": {
            "label": "Enable Subseed scheduling",
            "type": "checkbox",
            "value": False,
            "info": "Aim for more continuous similarity by mixing in a constant seed."
        },
        "subseed_schedule": {
            "label": "Subseed schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "subseed_strength_schedule": {
            "label": "Subseed strength schedule",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "enable_sampler_scheduling": {
            "label": "Enable sampler scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "sampler_schedule": {
            "label": "Sampler schedule",
            "type": "textbox",
            "value": '0: ("Euler")',
            "info": "allows keyframing of samplers. Use names as they appear in ui dropdown in 'run' tab"
        },
        "enable_scheduler_scheduling": {
            "label": "Enable scheduler scheduling",
            "type": "checkbox",
            "value": False,
            "info": "enables scheduling of scheduler schedules."
        },
        "scheduler_schedule": {
            "label": "Scheduler schedule",
            "type": "textbox",
            "value": '0: ("Simple")',
            "info": "allows keyframing of schedulers. Use names as they appear in ui dropdown in 'run' tab"
        },
        "use_noise_mask": {
            "label": "Use noise mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_schedule": {
            "label": "Mask schedule",
            "type": "textbox",
            "value": '0: ("{video_mask}")',
            "info": ""
        },
        "noise_mask_schedule": {
            "label": "Noise mask schedule",
            "type": "textbox",
            "value": '0: ("{video_mask}")',
            "info": ""
        },
        "enable_checkpoint_scheduling": {
            "label": "Enable checkpoint scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "checkpoint_schedule": {
            "label": "allows keyframing different sd models. Use *full* name as appears in ui dropdown",
            "type": "textbox",
            "value": '0: ("model1.ckpt"), 100: ("model2.safetensors")',
            "info": "allows keyframing different sd models. Use *full* name as appears in ui dropdown"
        },
        "enable_clipskip_scheduling": {
            "label": "Enable CLIP skip scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "clipskip_schedule": {
            "label": "CLIP skip schedule",
            "type": "textbox",
            "value": "0: (2)",
            "info": ""
        },
        "enable_noise_multiplier_scheduling": {
            "label": "Enable noise multiplier scheduling",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "noise_multiplier_schedule": {
            "label": "Noise multiplier schedule",
            "type": "textbox",
            "value": "0: (1.05)",
            "info": ""
        },
        "resume_from_timestring": {
            "label": "Resume from timestring",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "resume_timestring": {
            "label": "Resume timestring",
            "type": "textbox",
            "value": "20241111111111",
            "info": ""
        },
        "enable_ddim_eta_scheduling": {
            "label": "Enable DDIM ETA scheduling",
            "type": "checkbox",
            "value": False,
            "visible": False,
            "info": "noise multiplier; higher = more unpredictable results"
        },
        "ddim_eta_schedule": {
            "label": "DDIM ETA Schedule",
            "type": "textbox",
            "value": "0: (0)",
            "visible": False,
            "info": ""
        },
        "enable_ancestral_eta_scheduling": {
            "label": "Enable Ancestral ETA scheduling",
            "type": "checkbox",
            "value": False,
            "info": "noise multiplier; applies to Euler A and other samplers that have the letter 'a' in them"
        },
        "ancestral_eta_schedule": {
            "label": "Ancestral ETA Schedule",
            "type": "textbox",
            "value": "0: (1)",
            "visible": False,
            "info": ""
        },
        "amount_schedule": {
            "label": "Amount schedule",
            "type": "textbox",
            "value": "0: (0.1)",
            "info": ""
        },
        "kernel_schedule": {
            "label": "Kernel schedule",
            "type": "textbox",
            "value": "0: (5)",
            "info": ""
        },
        "sigma_schedule": {
            "label": "Sigma schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "threshold_schedule": {
            "label": "Threshold schedule",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "color_coherence": {
            "label": "Color coherence",
            "type": "dropdown",
            "choices": ['None', 'HSV', 'LAB', 'RGB', 'Video Input', 'Image'],
            "value": "None",
            "info": "choose an algorithm/ method for keeping color coherence across the animation"
        },
        "color_coherence_image_path": {
            "label": "Color coherence image path",
            "type": "textbox",
            "value": "",
            "info": ""
        },
        "color_coherence_video_every_N_frames": {
            "label": "Color coherence video every N frames",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "",
        },
        "color_force_grayscale": {
            "label": "Color force Grayscale",
            "type": "checkbox",
            "value": False,
            "info": "force all frames to be in grayscale"
        },
        "legacy_colormatch": {
            "label": "Legacy colormatch",
            "type": "checkbox",
            "value": False,
            "info": "apply colormatch before adding noise (use with CN's Tile)"
        },
        "keyframe_distribution": {
            "label": "Keyframe distribution.",
            "type": "dropdown",
            "choices": get_keyframe_distribution_list().values(),
            "value": "Keyframes Only",
            "info": "Allows for fast generations at high cadence or no cadence."
        },
        "diffusion_cadence": {
            "label": "Cadence",
            "type": "slider",
            "minimum": 1,
            "maximum": 200,
            "step": 1,
            "value": 10,
            "info": "# of in-between frames that will not be directly diffused"
        },
        "optical_flow_cadence": {
            "label": "Optical flow cadence",
            "type": "dropdown",
            "choices": ['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'],
            "value": "None",
            "info": "use optical flow estimation for your in-between (cadence) frames"
        },
        "cadence_flow_factor_schedule": {
            "label": "Cadence flow factor schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "optical_flow_redo_generation": {
            "label": "Optical flow generation",
            "type": "dropdown",
            "choices": ['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'],
            "value": "None",
            "info": "this option takes twice as long because it generates twice in order to capture the optical flow from the previous image to the first generation, then warps the previous image and redoes the generation"
        },
        "redo_flow_factor_schedule": {
            "label": "Generation flow factor schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "diffusion_redo": '0',
        "noise_type": {
            "label": "Noise type",
            "type": "radio",
            "choices": ['uniform', 'perlin'],
            "value": "perlin",
            "info": ""
        },
        "perlin_w": {
            "label": "Perlin W",
            "type": "slider",
            "minimum": 0.1,
            "maximum": 16,
            "step": 0.1,
            "value": 8,
            "visible": False
        },
        "perlin_h": {
            "label": "Perlin H",
            "type": "slider",
            "minimum": 0.1,
            "maximum": 16,
            "step": 0.1,
            "value": 8,
            "visible": False
        },
        "perlin_octaves": {
            "label": "Perlin octaves",
            "type": "slider",
            "minimum": 1,
            "maximum": 7,
            "step": 1,
            "value": 4
        },
        "perlin_persistence": {
            "label": "Perlin persistence",
            "type": "slider",
            "minimum": 0,
            "maximum": 1,
            "step": 0.02,
            "value": 0.5
        },
        "use_depth_warping": {
            "label": "Use depth warping",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "depth_algorithm": {
            "label": "Depth Algorithm",
            "type": "dropdown",
            "choices": ['Depth-Anything-V2-Small', 'Midas-3-Hybrid'],
            "value": "Depth-Anything-V2-Small",
            "info": "Depth estimation algorithm: Depth-Anything-V2 (recommended) or MiDaS (fallback)"
        },
        "midas_weight": {
            "label": "MiDaS/Zoe weight",
            "type": "number",
            "precision": None,
            "value": 0.2,
            "info": "sets a midpoint at which a depth-map is to be drawn: range [-1 to +1]",
            "visible": False
        },
        "padding_mode": {
            "label": "Padding mode",
            "type": "radio",
            "choices": ['border', 'reflection', 'zeros'],
            "value": "border",
            "info": "Choose how to handle pixels outside the field of view as they come into the scene"
        },
        "sampling_mode": {
            "label": "Sampling mode",
            "type": "radio",
            "choices": ['bicubic', 'bilinear', 'nearest'],
            "value": "bicubic",
            "info": "Choose the sampling method: Bicubic for quality, Bilinear for speed, Nearest for simplicity."
        },
        "save_depth_maps": {
            "label": "Save 3D depth maps",
            "type": "checkbox",
            "value": False,
            "info": "save animation's depth maps as extra files"
        },
        "video_init_path": {
            "label": "Video init path/ URL",
            "type": "textbox",
            "value": 'https://deforum.github.io/a1/V1.mp4',
            "info": ""
        },
        "extract_nth_frame": {
            "label": "Extract nth frame",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": ""
        },
        "extract_from_frame": {
            "label": "Extract from frame",
            "type": "number",
            "precision": 0,
            "value": 0,
            "info": ""
        },
        "extract_to_frame": {
            "label": "Extract to frame",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": ""
        },
        "overwrite_extracted_frames": {
            "label": "Overwrite extracted frames",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "use_mask_video": {
            "label": "Use mask video",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "video_mask_path": {
            "label": "Video mask path",
            "type": "textbox",
            "value": 'https://deforum.github.io/a1/VM1.mp4',
            "info": ""
        },
        "hybrid_comp_alpha_schedule": {
            "label": "Comp alpha schedule",
            "type": "textbox",
            "value": "0:(0.5)",
            "info": ""
        },
        "hybrid_comp_mask_blend_alpha_schedule": {
            "label": "Comp mask blend alpha schedule",
            "type": "textbox",
            "value": "0:(0.5)",
            "info": ""
        },
        "hybrid_comp_mask_contrast_schedule": {
            "label": "Comp mask contrast schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": ""
        },
        "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": {
            "label": "Comp mask auto contrast cutoff high schedule",
            "type": "textbox",
            "value": "0:(100)",
            "info": ""
        },
        "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": {
            "label": "Comp mask auto contrast cutoff low schedule",
            "type": "textbox",
            "value": "0:(0)",
            "info": ""
        },
        "hybrid_flow_factor_schedule": {
            "label": "Flow factor schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": ""
        },
        "hybrid_generate_inputframes": {
            "label": "Generate inputframes",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "hybrid_generate_human_masks": {
            "label": "Generate human masks",
            "type": "radio",
            "choices": ['None', 'PNGs', 'Video', 'Both'],
            "value": "None",
            "info": ""
        },
        "hybrid_use_first_frame_as_init_image": {
            "label": "First frame as init image",
            "type": "checkbox",
            "value": True,
            "info": "",
            "visible": False
        },
        "hybrid_motion": {
            "label": "Hybrid motion",
            "type": "radio",
            "choices": ['None', 'Optical Flow', 'Perspective', 'Affine'],
            "value": "None",
            "info": ""
        },
        "hybrid_motion_use_prev_img": {
            "label": "Motion use prev img",
            "type": "checkbox",
            "value": False,
            "info": "",
            "visible": False
        },
        "hybrid_flow_consistency": {
            "label": "Flow consistency mask",
            "type": "checkbox",
            "value": False,
            "info": "",
            "visible": False
        },
        "hybrid_consistency_blur": {
            "label": "Consistency mask blur",
            "type": "slider",
            "minimum": 0,
            "maximum": 16,
            "step": 1,
            "value": 2,
            "visible": False
        },
        "hybrid_flow_method": {
            "label": "Flow method",
            "type": "radio",
            "choices": ['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'],
            "value": "RAFT",
            "info": "",
            "visible": False
        },
        "hybrid_composite": 'None',  # ['None', 'Normal', 'Before Motion', 'After Generation']
        "hybrid_use_init_image": {
            "label": "Use init image as video",
            "type": "checkbox",
            "value": False,
            "info": "",
        },
        "hybrid_comp_mask_type": {
            "label": "Comp mask type",
            "type": "radio",
            "choices": ['None', 'Depth', 'Video Depth', 'Blend', 'Difference'],
            "value": "None",
            "info": "",
            "visible": False
        },
        "hybrid_comp_mask_inverse": False,
        "hybrid_comp_mask_equalize": {
            "label": "Comp mask equalize",
            "type": "radio",
            "choices": ['None', 'Before', 'After', 'Both'],
            "value": "None",
            "info": "",
        },
        "hybrid_comp_mask_auto_contrast": False,
        "hybrid_comp_save_extra_frames": False
    }


def DeforumArgs():
    return {
        "W": {
            "label": "Width",
            "type": "slider",
            "value": 1024,
            "minimum": 128,
            "maximum": 2048,
            "step": 64,
            "info": "width in pixels. Use bigger sizes for Flux. Recommended = 1024"
        },
        "H": {
            "label": "Height", 
            "type": "slider",
            "value": 1024,
            "minimum": 128,
            "maximum": 2048,
            "step": 64,
            "info": "height in pixels. Use bigger sizes for Flux. Recommended = 1024"
        },
        "show_info_on_ui": {
            "type": "checkbox",
            "value": False,
            "label": "Show more info",
            "info": "adds additional explanatory text to various parameters"},
        "show_controlnet_tab": {
            "type": "checkbox",
            "value": False,
            "label": "Show ControlNet tab",
            "info": "Shows the ControlNet tab (experimental - not fully working yet)"
        },
        "tiling": {
            "label": "Tiling",
            "type": "checkbox",
            "value": False,
            "info": "enable for seamless-tiling of each generated image. Experimental"
        },
        "restore_faces": {
            "label": "Restore faces",
            "type": "checkbox",
            "value": False,
            "info": "enable to trigger webui's face restoration on each frame during the generation"
        },
        "seed_resize_from_w": {
            "label": "Resize seed from width",
            "type": "slider",
            "minimum": 0,
            "maximum": 2048,
            "step": 64,
            "value": 0,
        },
        "seed_resize_from_h": {
            "label": "Resize seed from height",
            "type": "slider",
            "minimum": 0,
            "maximum": 2048,
            "step": 64,
            "value": 0,
        },
        "seed": {
            "label": "Seed",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "Starting seed for the animation. -1 for random"
        },
        "sampler": {
            "label": "Sampler",
            "type": "dropdown",
            "choices": get_samplers_list().values(),
            "value": "Euler",
        },
        "scheduler": {
            "label": "Scheduler",
            "type": "dropdown",
            "choices": get_schedulers_list().values(),
            "value": "Simple",
        },
        "steps": {
            "label": "Steps",
            "type": "slider",
            "minimum": 1,
            "maximum": 200,
            "step": 1,
            "value": 20,
        },
        "batch_name": {
            "label": "Batch name",
            "type": "textbox",
            "value": "Deforum_{timestring}",
            "info": "output images will be placed in a folder with this name ({timestring} token will be replaced) inside the img2img output folder. Supports params placeholders. e.g {seed}, {w}, {h}, {prompts}"
        },
        "seed_behavior": {
            "label": "Seed behavior",
            "type": "radio",
            "choices": ['iter', 'fixed', 'random', 'ladder', 'alternate', 'schedule'],
            "value": "iter",
            "info": "controls the seed behavior that is used for animation. Hover on the options to see more info"
        },
        "seed_iter_N": {
            "label": "Seed iter N",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "for how many frames the same seed should stick before iterating to the next one"
        },
        "use_init": {
            "label": "Use init",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "strength": {
            "label": "Strength",
            "type": "slider",
            "minimum": 0,
            "maximum": 1,
            "step": 0.01,
            "value": 0.85,
            "info": "the inverse of denoise; lower values alter the init image more (high denoise); higher values alter it less (low denoise)"
        },
        "strength_0_no_init": {
            "label": "Strength 0 no init",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "init_image": {
            "label": "Init image URL",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/I1.png",
            "info": "Use web address or local path. Note: if the image box below is used then this field is ignored."
        },
        "init_image_box": {
            "label": "Init image box",
            "type": "image",
            "type_param": "pil",
            "source": "upload",
            "interactive": True,
            "info": ""
        },
        "use_mask": {
            "label": "Use mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "use_alpha_as_mask": {
            "label": "Use alpha as mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_file": {
            "label": "Mask file",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/M1.jpg",
            "info": ""
        },
        "invert_mask": {
            "label": "Invert mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_contrast_adjust": {
            "label": "Mask contrast adjust",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": ""
        },
        "mask_brightness_adjust": {
            "label": "Mask brightness adjust",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": ""
        },
        "overlay_mask": {
            "label": "Overlay mask",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "mask_overlay_blur": {
            "label": "Mask overlay blur",
            "type": "slider",
            "minimum": 0,
            "maximum": 64,
            "step": 1,
            "value": 4,
        },
        "fill": {
            "label": "Mask fill",
            "type": "radio",
            "type_param": "index",
            "choices": ['fill', 'original', 'latent noise', 'latent nothing'],
            "value": 'original',
            "info": ""
        },
        "full_res_mask": {
            "label": "Full res mask",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "full_res_mask_padding": {
            "label": "Full res mask padding",
            "type": "slider",
            "minimum": 0,
            "maximum": 512,
            "step": 1,
            "value": 4,
        },
        "reroll_blank_frames": {
            "label": "Reroll blank frames",
            "type": "radio",
            "choices": ['reroll', 'interrupt', 'ignore'],
            "value": "ignore",
            "info": ""
        },
        "reroll_patience": {
            "label": "Reroll patience",
            "type": "number",
            "precision": None,
            "value": 10,
            "info": ""
        },
        "motion_preview_mode": {
            "label": "Motion preview mode (dry run).",
            "type": "checkbox",
            "value": False,
            "info": "Preview motion only. Uses a static picture for init, and draw motion reference rectangle."
        },
    }


def LoopArgs():
    return {
        "use_looper": {
            "label": "Enable guided images mode",
            "type": "checkbox",
            "value": False,
        },
        "init_images": {
            "label": "Images to use for keyframe guidance",
            "type": "textbox",
            "lines": 9,
            "value": get_guided_imgs_default_json(),
        },
        "image_strength_schedule": {
            "label": "Image strength schedule",
            "type": "textbox",
            "value": "0:(0.85)",
        },
        "image_keyframe_strength_schedule": {
            "label": "Image strength schedule",
            "type": "textbox",
            "value": "0:(0.20)",
        },
        "blendFactorMax": {
            "label": "Blend factor max",
            "type": "textbox",
            "value": "0:(0.35)",
        },
        "blendFactorSlope": {
            "label": "Blend factor slope",
            "type": "textbox",
            "value": "0:(0.25)",
        },
        "tweening_frames_schedule": {
            "label": "Tweening frames schedule",
            "type": "textbox",
            "value": "0:(20)",
        },
        "color_correction_factor": {
            "label": "Color correction factor",
            "type": "textbox",
            "value": "0:(0.075)",
        }
    }


def ParseqArgs():
    return {
        "parseq_manifest": {
            "label": "Parseq manifest (JSON or URL)",
            "type": "textbox",
            "lines": 4,
            "value": None,
        },
        "parseq_use_deltas": {
            "label": "Use delta values for movement parameters (recommended)",
            "type": "checkbox",
            "value": True,
            "info": "Recommended. If you uncheck this, Parseq keyframe values as are treated as relative movement values instead of absolute."
        },
        "parseq_non_schedule_overrides": {
            "label": "Use FPS, max_frames and cadence from the Parseq manifest, if present (recommended)",
            "type": "checkbox",
            "value": True,
            "info": "Recommended. If you uncheck this, the FPS, max_frames and cadence in the Parseq doc are ignored, and the values in the A1111 UI are used instead."
        },
    }


def WanArgs():
    """Wan 2.1 Video Generation Arguments"""
    return {
        # Core Settings
        "wan_mode": {
            "label": "Wan Mode",
            "type": "radio",
            "choices": ["Disabled", "T2V Only", "I2V Chaining"],
            "value": "Disabled",
            "info": "Choose generation mode: T2V Only for independent clips, I2V Chaining for seamless transitions"
        },
        "wan_model_path": {
            "label": "Model Path",
            "type": "textbox",
            "lines": 1,
            "value": "models/wan",
            "info": "Directory containing Wan models (auto-discovered)"
        },
        "wan_model_name": {
            "label": "Model Selection",
            "type": "dropdown",
            "choices": ["Auto-Select"],
            "value": "Auto-Select",
            "info": "Wan model to use (auto-populated from model directory)"
        },
        "wan_enable_prompt_enhancement": {
            "label": "Enable Prompt Enhancement",
            "type": "checkbox",
            "value": False,
            "info": "Use QwenPromptExpander to automatically enhance prompts for better video quality"
        },
        "wan_qwen_model": {
            "label": "Qwen Model",
            "type": "dropdown",
            "choices": ["Auto-Select", "QwenVL2.5_3B", "QwenVL2.5_7B", "Qwen2.5_3B", "Qwen2.5_7B", "Qwen2.5_14B"],
            "value": "Auto-Select",
            "info": "Qwen model for prompt enhancement (auto-selected based on VRAM)"
        },
        "wan_enable_movement_analysis": {
            "label": "Enable Movement Analysis",
            "type": "checkbox",
            "value": True,
            "info": "Analyze Deforum movement schedules and add to prompts"
        },
        "wan_movement_sensitivity": {
            "label": "Movement Sensitivity",
            "type": "slider",
            "minimum": 0.1,
            "maximum": 5.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Sensitivity for detecting camera movements"
        },
        "wan_style_prompt": {
            "label": "Global Style Prompt",
            "type": "textbox",
            "lines": 2,
            "value": "",
            "info": "Style elements to add to all prompts (e.g., 'cinematic, high quality')"
        },
        "wan_style_strength": {
            "label": "Style Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.1,
            "value": 0.5,
            "info": "How strongly to apply style enhancements"
        },
        "wan_i2v_strength": {
            "label": "I2V Strength Override",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.05,
            "value": 0.8,
            "info": "Override I2V strength for better continuity (leave at 0.8 for automatic)"
        }
    }


def DeforumOutputArgs():
    return {
        "skip_video_creation": {
            "label": "Skip video creation",
            "type": "checkbox",
            "value": False,
            "info": "If enabled, only images will be saved"
        },
        "fps": {
            "label": "FPS",
            "type": "slider",
            "minimum": 1,
            "maximum": 240,
            "step": 1,
            "value": 60,
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
        },
        "delete_input_frames": {
            "label": "Delete All Inputframes",
            "type": "checkbox",
            "value": False,
            "info": "auto-delete inputframes (incl CN ones) when video is ready"
        },
        "image_path": {
            "label": "Image path",
            "type": "textbox",
            "value": "C:/SD/20241111111111_%09d.png",
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
        "r_upscale_video": {
            "label": "Upscale",
            "type": "checkbox",
            "value": False,
            "info": "upscale output imgs when run is finished"
        },
        "r_upscale_factor": {
            "label": "Upscale factor",
            "type": "dropdown",
            "choices": ['x2', 'x3', 'x4'],
            "value": "x2",
        },
        "r_upscale_model": {
            "label": "Upscale model",
            "type": "dropdown",
            "choices": ['realesr-animevideov3', 'realesrgan-x4plus', 'realesrgan-x4plus-anime'],
            "value": 'realesr-animevideov3',
        },
        "r_upscale_keep_imgs": {
            "label": "Keep Imgs",
            "type": "checkbox",
            "value": True,
            "info": "don't delete upscaled imgs",
        },
        "store_frames_in_ram": {
            "label": "Store frames in ram",
            "type": "checkbox",
            "value": False,
            "info": "auto-delete imgs when video is ready",
            "visible": False
        },
        "frame_interpolation_engine": {
            "label": "Engine",
            "type": "radio",
            "choices": ['None', 'RIFE v4.6', 'FILM'],
            "value": "None",
            "info": "select the frame interpolation engine. hover on the options for more info"
        },
        "frame_interpolation_x_amount": {
            "label": "Interp X",
            "type": "slider",
            "minimum": 2,
            "maximum": 10,
            "step": 1,
            "value": 2,
        },
        "frame_interpolation_slow_mo_enabled": {
            "label": "Slow-Mo",
            "type": "checkbox",
            "value": False,
            "visible": False,
            "info": "Slow-Mo the interpolated video, audio will not be used if enabled",
        },
        "frame_interpolation_slow_mo_amount": {
            "label": "Slow-Mo X",
            "type": "slider",
            "minimum": 2,
            "maximum": 10,
            "step": 1,
            "value": 2,
        },
        "frame_interpolation_keep_imgs": {
            "label": "Keep Imgs",
            "type": "checkbox",
            "value": False,
            "info": "Keep interpolated images on disk",
            "visible": False
        },
        "frame_interpolation_use_upscaled": {
            "label": "Use Upscaled",
            "type": "checkbox",
            "value": False,
            "info": "Interpolate upscaled images, if available",
            "visible": False
        },
    }


def get_component_names():
    return ['override_settings_with_file', 'custom_settings_file', *DeforumAnimArgs().keys(), 'animation_prompts',
            'animation_prompts_positive', 'animation_prompts_negative',
            *DeforumArgs().keys(), *DeforumOutputArgs().keys(), *ParseqArgs().keys(), *LoopArgs().keys(),
            *controlnet_component_names(), *WanArgs().keys()]


def get_settings_component_names():
    return [name for name in get_component_names()]


def pack_args(args_dict, keys_function):
    return {name: args_dict[name] for name in keys_function()}


def process_args(args_dict_main, run_id):
    from .settings import load_args
    override_settings_with_file = args_dict_main['override_settings_with_file']
    custom_settings_file = args_dict_main['custom_settings_file']
    p = args_dict_main['p']

    root = SimpleNamespace(**RootArgs())
    args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumArgs()})
    anim_args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumAnimArgs()})
    video_args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumOutputArgs()})
    parseq_args = SimpleNamespace(**{name: args_dict_main[name] for name in ParseqArgs()})
    loop_args = SimpleNamespace(**{name: args_dict_main[name] for name in LoopArgs()})
    controlnet_args = SimpleNamespace(**{name: args_dict_main[name] for name in controlnet_component_names()})

    root.animation_prompts = json.loads(args_dict_main['animation_prompts'])

    args_loaded_ok = True
    if override_settings_with_file:
        args_loaded_ok = load_args(args_dict_main, args, anim_args, parseq_args, loop_args, controlnet_args, custom_settings_file, root, run_id)

    positive_prompts = args_dict_main['animation_prompts_positive']
    negative_prompts = args_dict_main['animation_prompts_negative']
    negative_prompts = negative_prompts.replace('--neg',
                                                '')  # remove --neg from negative_prompts if received by mistake
    root.prompt_keyframes = [key for key in root.animation_prompts.keys()]
    root.animation_prompts = {key: f"{positive_prompts} {val} {'' if '--neg' in val else '--neg'} {negative_prompts}"
                              for key, val in root.animation_prompts.items()}

    if args.seed == -1:
        root.raw_seed = -1
    args.seed = get_fixed_seed(args.seed)
    if root.raw_seed != -1:
        root.raw_seed = args.seed
    root.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))
    args.prompts = json.loads(args_dict_main['animation_prompts'])
    args.positive_prompts = args_dict_main['animation_prompts_positive']
    args.negative_prompts = args_dict_main['animation_prompts_negative']

    if not args.use_init and not anim_args.hybrid_use_init_image:
        args.init_image = None
        args.init_image_box = None

    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True

    additional_substitutions = SimpleNamespace(date=time.strftime('%Y%m%d'), time=time.strftime('%H%M%S'))
    current_arg_list = [args, anim_args, video_args, parseq_args, root, additional_substitutions]
    full_base_folder_path = os.path.join(os.getcwd(), p.outpath_samples)
    root.raw_batch_name = args.batch_name
    args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
    args.outdir = os.path.join(p.outpath_samples, str(args.batch_name))
    args.outdir = os.path.join(os.getcwd(), args.outdir)
    args.outdir = os.path.realpath(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    default_img = Image.open(os.path.join(pathlib.Path(__file__).parent.absolute(), '114763196.jpg'))
    assert default_img is not None
    default_img = default_img.resize((args.W, args.H))
    root.default_img = default_img

    return args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args
