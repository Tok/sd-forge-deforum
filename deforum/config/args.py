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
from deforum.integrations.controlnet.legacy_controlnet_stubs import controlnet_component_names
from deforum.utils.general import get_os, substitute_placeholders


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
        "render_mode": {
            "label": "Render Mode",
            "type": "radio",
            "choices": ['Classic 3D', 'New 3D', 'Keyframes Only', 'Flux + Interpolation'],
            "value": "Keyframes Only",
            "info": "Primary workflow selector: Classic 3D (fixed cadence, RAFT/ControlNet), New 3D (keyframe redistribution, dual strength), Keyframes Only (depth tweening), Flux + Interpolation (multi-method AI interpolation)"
        },
        "animation_mode": {
            "label": "Animation mode (Legacy - use Render Mode instead)",
            "type": "radio",
            "choices": ['3D', 'Interpolation', 'Flux + Interpolation'],
            "value": "3D",
            "info": "[DEPRECATED] Use Render Mode instead. This field is maintained for backward compatibility."
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
            "label": "Strength schedule (normal/tween frames)",
            "type": "textbox",
            "value": "0: (0.2)",
            "info": "Amount of presence of previous frame to influence next frame for TWEEN frames (non-keyframes). Controls steps: [steps - (strength * steps)]. New 3D default: 0.2. Should be LOWER than keyframe strength."
        },
        "keyframe_strength_schedule": {
            "label": "Strength schedule (keyframes)",
            "type": "textbox",
            "value": "0: (0.85)",
            "info": "Like 'Strength schedule' but only for frames with an entry in 'prompts' (keyframes). Should be HIGHER than normal strength. Keyframes Only/New 3D default: 0.85. At 0 it generates a totally new image on every prompt change. Ignored if Parseq is used or when Classic 3D mode active."
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
            "choices": ['None', 'HSV', 'LAB', 'RGB', 'Image'],
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
        "enable_wan_flf2v": {
            "label": "Enable Wan FLF2V for Tweens",
            "type": "checkbox",
            "value": False,
            "info": "Use Wan AI video interpolation (FLF2V) instead of depth-based tweening. Requires Wan FLF2V model. Auto-chains for long sections."
        },
        "wan_flf2v_chunk_size": {
            "label": "Wan FLF2V Chunk Size",
            "type": "slider",
            "minimum": 21,
            "maximum": 161,
            "step": 20,
            "value": 81,
            "info": "Max frames per FLF2V call. For longer sections, generates depth-tween intermediate keyframes and chains FLF2V between them. Must be 4n+1 format (21, 41, 61, 81, 101, 121, 141, 161)."
        },
        "keyframe_type_schedule": {
            "label": "Keyframe Type Schedule",
            "type": "textbox",
            "value": "0:(tween)",
            "info": "Per-keyframe tween interpolation method: 'tween' (depth-based), 'flf2v' (Wan FLF2V), or 'auto' (decide based on tween count). Example: 0:(tween), 60:(flf2v), 120:(auto)"
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
            "choices": ['None', 'RAFT'],
            "value": "None",
            "info": "Use optical flow estimation for in-between (cadence) frames. ⚠️ WARNING: Can produce 'smear-core' artifacts with many cadence frames. Works best with low cadence (2-3 frames). Experimental feature - use with caution."
        },
        "raft_model_size": {
            "label": "RAFT model size",
            "type": "dropdown",
            "choices": ['Large', 'Small'],
            "value": "Small",
            "info": "RAFT model size - Large (best quality, slower) or Small (faster, good quality)"
        },
        "raft_flow_iterations": {
            "label": "RAFT flow iterations",
            "type": "slider",
            "minimum": 6,
            "maximum": 50,
            "step": 1,
            "value": 12,
            "info": "Number of flow refinement iterations - Higher values are more accurate but slower (12 is default, 20-30 for best quality)"
        },
        "show_flow_arrows": {
            "label": "Show flow arrows on depth preview",
            "type": "checkbox",
            "value": True,
            "info": "Draw green arrows on depth maps showing optical flow motion vectors (helps visualize tween movement)"
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
            "choices": ['None', 'RAFT'],
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
            "choices": ['Depth-Anything-V2-Small', 'Depth-Anything-V2-Base', 'Depth-Anything-V2-Large'],
            "value": "Depth-Anything-V2-Small",
            "info": "Depth Anything V2 model size - Small (fastest), Base (balanced), Large (best quality)"
        },
        "midas_weight": {
            "label": "Depth weight (legacy, not used)",
            "type": "number",
            "precision": None,
            "value": 0.2,
            "info": "Legacy parameter from old depth models (MiDaS/Zoe), no longer used with Depth-Anything V2",
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
        "enable_flux_controlnet": {
            "label": "Enable Flux ControlNet",
            "type": "checkbox",
            "value": False,
            "info": "Use Flux ControlNet for keyframe generation (Canny or Depth control)"
        },
        "flux_controlnet_type": {
            "label": "ControlNet Type",
            "type": "dropdown",
            "choices": ['canny', 'depth'],
            "value": "canny",
            "info": "Canny: edge detection from previous frame, Depth: use Depth-Anything V2 depth maps"
        },
        "flux_controlnet_model": {
            "label": "ControlNet Model",
            "type": "dropdown",
            "choices": ['instantx', 'xlabs', 'bfl', 'shakker'],
            "value": "instantx",
            "info": "Model provider: InstantX (default), XLabs-AI, BFL (official), Shakker Labs (Depth recommended)"
        },
        "flux_controlnet_strength": {
            "label": "ControlNet Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.05,
            "value": 0.7,
            "info": "How strongly ControlNet influences generation (0.0 = no influence, 1.0 = full control)"
        },
        "flux_controlnet_canny_low": {
            "label": "Canny Low Threshold",
            "type": "slider",
            "minimum": 0,
            "maximum": 255,
            "step": 5,
            "value": 100,
            "info": "Canny edge detection low threshold (lower = more edges)"
        },
        "flux_controlnet_canny_high": {
            "label": "Canny High Threshold",
            "type": "slider",
            "minimum": 0,
            "maximum": 255,
            "step": 5,
            "value": 200,
            "info": "Canny edge detection high threshold (higher = fewer edges)"
        },
        "flux_guidance_scale": {
            "label": "Flux Guidance Scale",
            "type": "slider",
            "minimum": 1.0,
            "maximum": 10.0,
            "step": 0.5,
            "value": 3.5,
            "info": "Guidance scale for Flux generation (3.5 recommended)"
        },
        "flux_base_model": {
            "label": "Flux Base Model",
            "type": "textbox",
            "value": "black-forest-labs/FLUX.1-dev",
            "info": "HuggingFace model ID for Flux base - requires 'huggingface-cli login' and accepting FLUX.1-dev license at https://huggingface.co/black-forest-labs/FLUX.1-dev (TEMP: Will integrate with Forge's loaded model in future)"
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
    }


def DeforumArgs():
    return {
        "W": {
            "label": "Width",
            "type": "slider",
            "minimum": 64,
            "maximum": 2048,
            "step": 4,
            "value": 1280,
        },
        "H": {
            "label": "Height",
            "type": "slider",
            "minimum": 64,
            "maximum": 2048,
            "step": 4,
            "value": 720,
        },
        "show_info_on_ui": True,
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
            "info": "output images will be placed in a folder with this name ({timestring} token will be replaced) inside the outputs/deforum folder. Supports params placeholders. e.g {seed}, {w}, {h}, {prompts}"
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
            "value": "",
            "info": "Path or URL to mask image"
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


def AudioSyncArgs():
    """Audio event detection and prompt synchronization arguments"""
    return {
        "enable_audio_sync": {
            "label": "Enable Audio Sync",
            "type": "checkbox",
            "value": False,
            "info": "Enable audio event detection for prompt synchronization. Disabled when Parseq is active (Parseq has its own audio features)."
        },
        "audio_detection_method": {
            "label": "Detection Method",
            "type": "dropdown",
            "choices": ["onset", "beat", "bass"],
            "value": "onset",
            "info": "Event detection method: 'onset' (general transients/kicks/snares), 'beat' (rhythmic pulse/BPM), 'bass' (low-frequency energy peaks)"
        },
        "audio_frequency_band": {
            "label": "Frequency Band",
            "type": "dropdown",
            "choices": ["bass", "mid", "high", "full"],
            "value": "bass",
            "info": "Frequency range to isolate: 'bass' (20-250Hz for kicks), 'mid' (250-2000Hz for snares/claps), 'high' (2000-8000Hz for hi-hats), 'full' (no filtering)"
        },
        "audio_lowpass_cutoff": {
            "label": "Lowpass Cutoff (Hz)",
            "type": "slider",
            "minimum": 20,
            "maximum": 2000,
            "step": 10,
            "value": 250,
            "info": "Lowpass filter cutoff frequency in Hz. Used when frequency_band='bass'. Typical bass kicks: 60-250Hz"
        },
        "audio_distortion_gain": {
            "label": "Distortion Gain",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.1,
            "value": 0.0,
            "info": "Distortion amount to emphasize transients (0.0=none, 1.0=moderate, 2.0=heavy). Helps isolate bass kicks and percussive hits"
        },
        "audio_distortion_type": {
            "label": "Distortion Type",
            "type": "dropdown",
            "choices": ["tanh", "hard", "arctan"],
            "value": "tanh",
            "info": "Distortion curve: 'tanh' (smooth soft clipping, musical), 'hard' (aggressive clipping), 'arctan' (gentle compression)"
        },
        "audio_sensitivity": {
            "label": "Detection Sensitivity",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.05,
            "value": 0.5,
            "info": "Event detection sensitivity (0.0-1.0). Higher = more sensitive, detects weaker events. Start with 0.5 and adjust"
        },
        "audio_intensity_threshold": {
            "label": "Intensity Threshold",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.05,
            "value": 0.5,
            "info": "Minimum event intensity to generate keyframe (0.0-1.0). Filters out weak events. Lower = more keyframes"
        },
        "audio_min_spacing_frames": {
            "label": "Minimum Keyframe Spacing (frames)",
            "type": "number",
            "precision": 0,
            "value": 12,
            "info": "Minimum frames between keyframes. Prevents keyframe spam. Example: 12 frames @ 24fps = 0.5 seconds. When events are closer, strongest event is kept"
        },
        "audio_apply_to_prompts": {
            "label": "Apply Events to Prompts",
            "type": "checkbox",
            "value": True,
            "info": "Generate prompt keyframes from audio events. When disabled, events can still be visualized and analyzed"
        },
    }


def WanArgs():
    """Wan 2.1 video generation arguments - Updated to integrate with Deforum schedules"""
    return {
        "wan_t2v_model": {
            "label": "Wan 2.2 Model Selection",
            "type": "dropdown",
            "choices": ["Auto-Detect (Recommended)", "TI2V-5B", "TI2V-A14B", "Custom Path"],
            "value": "TI2V-5B",
            "info": "Wan 2.2 unified T2V+I2V model. Auto-Detect: automatically selects best available model. TI2V-5B: 16GB+ VRAM with CPU offload, 720p@24fps. TI2V-A14B: 32GB+ VRAM, MoE architecture, highest quality."
        },
        "wan_auto_download": {
            "label": "Auto-Download Models",
            "type": "checkbox",
            "value": True,
            "info": "Automatically download missing models from HuggingFace (recommended for first-time setup)"
        },
        "wan_model_path": {
            "label": "Custom Model Path",
            "type": "textbox", 
            "value": "models/Deforum/wan",
            "info": "Custom path to Wan model (used when 'Custom Path' is selected)"
        },
        "wan_resolution": {
            "label": "Wan Resolution ⚠️ Set FPS to 24 in Output tab!",
            "type": "dropdown",
            "choices": ["1280x736 (Landscape, 720p)", "736x1280 (Portrait, 720p)", "1024x1024 (Square, 1K)", "1280x704 (Landscape, Letterbox)"],
            "value": "1280x736 (Landscape, 720p)",
            "info": "Wan 2.2 TI2V models require resolutions divisible by 32 (VAE=16x * Patch=2x). These are optimized for 720p.\n\n⚠️ IMPORTANT: Wan 2.2 is trained at 24 FPS! Set FPS to 24 in the Output tab for best results. Deforum defaults to 60 FPS which may cause temporal artifacts. Since frames are saved individually, you can adjust FPS during video compilation."
        },
        "wan_seed": {
            "label": "Wan Seed",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "Seed for Wan generation. -1 for random, 0+ for fixed seed"
        },
        "wan_inference_steps": {
            "label": "Inference Steps",
            "type": "slider",
            "minimum": 5,
            "maximum": 100,
            "step": 1,
            "value": 30,
            "info": "Number of inference steps for Wan generation. Lower values (5-15) for quick testing, higher values (30-50) for quality"
        },
        "wan_strength_override": {
            "label": "I2V Strength Override",
            "type": "checkbox",
            "value": True,
            "info": "Use fixed I2V strength for clip-to-clip continuity. Recommended for Wan 2.2 to maintain consistent visual flow between clips."
        },
        "wan_fixed_strength": {
            "label": "I2V Continuity Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.05,
            "value": 0.5,
            "info": "Controls how strongly each clip continues from the last frame. 0.5-0.7: balanced (recommended for prompt changes), 0.85-0.95: strong continuity, 0.0-0.3: creative variation. Lower values = more prompt freedom."
        },
        "wan_guidance_override": {
            "label": "Guidance Scale Override",
            "type": "checkbox",
            "value": True,
            "info": "Override Deforum CFG scale schedule with fixed value (recommended for consistent Wan generation)"
        },
        "wan_guidance_scale": {
            "label": "Fixed Guidance Scale (CFG)",
            "type": "slider",
            "minimum": 1.0,
            "maximum": 20.0,
            "step": 0.5,
            "value": 3.5,
            "info": "Wan 2.2 recommended CFG: 3.5 (higher values like 6+ tend to 'cook' outputs). Range 2.5-5.0 works best. Only used when Guidance Scale Override is enabled."
        },
        
          # FLF2V Interpolation Settings
        "wan_flf2v_guidance_scale": {
            "label": "FLF2V Guidance Scale",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 7.0,
            "step": 0.1,
            "value": 3.5,
            "info": "Guidance scale for FLF2V interpolation. Recommended: 3.5 (default) for smooth visual morphing throughout the clip. Higher values (5.5+) may cause 'mode collapse' where frames stick to the first image with sudden transition at the end. NEVER use 0.0 (breaks last_image conditioning). Range: 2.5-7.0."
        },
        "wan_flf2v_prompt_mode": {
            "label": "FLF2V Prompt Mode",
            "type": "dropdown",
            "choices": ["none", "first", "last", "blend"],
            "value": "blend",
            "info": "How to use prompts for FLF2V interpolation: 'blend' (RECOMMENDED - combine both prompts describing transition), 'last' (use end keyframe prompt as target), 'first' (use start keyframe prompt), 'none' (empty prompt, may not interpolate correctly)"
        },
        "flux_flf2v_interpolation_method": {
            "label": "FLF2V Interpolation Method",
            "type": "dropdown",
            "choices": ["Wan", "FILM"],
            "value": "Wan",
            "info": "Interpolation method for Flux FLF2V mode: 'Wan' (AI-generated video, recommended for large motion), 'FILM' (smearcore - sharp motion mixing like dragging paint, Google's ML interpolation). Note: RIFE is available in post-processing for framerate doubling."
        },

        # Advanced Generation Settings
        "wan_negative_prompt": {
            "label": "Negative Prompt",
            "type": "textbox",
            "value": "blurry, low quality, distorted, watermark, text, worst quality",
            "lines": 3,
            "info": "Negative prompt for Wan 2.2 generation. Describes what you don't want in the video."
        },
        "wan_sampler": {
            "label": "Sampler",
            "type": "dropdown",
            "choices": ["auto", "uni_pc", "euler", "ddim", "pndm"],
            "value": "auto",
            "info": "Sampling method for Wan 2.2. Recommended: 'auto' (uses uni_pc), or 'euler' for faster generation. uni_pc generally provides best quality."
        },
        "wan_scheduler": {
            "label": "Scheduler",
            "type": "dropdown",
            "choices": ["auto", "simple", "normal", "karras", "exponential"],
            "value": "auto",
            "info": "Noise schedule for Wan 2.2. Recommended: 'auto' (uses simple). 'simple' works best for most cases."
        },

        "wan_frame_overlap": {
            "label": "Frame Overlap",
            "type": "slider",
            "minimum": 0,
            "maximum": 10,
            "step": 1,
            "value": 2,
            "info": "Number of overlapping frames between clips for smoother transitions"
        },
        "wan_motion_strength": {
            "label": "Motion Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Strength of motion in generated videos"
        },
        "wan_enable_interpolation": {
            "label": "Enable Interpolation",
            "type": "checkbox",
            "value": True,
            "info": "Enable frame interpolation between clips"
        },
        "wan_interpolation_strength": {
            "label": "Interpolation Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.1,
            "value": 0.5,
            "info": "Strength of interpolation between consecutive clips"
        },
        "wan_flash_attention_mode": {
            "label": "Flash Attention Mode",
            "type": "dropdown",
            "choices": ["Auto (Recommended)", "Force Flash Attention", "Force PyTorch Fallback"],
            "value": "Auto (Recommended)",
            "info": "Flash Attention mode: Auto tries Flash Attention then falls back to PyTorch. Force options override detection."
        },
        
        # Prompt Enhancement with QwenPromptExpander
        "wan_qwen_model": {
            "label": "Qwen Model",
            "type": "dropdown",
            "choices": ["QwenVL2.5_7B", "QwenVL2.5_3B", "Qwen2.5_14B", "Qwen2.5_7B", "Qwen2.5_3B", "Auto-Select"],
            "value": "Auto-Select",
            "info": "Qwen model for prompt enhancement. VL models support image+text, text-only models are faster. Auto-Select chooses based on available VRAM."
        },
        "wan_qwen_auto_download": {
            "label": "Auto-Download Qwen Models",
            "type": "checkbox",
            "value": True,
            "info": "Automatically download Qwen models if not found locally (recommended for first-time setup)"
        },
        "wan_qwen_language": {
            "label": "Enhanced Prompt Language",
            "type": "dropdown",
            "choices": ["English", "Chinese"],
            "value": "English",
            "info": "Language for enhanced prompts. English works best with most models, Chinese for specialized use cases."
        },
        "wan_enhanced_prompts": {
            "label": "Enhanced Prompts (Editable)",
            "type": "textbox",
            "value": "",
            "lines": 10,
            "info": "Enhanced prompts generated by QwenPromptExpander. You can manually edit these before generation."
        },
        "wan_movement_description": {
            "label": "Movement Description (Auto-Generated)",
            "type": "textbox",
            "value": "",
            "lines": 3,
            "info": "Auto-generated movement description from Deforum schedules (translation, rotation, zoom). Appended to enhanced prompts."
        },
        "wan_movement_sensitivity": {
            "label": "Movement Sensitivity",
            "type": "slider",
            "minimum": 0.1,
            "maximum": 2.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Sensitivity for movement detection. Higher values detect smaller movements, lower values only detect significant motion."
        },
        
        # Advanced Override Settings (moved wan_motion_strength here)
        "wan_motion_strength_override": {
            "label": "Motion Strength Override",
            "type": "checkbox",
            "value": False,
            "info": "Override dynamic motion strength calculation with fixed value"
        },
        "wan_motion_strength": {
            "label": "Fixed Motion Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Fixed motion strength value (only used when Motion Strength Override is enabled). Dynamic calculation from schedules is recommended."
        },

        # VRAM Optimization Settings
        "wan_t5_cpu_offload": {
            "label": "T5 CPU Offload",
            "type": "checkbox",
            "value": False,
            "info": "Keep T5 text encoder on CPU to save ~3-4GB VRAM (slightly slower text encoding). Helpful for 16GB GPUs."
        },
        "wan_gradient_checkpointing": {
            "label": "Gradient Checkpointing",
            "type": "checkbox",
            "value": False,
            "info": "Trade compute speed for VRAM (~15-20% slower, saves ~2-3GB). Enable if you get OOM errors on 16GB GPUs."
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
        "prompt_authored_fps": {
            "label": "Prompt Authored FPS",
            "type": "slider",
            "minimum": 1,
            "maximum": 240,
            "step": 1,
            "value": 0,  # 0 = auto-detect (use current fps)
            "info": "FPS that prompts were originally authored at. Set to 0 for auto (uses current FPS). If prompts were synced to audio at 60 FPS but you want 24 FPS video, set this to 60 and frame numbers will auto-convert."
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
    # TEMPORARILY DISABLED: ControlNet support disabled until Flux-specific reimplementation
    # Re-enable Wan components (UI level, imports still isolated)
    return ['override_settings_with_file', 'custom_settings_file', *DeforumAnimArgs().keys(), 'animation_prompts',
            'animation_prompts_positive', 'animation_prompts_negative',
            *DeforumArgs().keys(), *DeforumOutputArgs().keys(), *ParseqArgs().keys(), *AudioSyncArgs().keys(), *LoopArgs().keys(),
            # *controlnet_component_names(),  # Disabled - ControlNet temporarily removed
            *WanArgs().keys()]


def get_settings_component_names():
    return [name for name in get_component_names()]


def pack_args(args_dict, keys_function):
    # Use .get() to handle missing keys gracefully (e.g., when ControlNet is disabled)
    return {name: args_dict[name] for name in keys_function() if name in args_dict}


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
    audio_sync_args = SimpleNamespace(**{name: args_dict_main[name] for name in AudioSyncArgs()})
    loop_args = SimpleNamespace(**{name: args_dict_main[name] for name in LoopArgs()})
    wan_args = SimpleNamespace(**{name: args_dict_main[name] for name in WanArgs()})
    # TEMPORARILY DISABLED: ControlNet support disabled until Flux-specific reimplementation
    # controlnet_args = SimpleNamespace(**{name: args_dict_main[name] for name in controlnet_component_names()})
    controlnet_args = SimpleNamespace()  # Empty namespace for backwards compatibility

    root.animation_prompts = json.loads(args_dict_main['animation_prompts'])

    # Convert prompt frame numbers if authored at different FPS (like shakify does)
    prompt_authored_fps = video_args.prompt_authored_fps if hasattr(video_args, 'prompt_authored_fps') else 0
    current_fps = video_args.fps

    if prompt_authored_fps > 0 and prompt_authored_fps != current_fps:
        # FPS conversion needed - use shakify formula: new_frame = old_frame * (target_fps / source_fps)
        fps_ratio = current_fps / prompt_authored_fps
        converted_prompts = {}

        print(f"\n🔄 Converting prompt frame numbers from {prompt_authored_fps} FPS → {current_fps} FPS (ratio: {fps_ratio:.4f})")

        for frame_key, prompt_text in root.animation_prompts.items():
            try:
                old_frame = int(frame_key)
                new_frame = int(old_frame * fps_ratio)
                converted_prompts[str(new_frame)] = prompt_text
                print(f"   Frame {old_frame} → {new_frame}")
            except ValueError:
                # Non-numeric key (e.g., "max_f"), keep as-is
                converted_prompts[frame_key] = prompt_text

        root.animation_prompts = converted_prompts
        print(f"✅ Converted {len(converted_prompts)} prompt frame numbers\n")

    args_loaded_ok = True
    if override_settings_with_file:
        args_loaded_ok = load_args(args_dict_main, args, anim_args, parseq_args, audio_sync_args, loop_args, controlnet_args, wan_args, video_args, custom_settings_file, root, run_id)

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

    if not args.use_init:
        args.init_image = None
        args.init_image_box = None

    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True

    additional_substitutions = SimpleNamespace(date=time.strftime('%Y%m%d'), time=time.strftime('%H%M%S'))
    current_arg_list = [args, anim_args, video_args, parseq_args, root, additional_substitutions]

    # Use dedicated Deforum output directory instead of img2img folder
    # Allow override via outdir_samples for test isolation
    # Check opts.data first since that's where options_overrides are set
    if 'outdir_samples' in sh.opts.data and sh.opts.data['outdir_samples']:
        deforum_outpath = sh.opts.data['outdir_samples']
    else:
        deforum_outpath = os.path.join(os.getcwd(), 'outputs', 'deforum')
    full_base_folder_path = deforum_outpath
    root.raw_batch_name = args.batch_name
    args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
    args.outdir = os.path.join(deforum_outpath, str(args.batch_name))
    args.outdir = os.path.realpath(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    # Load Deforum logo from project root (deforum/config/ -> deforum/ -> extension root)
    extension_root = pathlib.Path(__file__).parent.parent.parent
    default_img = Image.open(os.path.join(extension_root, 'deforum-logo.jpg'))
    assert default_img is not None
    default_img = default_img.resize((args.W, args.H))
    root.default_img = default_img

    return args_loaded_ok, root, args, anim_args, video_args, parseq_args, audio_sync_args, loop_args, controlnet_args, wan_args
