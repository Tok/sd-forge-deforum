"""
Argument Defaults - COMPLETE VERSION
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
    """Animation argument defaults with UI component definitions - COMPLETE VERSION."""
    return {
        # CORE ANIMATION SETTINGS
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
        "diffusion_cadence": {
            "label": "Diffusion cadence",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "Frequency of diffusion sampling"
        },
        
        # 2D/3D MOTION PARAMETERS
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
        },
        
        # PERSPECTIVE FLIP CONTROLS
        "enable_perspective_flip": {
            "label": "Enable perspective flip",
            "type": "checkbox",
            "value": False,
            "info": "enable perspective flip animation"
        },
        "perspective_flip_theta": {
            "label": "Perspective flip theta",
            "type": "textbox",
            "value": "0: (0)",
            "info": "perspective flip theta schedule"
        },
        "perspective_flip_phi": {
            "label": "Perspective flip phi",
            "type": "textbox",
            "value": "0: (0)",
            "info": "perspective flip phi schedule"
        },
        "perspective_flip_gamma": {
            "label": "Perspective flip gamma",
            "type": "textbox",
            "value": "0: (0)",
            "info": "perspective flip gamma schedule"
        },
        "perspective_flip_fv": {
            "label": "Perspective flip fv",
            "type": "textbox",
            "value": "0: (53)",
            "info": "perspective flip field of view schedule"
        },
        
        # NOISE SETTINGS
        "noise_type": {
            "label": "Noise type",
            "type": "dropdown",
            "choices": ["perlin", "uniform"],
            "value": "perlin",
            "info": "noise type for animation"
        },
        "noise_schedule": {
            "label": "Noise schedule",
            "type": "textbox",
            "value": "0: (0.02)",
            "info": "amount of noise to add per frame"
        },
        "perlin_octaves": {
            "label": "Perlin octaves",
            "type": "number",
            "precision": 0,
            "value": 4,
            "info": "octaves for perlin noise"
        },
        "perlin_persistence": {
            "label": "Perlin persistence",
            "type": "number",
            "precision": None,
            "value": 0.5,
            "info": "persistence for perlin noise"
        },
        "perlin_w": {
            "label": "Perlin width",
            "type": "number",
            "precision": 0,
            "value": 8,
            "info": "width for perlin noise"
        },
        "perlin_h": {
            "label": "Perlin height",
            "type": "number",
            "precision": 0,
            "value": 8,
            "info": "height for perlin noise"
        },
        
        # COLOR COHERENCE SETTINGS
        "color_coherence": {
            "label": "Color coherence",
            "type": "dropdown",
            "choices": ["None", "Match Frame 0 HSV", "Match Frame 0 LAB", "Match Frame 0 RGB"],
            "value": "None",
            "info": "color coherence mode"
        },
        "color_force_grayscale": {
            "label": "Force grayscale",
            "type": "checkbox",
            "value": False,
            "info": "force grayscale output"
        },
        "color_coherence_image_path": {
            "label": "Color coherence image path",
            "type": "textbox",
            "value": "",
            "info": "path to image for color matching"
        },
        "color_coherence_video_every_N_frames": {
            "label": "Color coherence video every N frames",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "apply color coherence every N frames"
        },
        
        # OPTICAL FLOW SETTINGS
        "optical_flow_cadence": {
            "label": "Optical flow cadence",
            "type": "dropdown",
            "choices": ["None", "1", "2", "3", "4", "5"],
            "value": "None",
            "info": "optical flow cadence"
        },
        "cadence_flow_factor_schedule": {
            "label": "Cadence flow factor schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": "cadence flow factor schedule"
        },
        "optical_flow_redo_generation": {
            "label": "Optical flow redo generation",
            "type": "dropdown",
            "choices": ["None", "1", "2", "3"],
            "value": "None",
            "info": "optical flow redo generation"
        },
        "redo_flow_factor_schedule": {
            "label": "Redo flow factor schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": "redo flow factor schedule"
        },
        "diffusion_redo": {
            "label": "Diffusion redo",
            "type": "number",
            "precision": 0,
            "value": 0,
            "info": "diffusion redo amount"
        },
        
        # SCHEDULING SETTINGS
        "enable_ddim_eta_scheduling": {
            "label": "Enable DDIM eta scheduling",
            "type": "checkbox",
            "value": False,
            "info": "enable DDIM eta scheduling"
        },
        "ddim_eta_schedule": {
            "label": "DDIM eta schedule",
            "type": "textbox",
            "value": "0: (0.0)",
            "info": "DDIM eta schedule"
        },
        "enable_ancestral_eta_scheduling": {
            "label": "Enable ancestral eta scheduling",
            "type": "checkbox",
            "value": False,
            "info": "enable ancestral eta scheduling"
        },
        "ancestral_eta_schedule": {
            "label": "Ancestral eta schedule",
            "type": "textbox",
            "value": "0: (1.0)",
            "info": "ancestral eta schedule"
        },
        "strength_schedule": {
            "label": "Strength schedule",
            "type": "textbox",
            "value": "0: (0.65)",
            "info": "strength schedule for keyframes"
        },
        "cfg_scale_schedule": {
            "label": "CFG scale schedule",
            "type": "textbox",
            "value": "0: (7.0)",
            "info": "CFG scale schedule"
        },
        "distilled_cfg_scale_schedule": {
            "label": "Distilled CFG scale schedule",
            "type": "textbox",
            "value": "0: (7.0)",
            "info": "distilled CFG scale schedule"
        },
        "seed_schedule": {
            "label": "Seed schedule",
            "type": "textbox",
            "value": "0: (-1)",
            "info": "seed schedule"
        },
        "contrast_schedule": {
            "label": "Contrast schedule",
            "type": "textbox",
            "value": "0: (1.0)",
            "info": "contrast schedule"
        },
        "steps_schedule": {
            "label": "Steps schedule",
            "type": "textbox",
            "value": "0: (25)",
            "info": "steps schedule"
        },
        
        # DEPTH WARPING SETTINGS
        "use_depth_warping": {
            "label": "Use depth warping",
            "type": "checkbox",
            "value": True,
            "info": "enable depth warping"
        },
        "depth_algorithm": {
            "label": "Depth algorithm",
            "type": "dropdown",
            "choices": ["Zoe", "MiDaS", "LeReS"],
            "value": "Zoe",
            "info": "depth estimation algorithm"
        },
        "midas_weight": {
            "label": "MiDaS weight",
            "type": "number",
            "precision": None,
            "value": 0.3,
            "info": "MiDaS model weight"
        },
        "aspect_ratio_use_old_formula": {
            "label": "Aspect ratio use old formula",
            "type": "checkbox",
            "value": False,
            "info": "use old aspect ratio formula"
        },
        "aspect_ratio_schedule": {
            "label": "Aspect ratio schedule",
            "type": "textbox",
            "value": "0: (1.0)",
            "info": "aspect ratio schedule"
        },
        
        # VIDEO SETTINGS
        "video_init_path": {
            "label": "Video init path",
            "type": "textbox",
            "value": "",
            "info": "path to input video"
        },
        "extract_from_frame": {
            "label": "Extract from frame",
            "type": "number",
            "precision": 0,
            "value": 0,
            "info": "start frame for video extraction"
        },
        "extract_to_frame": {
            "label": "Extract to frame",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "end frame for video extraction"
        },
        "extract_nth_frame": {
            "label": "Extract nth frame",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "extract every nth frame"
        },
        "overwrite_extracted_frames": {
            "label": "Overwrite extracted frames",
            "type": "checkbox",
            "value": False,
            "info": "overwrite existing extracted frames"
        },
        "use_mask_video": {
            "label": "Use mask video",
            "type": "checkbox",
            "value": False,
            "info": "use video as mask"
        },
        "video_mask_path": {
            "label": "Video mask path",
            "type": "textbox",
            "value": "",
            "info": "path to mask video"
        },
        
        # RESUME SETTINGS
        "resume_from_timestring": {
            "label": "Resume from timestring",
            "type": "checkbox",
            "value": False,
            "info": "resume animation from timestring"
        },
        "resume_timestring": {
            "label": "Resume timestring",
            "type": "textbox",
            "value": "",
            "info": "timestring to resume from"
        },
        
        # MEMORY SETTINGS
        "store_frames_in_ram": {
            "label": "Store frames in RAM",
            "type": "checkbox",
            "value": False,
            "info": "store frames in RAM instead of disk"
        }
    }


def DeforumArgs():
    """Core generation argument defaults with UI component definitions - COMPLETE VERSION."""
    return {
        # BASIC GENERATION SETTINGS
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
        },
        "distilled_cfg_scale": {
            "label": "Distilled CFG Scale",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 30.0,
            "step": 0.5,
            "value": 8.0,
            "info": "Distilled CFG scale"
        },
        "sampler": {
            "label": "Sampler",
            "type": "dropdown",
            "choices": get_sampler_options(),
            "value": "Euler a",
            "info": "Sampling method"
        },
        "scheduler": {
            "label": "Scheduler",
            "type": "dropdown",
            "choices": get_scheduler_options(),
            "value": "Simple",
            "info": "Noise scheduling method"
        },
        "checkpoint": {
            "label": "Checkpoint",
            "type": "dropdown",
            "choices": ["Use Model from WebUI"],
            "value": "Use Model from WebUI",
            "info": "Model checkpoint to use"
        },
        "clip_skip": {
            "label": "Clip skip",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "Number of CLIP layers to skip"
        },
        
        # IMAGE SETTINGS
        "tiling": {
            "label": "Tiling",
            "type": "checkbox",
            "value": False,
            "info": "Enable tiling"
        },
        "restore_faces": {
            "label": "Restore faces",
            "type": "checkbox",
            "value": False,
            "info": "Apply face restoration"
        },
        
        # INITIALIZATION SETTINGS
        "use_init": {
            "label": "Use init image",
            "type": "checkbox",
            "value": False,
            "info": "Use an initialization image"
        },
        "init_image": {
            "label": "Init image",
            "type": "textbox",
            "value": "",
            "info": "Path to initialization image"
        },
        "strength": {
            "label": "Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01,
            "value": 0.65,
            "info": "How much to change the init image"
        },
        "strength_0_no_init": {
            "label": "Strength 0 no init",
            "type": "checkbox",
            "value": True,
            "info": "Use no init image when strength is 0"
        },
        "init_scale": {
            "label": "Init scale",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": "Scale factor for init image"
        },
        
        # MASK SETTINGS
        "use_mask": {
            "label": "Use mask",
            "type": "checkbox",
            "value": False,
            "info": "Use a mask for selective generation"
        },
        "use_alpha_as_mask": {
            "label": "Use alpha as mask",
            "type": "checkbox",
            "value": False,
            "info": "Use alpha channel as mask"
        },
        "invert_mask": {
            "label": "Invert mask",
            "type": "checkbox",
            "value": False,
            "info": "Invert the mask"
        },
        "overlay_mask": {
            "label": "Overlay mask",
            "type": "checkbox",
            "value": True,
            "info": "Overlay mask on image"
        },
        "mask_file": {
            "label": "Mask file",
            "type": "textbox",
            "value": "",
            "info": "Path to mask file"
        },
        "mask_brightness_adjust": {
            "label": "Mask brightness adjust",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": "Adjust mask brightness"
        },
        "mask_contrast_adjust": {
            "label": "Mask contrast adjust",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": "Adjust mask contrast"
        },
        "mask_overlay_blur": {
            "label": "Mask overlay blur",
            "type": "slider",
            "minimum": 0,
            "maximum": 64,
            "step": 1,
            "value": 4,
            "info": "Blur amount for mask overlay"
        },
        "fill": {
            "label": "Fill",
            "type": "radio",
            "choices": ["stretch", "fit", "crop"],
            "value": "stretch",
            "info": "How to handle image size differences"
        },
        "full_res_mask": {
            "label": "Full res mask",
            "type": "checkbox",
            "value": True,
            "info": "Use full resolution for masking"
        },
        "full_res_mask_padding": {
            "label": "Full res mask padding",
            "type": "slider",
            "minimum": 0,
            "maximum": 512,
            "step": 4,
            "value": 32,
            "info": "Padding for full resolution mask"
        },
        
        # PROMPT SETTINGS
        "prompt": {
            "label": "Prompt",
            "type": "textbox",
            "value": "",
            "info": "Text prompt for generation"
        },
        "negative_prompt": {
            "label": "Negative prompt",
            "type": "textbox",
            "value": "",
            "info": "Negative text prompt"
        },
        "prompts_path": {
            "label": "Prompts path",
            "type": "textbox",
            "value": "",
            "info": "Path to prompts file"
        },
        "negative_prompts_path": {
            "label": "Negative prompts path",
            "type": "textbox",
            "value": "",
            "info": "Path to negative prompts file"
        },
        
        # ERROR HANDLING
        "reroll_blank_frames": {
            "label": "Reroll blank frames",
            "type": "radio",
            "choices": ["ignore", "reroll", "interrupt"],
            "value": "ignore",
            "info": "How to handle blank frames"
        },
        "reroll_patience": {
            "label": "Reroll patience",
            "type": "slider",
            "minimum": 1.0,
            "maximum": 10.0,
            "step": 0.1,
            "value": 10.0,
            "info": "How long to wait before rerolling"
        },
        
        # MISC SETTINGS
        "seed_resize_from_w": {
            "label": "Seed resize from width",
            "type": "number",
            "precision": 0,
            "value": 0,
            "info": "Width to resize seed from"
        },
        "seed_resize_from_h": {
            "label": "Seed resize from height",
            "type": "number",
            "precision": 0,
            "value": 0,
            "info": "Height to resize seed from"
        },
        "noise_multiplier": {
            "label": "Noise multiplier",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": "Noise multiplier"
        },
        "ddim_eta": {
            "label": "DDIM eta",
            "type": "number",
            "precision": None,
            "value": 0.0,
            "info": "DDIM eta parameter"
        },
        "ancestral_eta": {
            "label": "Ancestral eta",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": "Ancestral eta parameter"
        },
        "subseed": {
            "label": "Subseed",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "Subseed for variation"
        },
        "subseed_strength": {
            "label": "Subseed strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01,
            "value": 0.0,
            "info": "Strength of subseed variation"
        },
        "seed_behavior": {
            "label": "Seed behavior",
            "type": "dropdown",
            "choices": ["iter", "fixed", "random"],
            "value": "iter",
            "info": "How seed changes between frames"
        },
        "seed_iter_N": {
            "label": "Seed iter N",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "Seed iteration step"
        },
        "use_areas": {
            "label": "Use areas",
            "type": "checkbox",
            "value": False,
            "info": "Use area-based prompting"
        },
        
        # UI SETTINGS
        "show_info_on_ui": {
            "label": "Show info on UI",
            "type": "checkbox",
            "value": False,
            "info": "Show information on UI"
        },
        "show_controlnet_tab": {
            "label": "Show ControlNet tab",
            "type": "checkbox",
            "value": False,
            "info": "Show ControlNet tab"
        },
        "motion_preview_mode": {
            "label": "Motion preview mode",
            "type": "checkbox",
            "value": False,
            "info": "Enable motion preview"
        },
        
        # OUTPUT SETTINGS
        "save_settings": {
            "label": "Save settings",
            "type": "checkbox",
            "value": True,
            "info": "Save settings file"
        },
        "save_sample": {
            "label": "Save sample",
            "type": "checkbox",
            "value": True,
            "info": "Save sample images"
        },
        "display_samples": {
            "label": "Display samples",
            "type": "checkbox",
            "value": True,
            "info": "Display sample images"
        },
        "save_sample_per_step": {
            "label": "Save sample per step",
            "type": "checkbox",
            "value": False,
            "info": "Save image at each step"
        },
        "show_sample_per_step": {
            "label": "Show sample per step",
            "type": "checkbox",
            "value": False,
            "info": "Show image at each step"
        },
        "override_these_with_webui": {
            "label": "Override with WebUI",
            "type": "checkbox",
            "value": False,
            "info": "Override settings with WebUI defaults"
        },
        "batch_name": {
            "label": "Batch name",
            "type": "textbox",
            "value": "Deforum",
            "info": "Name for batch of images"
        },
        "filename_format": {
            "label": "Filename format",
            "type": "textbox",
            "value": "{timestring}_{index:05}_{prompt}.png",
            "info": "Format for output filenames"
        },
        "init_image_box": {
            "label": "Init image box",
            "type": "textbox",
            "value": "",
            "info": "Bounding box for init image"
        }
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
        },
        "parseq_non_schedule_overrides": {
            "label": "Parseq non schedule overrides",
            "type": "checkbox",
            "value": False,
            "info": "Allow Parseq to override non-schedule settings"
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
        },
        "wan_auto_download": {
            "label": "Auto download models",
            "type": "checkbox",
            "value": True,
            "info": "Automatically download required models"
        },
        "wan_seed": {
            "label": "WAN seed",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "Seed for WAN generation"
        }
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
        },
        "delete_input_frames": {
            "label": "Delete input frames",
            "type": "checkbox",
            "value": False,
            "info": "Delete input frames after processing"
        },
        "skip_video_creation": {
            "label": "Skip video creation",
            "type": "checkbox",
            "value": False,
            "info": "Skip creating video output"
        },
        "save_images": {
            "label": "Save images",
            "type": "checkbox",
            "value": True,
            "info": "Save individual frame images"
        },
        "save_individual_images": {
            "label": "Save individual images",
            "type": "checkbox",
            "value": False,
            "info": "Save images individually"
        },
        "image_format": {
            "label": "Image format",
            "type": "dropdown",
            "choices": ["PNG", "JPEG"],
            "value": "PNG",
            "info": "Format for saved images"
        },
        "jpeg_quality": {
            "label": "JPEG quality",
            "type": "slider",
            "minimum": 1,
            "maximum": 100,
            "step": 1,
            "value": 85,
            "info": "Quality for JPEG images"
        },
        
        # UPSCALING SETTINGS
        "r_upscale_video": {
            "label": "Upscale video",
            "type": "checkbox",
            "value": False,
            "info": "Upscale video output"
        },
        "r_upscale_factor": {
            "label": "Upscale factor",
            "type": "dropdown",
            "choices": ["x2", "x4"],
            "value": "x2",
            "info": "Upscaling factor"
        },
        "r_upscale_model": {
            "label": "Upscale model",
            "type": "dropdown",
            "choices": ["RealESRGAN_x4plus", "RealESRGAN_x2plus"],
            "value": "RealESRGAN_x4plus",
            "info": "Upscaling model"
        },
        "r_upscale_keep_imgs": {
            "label": "Keep upscaled images",
            "type": "checkbox",
            "value": False,
            "info": "Keep upscaled image files"
        },
        
        # NCNN UPSCALING
        "ncnn_upscale_model": {
            "label": "NCNN upscale model",
            "type": "dropdown",
            "choices": ["None"],
            "value": "None",
            "info": "NCNN upscaling model"
        },
        "ncnn_upscale_factor": {
            "label": "NCNN upscale factor",
            "type": "dropdown",
            "choices": ["x2", "x4"],
            "value": "x2",
            "info": "NCNN upscaling factor"
        },
        
        # FRAME INTERPOLATION
        "frame_interpolation_engine": {
            "label": "Frame interpolation engine",
            "type": "dropdown",
            "choices": ["None", "RIFE", "FILM"],
            "value": "None",
            "info": "Frame interpolation method"
        },
        "frame_interpolation_x_amount": {
            "label": "Frame interpolation amount",
            "type": "number",
            "precision": 0,
            "value": 2,
            "info": "Frame interpolation multiplier"
        },
        "frame_interpolation_slow_mo_enabled": {
            "label": "Frame interpolation slow mo",
            "type": "checkbox",
            "value": False,
            "info": "Enable slow motion interpolation"
        },
        "frame_interpolation_slow_mo_amount": {
            "label": "Slow mo amount",
            "type": "number",
            "precision": 0,
            "value": 2,
            "info": "Slow motion multiplier"
        },
        "frame_interpolation_keep_imgs": {
            "label": "Keep interpolated images",
            "type": "checkbox",
            "value": False,
            "info": "Keep interpolated image files"
        },
        "frame_interpolation_use_upscaled": {
            "label": "Use upscaled for interpolation",
            "type": "checkbox",
            "value": False,
            "info": "Use upscaled images for interpolation"
        },
        
        # FFMPEG SETTINGS
        "ffmpeg_mode": {
            "label": "FFMPEG mode",
            "type": "dropdown",
            "choices": ["auto", "manual"],
            "value": "auto",
            "info": "FFMPEG processing mode"
        },
        "ffmpeg_outdir": {
            "label": "FFMPEG output directory",
            "type": "textbox",
            "value": "",
            "info": "Custom FFMPEG output directory"
        },
        "ffmpeg_crf": {
            "label": "FFMPEG CRF",
            "type": "slider",
            "minimum": 0,
            "maximum": 51,
            "step": 1,
            "value": 17,
            "info": "FFMPEG quality setting"
        },
        "ffmpeg_preset": {
            "label": "FFMPEG preset",
            "type": "dropdown",
            "choices": ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
            "value": "medium",
            "info": "FFMPEG encoding preset"
        },
        "max_video_frames": {
            "label": "Max video frames",
            "type": "number",
            "precision": 0,
            "value": 200,
            "info": "Maximum frames for video output"
        },
        "skip_video_for_run_all": {
            "label": "Skip video for run all",
            "type": "checkbox",
            "value": False,
            "info": "Skip video creation when running all"
        }
    }


def LoopArgs():
    """Loop/Guided Images argument defaults."""
    return {
        "use_looper": {
            "label": "Use looper",
            "type": "checkbox",
            "value": False,
            "info": "Enable guided images mode"
        },
        "init_images": {
            "label": "Init images",
            "type": "textbox",
            "value": "",
            "info": "JSON string of image paths for guided mode"
        },
        "image_strength_schedule": {
            "label": "Image strength schedule",
            "type": "textbox",
            "value": "0: (0.75)",
            "info": "Strength schedule for guided images"
        },
        "image_keyframe_strength_schedule": {
            "label": "Image keyframe strength schedule",
            "type": "textbox",
            "value": "0: (0.50)",
            "info": "Keyframe strength schedule for guided images"
        },
        "blendFactorMax": {
            "label": "Blend factor max",
            "type": "textbox",
            "value": "0: (0.35)",
            "info": "Maximum blend factor"
        },
        "blendFactorSlope": {
            "label": "Blend factor slope",
            "type": "textbox",
            "value": "0: (0.25)",
            "info": "Blend factor slope"
        },
        "tweening_frames_schedule": {
            "label": "Tweening frames schedule",
            "type": "textbox",
            "value": "0: (20)",
            "info": "Tweening frames schedule"
        },
        "color_correction_factor": {
            "label": "Color correction factor",
            "type": "textbox",
            "value": "0: (0.075)",
            "info": "Color correction factor"
        }
    }
