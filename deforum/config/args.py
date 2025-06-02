"""
Backward compatibility module for argument imports.

This module re-exports all argument classes and functions from the new config structure
to maintain compatibility with existing code that imports from deforum.config.args or
deforum.core.args.
"""

# Re-export everything from the new config structure
from .arguments import *
from .arg_defaults import *
from .arg_validation import *
from .arg_transformations import *

# Import the main dataclass definitions
try:
    from ..models.data_models import (
        DeforumArgs, DeforumAnimArgs, ParseqArgs, DeforumOutputArgs, 
        RootArgs, WanArgs, LoopArgs, ControlnetArgs
    )
except ImportError:
    # Fallback if models not available - create placeholder classes
    class LoopArgs:
        """Fallback LoopArgs for when data_models import fails"""
        def __init__(self):
            self.use_looper = False
            self.init_images = ""
            self.image_strength_schedule = "0: (0.75)"
            self.image_keyframe_strength_schedule = "0: (0.50)"
            self.blendFactorMax = "0: (0.35)"
            self.blendFactorSlope = "0: (0.25)"
            self.tweening_frames_schedule = "0: (20)"
            self.color_correction_factor = "0: (0.075)"
    
    class ControlnetArgs:
        """Fallback ControlnetArgs for when data_models import fails"""
        def __init__(self):
            # Initialize all CN attributes with defaults
            for i in range(1, 6):
                setattr(self, f'cn_{i}_enabled', False)
                setattr(self, f'cn_{i}_overwrite_frames', False)
                setattr(self, f'cn_{i}_vid_path', "")
                setattr(self, f'cn_{i}_mask_vid_path', "")
                setattr(self, f'cn_{i}_use_vid_as_input', False)
                setattr(self, f'cn_{i}_low_vram', False)
                setattr(self, f'cn_{i}_pixel_perfect', False)
                setattr(self, f'cn_{i}_module', "None")
                setattr(self, f'cn_{i}_model', "None")
                setattr(self, f'cn_{i}_weight', "0: (1.0)")
                setattr(self, f'cn_{i}_guidance_start', "0: (0.0)")
                setattr(self, f'cn_{i}_guidance_end', "0: (1.0)")
                setattr(self, f'cn_{i}_processor_res', 64)
                setattr(self, f'cn_{i}_threshold_a', 64)
                setattr(self, f'cn_{i}_threshold_b', 64)
                setattr(self, f'cn_{i}_resize_mode', "Scale to Fit (Inner Fit)")
                setattr(self, f'cn_{i}_control_mode', "Balanced")
                setattr(self, f'cn_{i}_loopback_mode', False)
    
    # Provide fallback classes for other imports if needed
    DeforumArgs = None
    DeforumAnimArgs = None
    ParseqArgs = None
    DeforumOutputArgs = None
    RootArgs = None
    WanArgs = None

# Re-export component name functions
try:
    from ..integrations.controlnet.core_integration import controlnet_component_names
except ImportError:
    def controlnet_component_names():
        return []

def get_component_names():
    """Get all component names for the UI"""
    components = []
    
    # Add main argument components
    components.extend([
        'seed', 'steps', 'sampler', 'scheduler', 'checkpoint', 'clip_skip',
        'W', 'H', 'strength', 'cfg_scale', 'distilled_cfg_scale', 'tiling',
        'restore_faces', 'seed_resize_from_w', 'seed_resize_from_h',
        'noise_multiplier', 'ddim_eta', 'ancestral_eta', 'overlay_mask',
        'mask_file', 'init_image', 'init_image_box', 'use_init', 'use_mask',
        'invert_mask', 'mask_brightness_adjust', 'mask_contrast_adjust',
        'mask_overlay_blur', 'fill', 'full_res_mask', 'full_res_mask_padding',
        'reroll_blank_frames', 'prompt', 'negative_prompt', 'subseed',
        'subseed_strength', 'seed_behavior', 'seed_iter_N', 'use_areas',
        'save_settings', 'save_sample', 'display_samples', 'save_sample_per_step',
        'show_sample_per_step', 'override_these_with_webui', 'batch_name',
        'filename_format', 'seed_behavior', 'seed_iter_N', 'animation_mode',
        'max_frames', 'border', 'angle', 'zoom', 'translation_x', 'translation_y',
        'translation_z', 'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z',
        'flip_2d_perspective', 'perspective_flip_theta', 'perspective_flip_phi',
        'perspective_flip_gamma', 'perspective_flip_fv', 'noise_schedule',
        'strength_schedule', 'contrast_schedule', 'cfg_scale_schedule',
        'distilled_cfg_scale_schedule', 'steps_schedule', 'seed_schedule',
        'motion_preview_mode', 'motion_preview_length', 'motion_preview_step'
    ])
    
    # Add ControlNet components
    components.extend(controlnet_component_names())
    
    return components

def process_args(*args, **kwargs):
    """Process arguments - placeholder for compatibility"""
    return args, kwargs

def get_settings_component_names():
    """Get settings component names"""
    return [
        'seed_behavior', 'save_settings', 'save_sample', 'display_samples', 
        'save_sample_per_step', 'show_sample_per_step', 'override_these_with_webui'
    ]

def set_arg_lists(*args, **kwargs):
    """Set argument lists - returns initialized argument objects"""
    try:
        # Try to create instances using the dataclasses
        from ..models.data_models import (
            DeforumArgs, DeforumAnimArgs, ParseqArgs, DeforumOutputArgs, 
            RootArgs, WanArgs, LoopArgs
        )
        
        d = DeforumArgs()
        da = DeforumAnimArgs() 
        dp = ParseqArgs()
        dv = DeforumOutputArgs()
        dr = RootArgs()
        dw = WanArgs()
        dloopArgs = LoopArgs()
        
        # Add critical attributes that the UI expects
        d.motion_preview_mode = {
            "label": "Motion preview mode",
            "type": "checkbox", 
            "value": False,
            "info": ""
        }
        d.show_info_on_ui = {
            "label": "Show info on UI",
            "type": "checkbox",
            "value": False,
            "info": ""
        }
        d.show_controlnet_tab = {
            "label": "Show ControlNet tab", 
            "type": "checkbox",
            "value": False,
            "info": ""
        }
        d.sampler = {
            "label": "Sampler",
            "type": "dropdown",
            "choices": ["Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a"],
            "value": "Euler a",
            "info": ""
        }
        d.scheduler = {
            "label": "Scheduler", 
            "type": "dropdown",
            "choices": ["Normal", "Karras", "Exponential"],
            "value": "Normal",
            "info": ""
        }
        d.steps = {
            "label": "Steps",
            "type": "slider",
            "minimum": 1,
            "maximum": 150,
            "step": 1,
            "value": 25,
            "info": ""
        }
        d.W = {
            "label": "Width",
            "type": "slider",
            "minimum": 64,
            "maximum": 8192,
            "step": 64,
            "value": 512,
            "info": ""
        }
        d.H = {
            "label": "Height",
            "type": "slider", 
            "minimum": 64,
            "maximum": 8192,
            "step": 64,
            "value": 512,
            "info": ""
        }
        d.seed = {
            "label": "Seed",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": ""
        }
        d.batch_name = {
            "label": "Batch name",
            "type": "textbox",
            "value": "Deforum",
            "info": ""
        }
        d.restore_faces = {
            "label": "Restore faces",
            "type": "checkbox",
            "value": False,
            "info": ""
        }
        d.tiling = {
            "label": "Tiling",
            "type": "checkbox", 
            "value": False,
            "info": ""
        }
        d.use_init = {
            "label": "Use init image",
            "type": "checkbox",
            "value": False,
            "info": ""
        }
        d.init_image = {
            "label": "Init image",
            "type": "textbox",
            "value": "",
            "info": ""
        }
        d.strength = {
            "label": "Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01,
            "value": 0.65,
            "info": ""
        }
        d.strength_0_no_init = {
            "label": "Strength 0 no init",
            "type": "checkbox",
            "value": True,
            "info": ""
        }
        d.init_scale = {
            "label": "Init scale",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": ""
        }
        d.use_mask = {
            "label": "Use mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        }
        d.use_alpha_as_mask = {
            "label": "Use alpha as mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        }
        d.invert_mask = {
            "label": "Invert mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        }
        d.overlay_mask = {
            "label": "Overlay mask",
            "type": "checkbox",
            "value": True,
            "info": ""
        }
        d.mask_file = {
            "label": "Mask file",
            "type": "textbox",
            "value": "",
            "info": ""
        }
        d.mask_brightness_adjust = {
            "label": "Mask brightness adjust",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": ""
        }
        d.mask_contrast_adjust = {
            "label": "Mask contrast adjust",
            "type": "number", 
            "precision": None,
            "value": 1.0,
            "info": ""
        }
        d.mask_overlay_blur = {
            "label": "Mask overlay blur",
            "type": "slider",
            "minimum": 0,
            "maximum": 64,
            "step": 1,
            "value": 4,
            "info": ""
        }
        d.fill = {
            "label": "Fill",
            "type": "radio",
            "choices": ["stretch", "fit", "crop"],
            "value": "stretch",
            "info": ""
        }
        d.full_res_mask = {
            "label": "Full res mask",
            "type": "checkbox",
            "value": True,
            "info": ""
        }
        d.full_res_mask_padding = {
            "label": "Full res mask padding",
            "type": "slider",
            "minimum": 0,
            "maximum": 512,
            "step": 4,
            "value": 32,
            "info": ""
        }
        d.reroll_blank_frames = {
            "label": "Reroll blank frames",
            "type": "radio",
            "choices": ["ignore", "reroll", "interrupt"], 
            "value": "ignore",
            "info": ""
        }
        d.reroll_patience = {
            "label": "Reroll patience",
            "type": "slider",
            "minimum": 1.0,
            "maximum": 10.0,
            "step": 0.1,
            "value": 10.0,
            "info": ""
        }
        d.seed_resize_from_w = {"label": "Seed resize from width", "type": "number", "value": 0, "info": ""}
        d.seed_resize_from_h = {"label": "Seed resize from height", "type": "number", "value": 0, "info": ""}
        d.noise_multiplier = {"label": "Noise multiplier", "type": "number", "value": 1.0, "info": ""}
        d.ddim_eta = {"label": "DDIM eta", "type": "number", "value": 0.0, "info": ""}
        d.ancestral_eta = {"label": "Ancestral eta", "type": "number", "value": 1.0, "info": ""}
        d.subseed = {"label": "Subseed", "type": "number", "value": -1, "info": ""}
        d.subseed_strength = {"label": "Subseed strength", "type": "slider", "minimum": 0.0, "maximum": 1.0, "step": 0.01, "value": 0.0, "info": ""}
        d.seed_behavior = {"label": "Seed behavior", "type": "dropdown", "choices": ["iter"], "value": "iter", "info": ""}
        d.seed_iter_N = {"label": "Seed iter N", "type": "number", "value": 1, "info": ""}
        d.use_areas = {"label": "Use areas", "type": "checkbox", "value": False, "info": ""}
        d.save_settings = {"label": "Save settings", "type": "checkbox", "value": True, "info": ""}
        d.save_sample = {"label": "Save sample", "type": "checkbox", "value": True, "info": ""}
        d.display_samples = {"label": "Display samples", "type": "checkbox", "value": True, "info": ""}
        d.save_sample_per_step = {"label": "Save sample per step", "type": "checkbox", "value": False, "info": ""}
        d.show_sample_per_step = {"label": "Show sample per step", "type": "checkbox", "value": False, "info": ""}
        d.override_these_with_webui = {"label": "Override with webui", "type": "checkbox", "value": False, "info": ""}
        d.filename_format = {"label": "Filename format", "type": "textbox", "value": "{timestring}_{index:05}_{prompt}.png", "info": ""}
        d.animation_mode = {"label": "Animation mode", "type": "dropdown", "choices": ["None", "2D", "3D"], "value": "None", "info": ""}
        d.border = {"label": "Border", "type": "dropdown", "choices": ["wrap", "reflect", "replicate", "zero"], "value": "wrap", "info": ""}
        d.max_frames = {"label": "Max frames", "type": "number", "value": 120, "info": ""}
        d.checkpoint = {"label": "Checkpoint", "type": "dropdown", "choices": ["Use Model from WebUI"], "value": "Use Model from WebUI", "info": ""}
        d.clip_skip = {"label": "Clip skip", "type": "number", "value": 1, "info": ""}
        d.cfg_scale = {"label": "CFG scale", "type": "slider", "minimum": 1.0, "maximum": 30.0, "step": 0.5, "value": 7.0, "info": ""}
        d.distilled_cfg_scale = {"label": "Distilled CFG scale", "type": "slider", "minimum": 1.0, "maximum": 30.0, "step": 0.5, "value": 7.0, "info": ""}
        d.prompt = {"label": "Prompt", "type": "textbox", "value": "", "info": ""}
        d.negative_prompt = {"label": "Negative prompt", "type": "textbox", "value": "", "info": ""}
        d.prompts_path = {"label": "Prompts path", "type": "textbox", "value": "", "info": ""}
        d.negative_prompts_path = {"label": "Negative prompts path", "type": "textbox", "value": "", "info": ""}
        d.init_image_box = {"label": "Init image box", "type": "textbox", "value": "", "info": ""}
        
        # Add missing file/path attributes that UI expects
        d.outdir = {"label": "Output directory", "type": "textbox", "value": "", "info": ""}
        d.negative_prompts_path = {"label": "Negative prompts path", "type": "textbox", "value": "", "info": ""}
        d.outdir_samples = {"label": "Output samples directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_grids = {"label": "Output grids directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_extras = {"label": "Output extras directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_img2img_samples = {"label": "Output img2img samples directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_img2img_grids = {"label": "Output img2img grids directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_save = {"label": "Output save directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_txt2img_samples = {"label": "Output txt2img samples directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_txt2img_grids = {"label": "Output txt2img grids directory", "type": "textbox", "value": "", "info": ""}
        
        # Animation args (da) - Essential for enable_ddim_eta_scheduling error
        da.enable_ddim_eta_scheduling = {"label": "Enable DDIM eta scheduling", "type": "checkbox", "value": False, "info": ""}
        da.enable_ancestral_eta_scheduling = {"label": "Enable ancestral eta scheduling", "type": "checkbox", "value": False, "info": ""}
        da.ddim_eta_schedule = {"label": "DDIM eta schedule", "type": "textbox", "value": "0: (0.0)", "info": ""}
        da.ancestral_eta_schedule = {"label": "Ancestral eta schedule", "type": "textbox", "value": "0: (1.0)", "info": ""}
        da.resume_from_timestring = {"label": "Resume from timestring", "type": "checkbox", "value": False, "info": ""}
        da.resume_timestring = {"label": "Resume timestring", "type": "textbox", "value": "", "info": ""}
        da.animation_mode = {"label": "Animation mode", "type": "dropdown", "choices": ["None", "2D", "3D"], "value": "None", "info": ""}
        da.border = {"label": "Border", "type": "dropdown", "choices": ["wrap", "reflect", "replicate", "zero"], "value": "wrap", "info": ""}
        da.diffusion_cadence = {"label": "Diffusion cadence", "type": "number", "value": 1, "info": ""}
        da.max_frames = {"label": "Max frames", "type": "number", "value": 120, "info": ""}
        da.keyframe_distribution = {"label": "Keyframe distribution", "type": "dropdown", "choices": ["Off", "Keyframes Only", "Additive", "Redistributed"], "value": "Redistributed", "info": ""}
        
        # Keyframe schedule attributes - CRITICAL MISSING ONES
        da.strength_schedule = {"label": "Strength schedule", "type": "textbox", "value": "0: (0.65)", "info": ""}
        da.keyframe_strength_schedule = {"label": "Keyframe strength schedule", "type": "textbox", "value": "0: (0.50)", "info": ""}
        da.cfg_scale_schedule = {"label": "CFG scale schedule", "type": "textbox", "value": "0: (7.0)", "info": ""}
        da.distilled_cfg_scale_schedule = {"label": "Distilled CFG scale schedule", "type": "textbox", "value": "0: (7.0)", "info": ""}
        da.enable_clipskip_scheduling = {"label": "Enable clip skip scheduling", "type": "checkbox", "value": False, "info": ""}
        da.clipskip_schedule = {"label": "Clip skip schedule", "type": "textbox", "value": "0: (1)", "info": ""}
        da.seed_schedule = {"label": "Seed schedule", "type": "textbox", "value": "0: (-1)", "info": ""}
        da.enable_subseed_scheduling = {"label": "Enable subseed scheduling", "type": "checkbox", "value": False, "info": ""}
        da.subseed_schedule = {"label": "Subseed schedule", "type": "textbox", "value": "0: (1)", "info": ""}
        da.subseed_strength_schedule = {"label": "Subseed strength schedule", "type": "textbox", "value": "0: (0.0)", "info": ""}
        da.enable_steps_scheduling = {"label": "Enable steps scheduling", "type": "checkbox", "value": False, "info": ""}
        da.steps_schedule = {"label": "Steps schedule", "type": "textbox", "value": "0: (25)", "info": ""}
        da.enable_sampler_scheduling = {"label": "Enable sampler scheduling", "type": "checkbox", "value": False, "info": ""}
        da.sampler_schedule = {"label": "Sampler schedule", "type": "textbox", "value": "0: (\"Euler a\")", "info": ""}
        da.enable_scheduler_scheduling = {"label": "Enable scheduler scheduling", "type": "checkbox", "value": False, "info": ""}
        da.scheduler_schedule = {"label": "Scheduler schedule", "type": "textbox", "value": "0: (\"Normal\")", "info": ""}
        da.enable_checkpoint_scheduling = {"label": "Enable checkpoint scheduling", "type": "checkbox", "value": False, "info": ""}
        da.checkpoint_schedule = {"label": "Checkpoint schedule", "type": "textbox", "value": "0: (\"model1.ckpt\")", "info": ""}
        
        # Motion and transformation attributes
        da.zoom = {"label": "Zoom", "type": "textbox", "value": "0: (1.0)", "info": ""}
        da.angle = {"label": "Angle", "type": "textbox", "value": "0: (0)", "info": ""}
        da.transform_center_x = {"label": "Transform center X", "type": "textbox", "value": "0: (0.5)", "info": ""}
        da.transform_center_y = {"label": "Transform center Y", "type": "textbox", "value": "0: (0.5)", "info": ""}
        da.translation_x = {"label": "Translation X", "type": "textbox", "value": "0: (0)", "info": ""}
        da.translation_y = {"label": "Translation Y", "type": "textbox", "value": "0: (0)", "info": ""}
        da.translation_z = {"label": "Translation Z", "type": "textbox", "value": "0: (0)", "info": ""}
        da.rotation_3d_x = {"label": "Rotation 3D X", "type": "textbox", "value": "0: (0)", "info": ""}
        da.rotation_3d_y = {"label": "Rotation 3D Y", "type": "textbox", "value": "0: (0)", "info": ""}
        da.rotation_3d_z = {"label": "Rotation 3D Z", "type": "textbox", "value": "0: (0)", "info": ""}
        
        # Perspective flip attributes
        da.enable_perspective_flip = {"label": "Enable perspective flip", "type": "checkbox", "value": False, "info": ""}
        da.perspective_flip_theta = {"label": "Perspective flip theta", "type": "textbox", "value": "0: (0)", "info": ""}
        da.perspective_flip_phi = {"label": "Perspective flip phi", "type": "textbox", "value": "0: (0)", "info": ""}
        da.perspective_flip_gamma = {"label": "Perspective flip gamma", "type": "textbox", "value": "0: (0)", "info": ""}
        da.perspective_flip_fv = {"label": "Perspective flip fv", "type": "textbox", "value": "0: (53)", "info": ""}
        
        # Camera shake attributes
        da.shake_name = {"label": "Shake name", "type": "textbox", "value": "INVESTIGATION", "info": ""}
        da.shake_intensity = {"label": "Shake intensity", "type": "number", "value": 1.0, "info": ""}
        da.shake_speed = {"label": "Shake speed", "type": "number", "value": 1.0, "info": ""}
        
        # Noise attributes
        da.noise_type = {"label": "Noise type", "type": "dropdown", "choices": ["perlin", "uniform"], "value": "perlin", "info": ""}
        da.noise_schedule = {"label": "Noise schedule", "type": "textbox", "value": "0: (0.02)", "info": ""}
        da.perlin_octaves = {"label": "Perlin octaves", "type": "number", "value": 4, "info": ""}
        da.perlin_persistence = {"label": "Perlin persistence", "type": "number", "value": 0.5, "info": ""}
        da.perlin_w = {"label": "Perlin width", "type": "number", "value": 8, "info": ""}
        da.perlin_h = {"label": "Perlin height", "type": "number", "value": 8, "info": ""}
        da.enable_noise_multiplier_scheduling = {"label": "Enable noise multiplier scheduling", "type": "checkbox", "value": False, "info": ""}
        da.noise_multiplier_schedule = {"label": "Noise multiplier schedule", "type": "textbox", "value": "0: (1.0)", "info": ""}
        
        # Coherence attributes
        da.color_coherence = {"label": "Color coherence", "type": "dropdown", "choices": ["None", "Match Frame 0 HSV", "Match Frame 0 LAB"], "value": "None", "info": ""}
        da.color_force_grayscale = {"label": "Color force grayscale", "type": "checkbox", "value": False, "info": ""}
        da.legacy_colormatch = {"label": "Legacy color match", "type": "checkbox", "value": False, "info": ""}
        da.color_coherence_image_path = {"label": "Color coherence image path", "type": "textbox", "value": "", "info": ""}
        da.color_coherence_video_every_N_frames = {"label": "Color coherence video every N frames", "type": "number", "value": 1, "info": ""}
        da.optical_flow_cadence = {"label": "Optical flow cadence", "type": "dropdown", "choices": ["None", "1", "2"], "value": "None", "info": ""}
        da.cadence_flow_factor_schedule = {"label": "Cadence flow factor schedule", "type": "textbox", "value": "0: (1)", "info": ""}
        da.optical_flow_redo_generation = {"label": "Optical flow redo generation", "type": "dropdown", "choices": ["None", "1", "2"], "value": "None", "info": ""}
        da.redo_flow_factor_schedule = {"label": "Redo flow factor schedule", "type": "textbox", "value": "0: (1)", "info": ""}
        da.contrast_schedule = {"label": "Contrast schedule", "type": "textbox", "value": "0: (1.0)", "info": ""}
        da.diffusion_redo = {"label": "Diffusion redo", "type": "number", "value": 0, "info": ""}
        
        # Anti-blur attributes
        da.amount_schedule = {"label": "Amount schedule", "type": "textbox", "value": "0: (0.1)", "info": ""}
        da.kernel_schedule = {"label": "Kernel schedule", "type": "textbox", "value": "0: (5)", "info": ""}
        da.sigma_schedule = {"label": "Sigma schedule", "type": "textbox", "value": "0: (1)", "info": ""}
        da.threshold_schedule = {"label": "Threshold schedule", "type": "textbox", "value": "0: (0)", "info": ""}
        
        # Depth warping attributes
        da.use_depth_warping = {"label": "Use depth warping", "type": "checkbox", "value": True, "info": ""}
        da.depth_algorithm = {"label": "Depth algorithm", "type": "dropdown", "choices": ["Zoe", "MiDaS"], "value": "Zoe", "info": ""}
        da.midas_weight = {"label": "MiDaS weight", "type": "number", "value": 0.3, "info": ""}
        da.padding_mode = {"label": "Padding mode", "type": "dropdown", "choices": ["border", "reflection"], "value": "border", "info": ""}
        da.sampling_mode = {"label": "Sampling mode", "type": "dropdown", "choices": ["bicubic", "bilinear"], "value": "bicubic", "info": ""}
        da.aspect_ratio_use_old_formula = {"label": "Aspect ratio use old formula", "type": "checkbox", "value": False, "info": ""}
        da.aspect_ratio_schedule = {"label": "Aspect ratio schedule", "type": "textbox", "value": "0: (1.0)", "info": ""}
        da.fov_schedule = {"label": "FOV schedule", "type": "textbox", "value": "0: (70)", "info": ""}
        da.near_schedule = {"label": "Near schedule", "type": "textbox", "value": "0: (200)", "info": ""}
        da.far_schedule = {"label": "Far schedule", "type": "textbox", "value": "0: (10000)", "info": ""}
        
        # Video initialization attributes (non-hybrid)
        da.video_init_path = {"label": "Video init path", "type": "textbox", "value": "", "info": ""}
        da.extract_from_frame = {"label": "Extract from frame", "type": "number", "value": 0, "info": ""}
        da.extract_to_frame = {"label": "Extract to frame", "type": "number", "value": -1, "info": ""}
        da.extract_nth_frame = {"label": "Extract nth frame", "type": "number", "value": 1, "info": ""}
        da.overwrite_extracted_frames = {"label": "Overwrite extracted frames", "type": "checkbox", "value": False, "info": ""}
        da.use_mask_video = {"label": "Use mask video", "type": "checkbox", "value": False, "info": ""}
        da.video_mask_path = {"label": "Video mask path", "type": "textbox", "value": "", "info": ""}
        
        # Additional animation attributes
        da.store_frames_in_ram = {"label": "Store frames in RAM", "type": "checkbox", "value": False, "info": ""}
        
        # Parseq args - basic set  
        dp.parseq_manifest = {"label": "Parseq manifest", "type": "textbox", "value": "", "info": ""}
        dp.parseq_use_deltas = {"label": "Use deltas", "type": "checkbox", "value": True, "info": ""}
        dp.parseq_non_schedule_overrides = {"label": "Parseq non schedule overrides", "type": "checkbox", "value": False, "info": ""}
        
        # Output args - basic set
        dv.fps = {"label": "FPS", "type": "number", "value": 15.0, "info": ""}
        dv.max_video_frames = {"label": "Max video frames", "type": "number", "value": 200, "info": ""}
        dv.add_soundtrack = {"label": "Add soundtrack", "type": "checkbox", "value": False, "info": ""}
        dv.soundtrack_path = {"label": "Soundtrack path", "type": "textbox", "value": "", "info": ""}
        dv.skip_video_for_run_all = {"label": "Skip video for run all", "type": "checkbox", "value": False, "info": ""}
        dv.delete_imgs = {"label": "Delete images", "type": "checkbox", "value": False, "info": ""}
        dv.delete_input_frames = {"label": "Delete input frames", "type": "checkbox", "value": False, "info": ""}
        dv.save_images = {"label": "Save images", "type": "checkbox", "value": True, "info": ""}
        dv.save_individual_images = {"label": "Save individual images", "type": "checkbox", "value": False, "info": ""}
        dv.image_format = {"label": "Image format", "type": "dropdown", "choices": ["PNG", "JPEG"], "value": "PNG", "info": ""}
        dv.jpeg_quality = {"label": "JPEG quality", "type": "slider", "minimum": 1, "maximum": 100, "step": 1, "value": 85, "info": ""}
        dv.ffmpeg_mode = {"label": "FFMPEG mode", "type": "dropdown", "choices": ["auto", "manual"], "value": "auto", "info": ""}
        dv.ffmpeg_outdir = {"label": "FFMPEG output directory", "type": "textbox", "value": "", "info": ""}
        dv.ffmpeg_crf = {"label": "FFMPEG CRF", "type": "slider", "minimum": 0, "maximum": 51, "step": 1, "value": 17, "info": ""}
        dv.ffmpeg_preset = {"label": "FFMPEG preset", "type": "dropdown", "choices": ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], "value": "medium", "info": ""}
        dv.frame_interpolation_engine = {"label": "Frame interpolation engine", "type": "dropdown", "choices": ["None", "RIFE", "FILM"], "value": "None", "info": ""}
        dv.frame_interpolation_x_amount = {"label": "Frame interpolation x amount", "type": "number", "value": 2, "info": ""}
        dv.frame_interpolation_slow_mo_enabled = {"label": "Frame interpolation slow mo enabled", "type": "checkbox", "value": False, "info": ""}
        dv.frame_interpolation_slow_mo_amount = {"label": "Frame interpolation slow mo amount", "type": "number", "value": 2, "info": ""}
        dv.frame_interpolation_keep_imgs = {"label": "Frame interpolation keep images", "type": "checkbox", "value": False, "info": ""}
        dv.frame_interpolation_use_upscaled = {"label": "Frame interpolation use upscaled", "type": "checkbox", "value": False, "info": ""}
        dv.film_interpolation_x_amount = {"label": "FILM interpolation x amount", "type": "number", "value": 2, "info": ""}
        dv.film_interpolation_slow_mo_enabled = {"label": "FILM interpolation slow mo enabled", "type": "checkbox", "value": False, "info": ""}
        dv.film_interpolation_slow_mo_amount = {"label": "FILM interpolation slow mo amount", "type": "number", "value": 2, "info": ""}
        dv.film_interpolation_keep_imgs = {"label": "FILM interpolation keep images", "type": "checkbox", "value": False, "info": ""}
        dv.r_upscale_video = {"label": "Real-ESRGAN upscale video", "type": "checkbox", "value": False, "info": ""}
        dv.r_upscale_factor = {"label": "Real-ESRGAN upscale factor", "type": "dropdown", "choices": ["x2", "x4"], "value": "x2", "info": ""}
        dv.r_upscale_model = {"label": "Real-ESRGAN upscale model", "type": "dropdown", "choices": ["RealESRGAN_x4plus", "RealESRGAN_x2plus"], "value": "RealESRGAN_x4plus", "info": ""}
        dv.r_upscale_keep_imgs = {"label": "Real-ESRGAN upscale keep images", "type": "checkbox", "value": False, "info": ""}
        
        # Root and WAN args - basic set
        dr.timestring = {"label": "Timestring", "type": "textbox", "value": "", "info": ""}
        dw.wan_mode = {"label": "WAN mode", "type": "dropdown", "choices": ["Off"], "value": "Off", "info": ""}
        
        # WAN video attributes - all the essential ones
        dw.wan_auto_download = {"label": "WAN auto download", "type": "checkbox", "value": True, "info": ""}
        dw.wan_preferred_size = {"label": "WAN preferred size", "type": "dropdown", "choices": ["512x512", "576x1024"], "value": "512x512", "info": ""}
        dw.wan_resolution = {"label": "WAN resolution", "type": "dropdown", "choices": ["512x512", "576x1024"], "value": "512x512", "info": ""}
        dw.wan_qwen_model = {"label": "WAN Qwen model", "type": "dropdown", "choices": ["Qwen/Qwen2-VL-2B-Instruct"], "value": "Qwen/Qwen2-VL-2B-Instruct", "info": ""}
        dw.wan_qwen_language = {"label": "WAN Qwen language", "type": "dropdown", "choices": ["English", "Chinese"], "value": "English", "info": ""}
        dw.wan_qwen_auto_download = {"label": "WAN Qwen auto download", "type": "checkbox", "value": True, "info": ""}
        dw.wan_t2v_model = {"label": "WAN T2V model", "type": "dropdown", "choices": ["cog-video-5b"], "value": "cog-video-5b", "info": ""}
        dw.wan_i2v_model = {"label": "WAN I2V model", "type": "dropdown", "choices": ["cog-video-5b-i2v"], "value": "cog-video-5b-i2v", "info": ""}
        dw.wan_model_path = {"label": "WAN model path", "type": "textbox", "value": "", "info": ""}
        dw.wan_strength_override = {"label": "WAN strength override", "type": "checkbox", "value": False, "info": ""}
        dw.wan_fixed_strength = {"label": "WAN fixed strength", "type": "number", "value": 0.8, "info": ""}
        dw.wan_guidance_override = {"label": "WAN guidance override", "type": "checkbox", "value": False, "info": ""}
        dw.wan_guidance_scale = {"label": "WAN guidance scale", "type": "number", "value": 6.0, "info": ""}
        dw.wan_motion_strength_override = {"label": "WAN motion strength override", "type": "checkbox", "value": False, "info": ""}
        dw.wan_motion_strength = {"label": "WAN motion strength", "type": "number", "value": 127, "info": ""}
        dw.wan_movement_sensitivity = {"label": "WAN movement sensitivity", "type": "number", "value": 127, "info": ""}
        dw.wan_frame_overlap = {"label": "WAN frame overlap", "type": "number", "value": 4, "info": ""}
        dw.wan_enable_interpolation = {"label": "WAN enable interpolation", "type": "checkbox", "value": False, "info": ""}
        dw.wan_interpolation_strength = {"label": "WAN interpolation strength", "type": "number", "value": 0.8, "info": ""}
        dw.wan_flash_attention_mode = {"label": "WAN flash attention mode", "type": "dropdown", "choices": ["auto", "enable", "disable"], "value": "auto", "info": ""}
        dw.wan_seed = {"label": "WAN seed", "type": "number", "precision": 0, "value": -1, "info": ""}
        
        # Loop args - ALL REQUIRED ATTRIBUTES
        dloopArgs.use_looper = {"label": "Use looper", "type": "checkbox", "value": False, "info": ""}
        dloopArgs.init_images = {"label": "Init images", "type": "textbox", "value": "", "info": ""}
        dloopArgs.image_strength_schedule = {"label": "Image strength schedule", "type": "textbox", "value": "0: (0.75)", "info": ""}
        dloopArgs.image_keyframe_strength_schedule = {"label": "Image keyframe strength schedule", "type": "textbox", "value": "0: (0.50)", "info": ""}
        dloopArgs.blendFactorMax = {"label": "Blend factor max", "type": "textbox", "value": "0: (0.35)", "info": ""}
        dloopArgs.blendFactorSlope = {"label": "Blend factor slope", "type": "textbox", "value": "0: (0.25)", "info": ""}
        dloopArgs.tweening_frames_schedule = {"label": "Tweening frames schedule", "type": "textbox", "value": "0: (20)", "info": ""}
        dloopArgs.color_correction_factor = {"label": "Color correction factor", "type": "textbox", "value": "0: (0.075)", "info": ""}
        
        return d, da, dp, dv, dr, dw, dloopArgs
        
    except (ImportError, TypeError) as e:
        # Fallback: create simple objects with comprehensive attributes
        print(f"⚠️ Using fallback argument objects: {e}")
        
        # Create simple namespace objects with basic attributes
        from types import SimpleNamespace
        
        d = SimpleNamespace()
        da = SimpleNamespace()
        dp = SimpleNamespace()
        dv = SimpleNamespace()
        dr = SimpleNamespace()
        dw = SimpleNamespace()
        dloopArgs = SimpleNamespace()
        
        # Essential UI attributes with proper dictionary structure
        d.motion_preview_mode = {"label": "Motion preview mode", "type": "checkbox", "value": False, "info": ""}
        d.show_info_on_ui = {"label": "Show info on UI", "type": "checkbox", "value": False, "info": ""}
        d.show_controlnet_tab = {"label": "Show ControlNet tab", "type": "checkbox", "value": False, "info": ""}
        d.sampler = {"label": "Sampler", "type": "dropdown", "choices": ["Euler a"], "value": "Euler a", "info": ""}
        d.scheduler = {"label": "Scheduler", "type": "dropdown", "choices": ["Normal"], "value": "Normal", "info": ""}
        d.steps = {"label": "Steps", "type": "slider", "minimum": 1, "maximum": 150, "step": 1, "value": 25, "info": ""}
        d.W = {"label": "Width", "type": "slider", "minimum": 64, "maximum": 8192, "step": 64, "value": 512, "info": ""}
        d.H = {"label": "Height", "type": "slider", "minimum": 64, "maximum": 8192, "step": 64, "value": 512, "info": ""}
        d.seed = {"label": "Seed", "type": "number", "precision": 0, "value": -1, "info": ""}
        d.batch_name = {"label": "Batch name", "type": "textbox", "value": "Deforum", "info": ""}
        d.restore_faces = {"label": "Restore faces", "type": "checkbox", "value": False, "info": ""}
        d.tiling = {"label": "Tiling", "type": "checkbox", "value": False, "info": ""}
        d.use_init = {"label": "Use init", "type": "checkbox", "value": False, "info": ""}
        d.init_image = {"label": "Init image", "type": "textbox", "value": "", "info": ""}
        d.strength = {"label": "Strength", "type": "slider", "minimum": 0.0, "maximum": 1.0, "step": 0.01, "value": 0.65, "info": ""}
        d.strength_0_no_init = {"label": "Strength 0 no init", "type": "checkbox", "value": True, "info": ""}
        d.init_scale = {"label": "Init scale", "type": "number", "value": 1.0, "info": ""}
        d.use_mask = {"label": "Use mask", "type": "checkbox", "value": False, "info": ""}
        d.use_alpha_as_mask = {"label": "Use alpha as mask", "type": "checkbox", "value": False, "info": ""}
        d.invert_mask = {"label": "Invert mask", "type": "checkbox", "value": False, "info": ""}
        d.overlay_mask = {"label": "Overlay mask", "type": "checkbox", "value": True, "info": ""}
        d.mask_file = {"label": "Mask file", "type": "textbox", "value": "", "info": ""}
        d.mask_brightness_adjust = {"label": "Mask brightness adjust", "type": "number", "value": 1.0, "info": ""}
        d.mask_contrast_adjust = {"label": "Mask contrast adjust", "type": "number", "value": 1.0, "info": ""}
        d.mask_overlay_blur = {"label": "Mask overlay blur", "type": "slider", "minimum": 0, "maximum": 64, "step": 1, "value": 4, "info": ""}
        d.fill = {"label": "Fill", "type": "radio", "choices": ["stretch"], "value": "stretch", "info": ""}
        d.full_res_mask = {"label": "Full res mask", "type": "checkbox", "value": True, "info": ""}
        d.full_res_mask_padding = {"label": "Full res mask padding", "type": "slider", "minimum": 0, "maximum": 512, "step": 4, "value": 32, "info": ""}
        d.reroll_blank_frames = {"label": "Reroll blank frames", "type": "radio", "choices": ["ignore"], "value": "ignore", "info": ""}
        d.reroll_patience = {"label": "Reroll patience", "type": "slider", "minimum": 1.0, "maximum": 10.0, "step": 0.1, "value": 10.0, "info": ""}
        d.seed_resize_from_w = {"label": "Seed resize from width", "type": "number", "value": 0, "info": ""}
        d.seed_resize_from_h = {"label": "Seed resize from height", "type": "number", "value": 0, "info": ""}
        d.noise_multiplier = {"label": "Noise multiplier", "type": "number", "value": 1.0, "info": ""}
        d.ddim_eta = {"label": "DDIM eta", "type": "number", "value": 0.0, "info": ""}
        d.ancestral_eta = {"label": "Ancestral eta", "type": "number", "value": 1.0, "info": ""}
        d.subseed = {"label": "Subseed", "type": "number", "value": -1, "info": ""}
        d.subseed_strength = {"label": "Subseed strength", "type": "slider", "minimum": 0.0, "maximum": 1.0, "step": 0.01, "value": 0.0, "info": ""}
        d.seed_behavior = {"label": "Seed behavior", "type": "dropdown", "choices": ["iter"], "value": "iter", "info": ""}
        d.seed_iter_N = {"label": "Seed iter N", "type": "number", "value": 1, "info": ""}
        d.use_areas = {"label": "Use areas", "type": "checkbox", "value": False, "info": ""}
        d.save_settings = {"label": "Save settings", "type": "checkbox", "value": True, "info": ""}
        d.save_sample = {"label": "Save sample", "type": "checkbox", "value": True, "info": ""}
        d.display_samples = {"label": "Display samples", "type": "checkbox", "value": True, "info": ""}
        d.save_sample_per_step = {"label": "Save sample per step", "type": "checkbox", "value": False, "info": ""}
        d.show_sample_per_step = {"label": "Show sample per step", "type": "checkbox", "value": False, "info": ""}
        d.override_these_with_webui = {"label": "Override with webui", "type": "checkbox", "value": False, "info": ""}
        d.filename_format = {"label": "Filename format", "type": "textbox", "value": "{timestring}_{index:05}_{prompt}.png", "info": ""}
        d.animation_mode = {"label": "Animation mode", "type": "dropdown", "choices": ["None", "2D", "3D"], "value": "None", "info": ""}
        d.border = {"label": "Border", "type": "dropdown", "choices": ["wrap"], "value": "wrap", "info": ""}
        d.max_frames = {"label": "Max frames", "type": "number", "value": 120, "info": ""}
        d.checkpoint = {"label": "Checkpoint", "type": "dropdown", "choices": ["Use Model from WebUI"], "value": "Use Model from WebUI", "info": ""}
        d.clip_skip = {"label": "Clip skip", "type": "number", "value": 1, "info": ""}
        d.cfg_scale = {"label": "CFG scale", "type": "slider", "minimum": 1.0, "maximum": 30.0, "step": 0.5, "value": 7.0, "info": ""}
        d.distilled_cfg_scale = {"label": "Distilled CFG scale", "type": "slider", "minimum": 1.0, "maximum": 30.0, "step": 0.5, "value": 7.0, "info": ""}
        d.prompt = {"label": "Prompt", "type": "textbox", "value": "", "info": ""}
        d.negative_prompt = {"label": "Negative prompt", "type": "textbox", "value": "", "info": ""}
        d.prompts_path = {"label": "Prompts path", "type": "textbox", "value": "", "info": ""}
        d.negative_prompts_path = {"label": "Negative prompts path", "type": "textbox", "value": "", "info": ""}
        d.init_image_box = {"label": "Init image box", "type": "textbox", "value": "", "info": ""}
        
        # Add missing file/path attributes that UI expects
        d.outdir = {"label": "Output directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_samples = {"label": "Output samples directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_grids = {"label": "Output grids directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_extras = {"label": "Output extras directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_img2img_samples = {"label": "Output img2img samples directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_img2img_grids = {"label": "Output img2img grids directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_save = {"label": "Output save directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_txt2img_samples = {"label": "Output txt2img samples directory", "type": "textbox", "value": "", "info": ""}
        d.outdir_txt2img_grids = {"label": "Output txt2img grids directory", "type": "textbox", "value": "", "info": ""}
        
        # Animation args (da) - Essential for enable_ddim_eta_scheduling error
        da.enable_ddim_eta_scheduling = {"label": "Enable DDIM eta scheduling", "type": "checkbox", "value": False, "info": ""}
        da.enable_ancestral_eta_scheduling = {"label": "Enable ancestral eta scheduling", "type": "checkbox", "value": False, "info": ""}
        da.ddim_eta_schedule = {"label": "DDIM eta schedule", "type": "textbox", "value": "0: (0.0)", "info": ""}
        da.ancestral_eta_schedule = {"label": "Ancestral eta schedule", "type": "textbox", "value": "0: (1.0)", "info": ""}
        da.resume_from_timestring = {"label": "Resume from timestring", "type": "checkbox", "value": False, "info": ""}
        da.resume_timestring = {"label": "Resume timestring", "type": "textbox", "value": "", "info": ""}
        da.animation_mode = {"label": "Animation mode", "type": "dropdown", "choices": ["None", "2D", "3D"], "value": "None", "info": ""}
        da.border = {"label": "Border", "type": "dropdown", "choices": ["wrap", "reflect", "replicate", "zero"], "value": "wrap", "info": ""}
        da.diffusion_cadence = {"label": "Diffusion cadence", "type": "number", "value": 1, "info": ""}
        da.max_frames = {"label": "Max frames", "type": "number", "value": 120, "info": ""}
        da.keyframe_distribution = {"label": "Keyframe distribution", "type": "dropdown", "choices": ["Off", "Keyframes Only", "Additive", "Redistributed"], "value": "Redistributed", "info": ""}
        
        # Keyframe schedule attributes - CRITICAL MISSING ONES
        da.strength_schedule = {"label": "Strength schedule", "type": "textbox", "value": "0: (0.65)", "info": ""}
        da.keyframe_strength_schedule = {"label": "Keyframe strength schedule", "type": "textbox", "value": "0: (0.50)", "info": ""}
        da.cfg_scale_schedule = {"label": "CFG scale schedule", "type": "textbox", "value": "0: (7.0)", "info": ""}
        da.distilled_cfg_scale_schedule = {"label": "Distilled CFG scale schedule", "type": "textbox", "value": "0: (7.0)", "info": ""}
        da.enable_clipskip_scheduling = {"label": "Enable clip skip scheduling", "type": "checkbox", "value": False, "info": ""}
        da.clipskip_schedule = {"label": "Clip skip schedule", "type": "textbox", "value": "0: (1)", "info": ""}
        da.seed_schedule = {"label": "Seed schedule", "type": "textbox", "value": "0: (-1)", "info": ""}
        da.enable_subseed_scheduling = {"label": "Enable subseed scheduling", "type": "checkbox", "value": False, "info": ""}
        da.subseed_schedule = {"label": "Subseed schedule", "type": "textbox", "value": "0: (1)", "info": ""}
        da.subseed_strength_schedule = {"label": "Subseed strength schedule", "type": "textbox", "value": "0: (0.0)", "info": ""}
        da.enable_steps_scheduling = {"label": "Enable steps scheduling", "type": "checkbox", "value": False, "info": ""}
        da.steps_schedule = {"label": "Steps schedule", "type": "textbox", "value": "0: (25)", "info": ""}
        da.enable_sampler_scheduling = {"label": "Enable sampler scheduling", "type": "checkbox", "value": False, "info": ""}
        da.sampler_schedule = {"label": "Sampler schedule", "type": "textbox", "value": "0: (\"Euler a\")", "info": ""}
        da.enable_scheduler_scheduling = {"label": "Enable scheduler scheduling", "type": "checkbox", "value": False, "info": ""}
        da.scheduler_schedule = {"label": "Scheduler schedule", "type": "textbox", "value": "0: (\"Normal\")", "info": ""}
        da.enable_checkpoint_scheduling = {"label": "Enable checkpoint scheduling", "type": "checkbox", "value": False, "info": ""}
        da.checkpoint_schedule = {"label": "Checkpoint schedule", "type": "textbox", "value": "0: (\"model1.ckpt\")", "info": ""}
        
        # Motion and transformation attributes
        da.zoom = {"label": "Zoom", "type": "textbox", "value": "0: (1.0)", "info": ""}
        da.angle = {"label": "Angle", "type": "textbox", "value": "0: (0)", "info": ""}
        da.transform_center_x = {"label": "Transform center X", "type": "textbox", "value": "0: (0.5)", "info": ""}
        da.transform_center_y = {"label": "Transform center Y", "type": "textbox", "value": "0: (0.5)", "info": ""}
        da.translation_x = {"label": "Translation X", "type": "textbox", "value": "0: (0)", "info": ""}
        da.translation_y = {"label": "Translation Y", "type": "textbox", "value": "0: (0)", "info": ""}
        da.translation_z = {"label": "Translation Z", "type": "textbox", "value": "0: (0)", "info": ""}
        da.rotation_3d_x = {"label": "Rotation 3D X", "type": "textbox", "value": "0: (0)", "info": ""}
        da.rotation_3d_y = {"label": "Rotation 3D Y", "type": "textbox", "value": "0: (0)", "info": ""}
        da.rotation_3d_z = {"label": "Rotation 3D Z", "type": "textbox", "value": "0: (0)", "info": ""}
        
        # Perspective flip attributes
        da.enable_perspective_flip = {"label": "Enable perspective flip", "type": "checkbox", "value": False, "info": ""}
        da.perspective_flip_theta = {"label": "Perspective flip theta", "type": "textbox", "value": "0: (0)", "info": ""}
        da.perspective_flip_phi = {"label": "Perspective flip phi", "type": "textbox", "value": "0: (0)", "info": ""}
        da.perspective_flip_gamma = {"label": "Perspective flip gamma", "type": "textbox", "value": "0: (0)", "info": ""}
        da.perspective_flip_fv = {"label": "Perspective flip fv", "type": "textbox", "value": "0: (53)", "info": ""}
        
        # Camera shake attributes
        da.shake_name = {"label": "Shake name", "type": "textbox", "value": "INVESTIGATION", "info": ""}
        da.shake_intensity = {"label": "Shake intensity", "type": "number", "value": 1.0, "info": ""}
        da.shake_speed = {"label": "Shake speed", "type": "number", "value": 1.0, "info": ""}
        
        # Noise attributes
        da.noise_type = {"label": "Noise type", "type": "dropdown", "choices": ["perlin", "uniform"], "value": "perlin", "info": ""}
        da.noise_schedule = {"label": "Noise schedule", "type": "textbox", "value": "0: (0.02)", "info": ""}
        da.perlin_octaves = {"label": "Perlin octaves", "type": "number", "value": 4, "info": ""}
        da.perlin_persistence = {"label": "Perlin persistence", "type": "number", "value": 0.5, "info": ""}
        da.perlin_w = {"label": "Perlin width", "type": "number", "value": 8, "info": ""}
        da.perlin_h = {"label": "Perlin height", "type": "number", "value": 8, "info": ""}
        da.enable_noise_multiplier_scheduling = {"label": "Enable noise multiplier scheduling", "type": "checkbox", "value": False, "info": ""}
        da.noise_multiplier_schedule = {"label": "Noise multiplier schedule", "type": "textbox", "value": "0: (1.0)", "info": ""}
        
        # Coherence attributes
        da.color_coherence = {"label": "Color coherence", "type": "dropdown", "choices": ["None", "Match Frame 0 HSV", "Match Frame 0 LAB"], "value": "None", "info": ""}
        da.color_force_grayscale = {"label": "Color force grayscale", "type": "checkbox", "value": False, "info": ""}
        da.legacy_colormatch = {"label": "Legacy color match", "type": "checkbox", "value": False, "info": ""}
        da.color_coherence_image_path = {"label": "Color coherence image path", "type": "textbox", "value": "", "info": ""}
        da.color_coherence_video_every_N_frames = {"label": "Color coherence video every N frames", "type": "number", "value": 1, "info": ""}
        da.optical_flow_cadence = {"label": "Optical flow cadence", "type": "dropdown", "choices": ["None", "1", "2"], "value": "None", "info": ""}
        da.cadence_flow_factor_schedule = {"label": "Cadence flow factor schedule", "type": "textbox", "value": "0: (1)", "info": ""}
        da.optical_flow_redo_generation = {"label": "Optical flow redo generation", "type": "dropdown", "choices": ["None", "1", "2"], "value": "None", "info": ""}
        da.redo_flow_factor_schedule = {"label": "Redo flow factor schedule", "type": "textbox", "value": "0: (1)", "info": ""}
        da.contrast_schedule = {"label": "Contrast schedule", "type": "textbox", "value": "0: (1.0)", "info": ""}
        da.diffusion_redo = {"label": "Diffusion redo", "type": "number", "value": 0, "info": ""}
        
        # Anti-blur attributes
        da.amount_schedule = {"label": "Amount schedule", "type": "textbox", "value": "0: (0.1)", "info": ""}
        da.kernel_schedule = {"label": "Kernel schedule", "type": "textbox", "value": "0: (5)", "info": ""}
        da.sigma_schedule = {"label": "Sigma schedule", "type": "textbox", "value": "0: (1)", "info": ""}
        da.threshold_schedule = {"label": "Threshold schedule", "type": "textbox", "value": "0: (0)", "info": ""}
        
        # Depth warping attributes
        da.use_depth_warping = {"label": "Use depth warping", "type": "checkbox", "value": True, "info": ""}
        da.depth_algorithm = {"label": "Depth algorithm", "type": "dropdown", "choices": ["Zoe", "MiDaS"], "value": "Zoe", "info": ""}
        da.midas_weight = {"label": "MiDaS weight", "type": "number", "value": 0.3, "info": ""}
        da.padding_mode = {"label": "Padding mode", "type": "dropdown", "choices": ["border", "reflection"], "value": "border", "info": ""}
        da.sampling_mode = {"label": "Sampling mode", "type": "dropdown", "choices": ["bicubic", "bilinear"], "value": "bicubic", "info": ""}
        da.aspect_ratio_use_old_formula = {"label": "Aspect ratio use old formula", "type": "checkbox", "value": False, "info": ""}
        da.aspect_ratio_schedule = {"label": "Aspect ratio schedule", "type": "textbox", "value": "0: (1.0)", "info": ""}
        da.fov_schedule = {"label": "FOV schedule", "type": "textbox", "value": "0: (70)", "info": ""}
        da.near_schedule = {"label": "Near schedule", "type": "textbox", "value": "0: (200)", "info": ""}
        da.far_schedule = {"label": "Far schedule", "type": "textbox", "value": "0: (10000)", "info": ""}
        
        # Video initialization attributes (non-hybrid)
        da.video_init_path = {"label": "Video init path", "type": "textbox", "value": "", "info": ""}
        da.extract_from_frame = {"label": "Extract from frame", "type": "number", "value": 0, "info": ""}
        da.extract_to_frame = {"label": "Extract to frame", "type": "number", "value": -1, "info": ""}
        da.extract_nth_frame = {"label": "Extract nth frame", "type": "number", "value": 1, "info": ""}
        da.overwrite_extracted_frames = {"label": "Overwrite extracted frames", "type": "checkbox", "value": False, "info": ""}
        da.use_mask_video = {"label": "Use mask video", "type": "checkbox", "value": False, "info": ""}
        da.video_mask_path = {"label": "Video mask path", "type": "textbox", "value": "", "info": ""}
        
        # Additional animation attributes
        da.store_frames_in_ram = {"label": "Store frames in RAM", "type": "checkbox", "value": False, "info": ""}
        
        # Parseq args - basic set  
        dp.parseq_manifest = {"label": "Parseq manifest", "type": "textbox", "value": "", "info": ""}
        dp.parseq_use_deltas = {"label": "Use deltas", "type": "checkbox", "value": True, "info": ""}
        dp.parseq_non_schedule_overrides = {"label": "Parseq non schedule overrides", "type": "checkbox", "value": False, "info": ""}
        
        # Output args - basic set
        dv.fps = {"label": "FPS", "type": "number", "value": 15.0, "info": ""}
        dv.max_video_frames = {"label": "Max video frames", "type": "number", "value": 200, "info": ""}
        dv.add_soundtrack = {"label": "Add soundtrack", "type": "checkbox", "value": False, "info": ""}
        dv.soundtrack_path = {"label": "Soundtrack path", "type": "textbox", "value": "", "info": ""}
        dv.skip_video_for_run_all = {"label": "Skip video for run all", "type": "checkbox", "value": False, "info": ""}
        dv.delete_imgs = {"label": "Delete images", "type": "checkbox", "value": False, "info": ""}
        dv.delete_input_frames = {"label": "Delete input frames", "type": "checkbox", "value": False, "info": ""}
        dv.save_images = {"label": "Save images", "type": "checkbox", "value": True, "info": ""}
        dv.save_individual_images = {"label": "Save individual images", "type": "checkbox", "value": False, "info": ""}
        dv.image_format = {"label": "Image format", "type": "dropdown", "choices": ["PNG", "JPEG"], "value": "PNG", "info": ""}
        dv.jpeg_quality = {"label": "JPEG quality", "type": "slider", "minimum": 1, "maximum": 100, "step": 1, "value": 85, "info": ""}
        dv.ffmpeg_mode = {"label": "FFMPEG mode", "type": "dropdown", "choices": ["auto", "manual"], "value": "auto", "info": ""}
        dv.ffmpeg_outdir = {"label": "FFMPEG output directory", "type": "textbox", "value": "", "info": ""}
        dv.ffmpeg_crf = {"label": "FFMPEG CRF", "type": "slider", "minimum": 0, "maximum": 51, "step": 1, "value": 17, "info": ""}
        dv.ffmpeg_preset = {"label": "FFMPEG preset", "type": "dropdown", "choices": ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], "value": "medium", "info": ""}
        dv.frame_interpolation_engine = {"label": "Frame interpolation engine", "type": "dropdown", "choices": ["None", "RIFE", "FILM"], "value": "None", "info": ""}
        dv.frame_interpolation_x_amount = {"label": "Frame interpolation x amount", "type": "number", "value": 2, "info": ""}
        dv.frame_interpolation_slow_mo_enabled = {"label": "Frame interpolation slow mo enabled", "type": "checkbox", "value": False, "info": ""}
        dv.frame_interpolation_slow_mo_amount = {"label": "Frame interpolation slow mo amount", "type": "number", "value": 2, "info": ""}
        dv.frame_interpolation_keep_imgs = {"label": "Frame interpolation keep images", "type": "checkbox", "value": False, "info": ""}
        dv.frame_interpolation_use_upscaled = {"label": "Frame interpolation use upscaled", "type": "checkbox", "value": False, "info": ""}
        dv.film_interpolation_x_amount = {"label": "FILM interpolation x amount", "type": "number", "value": 2, "info": ""}
        dv.film_interpolation_slow_mo_enabled = {"label": "FILM interpolation slow mo enabled", "type": "checkbox", "value": False, "info": ""}
        dv.film_interpolation_slow_mo_amount = {"label": "FILM interpolation slow mo amount", "type": "number", "value": 2, "info": ""}
        dv.film_interpolation_keep_imgs = {"label": "FILM interpolation keep images", "type": "checkbox", "value": False, "info": ""}
        dv.r_upscale_video = {"label": "Real-ESRGAN upscale video", "type": "checkbox", "value": False, "info": ""}
        dv.r_upscale_factor = {"label": "Real-ESRGAN upscale factor", "type": "dropdown", "choices": ["x2", "x4"], "value": "x2", "info": ""}
        dv.r_upscale_model = {"label": "Real-ESRGAN upscale model", "type": "dropdown", "choices": ["RealESRGAN_x4plus", "RealESRGAN_x2plus"], "value": "RealESRGAN_x4plus", "info": ""}
        dv.r_upscale_keep_imgs = {"label": "Real-ESRGAN upscale keep images", "type": "checkbox", "value": False, "info": ""}
        
        # Root and WAN args - basic set
        dr.timestring = {"label": "Timestring", "type": "textbox", "value": "", "info": ""}
        dw.wan_mode = {"label": "WAN mode", "type": "dropdown", "choices": ["Off"], "value": "Off", "info": ""}
        
        # WAN video attributes - all the essential ones
        dw.wan_auto_download = {"label": "WAN auto download", "type": "checkbox", "value": True, "info": ""}
        dw.wan_preferred_size = {"label": "WAN preferred size", "type": "dropdown", "choices": ["512x512", "576x1024"], "value": "512x512", "info": ""}
        dw.wan_resolution = {"label": "WAN resolution", "type": "dropdown", "choices": ["512x512", "576x1024"], "value": "512x512", "info": ""}
        dw.wan_qwen_model = {"label": "WAN Qwen model", "type": "dropdown", "choices": ["Qwen/Qwen2-VL-2B-Instruct"], "value": "Qwen/Qwen2-VL-2B-Instruct", "info": ""}
        dw.wan_qwen_language = {"label": "WAN Qwen language", "type": "dropdown", "choices": ["English", "Chinese"], "value": "English", "info": ""}
        dw.wan_qwen_auto_download = {"label": "WAN Qwen auto download", "type": "checkbox", "value": True, "info": ""}
        dw.wan_t2v_model = {"label": "WAN T2V model", "type": "dropdown", "choices": ["cog-video-5b"], "value": "cog-video-5b", "info": ""}
        dw.wan_i2v_model = {"label": "WAN I2V model", "type": "dropdown", "choices": ["cog-video-5b-i2v"], "value": "cog-video-5b-i2v", "info": ""}
        dw.wan_model_path = {"label": "WAN model path", "type": "textbox", "value": "", "info": ""}
        dw.wan_strength_override = {"label": "WAN strength override", "type": "checkbox", "value": False, "info": ""}
        dw.wan_fixed_strength = {"label": "WAN fixed strength", "type": "number", "value": 0.8, "info": ""}
        dw.wan_guidance_override = {"label": "WAN guidance override", "type": "checkbox", "value": False, "info": ""}
        dw.wan_guidance_scale = {"label": "WAN guidance scale", "type": "number", "value": 6.0, "info": ""}
        dw.wan_motion_strength_override = {"label": "WAN motion strength override", "type": "checkbox", "value": False, "info": ""}
        dw.wan_motion_strength = {"label": "WAN motion strength", "type": "number", "value": 127, "info": ""}
        dw.wan_movement_sensitivity = {"label": "WAN movement sensitivity", "type": "number", "value": 127, "info": ""}
        dw.wan_frame_overlap = {"label": "WAN frame overlap", "type": "number", "value": 4, "info": ""}
        dw.wan_enable_interpolation = {"label": "WAN enable interpolation", "type": "checkbox", "value": False, "info": ""}
        dw.wan_interpolation_strength = {"label": "WAN interpolation strength", "type": "number", "value": 0.8, "info": ""}
        dw.wan_flash_attention_mode = {"label": "WAN flash attention mode", "type": "dropdown", "choices": ["auto", "enable", "disable"], "value": "auto", "info": ""}
        dw.wan_seed = {"label": "WAN seed", "type": "number", "precision": 0, "value": -1, "info": ""}
        
        # Loop args - ALL REQUIRED ATTRIBUTES
        dloopArgs.use_looper = {"label": "Use looper", "type": "checkbox", "value": False, "info": ""}
        dloopArgs.init_images = {"label": "Init images", "type": "textbox", "value": "", "info": ""}
        dloopArgs.image_strength_schedule = {"label": "Image strength schedule", "type": "textbox", "value": "0: (0.75)", "info": ""}
        dloopArgs.image_keyframe_strength_schedule = {"label": "Image keyframe strength schedule", "type": "textbox", "value": "0: (0.50)", "info": ""}
        dloopArgs.blendFactorMax = {"label": "Blend factor max", "type": "textbox", "value": "0: (0.35)", "info": ""}
        dloopArgs.blendFactorSlope = {"label": "Blend factor slope", "type": "textbox", "value": "0: (0.25)", "info": ""}
        dloopArgs.tweening_frames_schedule = {"label": "Tweening frames schedule", "type": "textbox", "value": "0: (20)", "info": ""}
        dloopArgs.color_correction_factor = {"label": "Color correction factor", "type": "textbox", "value": "0: (0.075)", "info": ""}
        
        return d, da, dp, dv, dr, dw, dloopArgs

# Export everything for backward compatibility
__all__ = [
    'DeforumArgs', 'DeforumAnimArgs', 'ParseqArgs', 'DeforumOutputArgs',
    'RootArgs', 'WanArgs', 'LoopArgs', 'ControlnetArgs',
    'get_component_names', 'process_args', 'get_settings_component_names',
    'set_arg_lists', 'controlnet_component_names'
] 