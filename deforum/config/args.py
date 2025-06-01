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
    # Fallback if models not available
    pass

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
    """Set argument lists - placeholder for compatibility"""
    pass

# Export everything for backward compatibility
__all__ = [
    'DeforumArgs', 'DeforumAnimArgs', 'ParseqArgs', 'DeforumOutputArgs',
    'RootArgs', 'WanArgs', 'LoopArgs', 'ControlnetArgs',
    'get_component_names', 'process_args', 'get_settings_component_names',
    'set_arg_lists', 'controlnet_component_names'
] 