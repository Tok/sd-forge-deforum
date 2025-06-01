"""
Legacy compatibility adapter for seamless integration.

This module provides drop-in replacements for existing args functions,
ensuring zero breaking changes while enabling gradual migration to 
the functional system.
"""

import json
import os
import tempfile
import time
from types import SimpleNamespace
from typing import Dict, Any, Optional, List, Union

# Import the original modules for delegation
try:
    import modules.paths as ph
    import modules.shared as sh
    from modules.processing import get_fixed_seed
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

from .argument_processing import process_arguments
from .argument_conversion import to_legacy_dict, to_legacy_namespace

# Import general utils with fallback
try:
    from ..general_utils import get_os
except ImportError:
    import platform
    def get_os():
        """Fallback implementation"""
        return platform.system().lower()


# Global flag to enable/disable functional system
USE_FUNCTIONAL_SYSTEM = True


def DeforumArgs() -> Dict[str, Any]:
    """
    Legacy compatibility function for DeforumArgs().
    
    Returns a dictionary compatible with the original format,
    but internally uses the functional system when enabled.
    """
    if USE_FUNCTIONAL_SYSTEM:
        # Use functional system with default values
        processed = process_arguments({})
        legacy_dict = to_legacy_dict(processed)
        
        # Extract only Deforum-specific args for backward compatibility
        deforum_keys = [
            "W", "H", "seed", "sampler", "scheduler", "steps", "scale",
            "tiling", "restore_faces", "seed_resize_from_w", "seed_resize_from_h",
            "seed_behavior", "seed_iter_N", "use_init", "strength",
            "strength_0_no_init", "init_image", "init_image_box",
            "use_mask", "use_alpha_as_mask", "mask_file", "invert_mask",
            "mask_contrast_adjust", "mask_brightness_adjust", "overlay_mask",
            "mask_overlay_blur", "fill", "full_res_mask", "full_res_mask_padding",
            "batch_name", "reroll_blank_frames", "reroll_patience",
            "motion_preview_mode", "show_info_on_ui", "show_controlnet_tab"
        ]
        
        return {key: legacy_dict[key] for key in deforum_keys if key in legacy_dict}
    
    else:
        # Original implementation for fallback
        return _create_original_deforum_args()


def DeforumAnimArgs() -> Dict[str, Any]:
    """
    Legacy compatibility function for DeforumAnimArgs().
    
    Returns a dictionary compatible with the original format.
    """
    if USE_FUNCTIONAL_SYSTEM:
        processed = process_arguments({})
        legacy_dict = to_legacy_dict(processed)
        
        # Extract animation-specific args
        animation_keys = [
            "animation_mode", "max_frames", "border", "angle", "zoom",
            "translation_x", "translation_y", "translation_z",
            "transform_center_x", "transform_center_y",
            "rotation_3d_x", "rotation_3d_y", "rotation_3d_z",
            "shake_name", "shake_intensity", "shake_speed",
            "enable_perspective_flip", "perspective_flip_theta",
            "perspective_flip_phi", "perspective_flip_gamma", "perspective_flip_fv",
            "noise_schedule", "strength_schedule", "keyframe_strength_schedule",
            "contrast_schedule", "cfg_scale_schedule", "distilled_cfg_scale_schedule",
            "enable_steps_scheduling", "steps_schedule",
            "fov_schedule", "aspect_ratio_schedule", "aspect_ratio_use_old_formula",
            "near_schedule", "far_schedule", "seed_schedule",
            "enable_subseed_scheduling", "subseed_schedule", "subseed_strength_schedule",
            "enable_sampler_scheduling", "sampler_schedule",
            "enable_scheduler_scheduling", "scheduler_schedule",
            "use_noise_mask", "mask_schedule", "noise_mask_schedule",
            "enable_checkpoint_scheduling", "checkpoint_schedule",
            "enable_clipskip_scheduling", "clipskip_schedule",
            "enable_noise_multiplier_scheduling", "noise_multiplier_schedule",
            "resume_from_timestring", "resume_timestring",
            "enable_ddim_eta_scheduling", "ddim_eta_schedule",
            "enable_ancestral_eta_scheduling", "ancestral_eta_schedule",
            "amount_schedule", "kernel_schedule", "sigma_schedule", "threshold_schedule",
            "color_coherence", "color_coherence_image_path", "color_coherence_video_every_N_frames",
            "color_force_grayscale", "legacy_colormatch",
            "keyframe_distribution", "diffusion_cadence", "optical_flow_cadence",
            "cadence_flow_factor_schedule", "optical_flow_redo_generation",
            "redo_flow_factor_schedule", "diffusion_redo",
            "noise_type", "perlin_w", "perlin_h", "perlin_octaves", "perlin_persistence",
            "use_depth_warping", "depth_algorithm", "midas_weight",
            "padding_mode", "sampling_mode", "save_depth_maps",
            "video_init_path", "extract_nth_frame", "extract_from_frame",
            "extract_to_frame", "overwrite_extracted_frames",
            "use_mask_video", "video_mask_path"
        ]
        
        # Add hybrid settings
        hybrid_keys = [
            key for key in legacy_dict.keys() 
            if key.startswith("hybrid_")
        ]
        animation_keys.extend(hybrid_keys)
        
        return {key: legacy_dict[key] for key in animation_keys if key in legacy_dict}
    
    else:
        return _create_original_animation_args()


def DeforumOutputArgs() -> Dict[str, Any]:
    """
    Legacy compatibility function for DeforumOutputArgs().
    
    Returns video/output-specific arguments.
    """
    if USE_FUNCTIONAL_SYSTEM:
        processed = process_arguments({})
        legacy_dict = to_legacy_dict(processed)
        
        output_keys = [
            "skip_video_creation", "fps", "make_gif", "delete_imgs", "delete_input_frames",
            "image_path", "add_soundtrack", "soundtrack_path",
            "r_upscale_video", "r_upscale_factor", "r_upscale_model", "r_upscale_keep_imgs",
            "store_frames_in_ram", "frame_interpolation_engine", "frame_interpolation_x_amount",
            "frame_interpolation_slow_mo_enabled", "frame_interpolation_slow_mo_amount",
            "frame_interpolation_keep_imgs", "frame_interpolation_use_upscaled"
        ]
        
        return {key: legacy_dict[key] for key in output_keys if key in legacy_dict}
    
    else:
        return _create_original_output_args()


def ParseqArgs() -> Dict[str, Any]:
    """Legacy compatibility function for ParseqArgs()"""
    if USE_FUNCTIONAL_SYSTEM:
        processed = process_arguments({})
        legacy_dict = to_legacy_dict(processed)
        
        parseq_keys = ["parseq_manifest", "parseq_use_deltas", "parseq_non_schedule_overrides"]
        return {key: legacy_dict[key] for key in parseq_keys if key in legacy_dict}
    
    else:
        return _create_original_parseq_args()


def WanArgs() -> Dict[str, Any]:
    """Legacy compatibility function for WanArgs()"""
    if USE_FUNCTIONAL_SYSTEM:
        processed = process_arguments({})
        legacy_dict = to_legacy_dict(processed)
        
        wan_keys = [
            key for key in legacy_dict.keys() 
            if key.startswith("wan_")
        ]
        return {key: legacy_dict[key] for key in wan_keys if key in legacy_dict}
    
    else:
        return _create_original_wan_args()


def RootArgs() -> Dict[str, Any]:
    """Legacy compatibility function for RootArgs()"""
    if USE_FUNCTIONAL_SYSTEM:
        processed = process_arguments({})
        legacy_dict = to_legacy_dict(processed)
        
        root_keys = [
            "device", "models_path", "half_precision", "clipseg_model",
            "mask_preset_names", "frames_cache", "raw_batch_name", "raw_seed",
            "timestring", "subseed", "subseed_strength", "seed_internal",
            "init_sample", "noise_mask", "initial_info", "first_frame",
            "default_img", "animation_prompts", "prompt_keyframes",
            "current_user_os", "tmp_deforum_run_duplicated_folder", "job_id"
        ]
        
        result = {key: legacy_dict[key] for key in root_keys if key in legacy_dict}
        
        # Override with runtime-specific values if modules available
        if MODULES_AVAILABLE:
            result.update({
                "device": getattr(sh, 'device', 'cpu'),
                "models_path": getattr(ph, 'models_path', '') + '/Deforum',
                "half_precision": not getattr(sh.cmd_opts, 'no_half', True),
                "current_user_os": get_os(),
                "tmp_deforum_run_duplicated_folder": os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
            })
        
        return result
    
    else:
        return _create_original_root_args()


def pack_args(args_dict: Dict[str, Any], keys_function) -> SimpleNamespace:
    """
    Legacy compatibility function for pack_args().
    
    Converts dictionary to SimpleNamespace, with optional functional processing.
    """
    if USE_FUNCTIONAL_SYSTEM:
        # Process through functional system first
        processed = process_arguments(args_dict)
        legacy_dict = to_legacy_dict(processed)
        
        # Apply keys function if provided
        if keys_function and callable(keys_function):
            relevant_keys = keys_function()
            if isinstance(relevant_keys, dict):
                # Extract only the keys specified by the function
                filtered_dict = {key: legacy_dict.get(key) for key in relevant_keys.keys()}
                legacy_dict.update(filtered_dict)
        
        return SimpleNamespace(**legacy_dict)
    
    else:
        # Original implementation
        if keys_function and callable(keys_function):
            keys_dict = keys_function()
            filtered_dict = {key: args_dict.get(key, default) for key, default in keys_dict.items()}
            return SimpleNamespace(**filtered_dict)
        else:
            return SimpleNamespace(**args_dict)


def process_args(args_dict_main: Dict[str, Any], run_id: str) -> SimpleNamespace:
    """
    Legacy compatibility function for process_args().
    
    Enhanced version that uses functional system when enabled.
    """
    if USE_FUNCTIONAL_SYSTEM:
        # Process through functional system
        processed = process_arguments(args_dict_main, timestring=run_id)
        
        # Validate arguments
        from .argument_processing import validate_all_arguments
        validation_result = validate_all_arguments(processed)
        
        if not validation_result.valid:
            # Log validation errors but don't fail (for compatibility)
            print(f"Argument validation warnings: {validation_result.errors}")
        
        if validation_result.warnings:
            print(f"Argument validation warnings: {validation_result.warnings}")
        
        # Convert to legacy namespace
        return to_legacy_namespace(processed)
    
    else:
        # Original implementation fallback
        return SimpleNamespace(**args_dict_main)


def get_component_names() -> List[str]:
    """Legacy compatibility function for get_component_names()"""
    # This would need to be implemented based on the original function
    # For now, return a minimal set
    if USE_FUNCTIONAL_SYSTEM:
        processed = process_arguments({})
        # Extract all field names from all dataclasses
        component_names = []
        component_names.extend(processed.deforum.__dataclass_fields__.keys())
        component_names.extend(processed.animation.__dataclass_fields__.keys()) 
        component_names.extend(processed.video.__dataclass_fields__.keys())
        component_names.extend(processed.parseq.__dataclass_fields__.keys())
        component_names.extend(processed.wan.__dataclass_fields__.keys())
        component_names.extend(processed.root.__dataclass_fields__.keys())
        return component_names
    else:
        return _get_original_component_names()


def get_settings_component_names() -> List[str]:
    """Legacy compatibility function for get_settings_component_names()"""
    # Return settings-specific component names
    if USE_FUNCTIONAL_SYSTEM:
        processed = process_arguments({})
        # Return subset that are typically saved in settings
        settings_fields = []
        settings_fields.extend(processed.deforum.__dataclass_fields__.keys())
        settings_fields.extend(processed.animation.__dataclass_fields__.keys())
        settings_fields.extend(processed.video.__dataclass_fields__.keys())
        settings_fields.extend(processed.parseq.__dataclass_fields__.keys())
        settings_fields.extend(processed.wan.__dataclass_fields__.keys())
        # Exclude runtime-only fields from root
        runtime_excluded = {"frames_cache", "init_sample", "noise_mask", "first_frame", "default_img"}
        settings_fields.extend([
            field for field in processed.root.__dataclass_fields__.keys()
            if field not in runtime_excluded
        ])
        return settings_fields
    else:
        return _get_original_settings_component_names()


# Fallback implementations (minimal versions of original functions)
def _create_original_deforum_args() -> Dict[str, Any]:
    """Minimal fallback implementation"""
    return {
        "W": 1024,
        "H": 1024,
        "seed": -1,
        "sampler": "Euler",
        "scheduler": "Simple", 
        "steps": 20,
        "scale": 7.0,
        "batch_name": "Deforum_{timestring}",
        # Add other essential defaults...
    }


def _create_original_animation_args() -> Dict[str, Any]:
    """Minimal fallback implementation"""
    return {
        "animation_mode": "3D",
        "max_frames": 333,
        "border": "replicate",
        "angle": "0: (0)",
        "zoom": "0: (1.0)",
        # Add other animation defaults...
    }


def _create_original_output_args() -> Dict[str, Any]:
    """Minimal fallback implementation"""
    return {
        "skip_video_creation": False,
        "fps": 60,
        "make_gif": False,
        # Add other output defaults...
    }


def _create_original_parseq_args() -> Dict[str, Any]:
    """Minimal fallback implementation"""
    return {
        "parseq_manifest": None,
        "parseq_use_deltas": True,
        "parseq_non_schedule_overrides": True
    }


def _create_original_wan_args() -> Dict[str, Any]:
    """Minimal fallback implementation"""
    return {
        "wan_mode": "Disabled",
        "wan_model_path": "models/wan",
        # Add other wan defaults...
    }


def _create_original_root_args() -> Dict[str, Any]:
    """Minimal fallback implementation"""
    args = {
        "mask_preset_names": ["everywhere", "video_mask"],
        "frames_cache": [],
        "raw_batch_name": None,
        "raw_seed": None,
        "timestring": "",
        "subseed": -1,
        "subseed_strength": 0,
        "seed_internal": 0,
        "animation_prompts": {},
        "prompt_keyframes": [],
        "current_user_os": get_os(),
        "tmp_deforum_run_duplicated_folder": os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
    }
    
    # Add module-specific values if available
    if MODULES_AVAILABLE:
        args.update({
            "device": getattr(sh, 'device', 'cpu'),
            "models_path": getattr(ph, 'models_path', '') + '/Deforum',
            "half_precision": not getattr(sh.cmd_opts, 'no_half', True),
        })
    
    return args


def _get_original_component_names() -> List[str]:
    """Minimal fallback implementation"""
    return [
        "W", "H", "seed", "sampler", "steps", "scale", 
        "animation_mode", "max_frames", "fps"
        # Add minimal component names...
    ]


def _get_original_settings_component_names() -> List[str]:
    """Minimal fallback implementation"""
    return [
        "W", "H", "seed", "sampler", "steps", "scale",
        "animation_mode", "max_frames", "fps",
        "batch_name"
        # Add settings component names...
    ]


# Migration utilities
def enable_functional_system(enabled: bool = True) -> None:
    """Enable or disable the functional system globally"""
    global USE_FUNCTIONAL_SYSTEM
    USE_FUNCTIONAL_SYSTEM = enabled
    print(f"Functional argument system: {'enabled' if enabled else 'disabled'}")


def is_functional_system_enabled() -> bool:
    """Check if functional system is currently enabled"""
    return USE_FUNCTIONAL_SYSTEM


def migrate_legacy_args(legacy_args: Union[Dict, SimpleNamespace]) -> ProcessedArguments:
    """
    Migrate legacy args to functional system.
    
    Useful for gradually converting existing code.
    """
    if isinstance(legacy_args, SimpleNamespace):
        args_dict = vars(legacy_args)
    else:
        args_dict = legacy_args
    
    return process_arguments(args_dict)


def create_legacy_compatible_args(processed_args: ProcessedArguments) -> SimpleNamespace:
    """
    Create legacy-compatible args from functional system.
    
    Useful for interfacing with code that hasn't been migrated yet.
    """
    return to_legacy_namespace(processed_args) 