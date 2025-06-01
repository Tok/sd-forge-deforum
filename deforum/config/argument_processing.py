"""
Pure functional argument processing and validation.

This module provides pure functions for processing, validating, and transforming
argument configurations in a functional style.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import replace

from .argument_models import (
    ProcessedArguments, ArgumentValidationResult,
    DeforumGenerationArgs, DeforumAnimationArgs, DeforumVideoArgs,
    ParseqArgs, WanArgs, RootArgs
)

from .argument_conversion import (
    create_deforum_args_from_dict, create_animation_args_from_dict,
    create_video_args_from_dict, create_parseq_args_from_dict,
    create_wan_args_from_dict, create_root_args_from_dict
)

# Import general utils with fallback
try:
    from ..general_utils import substitute_placeholders
except ImportError:
    def substitute_placeholders(text: str, replacements: Dict[str, Any]) -> str:
        """Fallback implementation for placeholder substitution"""
        result = text
        for key, value in replacements.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result


def validate_all_arguments(args: ProcessedArguments) -> ArgumentValidationResult:
    """Pure function: validate all processed arguments"""
    errors = []
    warnings = []
    
    try:
        # Validation happens in __post_init__ of each dataclass
        # If we got here, basic validation passed
        
        # Add additional cross-argument validation
        cross_validation_errors = validate_cross_dependencies(args)
        errors.extend(cross_validation_errors)
        
        # Check for potential issues that are warnings, not errors
        validation_warnings = generate_validation_warnings(args)
        warnings.extend(validation_warnings)
        
        return ArgumentValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validated_args=args if len(errors) == 0 else None
        )
        
    except Exception as e:
        return ArgumentValidationResult(
            valid=False,
            errors=[f"Validation error: {str(e)}"],
            warnings=warnings,
            validated_args=None
        )


def validate_cross_dependencies(args: ProcessedArguments) -> List[str]:
    """Pure function: validate cross-dependencies between arguments"""
    errors = []
    
    # Validate init image dependencies
    if args.deforum.use_init and not args.deforum.init_image:
        errors.append("use_init is True but no init_image provided")
    
    # Validate mask dependencies
    if args.deforum.use_mask and not args.deforum.mask_file:
        errors.append("use_mask is True but no mask_file provided")
    
    # Validate video init dependencies
    if args.animation.video_init_path and args.animation.extract_to_frame != -1:
        if args.animation.extract_from_frame >= args.animation.extract_to_frame:
            errors.append("extract_from_frame must be less than extract_to_frame")
    
    # Validate frame count vs max_frames
    if args.animation.extract_to_frame > 0 and args.animation.extract_to_frame < args.animation.max_frames:
        pass  # This is just a warning, not an error
    
    # Validate Wan mode dependencies
    if args.wan.wan_mode != "Disabled":
        if not args.wan.wan_model_path:
            errors.append("Wan mode enabled but no model path provided")
    
    # Validate ControlNet dependencies
    if args.deforum.show_controlnet_tab:
        # Add ControlNet-specific validations here
        pass
    
    return errors


def generate_validation_warnings(args: ProcessedArguments) -> List[str]:
    """Pure function: generate validation warnings for potential issues"""
    warnings = []
    
    # Check for performance warnings
    if args.deforum.width * args.deforum.height > 1024 * 1024:
        warnings.append("Large image dimensions may cause memory issues")
    
    if args.animation.max_frames > 1000:
        warnings.append("Large frame count may take a very long time to render")
    
    # Check for quality warnings
    if args.deforum.steps < 10:
        warnings.append("Low step count may result in poor quality")
    
    if args.deforum.cfg_scale > 20:
        warnings.append("Very high CFG scale may cause artifacts")
    
    # Check for compatibility warnings
    if args.animation.use_depth_warping and args.animation.animation_mode.value == "2D":
        warnings.append("Depth warping has no effect in 2D animation mode")
    
    if args.video.fps > 60:
        warnings.append("High FPS may not be supported by all video players")
    
    return warnings


def process_arguments(args_dict: Dict[str, Any], 
                     timestring: Optional[str] = None,
                     overrides: Optional[Dict[str, Any]] = None) -> ProcessedArguments:
    """Pure function: process raw arguments into validated immutable structures"""
    start_time = time.time()
    
    # Apply overrides if provided
    if overrides:
        processed_dict = apply_overrides_to_dict(args_dict, overrides)
    else:
        processed_dict = args_dict.copy()
        overrides = {}
    
    # Generate timestring if not provided
    if timestring is None:
        timestring = generate_timestring()
    
    # Add timestring to processed dict for placeholder substitution
    processed_dict["timestring"] = timestring
    
    # Apply placeholder substitution
    processed_dict = substitute_argument_placeholders(processed_dict)
    
    # Create immutable argument objects
    deforum_args = create_deforum_args_from_dict(processed_dict)
    animation_args = create_animation_args_from_dict(processed_dict)
    video_args = create_video_args_from_dict(processed_dict)
    parseq_args = create_parseq_args_from_dict(processed_dict)
    wan_args = create_wan_args_from_dict(processed_dict)
    root_args = create_root_args_from_dict(processed_dict)
    
    # Ensure timestring is properly set
    root_args = replace(root_args, timestring=timestring)
    
    processing_time = time.time() - start_time
    
    return ProcessedArguments(
        deforum=deforum_args,
        animation=animation_args,
        video=video_args,
        parseq=parseq_args,
        wan=wan_args,
        root=root_args,
        processing_time=processing_time,
        applied_overrides=overrides
    )


def apply_overrides_to_dict(base_dict: Dict[str, Any], 
                           overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function: apply argument overrides to base dictionary"""
    result = base_dict.copy()
    result.update(overrides)
    return result


def substitute_argument_placeholders(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function: substitute placeholders in argument values"""
    result = {}
    
    for key, value in args_dict.items():
        if isinstance(value, str):
            # Use the existing substitute_placeholders function
            result[key] = substitute_placeholders(value, args_dict)
        else:
            result[key] = value
    
    return result


def generate_timestring() -> str:
    """Pure function: generate timestring for current time"""
    return time.strftime("%Y%m%d%H%M%S")


def merge_arguments(base_args: ProcessedArguments, 
                   override_args: ProcessedArguments) -> ProcessedArguments:
    """Pure function: merge two ProcessedArguments, with override taking precedence"""
    # Use dataclass replace to create new immutable objects with overridden values
    merged_deforum = merge_dataclass_args(base_args.deforum, override_args.deforum)
    merged_animation = merge_dataclass_args(base_args.animation, override_args.animation)
    merged_video = merge_dataclass_args(base_args.video, override_args.video)
    merged_parseq = merge_dataclass_args(base_args.parseq, override_args.parseq)
    merged_wan = merge_dataclass_args(base_args.wan, override_args.wan)
    merged_root = merge_dataclass_args(base_args.root, override_args.root)
    
    # Merge metadata
    merged_overrides = {**base_args.applied_overrides, **override_args.applied_overrides}
    merged_warnings = base_args.validation_warnings + override_args.validation_warnings
    
    return ProcessedArguments(
        deforum=merged_deforum,
        animation=merged_animation,
        video=merged_video,
        parseq=merged_parseq,
        wan=merged_wan,
        root=merged_root,
        processing_time=override_args.processing_time,
        validation_warnings=merged_warnings,
        applied_overrides=merged_overrides
    )


def merge_dataclass_args(base_args, override_args):
    """Pure function: merge two dataclass instances"""
    # Get all fields from override that are not None/default
    override_dict = {}
    
    for field_name in base_args.__dataclass_fields__.keys():
        override_value = getattr(override_args, field_name)
        base_value = getattr(base_args, field_name)
        
        # Only override if the value is different from the default
        if override_value != base_value:
            override_dict[field_name] = override_value
    
    # Create new instance with overridden values
    return replace(base_args, **override_dict)


def apply_argument_overrides(args: ProcessedArguments, 
                           overrides: Dict[str, Any]) -> ProcessedArguments:
    """Pure function: apply specific overrides to processed arguments"""
    # Convert back to dict, apply overrides, then re-process
    from .argument_conversion import to_legacy_dict
    
    legacy_dict = to_legacy_dict(args)
    overridden_dict = apply_overrides_to_dict(legacy_dict, overrides)
    
    # Re-process with new overrides
    return process_arguments(
        overridden_dict, 
        timestring=args.root.timestring,
        overrides=overrides
    )


def extract_schedule_strings(args: ProcessedArguments) -> Dict[str, str]:
    """Pure function: extract all schedule strings for schedule system processing"""
    schedules = {}
    
    # Animation schedules
    animation_schedules = [
        "angle", "zoom", "translation_x", "translation_y", "translation_z",
        "transform_center_x", "transform_center_y", 
        "rotation_3d_x", "rotation_3d_y", "rotation_3d_z",
        "perspective_flip_theta", "perspective_flip_phi", "perspective_flip_gamma", "perspective_flip_fv",
        "noise_schedule", "strength_schedule", "keyframe_strength_schedule",
        "contrast_schedule", "cfg_scale_schedule", "distilled_cfg_scale_schedule",
        "steps_schedule", "fov_schedule", "aspect_ratio_schedule", 
        "near_schedule", "far_schedule", "seed_schedule",
        "subseed_schedule", "subseed_strength_schedule",
        "sampler_schedule", "scheduler_schedule",
        "mask_schedule", "noise_mask_schedule", "checkpoint_schedule",
        "clipskip_schedule", "noise_multiplier_schedule",
        "ddim_eta_schedule", "ancestral_eta_schedule",
        "amount_schedule", "kernel_schedule", "sigma_schedule", "threshold_schedule",
        "cadence_flow_factor_schedule", "redo_flow_factor_schedule",
        "hybrid_comp_alpha_schedule", "hybrid_comp_mask_blend_alpha_schedule",
        "hybrid_comp_mask_contrast_schedule", "hybrid_comp_mask_auto_contrast_cutoff_high_schedule",
        "hybrid_comp_mask_auto_contrast_cutoff_low_schedule", "hybrid_flow_factor_schedule"
    ]
    
    for schedule_name in animation_schedules:
        if hasattr(args.animation, schedule_name):
            schedules[schedule_name] = getattr(args.animation, schedule_name)
    
    return schedules


def validate_schedule_syntax(schedule_string: str) -> Tuple[bool, List[str]]:
    """Pure function: validate schedule string syntax"""
    errors = []
    
    if not schedule_string or not schedule_string.strip():
        errors.append("Empty schedule string")
        return False, errors
    
    # Basic validation - actual parsing would be done by schedule system
    try:
        # Check for basic format: "frame: (value)"
        if ":" not in schedule_string:
            errors.append("Schedule must contain frame:value pairs")
        
        # Check for balanced parentheses
        open_parens = schedule_string.count("(")
        close_parens = schedule_string.count(")")
        if open_parens != close_parens:
            errors.append("Unbalanced parentheses in schedule")
        
    except Exception as e:
        errors.append(f"Schedule syntax error: {str(e)}")
    
    return len(errors) == 0, errors


def get_argument_summary(args: ProcessedArguments) -> Dict[str, Any]:
    """Pure function: generate summary statistics for processed arguments"""
    summary = {
        "total_args": sum([
            len(args.deforum.__dataclass_fields__),
            len(args.animation.__dataclass_fields__),
            len(args.video.__dataclass_fields__),
            len(args.parseq.__dataclass_fields__),
            len(args.wan.__dataclass_fields__),
            len(args.root.__dataclass_fields__)
        ]),
        "processing_time": args.processing_time,
        "applied_overrides_count": len(args.applied_overrides),
        "validation_warnings_count": len(args.validation_warnings),
        "animation_mode": args.animation.animation_mode.value,
        "max_frames": args.animation.max_frames,
        "resolution": f"{args.deforum.width}x{args.deforum.height}",
        "wan_enabled": args.wan.wan_mode != "Disabled",
        "video_creation": not args.video.skip_video_creation,
        "uses_init_image": args.deforum.use_init,
        "uses_mask": args.deforum.use_mask,
        "uses_depth_warping": args.animation.use_depth_warping,
    }
    
    return summary 