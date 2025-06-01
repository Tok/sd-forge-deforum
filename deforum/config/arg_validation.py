"""
Argument Validation
Contains validation logic and sanitization for all Deforum arguments
"""

import json
import os
from typing import Dict, Any, Tuple, Optional


def validate_animation_args(anim_args: Dict[str, Any]) -> Tuple[bool, list]:
    """Validate animation arguments for consistency and correctness.
    
    Args:
        anim_args: Dictionary of animation arguments
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Validate max_frames
    if anim_args.get('max_frames', 0) <= 0:
        errors.append("max_frames must be greater than 0")
    
    # Validate animation mode
    valid_modes = ['2D', '3D', 'Interpolation', 'Wan Video']
    if anim_args.get('animation_mode') not in valid_modes:
        errors.append(f"animation_mode must be one of {valid_modes}")
    
    # Validate border mode
    valid_borders = ['replicate', 'wrap']
    if anim_args.get('border') not in valid_borders:
        errors.append(f"border must be one of {valid_borders}")
    
    return len(errors) == 0, errors


def validate_generation_args(args: Dict[str, Any]) -> Tuple[bool, list]:
    """Validate core generation arguments.
    
    Args:
        args: Dictionary of generation arguments
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Validate dimensions
    width = args.get('W', 0)
    height = args.get('H', 0)
    
    if width <= 0 or width % 64 != 0:
        errors.append("Width must be positive and divisible by 64")
    
    if height <= 0 or height % 64 != 0:
        errors.append("Height must be positive and divisible by 64")
    
    # Validate steps
    steps = args.get('steps', 0)
    if steps <= 0 or steps > 150:
        errors.append("Steps must be between 1 and 150")
    
    # Validate CFG scale
    cfg_scale = args.get('cfg_scale', 0)
    if cfg_scale < 0 or cfg_scale > 30:
        errors.append("CFG scale must be between 0 and 30")
    
    # Validate strength
    strength = args.get('strength', 0)
    if strength < 0 or strength > 1:
        errors.append("Strength must be between 0 and 1")
    
    return len(errors) == 0, errors


def validate_output_args(video_args: Dict[str, Any]) -> Tuple[bool, list]:
    """Validate output and video processing arguments.
    
    Args:
        video_args: Dictionary of video output arguments
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Validate FPS
    fps = video_args.get('fps', 0)
    if fps <= 0 or fps > 120:
        errors.append("FPS must be between 1 and 120")
    
    # Validate soundtrack path if specified
    soundtrack_path = video_args.get('soundtrack_path', '')
    add_soundtrack = video_args.get('add_soundtrack', 'None')
    
    if add_soundtrack == 'File' and soundtrack_path:
        # Check if it's a URL or file path
        if not (soundtrack_path.startswith(('http://', 'https://')) or os.path.exists(soundtrack_path)):
            errors.append("Soundtrack path must be a valid URL or existing file")
    
    # Validate interpolation settings
    interp_engine = video_args.get('frame_interpolation_engine', 'None')
    valid_engines = ['None', 'RIFE v4.6', 'FILM']
    if interp_engine not in valid_engines:
        errors.append(f"Frame interpolation engine must be one of {valid_engines}")
    
    interp_amount = video_args.get('frame_interpolation_x_amount', 2)
    if interp_amount < 2 or interp_amount > 10:
        errors.append("Frame interpolation amount must be between 2 and 10")
    
    return len(errors) == 0, errors


def validate_parseq_args(parseq_args: Dict[str, Any]) -> Tuple[bool, list]:
    """Validate Parseq integration arguments.
    
    Args:
        parseq_args: Dictionary of Parseq arguments
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Validate parseq manifest JSON
    manifest = parseq_args.get('parseq_manifest', '')
    if manifest:
        try:
            json.loads(manifest)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid Parseq manifest JSON: {e}")
    
    return len(errors) == 0, errors


def validate_wan_args(wan_args: Dict[str, Any]) -> Tuple[bool, list]:
    """Validate WAN AI integration arguments.
    
    Args:
        wan_args: Dictionary of WAN arguments
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Validate WAN mode
    valid_modes = ['Video Input', 'Image Animation']
    wan_mode = wan_args.get('wan_mode', '')
    if wan_mode not in valid_modes:
        errors.append(f"WAN mode must be one of {valid_modes}")
    
    return len(errors) == 0, errors


def sanitize_strength(strength: float) -> float:
    """Sanitize strength value to valid range.
    
    Args:
        strength: Raw strength value
        
    Returns:
        Clamped strength value between 0.0 and 1.0
    """
    return max(0.0, min(1.0, strength))


def sanitize_seed(seed: int) -> int:
    """Sanitize seed value.
    
    Args:
        seed: Raw seed value
        
    Returns:
        Valid seed value
    """
    if seed == -1:
        return seed  # Keep -1 for random seed
    return max(0, seed)


def validate_all_args(args_dict: Dict[str, Any]) -> Tuple[bool, Dict[str, list]]:
    """Validate all argument categories.
    
    Args:
        args_dict: Dictionary containing all arguments
        
    Returns:
        Tuple of (all_valid, errors_by_category)
    """
    all_errors = {}
    all_valid = True
    
    # Validate each category
    categories = [
        ('animation', validate_animation_args),
        ('generation', validate_generation_args),
        ('output', validate_output_args),
        ('parseq', validate_parseq_args),
        ('wan', validate_wan_args)
    ]
    
    for category, validator in categories:
        is_valid, errors = validator(args_dict)
        if not is_valid:
            all_errors[category] = errors
            all_valid = False
    
    return all_valid, all_errors 