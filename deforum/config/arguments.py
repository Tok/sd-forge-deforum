"""
Arguments Module - Backward Compatibility Layer
Provides unified interface to argument system components
"""

# Import all functionality from split modules
from .arg_defaults import (
    RootArgs, DeforumAnimArgs, DeforumArgs, ParseqArgs, WanArgs, DeforumOutputArgs
)
from .arg_validation import (
    validate_animation_args, validate_generation_args, validate_output_args,
    validate_parseq_args, validate_wan_args, validate_all_args,
    sanitize_strength, sanitize_seed
)
from .arg_transformations import (
    get_component_names, get_settings_component_names, pack_args, process_args
)

# Re-export for backward compatibility
__all__ = [
    # Default functions
    'RootArgs', 'DeforumAnimArgs', 'DeforumArgs', 'ParseqArgs', 'WanArgs', 'DeforumOutputArgs',
    
    # Validation functions
    'validate_animation_args', 'validate_generation_args', 'validate_output_args',
    'validate_parseq_args', 'validate_wan_args', 'validate_all_args',
    'sanitize_strength', 'sanitize_seed',
    
    # Transformation functions
    'get_component_names', 'get_settings_component_names', 'pack_args', 'process_args'
]
