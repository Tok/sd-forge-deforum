"""
Deforum Configuration Package

Configuration management for Deforum:
- Argument parsing and validation
- Settings management
- Default configurations
- Configuration transformations

All configuration follows immutable patterns with functional transformations.
"""

from .arguments import *
from .settings import *
from .defaults import *

__all__ = [
    # Argument processing
    'parse_arguments',
    'validate_arguments',
    'transform_legacy_args',
    
    # Settings management
    'load_settings',
    'save_settings',
    'get_setting',
    'update_settings',
    
    # Default configurations
    'get_default_args',
    'get_default_animation_args',
    'get_default_video_args',
    
    # Configuration utilities
    'merge_configurations',
    'validate_configuration',
] 