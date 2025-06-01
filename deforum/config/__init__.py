"""
Deforum Configuration Package

Configuration management for Deforum:
- Argument parsing and validation
- Settings management
- Default configurations
- Configuration transformations

All configuration follows immutable patterns with functional transformations.
"""

# Core arguments always available
try:
    from .arguments import *
    ARGUMENTS_AVAILABLE = True
except ImportError:
    ARGUMENTS_AVAILABLE = False

# Settings and defaults with conditional import
try:
    from .settings import *
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

try:
    from .defaults import *
    DEFAULTS_AVAILABLE = True
except ImportError:
    DEFAULTS_AVAILABLE = False

# Export what's available
__all__ = []

if ARGUMENTS_AVAILABLE:
    __all__.extend([
        'RootArgs', 'DeforumAnimArgs', 'DeforumArgs', 'ParseqArgs', 'WanArgs', 'DeforumOutputArgs',
        'get_component_names', 'get_settings_component_names', 'pack_args', 'process_args',
        'validate_animation_args', 'validate_generation_args', 'validate_output_args',
        'validate_parseq_args', 'validate_wan_args', 'validate_all_args',
        'sanitize_strength', 'sanitize_seed'
    ])

if SETTINGS_AVAILABLE:
    __all__.extend([
        'load_settings', 'save_settings', 'get_setting', 'update_settings'
    ])

if DEFAULTS_AVAILABLE:
    __all__.extend([
        'get_default_args', 'get_default_animation_args', 'get_default_video_args'
    ])

# Configuration utilities
__all__.extend([
    'merge_configurations',
    'validate_configuration',
]) 