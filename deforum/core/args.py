"""
Backward compatibility module for core.args imports.

This module re-exports everything from the config.args module to maintain
compatibility with existing code that imports from deforum.core.args.
"""

# Re-export everything from the new config structure
from ..config.args import *

# Ensure all exports are available
__all__ = [
    'DeforumArgs', 'DeforumAnimArgs', 'ParseqArgs', 'DeforumOutputArgs',
    'RootArgs', 'WanArgs', 'LoopArgs', 'ControlnetArgs',
    'get_component_names', 'process_args', 'get_settings_component_names',
    'set_arg_lists', 'controlnet_component_names'
] 