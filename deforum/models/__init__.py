"""
Immutable data models and schedules for Deforum.

This package provides immutable, type-safe data structures to replace
the mutable SimpleNamespace and dictionary patterns used throughout Deforum.
"""

from .data_models import (
    AnimationArgs, DeforumArgs, VideoArgs, ParseqArgs, WanArgs, RootArgs,
    ProcessingResult, UIDefaults, SettingsState, ExternalLibraryArgs, TestFixtureArgs,
    create_animation_args_from_dict, create_deforum_args_from_dict, create_video_args_from_dict,
    create_parseq_args_from_dict, create_wan_args_from_dict, create_root_args_from_dict,
    validate_processing_result, validate_ui_defaults,
    ImageTuple, StringTuple, FloatTuple
)

from .schedule_models import (
    AnimationSchedules, ControlNetSchedules, LooperSchedules, ParseqScheduleData
)

__all__ = [
    # Data models
    'AnimationArgs', 'DeforumArgs', 'VideoArgs', 'ParseqArgs', 'WanArgs', 'RootArgs',
    'ProcessingResult', 'UIDefaults', 'SettingsState', 'ExternalLibraryArgs', 'TestFixtureArgs',
    
    # Schedule models
    'AnimationSchedules', 'ControlNetSchedules', 'LooperSchedules', 'ParseqScheduleData',
    
    # Factory functions
    'create_animation_args_from_dict', 'create_deforum_args_from_dict', 'create_video_args_from_dict',
    'create_parseq_args_from_dict', 'create_wan_args_from_dict', 'create_root_args_from_dict',
    
    # Validation functions
    'validate_processing_result', 'validate_ui_defaults',
    
    # Type aliases
    'ImageTuple', 'StringTuple', 'FloatTuple'
] 