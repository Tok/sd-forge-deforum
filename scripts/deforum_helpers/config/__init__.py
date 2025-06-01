"""
Functional configuration system for Deforum.

This module provides immutable data structures and pure functions for handling
all Deforum configuration, replacing the mutable dictionary-based args system.

Key principles:
- Immutable data structures (frozen dataclasses)
- Pure functions for validation and conversion
- Separation of concerns (data, UI metadata, validation, processing)
- Type safety throughout
- Zero breaking changes with legacy compatibility
"""

from .argument_models import (
    # Core immutable models
    DeforumGenerationArgs,
    DeforumAnimationArgs,
    DeforumVideoArgs,
    ParseqArgs,
    WanArgs,
    RootArgs,
    
    # Processing result types
    ProcessedArguments,
    ArgumentValidationResult,
)

from .argument_conversion import (
    # Legacy compatibility functions
    create_deforum_args_from_dict,
    create_animation_args_from_dict,
    create_video_args_from_dict,
    create_parseq_args_from_dict,
    create_wan_args_from_dict,
    create_root_args_from_dict,
    
    # Conversion to legacy format
    to_legacy_dict,
    to_legacy_namespace,
)

from .argument_processing import (
    # Pure processing functions
    process_arguments,
    validate_all_arguments,
    merge_arguments,
    apply_argument_overrides,
)

__all__ = [
    # Models
    "DeforumGenerationArgs",
    "DeforumAnimationArgs", 
    "DeforumVideoArgs",
    "ParseqArgs",
    "WanArgs",
    "RootArgs",
    "ProcessedArguments",
    "ArgumentValidationResult",
    
    # Conversion
    "create_deforum_args_from_dict",
    "create_animation_args_from_dict", 
    "create_video_args_from_dict",
    "create_parseq_args_from_dict",
    "create_wan_args_from_dict",
    "create_root_args_from_dict",
    "to_legacy_dict",
    "to_legacy_namespace",
    
    # Processing
    "process_arguments",
    "validate_all_arguments",
    "merge_arguments",
    "apply_argument_overrides",
] 