"""
Prompt Enhancement Module

Modular prompt enhancement system following functional programming principles.
This module provides a clean interface to the underlying enhancement components.
"""

# Import main data models and types
from .data_models import (
    PromptLanguage,
    PromptStyle,
    ModelType,
    ModelSpec,
    PromptEnhancementRequest,
    PromptEnhancementResult,
    EnhancementConfig,
    MODEL_SPECS
)

# Import model interface
from .model_interface import (
    ModelInferenceService,
    auto_select_model,
    get_system_prompt,
    validate_model_selection,
    get_model_spec,
    get_available_models,
    filter_models_by_vram,
    get_models_by_type
)

# Import core processing functions
from .core_processing import (
    normalize_language,
    normalize_style,
    validate_prompts_dict,
    build_style_theme_modifier,
    apply_style_to_prompt,
    clean_prompt_text,
    extract_quoted_text,
    restore_quoted_text,
    validate_prompt_length,
    calculate_prompt_complexity,
    detect_prompt_language_heuristic,
    merge_prompt_segments,
    split_prompt_by_frames,
    create_prompt_request_from_legacy
)

# Import main enhancement functions
from .enhancement_engine import (
    enhance_single_prompt,
    enhance_prompts_batch,
    enhance_prompts_main as enhance_prompts,
    enhance_prompts_simple,
    enhance_prompts_legacy_interface,
    validate_enhancement_request,
    estimate_processing_time,
    calculate_enhancement_quality_score,
    batch_quality_analysis
)

# Import analysis and reporting
from .analysis_reporting import (
    get_enhancement_statistics,
    format_enhancement_report,
    calculate_processing_metrics,
    analyze_enhancement_quality,
    generate_detailed_report,
    compare_enhancement_results,
    export_results_to_dict
)

# Define public API
__all__ = [
    # Data models
    "PromptLanguage",
    "PromptStyle", 
    "ModelType",
    "ModelSpec",
    "PromptEnhancementRequest",
    "PromptEnhancementResult",
    "EnhancementConfig",
    "MODEL_SPECS",
    
    # Model interface
    "ModelInferenceService",
    "auto_select_model",
    "get_system_prompt",
    "validate_model_selection",
    "get_model_spec",
    "get_available_models",
    "filter_models_by_vram",
    "get_models_by_type",
    
    # Core processing
    "normalize_language",
    "normalize_style",
    "validate_prompts_dict",
    "build_style_theme_modifier",
    "apply_style_to_prompt",
    "clean_prompt_text",
    "extract_quoted_text",
    "restore_quoted_text",
    "validate_prompt_length",
    "calculate_prompt_complexity",
    "detect_prompt_language_heuristic",
    "merge_prompt_segments",
    "split_prompt_by_frames",
    "create_prompt_request_from_legacy",
    
    # Main enhancement functions
    "enhance_single_prompt",
    "enhance_prompts_batch",
    "enhance_prompts",
    "enhance_prompts_simple",
    "enhance_prompts_legacy_interface",
    "validate_enhancement_request",
    "estimate_processing_time",
    "calculate_enhancement_quality_score",
    "batch_quality_analysis",
    
    # Analysis and reporting
    "get_enhancement_statistics",
    "format_enhancement_report",
    "calculate_processing_metrics",
    "analyze_enhancement_quality",
    "generate_detailed_report",
    "compare_enhancement_results",
    "export_results_to_dict"
]

# Version info
__version__ = "2.10.2"
__author__ = "Deforum Team"
__description__ = "Modular prompt enhancement system with functional programming principles"
