"""
Functional Rendering System for Deforum

This module provides a functional approach to video frame rendering with:
- Immutable frame state management
- Pure rendering functions
- Composable processing pipeline
- Clean separation of concerns
- Functional error handling

The system maintains backward compatibility while providing a cleaner,
more testable architecture for frame generation and processing.
"""

from .frame_models import (
    FrameState,
    RenderContext,
    FrameResult,
    RenderingError,
    ProcessingStage,
    FrameMetadata
)

from .frame_processing import (
    create_frame_state,
    process_frame,
    apply_frame_transformations,
    validate_frame_state,
    merge_frame_results
)

from .rendering_pipeline import (
    RenderingPipeline,
    create_rendering_pipeline,
    execute_pipeline,
    pipeline_with_error_handling
)

from .legacy_renderer import (
    functional_render_animation,
    convert_legacy_args_to_context,
    enable_functional_rendering
)

__all__ = [
    # Core data models
    'FrameState',
    'RenderContext', 
    'FrameResult',
    'RenderingError',
    'ProcessingStage',
    'FrameMetadata',
    
    # Frame processing functions
    'create_frame_state',
    'process_frame',
    'apply_frame_transformations',
    'validate_frame_state',
    'merge_frame_results',
    
    # Pipeline system
    'RenderingPipeline',
    'create_rendering_pipeline',
    'execute_pipeline',
    'pipeline_with_error_handling',
    
    # Legacy compatibility
    'functional_render_animation',
    'convert_legacy_args_to_context',
    'enable_functional_rendering'
]
