"""
Functional Rendering Pipeline

This module implements a composable, functional rendering pipeline that
processes frames through a series of pure transformations. The pipeline
supports error handling, parallel processing, and progress tracking.
"""

import time
from typing import Callable, List, Tuple, Optional, Dict, Any, Iterator
from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

from .frame_models import (
    FrameState, FrameResult, RenderContext, RenderingSession,
    ProcessingStage, RenderingError, ModelState
)
from .frame_processing import (
    create_frame_state, process_frame, validate_frame_state,
    merge_frame_results
)

# Type aliases for pipeline functions
FrameProcessor = Callable[[FrameState, RenderContext], FrameResult]
PipelineStage = Callable[[FrameState], FrameResult]
ErrorHandler = Callable[[RenderingError, FrameState], FrameResult]
ProgressCallback = Callable[[int, int, FrameResult], None]


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for rendering pipeline."""
    max_workers: int = 1  # Number of parallel workers
    enable_progress_tracking: bool = True
    enable_error_recovery: bool = True
    checkpoint_interval: int = 10  # Save progress every N frames
    memory_limit_mb: Optional[int] = None
    timeout_seconds: Optional[float] = None


@dataclass(frozen=True)
class RenderingPipeline:
    """Immutable rendering pipeline configuration."""
    stages: Tuple[PipelineStage, ...] = field(default_factory=tuple)
    error_handlers: Dict[ProcessingStage, ErrorHandler] = field(default_factory=dict)
    config: PipelineConfig = field(default_factory=PipelineConfig)
    
    def with_stage(self, stage: PipelineStage) -> 'RenderingPipeline':
        """Return new pipeline with added stage."""
        new_stages = self.stages + (stage,)
        return replace(self, stages=new_stages)
    
    def with_error_handler(self, stage: ProcessingStage, handler: ErrorHandler) -> 'RenderingPipeline':
        """Return new pipeline with error handler for specific stage."""
        new_handlers = {**self.error_handlers, stage: handler}
        return replace(self, error_handlers=new_handlers)
    
    def with_config(self, **config_updates) -> 'RenderingPipeline':
        """Return new pipeline with updated configuration."""
        new_config = replace(self.config, **config_updates)
        return replace(self, config=new_config)


def create_rendering_pipeline(
    context: RenderContext,
    custom_stages: Optional[List[PipelineStage]] = None,
    config: Optional[PipelineConfig] = None
) -> RenderingPipeline:
    """
    Create a rendering pipeline based on context and configuration.
    
    Args:
        context: Rendering context
        custom_stages: Optional custom pipeline stages
        config: Optional pipeline configuration
        
    Returns:
        Configured RenderingPipeline
    """
    pipeline_config = config or PipelineConfig()
    
    # Default stages based on context
    default_stages = []
    
    if custom_stages:
        stages = tuple(custom_stages)
    else:
        # Create default processing stage
        def default_processing_stage(frame_state: FrameState) -> FrameResult:
            return process_frame(frame_state, context)
        
        stages = (default_processing_stage,)
    
    # Create error handlers
    error_handlers = {
        ProcessingStage.INITIALIZATION: _default_error_handler,
        ProcessingStage.GENERATION: _generation_error_handler,
        ProcessingStage.POST_PROCESSING: _default_error_handler,
    }
    
    return RenderingPipeline(
        stages=stages,
        error_handlers=error_handlers,
        config=pipeline_config
    )


def _default_error_handler(error: RenderingError, frame_state: FrameState) -> FrameResult:
    """Default error handler that logs and continues."""
    print(f"Warning: {error}")
    return FrameResult(
        frame_state=frame_state,
        success=False,
        error=error
    )


def _generation_error_handler(error: RenderingError, frame_state: FrameState) -> FrameResult:
    """Error handler for generation stage with retry logic."""
    print(f"Generation error: {error}")
    
    # Could implement retry logic here
    # For now, just return the error
    return FrameResult(
        frame_state=frame_state,
        success=False,
        error=error
    )


def execute_pipeline_stage(
    stage: PipelineStage,
    frame_state: FrameState,
    error_handlers: Dict[ProcessingStage, ErrorHandler]
) -> FrameResult:
    """
    Execute a single pipeline stage with error handling.
    
    Args:
        stage: Pipeline stage function
        frame_state: Current frame state
        error_handlers: Error handlers by stage
        
    Returns:
        FrameResult from stage execution
    """
    try:
        result = stage(frame_state)
        return result
    except Exception as e:
        # Create error and try to handle it
        error = RenderingError(
            message=str(e),
            stage=frame_state.stage,
            frame_idx=frame_state.metadata.frame_idx,
            context={'exception_type': type(e).__name__}
        )
        
        # Try to find appropriate error handler
        handler = error_handlers.get(frame_state.stage, _default_error_handler)
        return handler(error, frame_state)


def execute_pipeline(
    pipeline: RenderingPipeline,
    frame_states: List[FrameState],
    progress_callback: Optional[ProgressCallback] = None
) -> Tuple[FrameResult, ...]:
    """
    Execute rendering pipeline on multiple frames.
    
    Args:
        pipeline: Rendering pipeline to execute
        frame_states: List of frame states to process
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of frame results
    """
    if not frame_states:
        return tuple()
    
    results = []
    
    if pipeline.config.max_workers == 1:
        # Sequential processing
        for i, frame_state in enumerate(frame_states):
            result = _process_single_frame(pipeline, frame_state)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(frame_states), result)
    else:
        # Parallel processing
        results = _process_frames_parallel(
            pipeline, frame_states, progress_callback
        )
    
    return tuple(results)


def _process_single_frame(
    pipeline: RenderingPipeline,
    frame_state: FrameState
) -> FrameResult:
    """
    Process a single frame through the pipeline.
    
    Args:
        pipeline: Rendering pipeline
        frame_state: Frame state to process
        
    Returns:
        FrameResult from processing
    """
    current_state = frame_state
    
    # Execute each stage in sequence
    for stage in pipeline.stages:
        if current_state is None:
            break
            
        result = execute_pipeline_stage(
            stage, current_state, pipeline.error_handlers
        )
        
        if not result.success and not pipeline.config.enable_error_recovery:
            return result
            
        current_state = result.frame_state
    
    # Return final result
    return FrameResult(
        frame_state=current_state,
        success=True
    )


def _process_frames_parallel(
    pipeline: RenderingPipeline,
    frame_states: List[FrameState],
    progress_callback: Optional[ProgressCallback] = None
) -> List[FrameResult]:
    """
    Process frames in parallel using ThreadPoolExecutor.
    
    Args:
        pipeline: Rendering pipeline
        frame_states: Frame states to process
        progress_callback: Optional progress callback
        
    Returns:
        List of frame results in order
    """
    results = [None] * len(frame_states)
    completed_count = 0
    lock = threading.Lock()
    
    def process_with_index(index: int, frame_state: FrameState) -> Tuple[int, FrameResult]:
        result = _process_single_frame(pipeline, frame_state)
        return index, result
    
    def update_progress(result: FrameResult) -> None:
        nonlocal completed_count
        with lock:
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, len(frame_states), result)
    
    with ThreadPoolExecutor(max_workers=pipeline.config.max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_with_index, i, frame_state): i
            for i, frame_state in enumerate(frame_states)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            try:
                index, result = future.result(timeout=pipeline.config.timeout_seconds)
                results[index] = result
                update_progress(result)
                        
            except Exception as e:
                index = future_to_index[future]
                error_result = FrameResult(
                    frame_state=frame_states[index],
                    success=False,
                    error=RenderingError(
                        f"Parallel processing failed: {str(e)}",
                        ProcessingStage.INITIALIZATION,
                        frame_states[index].metadata.frame_idx
                    )
                )
                results[index] = error_result
                update_progress(error_result)
    
    return results


def pipeline_with_error_handling(
    base_pipeline: RenderingPipeline,
    retry_count: int = 3,
    fallback_processor: Optional[FrameProcessor] = None
) -> RenderingPipeline:
    """
    Wrap pipeline with enhanced error handling and retry logic.
    
    Args:
        base_pipeline: Base pipeline to wrap
        retry_count: Number of retries for failed frames
        fallback_processor: Optional fallback processor
        
    Returns:
        Enhanced pipeline with error handling
    """
    def retry_error_handler(error: RenderingError, frame_state: FrameState) -> FrameResult:
        """Error handler with retry logic."""
        for attempt in range(retry_count):
            try:
                # Try processing again
                result = _process_single_frame(base_pipeline, frame_state)
                if result.success:
                    return result.with_warning(f"Succeeded after {attempt + 1} retries")
            except Exception as e:
                if attempt == retry_count - 1:
                    # Last attempt failed
                    if fallback_processor:
                        try:
                            return fallback_processor(frame_state, RenderContext(
                                output_dir=Path("/tmp"),
                                timestring="fallback",
                                width=512, height=512,
                                max_frames=1, fps=30.0,
                                animation_mode="2D",
                                use_depth_warping=False,
                                save_depth_maps=False,
                                hybrid_composite="None",
                                hybrid_motion="None",
                                depth_algorithm="midas",
                                optical_flow_cadence="None",
                                diffusion_cadence=1
                            ))
                        except Exception:
                            pass
        
        # All retries failed
        return FrameResult(
            frame_state=frame_state,
            success=False,
            error=error
        )
    
    # Add retry handler to all stages
    enhanced_handlers = {
        stage: retry_error_handler
        for stage in ProcessingStage
    }
    
    return base_pipeline.with_error_handler(ProcessingStage.GENERATION, retry_error_handler)


def create_frame_generator(
    context: RenderContext,
    start_frame: int = 0,
    metadata_overrides: Optional[Dict[str, Any]] = None
) -> Iterator[FrameState]:
    """
    Create a generator that yields frame states for processing.
    
    Args:
        context: Rendering context
        start_frame: Starting frame index
        metadata_overrides: Optional metadata overrides
        
    Yields:
        FrameState objects for processing
    """
    for frame_idx in range(start_frame, context.max_frames):
        frame_metadata = metadata_overrides or {}
        frame_metadata['frame_idx'] = frame_idx
        
        yield create_frame_state(frame_idx, context, frame_metadata)


def render_animation_functional(
    context: RenderContext,
    pipeline: Optional[RenderingPipeline] = None,
    start_frame: int = 0,
    progress_callback: Optional[ProgressCallback] = None
) -> RenderingSession:
    """
    Render animation using functional pipeline approach.
    
    Args:
        context: Rendering context
        pipeline: Optional custom pipeline
        start_frame: Starting frame index
        progress_callback: Optional progress callback
        
    Returns:
        RenderingSession with results
    """
    start_time = time.time()
    
    # Create pipeline if not provided
    if pipeline is None:
        pipeline = create_rendering_pipeline(context)
    
    # Generate frame states
    frame_states = list(create_frame_generator(context, start_frame))
    
    # Execute pipeline
    frame_results = execute_pipeline(pipeline, frame_states, progress_callback)
    
    # Create session result
    model_state = ModelState()  # Would be populated with actual model state
    
    session = RenderingSession(
        context=context,
        model_state=model_state,
        frame_results=frame_results,
        session_start_time=start_time,
        total_processing_time=time.time() - start_time
    )
    
    return session


def create_progress_tracker(verbose: bool = True) -> ProgressCallback:
    """
    Create a progress tracking callback.
    
    Args:
        verbose: Whether to print detailed progress
        
    Returns:
        Progress callback function
    """
    def progress_callback(completed: int, total: int, result: FrameResult) -> None:
        if verbose:
            status = "✓" if result.success else "✗"
            percentage = (completed / total) * 100
            print(f"[{percentage:6.2f}%] Frame {result.frame_idx:3d}/{total:3d} {status}")
            
            if result.warnings:
                for warning in result.warnings:
                    print(f"  Warning: {warning}")
            
            if result.error:
                print(f"  Error: {result.error.message}")
    
    return progress_callback


# Utility functions for pipeline composition
def compose_pipelines(*pipelines: RenderingPipeline) -> RenderingPipeline:
    """
    Compose multiple pipelines into a single pipeline.
    
    Args:
        *pipelines: Pipelines to compose
        
    Returns:
        Composed pipeline
    """
    if not pipelines:
        return RenderingPipeline()
    
    if len(pipelines) == 1:
        return pipelines[0]
    
    # Combine all stages
    all_stages = []
    all_error_handlers = {}
    
    for pipeline in pipelines:
        all_stages.extend(pipeline.stages)
        all_error_handlers.update(pipeline.error_handlers)
    
    # Use config from first pipeline
    return RenderingPipeline(
        stages=tuple(all_stages),
        error_handlers=all_error_handlers,
        config=pipelines[0].config
    )


def create_conditional_stage(
    condition: Callable[[FrameState], bool],
    true_stage: PipelineStage,
    false_stage: Optional[PipelineStage] = None
) -> PipelineStage:
    """
    Create a conditional pipeline stage.
    
    Args:
        condition: Function to test frame state
        true_stage: Stage to execute if condition is true
        false_stage: Optional stage to execute if condition is false
        
    Returns:
        Conditional pipeline stage
    """
    def conditional_stage(frame_state: FrameState) -> FrameResult:
        if condition(frame_state):
            return true_stage(frame_state)
        elif false_stage:
            return false_stage(frame_state)
        else:
            # Pass through unchanged
            return FrameResult(frame_state=frame_state, success=True)
    
    return conditional_stage 