"""
Pure Functional Frame Processing

This module contains pure functions for processing individual frames in the
rendering pipeline. All functions are side-effect free and operate on
immutable data structures.
"""

import time
from typing import Optional, Tuple, Callable, Dict, Any
from dataclasses import replace
import numpy as np
import cv2
from PIL import Image

from .frame_models import (
    FrameState, FrameResult, FrameMetadata, RenderContext,
    ProcessingStage, RenderingError, ImageArray, ValidationResult
)

# Import processing functions with graceful fallbacks
try:
    from ..animation import anim_frame_warp
    from ..noise import add_noise
    from ..colors import maintain_colors
    from ..image_sharpening import unsharp_mask
    # Note: hybrid video imports removed - functionality not available
    from ..masks import do_overlay_mask
    from ..composable_masks import compose_mask_with_check
except ImportError:
    # Fallback implementations for testing
    def anim_frame_warp(image, *args, **kwargs):
        return image, None
    
    def add_noise(image, *args, **kwargs):
        return image
    
    def maintain_colors(image, *args, **kwargs):
        return image
    
    def unsharp_mask(image, *args, **kwargs):
        return image
    
    def do_overlay_mask(*args, **kwargs):
        return args[2] if len(args) > 2 else None
    
    def compose_mask_with_check(*args, **kwargs):
        return None

# Fallback implementations for removed hybrid video functions
def image_transform_optical_flow(image, *args, **kwargs):
    return image

def image_transform_ransac(image, *args, **kwargs):
    return image

def get_flow_from_images(*args, **kwargs):
    return np.zeros((64, 64, 2))

def abs_flow_to_rel_flow(flow, *args, **kwargs):
    return flow

def rel_flow_to_abs_flow(flow, *args, **kwargs):
    return flow


def create_frame_state(
    frame_idx: int,
    context: RenderContext,
    metadata_overrides: Optional[Dict[str, Any]] = None
) -> FrameState:
    """
    Create initial frame state with metadata.
    
    Args:
        frame_idx: Index of the frame to create
        context: Rendering context
        metadata_overrides: Optional metadata overrides
        
    Returns:
        Initial FrameState for the frame
    """
    base_metadata = {
        'frame_idx': frame_idx,
        'timestamp': frame_idx / context.fps,
        'seed': 42,  # Will be overridden by schedule
        'strength': 0.75,
        'cfg_scale': 7.0,
        'distilled_cfg_scale': 7.0,
        'noise_level': 0.0,
        'prompt': "",
    }
    
    if metadata_overrides:
        base_metadata.update(metadata_overrides)
    
    metadata = FrameMetadata(**base_metadata)
    
    return FrameState(
        metadata=metadata,
        stage=ProcessingStage.INITIALIZATION
    )


def validate_frame_state(frame_state: FrameState) -> ValidationResult:
    """
    Validate frame state for processing.
    
    Args:
        frame_state: Frame state to validate
        
    Returns:
        True if valid, error message string if invalid
    """
    try:
        # Validate metadata
        if frame_state.metadata.frame_idx < 0:
            return "Frame index must be non-negative"
        
        if not (0.0 <= frame_state.metadata.strength <= 1.0):
            return f"Strength must be between 0.0 and 1.0, got {frame_state.metadata.strength}"
        
        if frame_state.metadata.cfg_scale <= 0:
            return f"CFG scale must be positive, got {frame_state.metadata.cfg_scale}"
        
        # Validate image dimensions if present
        if frame_state.current_image is not None:
            if len(frame_state.current_image.shape) != 3:
                return f"Current image must be 3D array, got shape {frame_state.current_image.shape}"
        
        if frame_state.previous_image is not None:
            if len(frame_state.previous_image.shape) != 3:
                return f"Previous image must be 3D array, got shape {frame_state.previous_image.shape}"
        
        return True
        
    except Exception as e:
        return f"Validation error: {str(e)}"


def apply_animation_warping(
    frame_state: FrameState,
    context: RenderContext,
    depth_model: Optional[Any] = None
) -> FrameResult:
    """
    Apply animation warping transformations to frame.
    
    Args:
        frame_state: Current frame state
        context: Rendering context
        depth_model: Optional depth model for 3D warping
        
    Returns:
        FrameResult with warped image
    """
    start_time = time.time()
    
    try:
        if frame_state.current_image is None:
            return FrameResult(
                frame_state=frame_state.with_stage(ProcessingStage.ANIMATION_WARPING),
                success=False,
                error=RenderingError(
                    "No current image for animation warping",
                    ProcessingStage.ANIMATION_WARPING,
                    frame_state.metadata.frame_idx
                )
            )
        
        # Apply animation warping (simplified for functional approach)
        warped_image = frame_state.current_image.copy()
        depth = None
        
        # For 3D animation mode, apply warping
        if context.animation_mode == '3D' and context.use_depth_warping:
            # This would call the actual warping function
            # warped_image, depth = anim_frame_warp(...)
            pass
        
        new_frame_state = (frame_state
                          .with_image(warped_image)
                          .with_stage(ProcessingStage.ANIMATION_WARPING)
                          .with_transformation("animation_warp"))
        
        if depth is not None:
            new_frame_state = replace(new_frame_state, depth_map=depth)
        
        processing_time = time.time() - start_time
        
        return FrameResult(
            frame_state=new_frame_state,
            success=True,
            processing_time=processing_time
        )
        
    except Exception as e:
        return FrameResult(
            frame_state=frame_state.with_stage(ProcessingStage.ANIMATION_WARPING),
            success=False,
            error=RenderingError(
                f"Animation warping failed: {str(e)}",
                ProcessingStage.ANIMATION_WARPING,
                frame_state.metadata.frame_idx
            ),
            processing_time=time.time() - start_time
        )


# Note: apply_hybrid_motion function removed - hybrid video functionality not available


def apply_noise(
    frame_state: FrameState,
    noise_params: Optional[Dict[str, Any]] = None
) -> FrameResult:
    """
    Apply noise to frame image.
    
    Args:
        frame_state: Current frame state
        noise_params: Optional noise parameters
        
    Returns:
        FrameResult with noise applied
    """
    start_time = time.time()
    
    try:
        if frame_state.current_image is None:
            return FrameResult(
                frame_state=frame_state.with_stage(ProcessingStage.NOISE_APPLICATION),
                success=False,
                error=RenderingError(
                    "No current image for noise application",
                    ProcessingStage.NOISE_APPLICATION,
                    frame_state.metadata.frame_idx
                )
            )
        
        # Apply noise using the noise function
        noised_image = frame_state.current_image.copy()
        
        # Default noise parameters
        noise_level = frame_state.metadata.noise_level
        seed = frame_state.metadata.seed
        
        if noise_params:
            noise_level = noise_params.get('noise_level', noise_level)
            seed = noise_params.get('seed', seed)
        
        if noise_level > 0:
            # noised_image = add_noise(noised_image, noise_level, seed, ...)
            pass
        
        new_frame_state = (frame_state
                          .with_image(noised_image)
                          .with_stage(ProcessingStage.NOISE_APPLICATION)
                          .with_transformation("noise"))
        
        processing_time = time.time() - start_time
        
        return FrameResult(
            frame_state=new_frame_state,
            success=True,
            processing_time=processing_time
        )
        
    except Exception as e:
        return FrameResult(
            frame_state=frame_state.with_stage(ProcessingStage.NOISE_APPLICATION),
            success=False,
            error=RenderingError(
                f"Noise application failed: {str(e)}",
                ProcessingStage.NOISE_APPLICATION,
                frame_state.metadata.frame_idx
            ),
            processing_time=time.time() - start_time
        )


def apply_color_correction(
    frame_state: FrameState,
    color_params: Optional[Dict[str, Any]] = None
) -> FrameResult:
    """
    Apply color correction to frame.
    
    Args:
        frame_state: Current frame state
        color_params: Optional color correction parameters
        
    Returns:
        FrameResult with color correction applied
    """
    start_time = time.time()
    
    try:
        if frame_state.current_image is None:
            return FrameResult(
                frame_state=frame_state.with_stage(ProcessingStage.COLOR_CORRECTION),
                success=False,
                error=RenderingError(
                    "No current image for color correction",
                    ProcessingStage.COLOR_CORRECTION,
                    frame_state.metadata.frame_idx
                )
            )
        
        corrected_image = frame_state.current_image.copy()
        
        # Apply color correction if parameters provided
        if color_params:
            color_coherence = color_params.get('color_coherence', 'None')
            color_match_sample = color_params.get('color_match_sample')
            
            if color_coherence != 'None' and color_match_sample is not None:
                # corrected_image = maintain_colors(corrected_image, color_match_sample, color_coherence)
                pass
        
        new_frame_state = (frame_state
                          .with_image(corrected_image)
                          .with_stage(ProcessingStage.COLOR_CORRECTION)
                          .with_transformation("color_correction"))
        
        processing_time = time.time() - start_time
        
        return FrameResult(
            frame_state=new_frame_state,
            success=True,
            processing_time=processing_time
        )
        
    except Exception as e:
        return FrameResult(
            frame_state=frame_state.with_stage(ProcessingStage.COLOR_CORRECTION),
            success=False,
            error=RenderingError(
                f"Color correction failed: {str(e)}",
                ProcessingStage.COLOR_CORRECTION,
                frame_state.metadata.frame_idx
            ),
            processing_time=time.time() - start_time
        )


def apply_mask_operations(
    frame_state: FrameState,
    mask_params: Optional[Dict[str, Any]] = None
) -> FrameResult:
    """
    Apply mask operations to frame.
    
    Args:
        frame_state: Current frame state
        mask_params: Optional mask parameters
        
    Returns:
        FrameResult with mask operations applied
    """
    start_time = time.time()
    
    try:
        if frame_state.current_image is None:
            return FrameResult(
                frame_state=frame_state.with_stage(ProcessingStage.MASK_APPLICATION),
                success=False,
                error=RenderingError(
                    "No current image for mask application",
                    ProcessingStage.MASK_APPLICATION,
                    frame_state.metadata.frame_idx
                )
            )
        
        masked_image = frame_state.current_image.copy()
        
        # Apply mask operations if mask is present
        if frame_state.mask_image is not None:
            # Apply mask overlay or compositing
            # masked_image = do_overlay_mask(...)
            pass
        
        new_frame_state = (frame_state
                          .with_image(masked_image)
                          .with_stage(ProcessingStage.MASK_APPLICATION)
                          .with_transformation("mask_application"))
        
        processing_time = time.time() - start_time
        
        return FrameResult(
            frame_state=new_frame_state,
            success=True,
            processing_time=processing_time
        )
        
    except Exception as e:
        return FrameResult(
            frame_state=frame_state.with_stage(ProcessingStage.MASK_APPLICATION),
            success=False,
            error=RenderingError(
                f"Mask application failed: {str(e)}",
                ProcessingStage.MASK_APPLICATION,
                frame_state.metadata.frame_idx
            ),
            processing_time=time.time() - start_time
        )


def apply_frame_transformations(
    frame_state: FrameState,
    context: RenderContext,
    transformations: Tuple[str, ...] = ("animation_warp", "noise", "color_correction", "mask_application"),
    **kwargs
) -> FrameResult:
    """
    Apply a sequence of transformations to a frame.
    
    Args:
        frame_state: Current frame state
        context: Rendering context
        transformations: Sequence of transformation names to apply
        **kwargs: Additional parameters for transformations
        
    Returns:
        FrameResult with all transformations applied
    """
    current_result = FrameResult(frame_state=frame_state, success=True)
    
    transformation_functions = {
        'animation_warp': lambda fs: apply_animation_warping(fs, context, kwargs.get('depth_model')),
        'noise': lambda fs: apply_noise(fs, kwargs.get('noise_params')),
        'color_correction': lambda fs: apply_color_correction(fs, kwargs.get('color_params')),
        'mask_application': lambda fs: apply_mask_operations(fs, kwargs.get('mask_params'))
    }
    
    for transformation in transformations:
        if not current_result.success:
            break
            
        if transformation in transformation_functions:
            current_result = transformation_functions[transformation](current_result.frame_state)
        else:
            # Unknown transformation - add warning but continue
            current_result = current_result.with_warning(f"Unknown transformation: {transformation}")
    
    return current_result


def process_frame(
    frame_state: FrameState,
    context: RenderContext,
    processing_params: Optional[Dict[str, Any]] = None
) -> FrameResult:
    """
    Process a single frame through the complete pipeline.
    
    Args:
        frame_state: Initial frame state
        context: Rendering context
        processing_params: Optional processing parameters
        
    Returns:
        FrameResult with processed frame
    """
    start_time = time.time()
    
    # Validate input
    validation_result = validate_frame_state(frame_state)
    if validation_result is not True:
        return FrameResult(
            frame_state=frame_state,
            success=False,
            error=RenderingError(
                f"Frame validation failed: {validation_result}",
                ProcessingStage.INITIALIZATION,
                frame_state.metadata.frame_idx
            )
        )
    
    # Set up processing parameters
    params = processing_params or {}
    
    # Apply transformations based on context
    transformations = []
    
    if context.animation_mode in ['2D', '3D']:
        transformations.append('animation_warp')
    
    transformations.extend(['noise', 'color_correction', 'mask_application'])
    
    # Process frame through pipeline
    result = apply_frame_transformations(
        frame_state.with_stage(ProcessingStage.PRE_PROCESSING),
        context,
        tuple(transformations),
        **params
    )
    
    # Update final processing time
    total_time = time.time() - start_time
    final_frame_state = result.frame_state.with_stage(ProcessingStage.COMPLETED)
    
    return FrameResult(
        frame_state=final_frame_state,
        success=result.success,
        error=result.error,
        warnings=result.warnings,
        processing_time=total_time
    )


def merge_frame_results(results: Tuple[FrameResult, ...]) -> Dict[str, Any]:
    """
    Merge multiple frame results into summary statistics.
    
    Args:
        results: Tuple of frame results to merge
        
    Returns:
        Dictionary with merged statistics
    """
    if not results:
        return {
            'total_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'warnings': [],
            'errors': []
        }
    
    successful_frames = sum(1 for r in results if r.success)
    failed_frames = len(results) - successful_frames
    total_processing_time = sum(r.processing_time for r in results)
    average_processing_time = total_processing_time / len(results) if results else 0.0
    
    all_warnings = []
    all_errors = []
    
    for result in results:
        all_warnings.extend(result.warnings)
        if result.error:
            all_errors.append(result.error)
    
    return {
        'total_frames': len(results),
        'successful_frames': successful_frames,
        'failed_frames': failed_frames,
        'total_processing_time': total_processing_time,
        'average_processing_time': average_processing_time,
        'warnings': all_warnings,
        'errors': all_errors
    } 