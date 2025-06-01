"""
Legacy Rendering Adapter

This module provides backward compatibility with the existing render.py interface
while using the new functional rendering system under the hood. It converts
legacy arguments to the new functional format and provides drop-in replacements.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from types import SimpleNamespace

from .frame_models import (
    RenderContext, FrameMetadata, FrameState, ModelState,
    ProcessingStage, RenderingError
)
from .rendering_pipeline import (
    create_rendering_pipeline, render_animation_functional,
    create_progress_tracker, PipelineConfig
)
from .frame_processing import create_frame_state

# Global flag to enable/disable functional rendering
_FUNCTIONAL_RENDERING_ENABLED = False


def enable_functional_rendering(enabled: bool = True) -> None:
    """
    Enable or disable functional rendering system.
    
    Args:
        enabled: Whether to enable functional rendering
    """
    global _FUNCTIONAL_RENDERING_ENABLED
    _FUNCTIONAL_RENDERING_ENABLED = enabled
    print(f"Functional rendering {'enabled' if enabled else 'disabled'}")


def is_functional_rendering_enabled() -> bool:
    """Check if functional rendering is enabled."""
    return _FUNCTIONAL_RENDERING_ENABLED


def convert_legacy_args_to_context(
    args: Any,
    anim_args: Any,
    video_args: Any,
    root: Any
) -> RenderContext:
    """
    Convert legacy argument objects to functional RenderContext.
    
    Args:
        args: Legacy args object
        anim_args: Legacy animation args
        video_args: Legacy video args
        root: Legacy root object
        
    Returns:
        RenderContext for functional rendering
    """
    # Extract values with fallbacks for missing attributes
    output_dir = Path(getattr(args, 'outdir', '/tmp/deforum_output'))
    timestring = getattr(root, 'timestring', 'default')
    width = getattr(args, 'W', 512)
    height = getattr(args, 'H', 512)
    max_frames = getattr(anim_args, 'max_frames', 100)
    fps = getattr(video_args, 'fps', 30.0) if hasattr(video_args, 'fps') else 30.0
    
    # Animation configuration
    animation_mode = getattr(anim_args, 'animation_mode', '2D')
    use_depth_warping = getattr(anim_args, 'use_depth_warping', False)
    save_depth_maps = getattr(anim_args, 'save_depth_maps', False)
    hybrid_composite = getattr(anim_args, 'hybrid_composite', 'None')
    hybrid_motion = getattr(anim_args, 'hybrid_motion', 'None')
    
    # Model configuration
    depth_algorithm = getattr(anim_args, 'depth_algorithm', 'midas')
    optical_flow_cadence = getattr(anim_args, 'optical_flow_cadence', 'None')
    diffusion_cadence = getattr(anim_args, 'diffusion_cadence', 1)
    
    # Quality settings
    motion_preview_mode = getattr(args, 'motion_preview_mode', False)
    
    # Device configuration
    device = getattr(root, 'device', 'cuda')
    half_precision = getattr(root, 'half_precision', True)
    
    return RenderContext(
        output_dir=output_dir,
        timestring=timestring,
        width=width,
        height=height,
        max_frames=max_frames,
        fps=fps,
        animation_mode=animation_mode,
        use_depth_warping=use_depth_warping,
        save_depth_maps=save_depth_maps,
        hybrid_composite=hybrid_composite,
        hybrid_motion=hybrid_motion,
        depth_algorithm=depth_algorithm,
        optical_flow_cadence=optical_flow_cadence,
        diffusion_cadence=diffusion_cadence,
        motion_preview_mode=motion_preview_mode,
        device=device,
        half_precision=half_precision
    )


def convert_legacy_frame_args(
    frame_idx: int,
    args: Any,
    keys: Any,
    prompt_series: Any
) -> Dict[str, Any]:
    """
    Convert legacy frame arguments to metadata dictionary.
    
    Args:
        frame_idx: Frame index
        args: Legacy args object
        keys: Legacy keys object
        prompt_series: Legacy prompt series
        
    Returns:
        Dictionary of frame metadata
    """
    metadata = {
        'frame_idx': frame_idx,
        'timestamp': frame_idx / 30.0,  # Default FPS
        'seed': getattr(args, 'seed', 42),
        'strength': getattr(args, 'strength', 0.75),
        'cfg_scale': getattr(args, 'cfg_scale', 7.0),
        'distilled_cfg_scale': getattr(args, 'distilled_cfg_scale', 7.0),
        'noise_level': 0.0,
        'prompt': "",
    }
    
    # Extract scheduled values if available
    if hasattr(keys, 'strength_schedule_series') and frame_idx < len(keys.strength_schedule_series):
        metadata['strength'] = keys.strength_schedule_series[frame_idx]
    
    if hasattr(keys, 'cfg_scale_schedule_series') and frame_idx < len(keys.cfg_scale_schedule_series):
        metadata['cfg_scale'] = keys.cfg_scale_schedule_series[frame_idx]
    
    if hasattr(keys, 'seed_schedule_series') and frame_idx < len(keys.seed_schedule_series):
        metadata['seed'] = int(keys.seed_schedule_series[frame_idx])
    
    if hasattr(keys, 'noise_schedule_series') and frame_idx < len(keys.noise_schedule_series):
        metadata['noise_level'] = keys.noise_schedule_series[frame_idx]
    
    # Extract prompt
    if hasattr(prompt_series, '__getitem__') and frame_idx < len(prompt_series):
        metadata['prompt'] = str(prompt_series[frame_idx])
    elif hasattr(args, 'prompt'):
        metadata['prompt'] = str(args.prompt)
    
    return metadata


def functional_render_animation(
    args: Any,
    anim_args: Any,
    video_args: Any,
    parseq_args: Any,
    loop_args: Any,
    controlnet_args: Any,
    freeu_args: Any,
    kohya_hrfix_args: Any,
    root: Any
) -> None:
    """
    Functional replacement for the legacy render_animation function.
    
    This function provides the same interface as the original render_animation
    but uses the new functional rendering system internally.
    
    Args:
        args: Legacy args object
        anim_args: Legacy animation args
        video_args: Legacy video args
        parseq_args: Legacy parseq args
        loop_args: Legacy loop args
        controlnet_args: Legacy controlnet args
        freeu_args: Legacy freeu args
        kohya_hrfix_args: Legacy kohya hrfix args
        root: Legacy root object
    """
    if not _FUNCTIONAL_RENDERING_ENABLED:
        # Fall back to original implementation
        from ..render import render_animation as legacy_render_animation
        return legacy_render_animation(
            args, anim_args, video_args, parseq_args, loop_args,
            controlnet_args, freeu_args, kohya_hrfix_args, root
        )
    
    print("Using functional rendering system...")
    
    try:
        # Convert legacy arguments to functional format
        context = convert_legacy_args_to_context(args, anim_args, video_args, root)
        
        # Create pipeline configuration
        config = PipelineConfig(
            max_workers=1,  # Start with sequential processing
            enable_progress_tracking=True,
            enable_error_recovery=True
        )
        
        # Create rendering pipeline
        pipeline = create_rendering_pipeline(context, config=config)
        
        # Create progress tracker
        progress_callback = create_progress_tracker(verbose=True)
        
        # Execute functional rendering
        session = render_animation_functional(
            context=context,
            pipeline=pipeline,
            start_frame=0,
            progress_callback=progress_callback
        )
        
        # Report results
        print(f"\nFunctional rendering completed:")
        print(f"  Total frames: {session.context.max_frames}")
        print(f"  Successful frames: {session.completed_frames}")
        print(f"  Failed frames: {session.failed_frames}")
        print(f"  Total processing time: {session.total_processing_time:.2f}s")
        
        if session.failed_frames > 0:
            print(f"  Warning: {session.failed_frames} frames failed to render")
            
            # Show first few errors
            errors = [r.error for r in session.frame_results if r.error][:3]
            for error in errors:
                print(f"    Frame {error.frame_idx}: {error.message}")
        
    except Exception as e:
        print(f"Functional rendering failed: {e}")
        print("Falling back to legacy rendering...")
        
        # Fall back to original implementation
        from ..render import render_animation as legacy_render_animation
        return legacy_render_animation(
            args, anim_args, video_args, parseq_args, loop_args,
            controlnet_args, freeu_args, kohya_hrfix_args, root
        )


def create_legacy_compatible_pipeline():
    """
    Create a pipeline that's compatible with legacy rendering expectations.
    
    Returns:
        RenderingPipeline configured for legacy compatibility
    """
    # This would create a pipeline that mimics the exact behavior
    # of the legacy rendering system
    pass


def migrate_legacy_settings(legacy_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate legacy settings to functional rendering format.
    
    Args:
        legacy_settings: Legacy settings dictionary
        
    Returns:
        Migrated settings for functional rendering
    """
    functional_settings = {}
    
    # Map legacy setting names to functional equivalents
    setting_mappings = {
        'animation_mode': 'animation_mode',
        'max_frames': 'max_frames',
        'strength_schedule': 'strength_schedule',
        'cfg_scale_schedule': 'cfg_scale_schedule',
        'seed_schedule': 'seed_schedule',
        'noise_schedule': 'noise_schedule',
        'use_depth_warping': 'use_depth_warping',
        'save_depth_maps': 'save_depth_maps',
        'hybrid_composite': 'hybrid_composite',
        'hybrid_motion': 'hybrid_motion',
    }
    
    for legacy_key, functional_key in setting_mappings.items():
        if legacy_key in legacy_settings:
            functional_settings[functional_key] = legacy_settings[legacy_key]
    
    return functional_settings


def validate_legacy_compatibility(args: Any, anim_args: Any) -> Tuple[bool, str]:
    """
    Validate that legacy arguments are compatible with functional rendering.
    
    Args:
        args: Legacy args object
        anim_args: Legacy animation args
        
    Returns:
        Tuple of (is_compatible, error_message)
    """
    try:
        # Check for required attributes
        required_attrs = [
            ('args', args, ['outdir', 'W', 'H']),
            ('anim_args', anim_args, ['animation_mode', 'max_frames'])
        ]
        
        for obj_name, obj, attrs in required_attrs:
            for attr in attrs:
                if not hasattr(obj, attr):
                    return False, f"Missing required attribute: {obj_name}.{attr}"
        
        # Check for unsupported features
        if hasattr(anim_args, 'animation_mode'):
            if anim_args.animation_mode not in ['2D', '3D', 'Video Input']:
                return False, f"Unsupported animation mode: {anim_args.animation_mode}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# Utility functions for gradual migration
def create_hybrid_renderer(use_functional_for_frames: Optional[set] = None):
    """
    Create a hybrid renderer that uses functional rendering for specific frames
    and legacy rendering for others. Useful for gradual migration.
    
    Args:
        use_functional_for_frames: Set of frame indices to render functionally
        
    Returns:
        Hybrid rendering function
    """
    def hybrid_render(args, anim_args, video_args, parseq_args, loop_args,
                     controlnet_args, freeu_args, kohya_hrfix_args, root):
        
        if use_functional_for_frames is None:
            # Use functional rendering for all frames
            return functional_render_animation(
                args, anim_args, video_args, parseq_args, loop_args,
                controlnet_args, freeu_args, kohya_hrfix_args, root
            )
        
        # Hybrid approach - would need more complex implementation
        # For now, just use functional if enabled
        if _FUNCTIONAL_RENDERING_ENABLED:
            return functional_render_animation(
                args, anim_args, video_args, parseq_args, loop_args,
                controlnet_args, freeu_args, kohya_hrfix_args, root
            )
        else:
            from ..render import render_animation as legacy_render_animation
            return legacy_render_animation(
                args, anim_args, video_args, parseq_args, loop_args,
                controlnet_args, freeu_args, kohya_hrfix_args, root
            )
    
    return hybrid_render 