"""
Immutable Frame State Models for Functional Rendering

This module defines the core data structures for the functional rendering system.
All models are immutable (frozen dataclasses) to ensure functional programming
principles and prevent accidental state mutations.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum
from pathlib import Path
import numpy as np
from PIL import Image


class ProcessingStage(Enum):
    """Enumeration of frame processing stages."""
    INITIALIZATION = "initialization"
    PRE_PROCESSING = "pre_processing"
    ANIMATION_WARPING = "animation_warping"
    HYBRID_MOTION = "hybrid_motion"
    NOISE_APPLICATION = "noise_application"
    GENERATION = "generation"
    POST_PROCESSING = "post_processing"
    COLOR_CORRECTION = "color_correction"
    MASK_APPLICATION = "mask_application"
    SAVING = "saving"
    COMPLETED = "completed"


class RenderingError(Exception):
    """Custom exception for rendering errors with context."""
    
    def __init__(self, message: str, stage: ProcessingStage, frame_idx: int, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.stage = stage
        self.frame_idx = frame_idx
        self.context = context or {}
        super().__init__(f"Frame {frame_idx} at {stage.value}: {message}")


@dataclass(frozen=True)
class FrameMetadata:
    """Immutable metadata for a single frame."""
    frame_idx: int
    timestamp: float
    seed: int
    strength: float
    cfg_scale: float
    distilled_cfg_scale: float
    noise_level: float
    prompt: str
    negative_prompt: str = ""
    sampler_name: Optional[str] = None
    scheduler_name: Optional[str] = None
    checkpoint: Optional[str] = None
    
    def __post_init__(self):
        """Validate frame metadata."""
        if self.frame_idx < 0:
            raise ValueError(f"Frame index must be non-negative, got {self.frame_idx}")
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {self.strength}")
        if self.cfg_scale <= 0:
            raise ValueError(f"CFG scale must be positive, got {self.cfg_scale}")


@dataclass(frozen=True)
class FrameState:
    """Immutable state for a single frame in the rendering pipeline."""
    metadata: FrameMetadata
    current_image: Optional[np.ndarray] = None
    previous_image: Optional[np.ndarray] = None
    init_image: Optional[np.ndarray] = None
    mask_image: Optional[np.ndarray] = None
    noise_mask: Optional[np.ndarray] = None
    depth_map: Optional[np.ndarray] = None
    flow_map: Optional[np.ndarray] = None
    
    # Processing state
    stage: ProcessingStage = ProcessingStage.INITIALIZATION
    transformations_applied: Tuple[str, ...] = field(default_factory=tuple)
    processing_time: float = 0.0
    
    # Animation parameters
    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.0
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    
    # Hybrid motion parameters
    hybrid_alpha: float = 1.0
    hybrid_flow_factor: float = 1.0
    
    def with_image(self, image: np.ndarray) -> 'FrameState':
        """Return new FrameState with updated current image."""
        return replace(self, current_image=image)
    
    def with_stage(self, stage: ProcessingStage) -> 'FrameState':
        """Return new FrameState with updated processing stage."""
        return replace(self, stage=stage)
    
    def with_transformation(self, transformation: str) -> 'FrameState':
        """Return new FrameState with added transformation."""
        new_transformations = self.transformations_applied + (transformation,)
        return replace(self, transformations_applied=new_transformations)
    
    def with_metadata(self, **kwargs) -> 'FrameState':
        """Return new FrameState with updated metadata."""
        new_metadata = replace(self.metadata, **kwargs)
        return replace(self, metadata=new_metadata)


@dataclass(frozen=True)
class RenderContext:
    """Immutable context for the entire rendering session."""
    # Output configuration
    output_dir: Path
    timestring: str
    width: int
    height: int
    max_frames: int
    fps: float
    
    # Animation configuration
    animation_mode: str
    use_depth_warping: bool
    save_depth_maps: bool
    hybrid_composite: str
    hybrid_motion: str
    
    # Model configuration
    depth_algorithm: str
    optical_flow_cadence: str
    diffusion_cadence: int
    
    # Quality settings
    motion_preview_mode: bool = False
    save_gen_info_as_srt: bool = False
    
    # Device configuration
    device: str = "cuda"
    half_precision: bool = True
    keep_models_in_vram: bool = False
    
    def __post_init__(self):
        """Validate render context."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Dimensions must be positive, got {self.width}x{self.height}")
        if self.max_frames <= 0:
            raise ValueError(f"Max frames must be positive, got {self.max_frames}")
        if self.fps <= 0:
            raise ValueError(f"FPS must be positive, got {self.fps}")


@dataclass(frozen=True)
class FrameResult:
    """Immutable result of frame processing."""
    frame_state: FrameState
    success: bool
    error: Optional[RenderingError] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    processing_time: float = 0.0
    memory_usage: Optional[int] = None
    
    @property
    def frame_idx(self) -> int:
        """Get frame index from metadata."""
        return self.frame_state.metadata.frame_idx
    
    @property
    def stage(self) -> ProcessingStage:
        """Get current processing stage."""
        return self.frame_state.stage
    
    def with_warning(self, warning: str) -> 'FrameResult':
        """Return new FrameResult with added warning."""
        new_warnings = self.warnings + (warning,)
        return replace(self, warnings=new_warnings)
    
    def with_error(self, error: RenderingError) -> 'FrameResult':
        """Return new FrameResult with error."""
        return replace(self, success=False, error=error)


@dataclass(frozen=True)
class ModelState:
    """Immutable state for models used in rendering."""
    depth_model_loaded: bool = False
    raft_model_loaded: bool = False
    controlnet_enabled: bool = False
    current_checkpoint: Optional[str] = None
    
    # Model memory management
    models_in_vram: Tuple[str, ...] = field(default_factory=tuple)
    memory_usage: Dict[str, int] = field(default_factory=dict)
    
    def with_model_loaded(self, model_name: str) -> 'ModelState':
        """Return new ModelState with model marked as loaded."""
        new_models = self.models_in_vram + (model_name,)
        return replace(self, models_in_vram=new_models)
    
    def with_model_unloaded(self, model_name: str) -> 'ModelState':
        """Return new ModelState with model removed."""
        new_models = tuple(m for m in self.models_in_vram if m != model_name)
        return replace(self, models_in_vram=new_models)


@dataclass(frozen=True)
class RenderingSession:
    """Immutable state for an entire rendering session."""
    context: RenderContext
    model_state: ModelState
    frame_results: Tuple[FrameResult, ...] = field(default_factory=tuple)
    session_start_time: float = 0.0
    total_processing_time: float = 0.0
    
    @property
    def completed_frames(self) -> int:
        """Get number of successfully completed frames."""
        return sum(1 for result in self.frame_results if result.success)
    
    @property
    def failed_frames(self) -> int:
        """Get number of failed frames."""
        return sum(1 for result in self.frame_results if not result.success)
    
    @property
    def current_frame_idx(self) -> int:
        """Get the index of the next frame to process."""
        return len(self.frame_results)
    
    def with_frame_result(self, result: FrameResult) -> 'RenderingSession':
        """Return new RenderingSession with added frame result."""
        new_results = self.frame_results + (result,)
        new_total_time = self.total_processing_time + result.processing_time
        return replace(
            self, 
            frame_results=new_results,
            total_processing_time=new_total_time
        )
    
    def with_model_state(self, model_state: ModelState) -> 'RenderingSession':
        """Return new RenderingSession with updated model state."""
        return replace(self, model_state=model_state)


# Type aliases for better readability
ImageArray = np.ndarray
FlowArray = np.ndarray
DepthArray = np.ndarray
MaskArray = np.ndarray

# Result types for functional error handling
ProcessingResult = Union[FrameResult, RenderingError]
ValidationResult = Union[bool, str]  # True for valid, string for error message 