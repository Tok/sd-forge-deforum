"""
Immutable data models for Deforum to replace mutable SimpleNamespace objects.

This module provides type-safe, validated, immutable data structures that replace
the current dictionary-based configuration pattern. These models enforce data
integrity and make the codebase more testable and maintainable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
import re
import json
from enum import Enum
import numpy as np
from PIL import Image

# Import legacy args for backward compatibility with tests
try:
    from .args import DeforumAnimArgs, DeforumArgs as LegacyDeforumArgs, ParseqArgs as LegacyParseqArgs, WanArgs as LegacyWanArgs, RootArgs as LegacyRootArgs
except ImportError:
    # Fallback if circular import issues
    DeforumAnimArgs = None
    LegacyDeforumArgs = None
    LegacyParseqArgs = None
    LegacyWanArgs = None
    LegacyRootArgs = None


class AnimationMode(Enum):
    """Animation mode enumeration"""
    TWO_D = "2D"
    THREE_D = "3D"
    INTERPOLATION = "Interpolation"
    WAN_VIDEO = "Wan Video"


class ColorCoherence(Enum):
    """Color coherence algorithm enumeration"""
    NONE = "None"
    HSV = "HSV"
    LAB = "LAB"
    RGB = "RGB"
    IMAGE = "Image"


class BorderMode(Enum):
    """Border mode enumeration"""
    REPLICATE = "replicate"
    WRAP = "wrap"


class PaddingMode(Enum):
    """Padding mode enumeration"""
    BORDER = "border"
    REFLECTION = "reflection"
    ZEROS = "zeros"


class SamplingMode(Enum):
    """Sampling mode enumeration"""
    BICUBIC = "bicubic"
    BILINEAR = "bilinear"
    NEAREST = "nearest"


class DepthAlgorithm(Enum):
    """Depth algorithm enumeration"""
    DEPTH_ANYTHING_V2_SMALL = "Depth-Anything-V2-Small"
    MIDAS_3_HYBRID = "Midas-3-Hybrid"


def validate_schedule_string(schedule_str: str, parameter_name: str = "schedule") -> None:
    """
    Validate that a schedule string follows the correct format: "frame:(value)"
    
    Args:
        schedule_str: The schedule string to validate
        parameter_name: Name of the parameter for error messages
        
    Raises:
        ValueError: If the schedule string format is invalid
    """
    if not schedule_str or not isinstance(schedule_str, str):
        raise ValueError(f"{parameter_name} must follow format 'frame:(value)' or 'frame1:(value1), frame2:(value2)'. Got: {schedule_str}")
    
    # Basic pattern that handles nested parentheses and complex expressions
    # Format: frame:(expression), frame:(expression), ...
    # The expression can contain any content including nested parentheses
    # More strict: ensure proper comma separation and no extra content
    pattern = r'^\s*[^:,]+\s*:\s*\([^)]*(?:\([^)]*\)[^)]*)*\)\s*(?:,\s*[^:,]+\s*:\s*\([^)]*(?:\([^)]*\)[^)]*)*\)\s*)*$'
    
    if not re.match(pattern, schedule_str.strip()):
        raise ValueError(
            f"{parameter_name} must follow format 'frame:(value)' or 'frame1:(value1), frame2:(value2)'. "
            f"Got: {schedule_str}"
        )


def validate_positive_int(value: int, parameter_name: str = "value") -> None:
    """Validate that a value is a positive integer"""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{parameter_name} must be a positive integer, got: {value}")


def validate_non_negative_number(value: Union[int, float], parameter_name: str = "value") -> None:
    """Validate that a value is a non-negative number"""
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{parameter_name} must be non-negative, got: {value}")


def validate_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float], parameter_name: str = "value") -> None:
    """Validate that a value is within a specified range"""
    if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
        raise ValueError(f"{parameter_name} must be between {min_val} and {max_val}, got: {value}")


@dataclass(frozen=True)
class AnimationArgs:
    """Immutable animation arguments with validation"""
    
    # Core animation settings
    animation_mode: AnimationMode = AnimationMode.THREE_D
    max_frames: int = 333
    border: BorderMode = BorderMode.REPLICATE
    
    # Movement schedules
    angle: str = "0: (0)"
    zoom: str = "0: (1.0)"
    translation_x: str = "0: (0)"
    translation_y: str = "0: (0)"
    translation_z: str = "0: (0)"
    rotation_3d_x: str = "0: (0)"
    rotation_3d_y: str = "0: (0)"
    rotation_3d_z: str = "0: (0)"
    
    # Perspective flip
    perspective_flip_theta: str = "0: (0)"
    perspective_flip_phi: str = "0: (0)"
    perspective_flip_gamma: str = "0: (0)"
    perspective_flip_fv: str = "0: (53)"
    
    # Transform center
    transform_center_x: str = "0: (0.5)"
    transform_center_y: str = "0: (0.5)"
    
    # Schedules
    noise_schedule: str = "0: (0.02)"
    strength_schedule: str = "0: (0.65)"
    contrast_schedule: str = "0: (1.0)"
    cfg_scale_schedule: str = "0: (7)"
    
    # 3D settings
    fov_schedule: str = "0: (70)"
    aspect_ratio_schedule: str = "0: (1.0)"
    aspect_ratio_use_old_formula: bool = False
    near_schedule: str = "0: (200)"
    far_schedule: str = "0: (10000)"
    
    # Seed scheduling
    seed_schedule: str = '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)'
    enable_subseed_scheduling: bool = False
    subseed_schedule: str = "0: (1)"
    
    # Sampling scheduling
    enable_steps_scheduling: bool = False
    steps_schedule: str = "0: (25)"
    
    # Sampler scheduling
    enable_sampler_scheduling: bool = False
    sampler_schedule: str = "0: (\"Euler a\")"
    
    # CFG scheduling
    enable_cfg_scheduling: bool = False
    
    # Checkpoint scheduling
    enable_checkpoint_scheduling: bool = False
    checkpoint_schedule: str = "0: (\"model1.ckpt\"), 100: (\"model2.safetensors\")"
    
    # Clip skip scheduling
    enable_clipskip_scheduling: bool = False
    clipskip_schedule: str = "0: (2)"
    
    # Noise settings
    noise_type: str = "perlin"
    perlin_w: int = 8
    perlin_h: int = 8
    perlin_octaves: int = 4
    perlin_persistence: float = 0.5
    
    # Coherence settings
    color_coherence: ColorCoherence = ColorCoherence.NONE
    color_coherence_image_path: str = ""
    color_coherence_video_every_N_frames: int = 1
    
    # Optical flow settings
    optical_flow_cadence: str = "None"
    optical_flow_redo_generation: str = "None"
    optical_flow_redo_generation_image: str = ""
    
    # Depth settings
    depth_algorithm: DepthAlgorithm = DepthAlgorithm.DEPTH_ANYTHING_V2_SMALL
    midas_weight: float = 0.2
    padding_mode: PaddingMode = PaddingMode.BORDER
    sampling_mode: SamplingMode = SamplingMode.BICUBIC
    save_depth_maps: bool = False
    
    # Video init settings
    video_init_path: str = 'https://deforum.github.io/a1/V1.mp4'
    extract_nth_frame: int = 1
    extract_from_frame: int = 0
    extract_to_frame: int = -1
    overwrite_extracted_frames: bool = False
    
    # Video mask settings
    use_mask_video: bool = False
    video_mask_path: str = 'https://deforum.github.io/a1/VM1.mp4'
    
    # Hybrid settings
    hybrid_comp_alpha_schedule: str = "0:(0.5)"
    hybrid_comp_mask_blend_alpha_schedule: str = "0:(0.5)"
    hybrid_comp_mask_contrast_schedule: str = "0:(1)"
    hybrid_comp_mask_auto_contrast_cutoff_high_schedule: str = "0:(100)"
    hybrid_comp_mask_auto_contrast_cutoff_low_schedule: str = "0:(0)"
    hybrid_flow_method: str = "RAFT"
    hybrid_composite: str = "None"
    hybrid_use_init_image: bool = False
    hybrid_comp_mask_type: str = "None"
    hybrid_comp_mask_inverse: bool = False
    hybrid_comp_mask_equalize: str = "None"
    hybrid_comp_mask_auto_contrast: bool = False
    hybrid_comp_save_extra_frames: bool = False
    
    # Additional schedules
    diffusion_cadence: str = "1"
    optical_flow_cadence_schedule: str = "0: (1)"
    cadence_flow_factor_schedule: str = "0: (1)"
    redo_flow_factor_schedule: str = "0: (1)"
    mask_schedule: str = "0: (\"\")"
    noise_mask_schedule: str = "0: (\"\")"
    amount_schedule: str = "0: (0.1)"
    kernel_schedule: str = "0: (5)"
    sigma_schedule: str = "0: (1)"
    threshold_schedule: str = "0: (0)"
    
    # Ancestral ETA scheduling
    enable_ancestral_eta_scheduling: bool = False
    ancestral_eta_schedule: str = "0: (1)"
    
    def __post_init__(self):
        """Validate all schedule strings and numerical values"""
        # Validate frame count
        validate_positive_int(self.max_frames, "max_frames")
        
        # Validate schedule strings
        schedule_fields = [
            'angle', 'zoom', 'translation_x', 'translation_y', 'translation_z',
            'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z',
            'perspective_flip_theta', 'perspective_flip_phi', 'perspective_flip_gamma', 'perspective_flip_fv',
            'transform_center_x', 'transform_center_y',
            'noise_schedule', 'strength_schedule', 'contrast_schedule', 'cfg_scale_schedule',
            'fov_schedule', 'aspect_ratio_schedule', 'near_schedule', 'far_schedule',
            'subseed_schedule', 'steps_schedule', 'clipskip_schedule',
            'hybrid_comp_alpha_schedule', 'hybrid_comp_mask_blend_alpha_schedule',
            'hybrid_comp_mask_contrast_schedule', 'hybrid_comp_mask_auto_contrast_cutoff_high_schedule',
            'hybrid_comp_mask_auto_contrast_cutoff_low_schedule',
            'optical_flow_cadence_schedule', 'cadence_flow_factor_schedule', 'redo_flow_factor_schedule',
            'amount_schedule', 'kernel_schedule', 'sigma_schedule', 'threshold_schedule', 'ancestral_eta_schedule'
        ]
        
        for field_name in schedule_fields:
            field_value = getattr(self, field_name)
            if field_value:  # Only validate non-empty schedules
                validate_schedule_string(field_value, field_name)
        
        # Validate numerical ranges
        validate_range(self.midas_weight, -1.0, 1.0, "midas_weight")
        validate_positive_int(self.extract_nth_frame, "extract_nth_frame")
        validate_non_negative_number(self.extract_from_frame, "extract_from_frame")
        validate_positive_int(self.color_coherence_video_every_N_frames, "color_coherence_video_every_N_frames")
        validate_positive_int(self.perlin_w, "perlin_w")
        validate_positive_int(self.perlin_h, "perlin_h")
        validate_positive_int(self.perlin_octaves, "perlin_octaves")
        validate_range(self.perlin_persistence, 0.0, 1.0, "perlin_persistence")


@dataclass(frozen=True)
class DeforumArgs:
    """Immutable Deforum arguments with validation"""
    
    # Core generation settings
    W: int = 512
    H: int = 512
    seed: int = -1
    sampler: str = "Euler a"
    steps: int = 25
    scale: float = 7.0
    ddim_eta: float = 0.0
    dynamic_threshold: Optional[float] = None
    static_threshold: Optional[float] = None
    
    # Init image settings
    use_init: bool = False
    strength: float = 0.65
    strength_0_no_init: bool = True
    init_image: Optional[str] = None
    init_image_box: Optional[str] = None
    use_mask: bool = False
    use_alpha_as_mask: bool = False
    mask_file: str = ""
    invert_mask: bool = False
    mask_overlay_blur: int = 4
    mask_contrast_adjust: float = 1.0
    mask_brightness_adjust: float = 1.0
    
    # Output settings
    batch_name: str = "Deforum"
    filename_format: str = "{timestring}_{index:05}_{seed}"
    save_samples: bool = True
    save_sample_per_step: bool = False
    show_sample_per_step: bool = False
    
    # Prompt settings
    prompts: Dict[str, str] = field(default_factory=dict)
    positive_prompts: str = ""
    negative_prompts: str = ""
    
    # Advanced settings
    clip_skip: int = 1
    
    def __post_init__(self):
        """Validate arguments"""
        validate_positive_int(self.W, "W")
        validate_positive_int(self.H, "H")
        validate_positive_int(self.steps, "steps")
        validate_range(self.scale, 1.0, 30.0, "scale")
        validate_range(self.strength, 0.0, 1.0, "strength")
        validate_range(self.ddim_eta, 0.0, 1.0, "ddim_eta")
        validate_non_negative_number(self.mask_overlay_blur, "mask_overlay_blur")
        validate_range(self.mask_contrast_adjust, 0.0, 2.0, "mask_contrast_adjust")
        validate_range(self.mask_brightness_adjust, 0.0, 2.0, "mask_brightness_adjust")
        validate_positive_int(self.clip_skip, "clip_skip")


@dataclass(frozen=True)
class VideoArgs:
    """Immutable video output arguments with validation"""
    
    fps: int = 30
    output_format: str = "mp4"
    ffmpeg_location: str = "ffmpeg"
    ffmpeg_crf: int = 17
    ffmpeg_preset: str = "slow"
    add_soundtrack: str = "None"
    soundtrack_path: str = ""
    use_manual_settings: bool = False
    
    def __post_init__(self):
        """Validate video arguments"""
        validate_positive_int(self.fps, "fps")
        validate_range(self.ffmpeg_crf, 0, 51, "ffmpeg_crf")


@dataclass(frozen=True)
class ParseqArgs:
    """Immutable Parseq arguments with validation"""
    
    parseq_manifest: Optional[str] = None
    parseq_use_deltas: bool = True
    parseq_non_schedule_overrides: bool = True
    
    def __post_init__(self):
        """Validate Parseq arguments"""
        if self.parseq_manifest is not None and self.parseq_manifest.strip():
            # Basic validation - should be JSON or URL
            manifest = self.parseq_manifest.strip()
            if not (manifest.startswith('{') or manifest.startswith('http')):
                raise ValueError("parseq_manifest must be valid JSON or URL")


@dataclass(frozen=True) 
class WanArgs:
    """Immutable Wan arguments with validation"""
    
    # Core settings
    wan_mode: str = "Disabled"
    wan_model_path: str = "models/wan"
    wan_model_name: str = "Auto-Select"
    wan_enable_prompt_enhancement: bool = False
    wan_qwen_model: str = "Auto-Select"
    wan_qwen_language: str = "English"
    wan_auto_download: bool = True
    wan_preferred_size: str = "1.3B VACE (Recommended)"
    wan_enable_movement_analysis: bool = True
    wan_movement_sensitivity: float = 1.0
    wan_style_prompt: str = ""
    wan_style_strength: float = 0.5
    wan_i2v_strength: float = 0.8
    
    # Model settings
    wan_t2v_model: str = "1.3B VACE"
    wan_i2v_model: str = "Use Primary Model"
    wan_resolution: str = "864x480 (Landscape)"
    wan_seed: int = -1
    wan_guidance_scale: float = 7.5
    
    # Advanced settings
    wan_strength_override: bool = True
    wan_fixed_strength: float = 1.0
    wan_guidance_override: bool = True
    wan_motion_strength_override: bool = False
    wan_motion_strength: float = 1.0
    wan_frame_overlap: int = 2
    wan_enable_interpolation: bool = True
    wan_interpolation_strength: float = 0.5
    wan_flash_attention_mode: str = "Auto (Recommended)"
    wan_qwen_auto_download: bool = True
    
    # Additional fields for runtime state (should be set by system, not user)
    wan_enhanced_prompts: str = ""
    wan_movement_description: str = ""
    
    def __post_init__(self):
        """Validate Wan arguments"""
        validate_range(self.wan_movement_sensitivity, 0.1, 5.0, "wan_movement_sensitivity")
        validate_range(self.wan_style_strength, 0.0, 1.0, "wan_style_strength")
        validate_range(self.wan_i2v_strength, 0.0, 1.0, "wan_i2v_strength")
        validate_range(self.wan_guidance_scale, 1.0, 20.0, "wan_guidance_scale")
        validate_range(self.wan_fixed_strength, 0.0, 1.0, "wan_fixed_strength")
        validate_range(self.wan_motion_strength, 0.0, 2.0, "wan_motion_strength")
        validate_range(self.wan_frame_overlap, 0, 10, "wan_frame_overlap")
        validate_range(self.wan_interpolation_strength, 0.0, 1.0, "wan_interpolation_strength")
        
        # Validate choices
        valid_modes = ["Disabled", "T2V Only", "I2V Chaining"]
        if self.wan_mode not in valid_modes:
            raise ValueError(f"wan_mode must be one of {valid_modes}, got: {self.wan_mode}")
        
        valid_languages = ["English", "Chinese"]
        if self.wan_qwen_language not in valid_languages:
            raise ValueError(f"wan_qwen_language must be one of {valid_languages}, got: {self.wan_qwen_language}")


@dataclass(frozen=True)
class RootArgs:
    """Immutable root arguments - runtime state and shared data"""
    
    device: Optional[str] = None
    models_path: str = ""
    half_precision: bool = True
    clipseg_model: Optional[Any] = None
    mask_preset_names: Tuple[str, ...] = ("everywhere", "video_mask")
    frames_cache: Tuple[Any, ...] = field(default_factory=tuple)
    raw_batch_name: Optional[str] = None
    raw_seed: Optional[int] = None
    timestring: str = ""
    subseed: int = -1
    subseed_strength: float = 0.0
    seed_internal: int = 0
    init_sample: Optional[Any] = None
    noise_mask: Optional[Any] = None
    initial_info: Optional[str] = None
    first_frame: Optional[Any] = None
    animation_prompts: Dict[str, str] = field(default_factory=dict)
    prompt_keyframes: List[str] = field(default_factory=list)
    current_user_os: str = ""
    tmp_deforum_run_duplicated_folder: str = ""
    default_img: Optional[Any] = None
    
    # Runtime values that may be set during execution
    job_id: Optional[str] = None
    initial_ddim_eta: float = 0.0
    initial_ancestral_eta: float = 1.0


# Helper functions for creating data models from legacy dictionaries
def create_animation_args_from_dict(data: Dict[str, Any]) -> AnimationArgs:
    """Create AnimationArgs from legacy dictionary data"""
    # Extract only the fields that exist in AnimationArgs
    animation_fields = {field.name for field in AnimationArgs.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in animation_fields}
    
    # Handle enum conversions
    if 'animation_mode' in filtered_data:
        mode_str = filtered_data['animation_mode']
        filtered_data['animation_mode'] = AnimationMode(mode_str)
    
    if 'color_coherence' in filtered_data:
        coherence_str = filtered_data['color_coherence']
        filtered_data['color_coherence'] = ColorCoherence(coherence_str)
    
    if 'border' in filtered_data:
        border_str = filtered_data['border']
        filtered_data['border'] = BorderMode(border_str)
    
    if 'padding_mode' in filtered_data:
        padding_str = filtered_data['padding_mode']
        filtered_data['padding_mode'] = PaddingMode(padding_str)
    
    if 'sampling_mode' in filtered_data:
        sampling_str = filtered_data['sampling_mode']
        filtered_data['sampling_mode'] = SamplingMode(sampling_str)
        
    if 'depth_algorithm' in filtered_data:
        depth_str = filtered_data['depth_algorithm']
        filtered_data['depth_algorithm'] = DepthAlgorithm(depth_str)
    
    return AnimationArgs(**filtered_data)


def create_deforum_args_from_dict(data: Dict[str, Any]) -> DeforumArgs:
    """Create DeforumArgs from legacy dictionary data"""
    deforum_fields = {field.name for field in DeforumArgs.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in deforum_fields}
    return DeforumArgs(**filtered_data)


def create_video_args_from_dict(data: Dict[str, Any]) -> VideoArgs:
    """Create VideoArgs from legacy dictionary data"""
    video_fields = {field.name for field in VideoArgs.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in video_fields}
    return VideoArgs(**filtered_data)


def create_parseq_args_from_dict(data: Dict[str, Any]) -> ParseqArgs:
    """Create ParseqArgs from legacy dictionary data"""
    parseq_fields = {field.name for field in ParseqArgs.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in parseq_fields}
    return ParseqArgs(**filtered_data)


def create_wan_args_from_dict(data: Dict[str, Any]) -> WanArgs:
    """Create WanArgs from legacy dictionary data"""
    wan_fields = {field.name for field in WanArgs.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in wan_fields}
    return WanArgs(**filtered_data)


def create_root_args_from_dict(data: Dict[str, Any]) -> RootArgs:
    """Create RootArgs from legacy dictionary data"""
    root_fields = {field.name for field in RootArgs.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in root_fields}
    
    # Convert list to tuple for immutable fields
    if 'mask_preset_names' in filtered_data and isinstance(filtered_data['mask_preset_names'], list):
        filtered_data['mask_preset_names'] = tuple(filtered_data['mask_preset_names'])
    
    if 'frames_cache' in filtered_data and isinstance(filtered_data['frames_cache'], list):
        filtered_data['frames_cache'] = tuple(filtered_data['frames_cache'])
        
    if 'prompt_keyframes' in filtered_data and not isinstance(filtered_data['prompt_keyframes'], list):
        filtered_data['prompt_keyframes'] = list(filtered_data['prompt_keyframes'])
    
    return RootArgs(**filtered_data)


@dataclass(frozen=True)
class ProcessingResult:
    """
    Immutable processing result to replace SimpleNamespace objects in generate.py.
    
    Replaces patterns like:
    processed = SimpleNamespace(images = [root.default_img], info = "Generating motion preview...")
    """
    images: Tuple[Image.Image, ...] = field(default_factory=tuple)
    info: str = ""
    success: bool = True
    processing_time: float = 0.0
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    seeds: Tuple[int, ...] = field(default_factory=tuple)
    prompts: Tuple[str, ...] = field(default_factory=tuple)
    
    @classmethod
    def create_motion_preview(cls, image: Image.Image) -> 'ProcessingResult':
        """Factory method for motion preview results"""
        return cls(
            images=(image,),
            info="Generating motion preview...",
            success=True
        )
    
    @classmethod
    def create_generation_result(cls, images: List[Image.Image], info: str) -> 'ProcessingResult':
        """Factory method for actual generation results"""
        return cls(
            images=tuple(images),
            info=info,
            success=True
        )
    
    def with_warning(self, warning: str) -> 'ProcessingResult':
        """Return new ProcessingResult with added warning"""
        new_warnings = self.warnings + (warning,)
        return ProcessingResult(
            images=self.images,
            info=self.info,
            success=self.success,
            processing_time=self.processing_time,
            warnings=new_warnings,
            seeds=self.seeds,
            prompts=self.prompts
        )


@dataclass(frozen=True)
class UIDefaults:
    """
    Immutable UI defaults to replace mutable SimpleNamespace objects in ui_left.py.
    
    Replaces patterns like:
    d = SimpleNamespace(**DeforumArgs())  # default args
    da = SimpleNamespace(**DeforumAnimArgs())  # default anim args
    """
    deforum_args: Dict[str, Any] = field(default_factory=dict)
    animation_args: Dict[str, Any] = field(default_factory=dict)
    video_args: Dict[str, Any] = field(default_factory=dict)
    parseq_args: Dict[str, Any] = field(default_factory=dict)
    wan_args: Dict[str, Any] = field(default_factory=dict)
    root_args: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_defaults(cls) -> 'UIDefaults':
        """Factory method to create UI defaults using functional args"""
        try:
            from .args import DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, ParseqArgs, WanArgs, RootArgs
            return cls(
                deforum_args=DeforumArgs(),
                animation_args=DeforumAnimArgs(),
                video_args=DeforumOutputArgs(),
                parseq_args=ParseqArgs(),
                wan_args=WanArgs(),
                root_args=RootArgs()
            )
        except ImportError:
            # Fallback for testing or when args not available
            return cls()


@dataclass(frozen=True)
class SettingsState:
    """
    Immutable settings state to replace mutable objects in settings.py.
    
    Replaces patterns like:
    wan_args = SimpleNamespace(**{name: args_dict_main[name] for name in WanArgs() if name in args_dict_main})
    """
    loaded_settings: Dict[str, Any] = field(default_factory=dict)
    validation_errors: Tuple[str, ...] = field(default_factory=tuple)
    last_modified: float = 0.0
    file_path: Optional[str] = None
    wan_args: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, settings_dict: Dict[str, Any], file_path: Optional[str] = None) -> 'SettingsState':
        """Factory method to create settings state from dictionary"""
        # Extract WAN args if present
        wan_keys = []
        try:
            from .args import WanArgs
            wan_keys = list(WanArgs().keys())
        except ImportError:
            pass
        
        wan_args = {name: settings_dict[name] for name in wan_keys if name in settings_dict}
        
        return cls(
            loaded_settings=settings_dict.copy(),
            file_path=file_path,
            wan_args=wan_args
        )
    
    def with_validation_error(self, error: str) -> 'SettingsState':
        """Return new SettingsState with added validation error"""
        new_errors = self.validation_errors + (error,)
        return SettingsState(
            loaded_settings=self.loaded_settings,
            validation_errors=new_errors,
            last_modified=self.last_modified,
            file_path=self.file_path,
            wan_args=self.wan_args
        )


@dataclass(frozen=True)
class ExternalLibraryArgs:
    """
    Immutable args for external libraries to replace SimpleNamespace objects.
    
    Replaces patterns like:
    args = SimpleNamespace() in RIFE and FILM libraries
    """
    # Common fields
    multi: float = 1.0
    video: str = ""
    output: str = ""
    img: str = ""
    exp: int = 1
    ratio: float = 0.0
    rthreshold: float = 0.02
    rmaxcycles: int = 8
    UHD: bool = False
    scale: float = 1.0
    fps: Optional[float] = None
    png: bool = False
    ext: str = "mp4"
    
    # RIFE-specific fields
    modelDir: Optional[str] = None
    fp16: bool = False
    deforum_models_path: Optional[str] = None
    raw_output_imgs_path: Optional[str] = None
    img_batch_id: Optional[str] = None
    ffmpeg_location: str = "ffmpeg"
    audio_track: Optional[str] = None
    interp_x_amount: int = 2
    slow_mo_enabled: bool = False
    slow_mo_x_amount: int = 2
    ffmpeg_crf: int = 17
    ffmpeg_preset: str = "veryslow"
    keep_imgs: bool = False
    orig_vid_name: Optional[str] = None
    
    # FILM-specific fields  
    model_path: Optional[str] = None
    input_folder: Optional[str] = None
    save_folder: Optional[str] = None
    inter_frames: Optional[int] = None
    
    @classmethod
    def create_rife_args(cls) -> 'ExternalLibraryArgs':
        """Factory method for RIFE-specific args"""
        return cls(
            multi=1.0,
            video="",
            output="",
            img="",
            exp=1,
            ratio=0.0,
            rthreshold=0.02,
            rmaxcycles=8,
            UHD=False,
            scale=1.0,
            fps=None,
            png=False,
            ext="mp4",
            fp16=False,
            interp_x_amount=2,
            slow_mo_enabled=False,
            slow_mo_x_amount=2,
            ffmpeg_crf=17,
            ffmpeg_preset="veryslow",
            keep_imgs=False,
            ffmpeg_location="ffmpeg"
        )
    
    @classmethod  
    def create_film_args(cls) -> 'ExternalLibraryArgs':
        """Factory method for FILM-specific args"""
        return cls(
            multi=1.0,
            video="",
            output="",
            img="",
            exp=1
        )


@dataclass(frozen=True)
class TestFixtureArgs:
    """
    Immutable args for test fixtures to replace SimpleNamespace objects.
    
    Replaces patterns in tests/conftest.py:
    args = SimpleNamespace()
    """
    W: int = 512
    H: int = 512
    seed: int = 42
    sampler: str = "Euler"
    steps: int = 20
    cfg_scale: float = 7.0
    strength: float = 0.85
    use_init: bool = False
    animation_mode: str = "3D"
    max_frames: int = 100
    fps: int = 30
    batch_name: str = "test_batch"
    
    @classmethod
    def create_minimal_args(cls) -> 'TestFixtureArgs':
        """Factory method for minimal test args"""
        return cls()
    
    @classmethod
    def create_animation_args(cls) -> 'TestFixtureArgs':
        """Factory method for animation test args"""
        return cls(
            max_frames=50,
            animation_mode="2D"
        )
    
    @classmethod
    def create_video_args(cls) -> 'TestFixtureArgs':
        """Factory method for video test args"""
        return cls(
            fps=24,
            max_frames=120
        )


# Type aliases for common patterns
ImageTuple = Tuple[Image.Image, ...]
StringTuple = Tuple[str, ...]
FloatTuple = Tuple[float, ...]

# Validation functions for data integrity
def validate_processing_result(result: ProcessingResult) -> bool:
    """Validate ProcessingResult data integrity"""
    if not isinstance(result.images, tuple):
        return False
    if not isinstance(result.info, str):
        return False
    if not isinstance(result.success, bool):
        return False
    if result.processing_time < 0:
        return False
    return True

def validate_ui_defaults(defaults: UIDefaults) -> bool:
    """Validate UIDefaults data integrity"""
    required_keys = ['deforum_args', 'animation_args', 'video_args', 'parseq_args', 'wan_args', 'root_args']
    for key in required_keys:
        if not hasattr(defaults, key):
            return False
        if not isinstance(getattr(defaults, key), dict):
            return False
    return True 