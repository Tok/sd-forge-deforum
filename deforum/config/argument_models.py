"""
Immutable argument models for Deforum configuration.

This module contains frozen dataclasses that replace the mutable dictionary-based
argument system, providing type safety, validation, and immutability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ..data_models import (
    AnimationMode, BorderMode, ColorCoherence, DepthAlgorithm, 
    PaddingMode, SamplingMode, validate_positive_int, validate_range,
    validate_non_negative_number
)


class SamplerType(Enum):
    """Sampler enumeration for type safety"""
    EULER = "Euler"
    EULER_A = "Euler a"
    HEUN = "Heun"
    DPM2 = "DPM2"
    DPM2_A = "DPM2 a"
    DPM_PLUS_PLUS_2S_A = "DPM++ 2S a"
    DPM_PLUS_PLUS_2M = "DPM++ 2M"
    DPM_PLUS_PLUS_SDE = "DPM++ SDE"
    DPM_FAST = "DPM fast"
    DPM_ADAPTIVE = "DPM adaptive"
    LMS = "LMS"
    PLMS = "PLMS"
    DDIM = "DDIM"
    DDPM = "DDPM"


class SchedulerType(Enum):
    """Scheduler enumeration for type safety"""
    SIMPLE = "Simple"
    NORMAL = "Normal"
    KARRAS = "Karras"
    EXPONENTIAL = "Exponential"
    POLYEXPONENTIAL = "Polyexponential"


class SeedBehavior(Enum):
    """Seed behavior enumeration"""
    ITER = "iter"
    FIXED = "fixed"
    RANDOM = "random"
    LADDER = "ladder"
    ALTERNATE = "alternate"
    SCHEDULE = "schedule"


class NoiseType(Enum):
    """Noise type enumeration"""
    UNIFORM = "uniform"
    PERLIN = "perlin"


class RerollBehavior(Enum):
    """Reroll behavior enumeration"""
    REROLL = "reroll"
    INTERRUPT = "interrupt"
    IGNORE = "ignore"


class MaskFill(Enum):
    """Mask fill enumeration"""
    FILL = "fill"
    ORIGINAL = "original"
    LATENT_NOISE = "latent noise"
    LATENT_NOTHING = "latent nothing"


@dataclass(frozen=True)
class DeforumGenerationArgs:
    """Immutable core generation arguments with validation"""
    
    # Image dimensions
    width: int = 1024
    height: int = 1024
    
    # Generation settings
    seed: int = -1
    sampler: SamplerType = SamplerType.EULER
    scheduler: SchedulerType = SchedulerType.SIMPLE
    steps: int = 20
    cfg_scale: float = 7.0
    
    # Advanced settings
    tiling: bool = False
    restore_faces: bool = False
    seed_resize_from_w: int = 0
    seed_resize_from_h: int = 0
    
    # Seed behavior
    seed_behavior: SeedBehavior = SeedBehavior.ITER
    seed_iter_n: int = 1
    
    # Init image settings
    use_init: bool = False
    strength: float = 0.85
    strength_0_no_init: bool = True
    init_image: Optional[str] = "https://deforum.github.io/a1/I1.png"
    init_image_box: Optional[Any] = None
    
    # Mask settings
    use_mask: bool = False
    use_alpha_as_mask: bool = False
    mask_file: str = "https://deforum.github.io/a1/M1.jpg"
    invert_mask: bool = False
    mask_contrast_adjust: float = 1.0
    mask_brightness_adjust: float = 1.0
    overlay_mask: bool = True
    mask_overlay_blur: int = 4
    fill: MaskFill = MaskFill.ORIGINAL
    full_res_mask: bool = True
    full_res_mask_padding: int = 4
    
    # Output settings
    batch_name: str = "Deforum_{timestring}"
    
    # Error handling
    reroll_blank_frames: RerollBehavior = RerollBehavior.IGNORE
    reroll_patience: int = 10
    
    # Preview mode
    motion_preview_mode: bool = False
    
    # UI settings
    show_info_on_ui: bool = False
    show_controlnet_tab: bool = False
    
    def __post_init__(self):
        """Validate all arguments"""
        validate_positive_int(self.width, "width")
        validate_positive_int(self.height, "height")
        validate_positive_int(self.steps, "steps")
        validate_range(self.cfg_scale, 1.0, 30.0, "cfg_scale")
        validate_range(self.strength, 0.0, 1.0, "strength")
        validate_non_negative_number(self.mask_overlay_blur, "mask_overlay_blur")
        validate_range(self.mask_contrast_adjust, 0.0, 2.0, "mask_contrast_adjust")
        validate_range(self.mask_brightness_adjust, 0.0, 2.0, "mask_brightness_adjust")
        validate_positive_int(self.seed_iter_n, "seed_iter_n")
        validate_positive_int(self.reroll_patience, "reroll_patience")
        
        # Validate dimensions are multiples of 8
        if self.width % 8 != 0:
            raise ValueError(f"Width must be multiple of 8, got {self.width}")
        if self.height % 8 != 0:
            raise ValueError(f"Height must be multiple of 8, got {self.height}")


@dataclass(frozen=True)
class DeforumAnimationArgs:
    """Immutable animation arguments with validation"""
    
    # Core animation settings
    animation_mode: AnimationMode = AnimationMode.THREE_D
    max_frames: int = 333
    border: BorderMode = BorderMode.REPLICATE
    
    # Movement schedules (as strings for now - will be parsed by schedule system)
    angle: str = "0: (0)"
    zoom: str = "0: (1.0)"
    translation_x: str = "0: (0)"
    translation_y: str = "0: (0)"
    translation_z: str = "0: (0)"
    transform_center_x: str = "0: (0.5)"
    transform_center_y: str = "0: (0.5)"
    rotation_3d_x: str = "0: (0)"
    rotation_3d_y: str = "0: (0)"
    rotation_3d_z: str = "0: (0)"
    
    # Camera shake
    shake_name: str = "INVESTIGATION"
    shake_intensity: float = 1.0
    shake_speed: float = 1.0
    
    # Perspective flip
    enable_perspective_flip: bool = False
    perspective_flip_theta: str = "0: (0)"
    perspective_flip_phi: str = "0: (0)"
    perspective_flip_gamma: str = "0: (0)"
    perspective_flip_fv: str = "0: (53)"
    
    # Schedules
    noise_schedule: str = "0: (0.065)"
    strength_schedule: str = "0: (0.85)"
    keyframe_strength_schedule: str = "0: (0.50)"
    contrast_schedule: str = "0: (1.0)"
    cfg_scale_schedule: str = "0: (1.0)"
    distilled_cfg_scale_schedule: str = "0: (3.5)"
    
    # Steps scheduling
    enable_steps_scheduling: bool = False
    steps_schedule: str = "0: (20)"
    
    # FOV and aspect ratio
    fov_schedule: str = "0: (70)"
    aspect_ratio_schedule: str = "0: (1.0)"
    aspect_ratio_use_old_formula: bool = False
    near_schedule: str = "0: (200)"
    far_schedule: str = "0: (10000)"
    
    # Seed scheduling
    seed_schedule: str = '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)'
    enable_subseed_scheduling: bool = False
    subseed_schedule: str = "0: (1)"
    subseed_strength_schedule: str = "0: (0)"
    
    # Sampler scheduling
    enable_sampler_scheduling: bool = False
    sampler_schedule: str = '0: ("Euler")'
    
    # Scheduler scheduling  
    enable_scheduler_scheduling: bool = False
    scheduler_schedule: str = '0: ("Simple")'
    
    # Mask scheduling
    use_noise_mask: bool = False
    mask_schedule: str = '0: ("{video_mask}")'
    noise_mask_schedule: str = '0: ("{video_mask}")'
    
    # Checkpoint scheduling
    enable_checkpoint_scheduling: bool = False
    checkpoint_schedule: str = '0: ("model1.ckpt"), 100: ("model2.safetensors")'
    
    # CLIP skip scheduling
    enable_clipskip_scheduling: bool = False
    clipskip_schedule: str = "0: (2)"
    
    # Noise multiplier scheduling
    enable_noise_multiplier_scheduling: bool = True
    noise_multiplier_schedule: str = "0: (1.05)"
    
    # Resume settings
    resume_from_timestring: bool = False
    resume_timestring: str = "20241111111111"
    
    # ETA scheduling
    enable_ddim_eta_scheduling: bool = False
    ddim_eta_schedule: str = "0: (0)"
    enable_ancestral_eta_scheduling: bool = False
    ancestral_eta_schedule: str = "0: (1)"
    
    # Additional schedules
    amount_schedule: str = "0: (0.1)"
    kernel_schedule: str = "0: (5)"
    sigma_schedule: str = "0: (1)"
    threshold_schedule: str = "0: (0)"
    
    # Color coherence
    color_coherence: ColorCoherence = ColorCoherence.NONE
    color_coherence_image_path: str = ""
    color_coherence_video_every_n_frames: int = 1
    color_force_grayscale: bool = False
    legacy_colormatch: bool = False
    
    # Cadence and optical flow
    keyframe_distribution: str = "Redistributed"
    diffusion_cadence: int = 10
    optical_flow_cadence: str = "None"
    cadence_flow_factor_schedule: str = "0: (1)"
    optical_flow_redo_generation: str = "None"
    redo_flow_factor_schedule: str = "0: (1)"
    diffusion_redo: str = "0"
    
    # Noise settings
    noise_type: NoiseType = NoiseType.PERLIN
    perlin_w: float = 8.0
    perlin_h: float = 8.0
    perlin_octaves: int = 4
    perlin_persistence: float = 0.5
    
    # Depth settings
    use_depth_warping: bool = True
    depth_algorithm: DepthAlgorithm = DepthAlgorithm.DEPTH_ANYTHING_V2_SMALL
    midas_weight: float = 0.2
    padding_mode: PaddingMode = PaddingMode.BORDER
    sampling_mode: SamplingMode = SamplingMode.BICUBIC
    save_depth_maps: bool = False
    
    # Video init
    video_init_path: str = 'https://deforum.github.io/a1/V1.mp4'
    extract_nth_frame: int = 1
    extract_from_frame: int = 0
    extract_to_frame: int = -1
    overwrite_extracted_frames: bool = False
    
    # Video mask
    use_mask_video: bool = False
    video_mask_path: str = 'https://deforum.github.io/a1/VM1.mp4'
    
    # Hybrid video settings
    hybrid_comp_alpha_schedule: str = "0:(0.5)"
    hybrid_comp_mask_blend_alpha_schedule: str = "0:(0.5)"
    hybrid_comp_mask_contrast_schedule: str = "0:(1)"
    hybrid_comp_mask_auto_contrast_cutoff_high_schedule: str = "0:(100)"
    hybrid_comp_mask_auto_contrast_cutoff_low_schedule: str = "0:(0)"
    hybrid_flow_factor_schedule: str = "0:(1)"
    hybrid_generate_inputframes: bool = False
    hybrid_generate_human_masks: str = "None"
    hybrid_use_first_frame_as_init_image: bool = True
    hybrid_motion: str = "None"
    hybrid_motion_use_prev_img: bool = False
    hybrid_flow_consistency: bool = False
    hybrid_consistency_blur: int = 2
    hybrid_flow_method: str = "RAFT"
    hybrid_composite: str = "None"
    hybrid_use_init_image: bool = False
    hybrid_comp_mask_type: str = "None"
    hybrid_comp_mask_inverse: bool = False
    hybrid_comp_mask_equalize: str = "None"
    hybrid_comp_mask_auto_contrast: bool = False
    hybrid_comp_save_extra_frames: bool = False
    
    def __post_init__(self):
        """Validate animation arguments"""
        validate_positive_int(self.max_frames, "max_frames")
        validate_range(self.shake_intensity, 0.0, 3.0, "shake_intensity")
        validate_range(self.shake_speed, 0.0, 3.0, "shake_speed")
        validate_positive_int(self.color_coherence_video_every_n_frames, "color_coherence_video_every_n_frames")
        validate_positive_int(self.diffusion_cadence, "diffusion_cadence")
        validate_positive_int(self.perlin_octaves, "perlin_octaves")
        validate_range(self.perlin_persistence, 0.0, 1.0, "perlin_persistence")
        validate_range(self.midas_weight, -1.0, 1.0, "midas_weight")
        validate_positive_int(self.extract_nth_frame, "extract_nth_frame")
        validate_non_negative_number(self.extract_from_frame, "extract_from_frame")
        validate_range(self.hybrid_consistency_blur, 0, 16, "hybrid_consistency_blur")


@dataclass(frozen=True)
class DeforumVideoArgs:
    """Immutable video output arguments with validation"""
    
    # Video creation
    skip_video_creation: bool = False
    fps: int = 60
    make_gif: bool = False
    
    # Cleanup options
    delete_imgs: bool = False
    delete_input_frames: bool = False
    
    # Output path
    image_path: str = "C:/SD/20241111111111_%09d.png"
    
    # Audio
    add_soundtrack: str = "File"  # 'None', 'File', 'Init Video'
    soundtrack_path: str = "https://ia801303.us.archive.org/26/items/amen-breaks/cw_amen13_173.mp3"
    
    # Upscaling
    r_upscale_video: bool = False
    r_upscale_factor: str = "x2"  # 'x2', 'x3', 'x4'
    r_upscale_model: str = 'realesr-animevideov3'
    r_upscale_keep_imgs: bool = True
    
    # Memory management
    store_frames_in_ram: bool = False
    
    # Frame interpolation
    frame_interpolation_engine: str = "None"  # 'None', 'RIFE v4.6', 'FILM'
    frame_interpolation_x_amount: int = 2
    frame_interpolation_slow_mo_enabled: bool = False
    frame_interpolation_slow_mo_amount: int = 2
    frame_interpolation_keep_imgs: bool = False
    frame_interpolation_use_upscaled: bool = False
    
    def __post_init__(self):
        """Validate video arguments"""
        validate_range(self.fps, 1, 240, "fps")
        validate_range(self.frame_interpolation_x_amount, 2, 10, "frame_interpolation_x_amount")
        validate_range(self.frame_interpolation_slow_mo_amount, 2, 10, "frame_interpolation_slow_mo_amount")


@dataclass(frozen=True)
class ParseqArgs:
    """Immutable Parseq arguments with validation"""
    
    parseq_manifest: Optional[str] = None
    parseq_use_deltas: bool = True
    parseq_non_schedule_overrides: bool = True
    
    def __post_init__(self):
        """Validate Parseq arguments"""
        if self.parseq_manifest is not None and self.parseq_manifest.strip():
            manifest = self.parseq_manifest.strip()
            if not (manifest.startswith('{') or manifest.startswith('http')):
                raise ValueError("parseq_manifest must be valid JSON or URL")


@dataclass(frozen=True)
class WanArgs:
    """Immutable Wan 2.1 Video Generation Arguments"""
    
    # Core settings
    wan_mode: str = "Disabled"  # "Disabled", "T2V Only", "I2V Chaining"
    wan_model_path: str = "models/wan"
    wan_model_name: str = "Auto-Select"
    wan_enable_prompt_enhancement: bool = False
    wan_qwen_model: str = "Auto-Select"
    wan_qwen_language: str = "English"  # "English", "Chinese"
    wan_auto_download: bool = True
    wan_preferred_size: str = "1.3B VACE (Recommended)"
    
    # Movement analysis
    wan_enable_movement_analysis: bool = True
    wan_movement_sensitivity: float = 1.0
    
    # Style settings
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


@dataclass(frozen=True) 
class RootArgs:
    """Immutable root arguments - runtime state and shared data"""
    
    # System settings
    device: Optional[str] = None
    models_path: str = ""
    half_precision: bool = True
    
    # Runtime data
    clipseg_model: Optional[Any] = None
    mask_preset_names: Tuple[str, ...] = ("everywhere", "video_mask")
    frames_cache: Tuple[Any, ...] = field(default_factory=tuple)
    
    # Batch and seed info
    raw_batch_name: Optional[str] = None
    raw_seed: Optional[int] = None
    timestring: str = ""
    
    # Subseed settings
    subseed: int = -1
    subseed_strength: float = 0.0
    seed_internal: int = 0
    
    # Generation state
    init_sample: Optional[Any] = None
    noise_mask: Optional[Any] = None
    initial_info: Optional[str] = None
    first_frame: Optional[Any] = None
    default_img: Optional[Any] = None
    
    # Prompt data
    animation_prompts: Dict[str, str] = field(default_factory=dict)
    prompt_keyframes: List[str] = field(default_factory=list)
    
    # System info
    current_user_os: str = ""
    tmp_deforum_run_duplicated_folder: str = ""
    
    # Job tracking
    job_id: Optional[str] = None


@dataclass(frozen=True)
class ProcessedArguments:
    """Immutable result of argument processing"""
    
    deforum: DeforumGenerationArgs
    animation: DeforumAnimationArgs
    video: DeforumVideoArgs
    parseq: ParseqArgs
    wan: WanArgs
    root: RootArgs
    
    # Processing metadata
    processing_time: float = 0.0
    validation_warnings: List[str] = field(default_factory=list)
    applied_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArgumentValidationResult:
    """Immutable result of argument validation"""
    
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_args: Optional[ProcessedArguments] = None 