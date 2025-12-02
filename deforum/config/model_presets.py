"""Model-specific preset configurations for optimal Deforum performance.

This module provides automatic detection of loaded models and applies
optimal defaults based on model type, step count, and render mode.
"""

from typing import Dict, Any, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Supported model types with distinct characteristics."""
    FLUX_DEV = "flux_dev"
    FLUX_SCHNELL = "flux_schnell"
    Z_IMAGE_TURBO = "z_image_turbo"
    LUMINA_2_0 = "lumina_2_0"
    SDXL = "sdxl"
    SD_1_5 = "sd_1_5"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelPreset:
    """Optimal configuration for a specific model."""
    model_type: ModelType
    steps: int
    guidance_scale: float
    scheduler: str
    sampler: str
    width: int
    height: int
    cfg_scale: float  # Deforum's CFG (different from guidance_scale for diffusers)
    strength_keyframe: float  # For keyframe diffusions
    strength_cadence: float   # For cadence diffusions (New 3D only)
    fps: int
    cadence: int
    notes: str
    # Additional diffusers-specific params
    torch_dtype: str = "bfloat16"
    low_cpu_mem_usage: bool = False
    attention_backend: Optional[str] = None  # "flash", "_flash_3", or None (SDPA)


# ============================================================================
# Model Detection Patterns
# ============================================================================

MODEL_DETECTION_PATTERNS = {
    # Flux models
    ModelType.FLUX_DEV: [
        "flux1-dev",
        "flux-dev",
        "FLUX.1-dev",
    ],
    ModelType.FLUX_SCHNELL: [
        "flux1-schnell",
        "flux-schnell",
        "FLUX.1-schnell",
    ],

    # Z-Image
    ModelType.Z_IMAGE_TURBO: [
        "z-image-turbo",
        "z_image_turbo",
        "Z-Image-Turbo",
    ],

    # Lumina
    ModelType.LUMINA_2_0: [
        "lumina",
        "Lumina",
        "neta-lumina",
    ],

    # Stable Diffusion XL
    ModelType.SDXL: [
        "sdxl",
        "SDXL",
        "stable-diffusion-xl",
    ],

    # Stable Diffusion 1.5
    ModelType.SD_1_5: [
        "sd-v1-5",
        "sd_v1_5",
        "stable-diffusion-v1-5",
    ],
}


# ============================================================================
# Optimal Presets by Model Type
# ============================================================================

OPTIMAL_PRESETS: Dict[ModelType, ModelPreset] = {
    ModelType.FLUX_DEV: ModelPreset(
        model_type=ModelType.FLUX_DEV,
        steps=20,
        guidance_scale=3.5,  # Diffusers guidance (img2img not used for txt2img Flux)
        scheduler="karras",  # Excellent for 20 steps
        sampler="euler",
        width=1024,
        height=1024,
        cfg_scale=1.0,  # Deforum CFG (Flux doesn't use CFG in txt2img)
        strength_keyframe=0.20,  # Low preservation → 16/20 steps
        strength_cadence=0.85,   # High preservation → 3/20 steps
        fps=60,
        cadence=5,
        torch_dtype="bfloat16",
        notes="Flux Dev: 20 steps, Karras scheduler, 1024x1024 native resolution. "
              "Guidance scale used only for img2img workflows (I2V chaining)."
    ),

    ModelType.FLUX_SCHNELL: ModelPreset(
        model_type=ModelType.FLUX_SCHNELL,
        steps=4,
        guidance_scale=0.0,  # Turbo model - NO GUIDANCE!
        scheduler="linear_quadratic",  # CRITICAL: Specifically designed for 4-6 steps
        sampler="euler",
        width=1024,
        height=1024,
        cfg_scale=1.0,
        strength_keyframe=0.25,  # 25% with 4 steps = 1 step denoising
        strength_cadence=0.75,   # 75% with 4 steps = 3 steps denoising
        fps=24,
        cadence=3,
        torch_dtype="bfloat16",
        notes="Flux Schnell: 4 steps ONLY, guidance_scale=0.0 (turbo), "
              "LINEAR QUADRATIC scheduler required for optimal 4-step quality. "
              "Coarse strength resolution (25% per step) - use Flux Dev for finer control."
    ),

    ModelType.Z_IMAGE_TURBO: ModelPreset(
        model_type=ModelType.Z_IMAGE_TURBO,
        steps=9,  # Results in 8 DiT forwards
        guidance_scale=0.0,  # Turbo model - NO GUIDANCE!
        scheduler="karras",  # Default, works well
        sampler="euler",
        width=1024,
        height=1024,
        cfg_scale=1.0,
        strength_keyframe=0.20,
        strength_cadence=0.85,
        fps=60,
        cadence=5,
        torch_dtype="bfloat16",
        low_cpu_mem_usage=False,
        notes="Z-Image Turbo: 9 steps (8 DiT forwards), guidance_scale=0.0 (turbo), "
              "1024x1024 native resolution. Fast generation with good quality."
    ),

    ModelType.LUMINA_2_0: ModelPreset(
        model_type=ModelType.LUMINA_2_0,
        steps=30,
        guidance_scale=4.5,  # Lumina uses 4.0-5.5 range
        scheduler="karras",  # Or exponential for long schedules
        sampler="euler",
        width=1024,
        height=1024,
        cfg_scale=4.5,  # Lumina benefits from CFG
        strength_keyframe=0.20,
        strength_cadence=0.85,
        fps=60,
        cadence=5,
        torch_dtype="float16",  # Lumina uses fp16 text encoder
        notes="Lumina 2.0: 30 steps, Karras or Exponential scheduler, "
              "anime-optimized, CFG 4.0-5.5, uses linear_quadratic internally. "
              "May override scheduler selection."
    ),

    ModelType.SDXL: ModelPreset(
        model_type=ModelType.SDXL,
        steps=30,
        guidance_scale=7.5,
        scheduler="karras",
        sampler="dpmpp_2m",
        width=1024,
        height=1024,
        cfg_scale=7.5,
        strength_keyframe=0.30,
        strength_cadence=0.85,
        fps=30,
        cadence=3,
        torch_dtype="float16",
        notes="SDXL: 30 steps, DPM++ 2M Karras, 1024x1024 native, CFG 7.5"
    ),

    ModelType.SD_1_5: ModelPreset(
        model_type=ModelType.SD_1_5,
        steps=25,
        guidance_scale=7.5,
        scheduler="karras",
        sampler="dpmpp_2m",
        width=512,
        height=512,
        cfg_scale=7.5,
        strength_keyframe=0.35,
        strength_cadence=0.85,
        fps=30,
        cadence=2,
        torch_dtype="float16",
        notes="SD 1.5: 25 steps, DPM++ 2M Karras, 512x512 native, CFG 7.5"
    ),
}


# ============================================================================
# Scheduler Recommendations by Step Count (from SCHEDULER_RECOMMENDATIONS.md)
# ============================================================================

SCHEDULER_BY_STEPS = {
    4: "linear_quadratic",   # CRITICAL for 4 steps (Flux Schnell)
    6: "linear_quadratic",   # Still good for 6 steps
    8: "kl_optimal",         # Information-theoretic optimal
    20: "karras",            # Default for Flux Dev
    30: "karras",            # Default for Lumina, or "exponential" for smoothness
}


def get_optimal_scheduler(steps: int, model_type: ModelType) -> str:
    """Get optimal scheduler based on step count and model type.

    Args:
        steps: Number of diffusion steps
        model_type: Type of model being used

    Returns:
        Recommended scheduler name
    """
    # Special cases
    if model_type == ModelType.FLUX_SCHNELL:
        return "linear_quadratic"  # Always use for Schnell

    # General recommendations by step count
    if steps <= 6:
        return "linear_quadratic"
    elif steps <= 10:
        return "kl_optimal"
    elif steps <= 25:
        return "karras"
    else:
        return "karras"  # Or "exponential" for very long schedules


# ============================================================================
# Model Detection
# ============================================================================

def detect_model_type(model_name: str) -> ModelType:
    """Detect model type from checkpoint filename or path.

    Args:
        model_name: Model checkpoint filename or path

    Returns:
        Detected ModelType enum
    """
    if not model_name:
        return ModelType.UNKNOWN

    model_name_lower = model_name.lower()

    for model_type, patterns in MODEL_DETECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in model_name_lower:
                return model_type

    return ModelType.UNKNOWN


def get_preset_for_model(model_name: str) -> Optional[ModelPreset]:
    """Get optimal preset for a loaded model.

    Args:
        model_name: Model checkpoint filename or path

    Returns:
        ModelPreset if detected, None otherwise
    """
    model_type = detect_model_type(model_name)

    if model_type == ModelType.UNKNOWN:
        return None

    return OPTIMAL_PRESETS.get(model_type)


def get_preset_adjustments_for_render_mode(
    preset: ModelPreset,
    render_mode: str
) -> Dict[str, Any]:
    """Adjust preset parameters based on render mode.

    Args:
        preset: Base model preset
        render_mode: Deforum render mode ('Classic 3D', 'New 3D', etc.)

    Returns:
        Dict of parameter adjustments
    """
    adjustments = {}

    if render_mode == "Classic 3D":
        # Fixed cadence, stable generation
        adjustments["fps"] = 24
        adjustments["cadence"] = 2
        adjustments["strength_cadence"] = 0.85  # High stability

    elif render_mode == "New 3D":
        # Keyframe redistribution, dual strength
        adjustments["fps"] = 60
        adjustments["cadence"] = 5  # Pseudo-cadence
        # Use preset's strength values (already optimized)

    elif render_mode == "Keyframes Only":
        # Only keyframes, depth tweening
        adjustments["fps"] = 60
        adjustments["cadence"] = None  # No cadence
        adjustments["strength_keyframe"] = 0.15  # Very low for dramatic changes

    elif render_mode == "Flux + Interpolation":
        # Keyframes + AI interpolation
        adjustments["fps"] = 24
        adjustments["cadence"] = None  # No cadence
        # Strength used for I2V chaining (Wan FLF2V)
        adjustments["strength_keyframe"] = 0.65  # For I2V continuity

    return adjustments


# ============================================================================
# Preset Application
# ============================================================================

def create_preset_message(preset: ModelPreset, render_mode: str) -> str:
    """Create user-friendly message describing preset.

    Args:
        preset: Model preset to describe
        render_mode: Current render mode

    Returns:
        Formatted message string
    """
    adjustments = get_preset_adjustments_for_render_mode(preset, render_mode)

    msg = f"""
🎯 Optimal Settings for {preset.model_type.value.upper()}

**Core Settings:**
• Steps: {preset.steps}
• Scheduler: {preset.scheduler.upper()}
• Resolution: {preset.width}x{preset.height}
• FPS: {adjustments.get('fps', preset.fps)}

**Strength (Inverted):**
• Keyframe: {adjustments.get('strength_keyframe', preset.strength_keyframe)} (preservation)
• Cadence: {adjustments.get('strength_cadence', preset.strength_cadence)} (preservation)

**Guidance:**
• CFG Scale: {preset.cfg_scale}
• Guidance (diffusers): {preset.guidance_scale}

**Notes:**
{preset.notes}

**Apply these settings?**
""".strip()

    return msg


def get_settings_dict_from_preset(
    preset: ModelPreset,
    render_mode: str
) -> Dict[str, Any]:
    """Convert preset to Deforum settings dict.

    Args:
        preset: Model preset
        render_mode: Current render mode

    Returns:
        Dict mapping Deforum component names to values
    """
    adjustments = get_preset_adjustments_for_render_mode(preset, render_mode)

    return {
        # Core sampling
        "steps": preset.steps,
        "sampler": preset.sampler,
        "scheduler": preset.scheduler,

        # Resolution
        "W": preset.width,
        "H": preset.height,

        # CFG
        "scale": preset.cfg_scale,

        # Strength (inverted semantics!)
        "strength": adjustments.get("strength_keyframe", preset.strength_keyframe),
        "strength_0_no_init": adjustments.get("strength_keyframe", preset.strength_keyframe),

        # Animation
        "fps": adjustments.get("fps", preset.fps),
        "cadence": adjustments.get("cadence", preset.cadence) if adjustments.get("cadence") is not None else preset.cadence,
    }
