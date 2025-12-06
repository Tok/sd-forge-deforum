"""Model-specific configuration and validation.

Provides optimal defaults and validation for different model types:
- Flux Dev / Schnell
- Lumina 2.0
- Z-Image-Turbo
- SD 1.5 / SDXL (fallback)

Usage:
    >>> config = get_model_config("Flux/flux1-dev-bnb-nf4-v2.safetensors")
    >>> config.cfg_scale  # 1.0 (ignored for Flux)
    >>> config.distilled_cfg_scale  # 3.5 (Flux-specific)

    >>> warnings = validate_settings(args, "Flux/flux1-dev-bnb-nf4-v2.safetensors")
    >>> for warning in warnings:
    ...     logger.warning(warning)
"""

from typing import NamedTuple, List, Optional
from deforum.utils.system.logging import get_logger

logger = get_logger()


class ModelConfig(NamedTuple):
    """Model-specific configuration."""
    model_type: str
    display_name: str

    # Sampling parameters
    recommended_steps: int
    min_steps: int
    max_steps: int

    # CFG parameters
    uses_cfg: bool  # Traditional CFG scale
    cfg_scale_default: float
    cfg_scale_min: float
    cfg_scale_max: float

    uses_distilled_cfg: bool  # Flux-style distilled CFG
    distilled_cfg_scale_default: float
    distilled_cfg_scale_min: float
    distilled_cfg_scale_max: float

    # Scheduler
    recommended_scheduler: str
    compatible_schedulers: List[str]

    # Sampler
    recommended_sampler: str
    compatible_samplers: List[str]

    # Additional notes
    notes: str


# Model configurations database
MODEL_CONFIGS = {
    "flux_dev": ModelConfig(
        model_type="flux_dev",
        display_name="Flux.1 Dev",
        recommended_steps=20,
        min_steps=8,
        max_steps=50,
        uses_cfg=False,  # Flux ignores traditional CFG
        cfg_scale_default=1.0,
        cfg_scale_min=1.0,
        cfg_scale_max=1.0,
        uses_distilled_cfg=True,
        distilled_cfg_scale_default=3.5,
        distilled_cfg_scale_min=1.0,
        distilled_cfg_scale_max=10.0,
        recommended_scheduler="simple",
        compatible_schedulers=["simple", "normal", "karras", "exponential"],
        recommended_sampler="euler",
        compatible_samplers=["euler", "dpmpp_2m"],
        notes="Flux.1 Dev uses distilled CFG (3.5 recommended). Traditional CFG is ignored."
    ),

    "flux_schnell": ModelConfig(
        model_type="flux_schnell",
        display_name="Flux.1 Schnell",
        recommended_steps=4,
        min_steps=1,
        max_steps=8,
        uses_cfg=False,
        cfg_scale_default=1.0,
        cfg_scale_min=1.0,
        cfg_scale_max=1.0,
        uses_distilled_cfg=True,
        distilled_cfg_scale_default=3.5,
        distilled_cfg_scale_min=1.0,
        distilled_cfg_scale_max=10.0,
        recommended_scheduler="simple",
        compatible_schedulers=["simple"],
        recommended_sampler="euler",
        compatible_samplers=["euler"],
        notes="Flux.1 Schnell is optimized for 1-4 steps. More steps waste computation."
    ),

    "lumina": ModelConfig(
        model_type="lumina",
        display_name="Lumina 2.0",
        recommended_steps=30,
        min_steps=20,
        max_steps=50,
        uses_cfg=True,
        cfg_scale_default=5.0,
        cfg_scale_min=4.0,
        cfg_scale_max=5.5,
        uses_distilled_cfg=False,
        distilled_cfg_scale_default=3.5,  # Ignored
        distilled_cfg_scale_min=1.0,
        distilled_cfg_scale_max=10.0,
        recommended_scheduler="linear_quadratic",
        compatible_schedulers=["linear_quadratic", "normal", "karras"],
        recommended_sampler="euler",
        compatible_samplers=["euler", "dpmpp_2m"],
        notes="Lumina 2.0 uses traditional CFG (4.0-5.5). Distilled CFG is ignored. Requires linear_quadratic scheduler."
    ),

    "z_image": ModelConfig(
        model_type="z_image",
        display_name="Z-Image-Turbo",
        recommended_steps=9,
        min_steps=4,
        max_steps=15,
        uses_cfg=True,
        cfg_scale_default=2.0,
        cfg_scale_min=1.0,
        cfg_scale_max=4.0,
        uses_distilled_cfg=False,
        distilled_cfg_scale_default=3.5,  # Ignored
        distilled_cfg_scale_min=1.0,
        distilled_cfg_scale_max=10.0,
        recommended_scheduler="simple",
        compatible_schedulers=["simple", "normal"],
        recommended_sampler="euler",
        compatible_samplers=["euler", "dpmpp_2m"],
        notes="Z-Image-Turbo uses traditional CFG (2.0 recommended). Distilled CFG is ignored. Optimized for 4-15 steps (9 recommended)."
    ),

    "sdxl": ModelConfig(
        model_type="sdxl",
        display_name="SDXL",
        recommended_steps=25,
        min_steps=15,
        max_steps=50,
        uses_cfg=True,
        cfg_scale_default=7.5,
        cfg_scale_min=4.0,
        cfg_scale_max=15.0,
        uses_distilled_cfg=False,
        distilled_cfg_scale_default=3.5,  # Ignored
        distilled_cfg_scale_min=1.0,
        distilled_cfg_scale_max=10.0,
        recommended_scheduler="normal",
        compatible_schedulers=["normal", "karras", "exponential", "simple"],
        recommended_sampler="dpmpp_2m",
        compatible_samplers=["euler_a", "dpmpp_2m", "dpmpp_2m_sde", "ddim"],
        notes="SDXL uses traditional CFG (7.5 recommended). Distilled CFG is ignored."
    ),

    "sd15": ModelConfig(
        model_type="sd15",
        display_name="SD 1.5",
        recommended_steps=25,
        min_steps=15,
        max_steps=50,
        uses_cfg=True,
        cfg_scale_default=7.5,
        cfg_scale_min=4.0,
        cfg_scale_max=15.0,
        uses_distilled_cfg=False,
        distilled_cfg_scale_default=3.5,  # Ignored
        distilled_cfg_scale_min=1.0,
        distilled_cfg_scale_max=10.0,
        recommended_scheduler="normal",
        compatible_schedulers=["normal", "karras", "exponential", "simple"],
        recommended_sampler="dpmpp_2m",
        compatible_samplers=["euler_a", "dpmpp_2m", "dpmpp_2m_sde", "ddim"],
        notes="SD 1.5 uses traditional CFG (7.5 recommended). Distilled CFG is ignored."
    ),

    "unknown": ModelConfig(
        model_type="unknown",
        display_name="Unknown Model",
        recommended_steps=20,
        min_steps=1,
        max_steps=150,
        uses_cfg=True,
        cfg_scale_default=7.5,
        cfg_scale_min=0.0,
        cfg_scale_max=30.0,
        uses_distilled_cfg=False,
        distilled_cfg_scale_default=3.5,
        distilled_cfg_scale_min=0.0,
        distilled_cfg_scale_max=30.0,
        recommended_scheduler="normal",
        compatible_schedulers=["normal", "karras", "exponential", "simple", "sgm_uniform", "linear_quadratic"],
        recommended_sampler="euler",
        compatible_samplers=["euler", "euler_a", "dpmpp_2m", "dpmpp_2m_sde", "ddim", "dpm_2", "dpm_2_a"],
        notes="Unknown model type - validation disabled. Please configure settings manually based on your model's requirements."
    ),
}


def detect_model_type_extended(model_name: str) -> str:
    """Detect model type from model filename (extended version).

    Args:
        model_name: SD model filename or path

    Returns:
        Model type key: "flux_dev", "flux_schnell", "lumina", "z_image", "sdxl", "sd15", "unknown"
    """
    if not model_name:
        return "unknown"  # No model name available

    model_lower = model_name.lower()

    # Check in priority order (most specific first)
    if "schnell" in model_lower:
        return "flux_schnell"
    elif "flux" in model_lower:
        return "flux_dev"
    elif "lumina" in model_lower or "neta" in model_lower:
        return "lumina"
    elif any(pattern in model_lower for pattern in ["z-image", "zimage", "z_image", "zit", "tongyi"]):
        return "z_image"
    elif any(pattern in model_lower for pattern in ["sdxl", "sd_xl", "sd-xl", "stable-diffusion-xl"]):
        return "sdxl"
    elif any(pattern in model_lower for pattern in ["sd15", "sd_15", "sd-15", "sd1.5", "sd-1.5"]):
        return "sd15"
    else:
        return "unknown"  # Unknown model - skip validation


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for detected model type.

    Uses runtime detection (is_*_model functions) first, falls back to name-based detection.

    Args:
        model_name: SD model filename or path

    Returns:
        ModelConfig with optimal settings for this model
    """
    # Try runtime detection first (more reliable)
    try:
        from deforum.utils.model_detection import is_flux_model, is_lumina_model, is_zimage_model, is_sdxl_model

        if is_lumina_model():
            return MODEL_CONFIGS["lumina"]
        if is_flux_model():
            # Distinguish Dev vs Schnell
            if "schnell" in model_name.lower():
                return MODEL_CONFIGS["flux_schnell"]
            return MODEL_CONFIGS["flux_dev"]
        if is_zimage_model():
            return MODEL_CONFIGS["z_image"]
        if is_sdxl_model():
            return MODEL_CONFIGS["sdxl"]
    except Exception as e:
        logger.debug(f"Runtime model detection failed, using name-based fallback: {e}")

    # Fall back to name-based detection
    model_type = detect_model_type_extended(model_name)
    return MODEL_CONFIGS[model_type]


def validate_settings(args, model_name: str) -> List[str]:
    """Validate settings against model-specific recommendations.

    Args:
        args: Deforum args namespace with settings
        model_name: Current model name

    Returns:
        List of warning messages (empty if all settings are optimal)
    """
    config = get_model_config(model_name)
    warnings = []

    # Get current values from args
    steps = getattr(args, 'steps', 20)
    cfg_scale = getattr(args, 'cfg_scale', 1.0)
    sampler = getattr(args, 'sampler', 'euler')
    scheduler = getattr(args, 'sampler_schedule_type', 'normal')

    # Access distilled_cfg from anim_args if available
    distilled_cfg = 3.5  # Default
    if hasattr(args, 'anim_args'):
        # Parse schedule string to get first value
        schedule_str = getattr(args.anim_args, 'distilled_cfg_scale_schedule', '0: (3.5)')
        try:
            # Extract first value from schedule (e.g., "0: (3.5)" -> 3.5)
            import re
            match = re.search(r'\(([\d.]+)\)', schedule_str)
            if match:
                distilled_cfg = float(match.group(1))
        except:
            pass

    # STEPS VALIDATION
    if steps < config.min_steps:
        warnings.append(
            f"⚠️ Steps too low for {config.display_name}: {steps} < {config.min_steps} (recommended: {config.recommended_steps})"
        )
    elif steps > config.max_steps:
        warnings.append(
            f"⚠️ Steps too high for {config.display_name}: {steps} > {config.max_steps} (recommended: {config.recommended_steps}). "
            f"This wastes computation without improving quality."
        )
    elif steps != config.recommended_steps:
        # Info-level note (not warning)
        logger.info(
            f"Steps: {steps} (recommended: {config.recommended_steps} for {config.display_name})"
        )

    # CFG SCALE VALIDATION
    if config.uses_cfg:
        if cfg_scale < config.cfg_scale_min or cfg_scale > config.cfg_scale_max:
            warnings.append(
                f"⚠️ CFG scale out of range for {config.display_name}: {cfg_scale} "
                f"(recommended range: {config.cfg_scale_min}-{config.cfg_scale_max})"
            )
    else:
        # Model doesn't use traditional CFG
        if cfg_scale != config.cfg_scale_default:
            warnings.append(
                f"⚠️ {config.display_name} ignores traditional CFG scale. "
                f"Setting cfg_scale={cfg_scale} has no effect (should be {config.cfg_scale_default})."
            )

    # DISTILLED CFG VALIDATION
    if config.uses_distilled_cfg:
        if distilled_cfg < config.distilled_cfg_scale_min or distilled_cfg > config.distilled_cfg_scale_max:
            warnings.append(
                f"⚠️ Distilled CFG scale out of range for {config.display_name}: {distilled_cfg} "
                f"(recommended range: {config.distilled_cfg_scale_min}-{config.distilled_cfg_scale_max})"
            )
    else:
        # Model doesn't use distilled CFG
        if distilled_cfg != 3.5:  # Only warn if user changed from default
            warnings.append(
                f"⚠️ {config.display_name} ignores distilled CFG scale. "
                f"Setting distilled_cfg_scale={distilled_cfg} has no effect (Flux-only parameter)."
            )

    # SCHEDULER VALIDATION
    scheduler_lower = scheduler.lower() if scheduler else ""
    compatible_schedulers_lower = [s.lower() for s in config.compatible_schedulers]
    if scheduler_lower not in compatible_schedulers_lower:
        warnings.append(
            f"⚠️ Scheduler '{scheduler}' may not work optimally with {config.display_name}. "
            f"Recommended: {config.recommended_scheduler} "
            f"(compatible: {', '.join(config.compatible_schedulers)})"
        )

    # SAMPLER VALIDATION (case-insensitive)
    sampler_lower = sampler.lower() if sampler else ""
    compatible_samplers_lower = [s.lower() for s in config.compatible_samplers]
    if sampler_lower not in compatible_samplers_lower:
        warnings.append(
            f"⚠️ Sampler '{sampler}' may not work optimally with {config.display_name}. "
            f"Recommended: {config.recommended_sampler} "
            f"(compatible: {', '.join(config.compatible_samplers)})"
        )

    return warnings


def log_model_config(model_name: str) -> None:
    """Log detected model configuration.

    Args:
        model_name: Current model name
    """
    config = get_model_config(model_name)

    logger.info(f"")
    logger.info(f"═══ Model Configuration ═══")
    logger.info(f"Detected: {config.display_name} ({config.model_type})")
    logger.info(f"Steps: {config.recommended_steps} (range: {config.min_steps}-{config.max_steps})")

    if config.uses_cfg:
        logger.info(f"CFG Scale: {config.cfg_scale_default} (range: {config.cfg_scale_min}-{config.cfg_scale_max})")
    else:
        logger.info(f"CFG Scale: IGNORED (model doesn't use traditional CFG)")

    if config.uses_distilled_cfg:
        logger.info(f"Distilled CFG: {config.distilled_cfg_scale_default} (range: {config.distilled_cfg_scale_min}-{config.distilled_cfg_scale_max})")
    else:
        logger.info(f"Distilled CFG: IGNORED (Flux-only parameter)")

    logger.info(f"Scheduler: {config.recommended_scheduler}")
    logger.info(f"Sampler: {config.recommended_sampler}")
    logger.info(f"Note: {config.notes}")
    logger.info(f"═══════════════════════════")
    logger.info(f"")
