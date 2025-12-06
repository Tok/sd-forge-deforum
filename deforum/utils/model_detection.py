"""Model Detection Utilities

Detect which diffusion model is currently loaded in Forge.
Used for model-specific tuning and compatibility handling.
"""

from typing import Any, Optional
from dataclasses import dataclass

from deforum.utils.system.logging import get_logger

logger = get_logger()


# Model-specific configuration constants
@dataclass(frozen=True)
class ModelConfig:
    """Configuration parameters for a specific model type."""

    name: str
    cfg_range: tuple[float, float]
    recommended_steps: int


LUMINA_CONFIG = ModelConfig(
    name="Lumina 2.0",
    cfg_range=(4.0, 5.5),
    recommended_steps=30,
)

FLUX_CONFIG = ModelConfig(
    name="Flux",
    cfg_range=(1.0, 3.5),
    recommended_steps=20,
)

DEFAULT_CONFIG = ModelConfig(
    name="Unknown",
    cfg_range=(7.0, 12.0),
    recommended_steps=20,
)


def _get_shared_module() -> Optional[Any]:
    """Safely import and return modules.shared.

    Returns:
        modules.shared module or None if unavailable
    """
    try:
        import modules.shared as shared
        return shared
    except ImportError:
        logger.debug("modules.shared not available")
        return None


def _get_model_class_name(model: Any) -> Optional[str]:
    """Extract class name from model object.

    Args:
        model: Model object

    Returns:
        Class name string or None
    """
    return model.__class__.__name__ if hasattr(model, '__class__') else None


def _get_checkpoint_name(shared: Any) -> Optional[str]:
    """Extract checkpoint name from shared module.

    Args:
        shared: modules.shared module

    Returns:
        Lowercase checkpoint name or None
    """
    if not (hasattr(shared, 'opts') and hasattr(shared.opts, 'sd_model_checkpoint')):
        return None
    return shared.opts.sd_model_checkpoint.lower()


def _check_diffusion_model_class(model: Any, target_class: str) -> bool:
    """Check if model's diffusion model matches target class name.

    Args:
        model: Model object to check
        target_class: Expected diffusion model class name

    Returns:
        True if match found, False otherwise
    """
    if not hasattr(model, 'forge_objects'):
        return False

    unet = model.forge_objects.unet
    if not (hasattr(unet, 'model') and hasattr(unet.model, 'diffusion_model')):
        return False

    diff_model = unet.model.diffusion_model
    return diff_model.__class__.__name__ == target_class


def is_lumina_model() -> bool:
    """Detect if Lumina 2.0 is the currently loaded model.

    Lumina requires special handling:
    - Different CFG scale ranges (4.0-5.5 vs Flux's 1.0-3.5)
    - Different scheduler preferences (linear_quadratic vs simple)
    - Different step counts (30 recommended vs 20 for Flux)
    - num_tokens parameter required in dynamic_args

    Returns:
        True if Lumina is loaded, False otherwise
    """
    try:
        shared = _get_shared_module()
        if shared is None or not hasattr(shared, 'sd_model'):
            return False

        model = shared.sd_model

        # Check 1: Model class name contains 'Lumina'
        class_name = _get_model_class_name(model)
        if class_name and 'Lumina' in class_name:
            logger.debug(f"Detected Lumina model via class name: {class_name}")
            return True

        # Check 2: Diffusion engine type is Lumina2NextDiT
        if _check_diffusion_model_class(model, 'Lumina2NextDiT'):
            logger.debug("Detected Lumina model via diffusion_model class")
            return True

        # Check 3: Checkpoint filename contains 'lumina' or 'neta'
        checkpoint_name = _get_checkpoint_name(shared)
        if checkpoint_name and ('lumina' in checkpoint_name or 'neta' in checkpoint_name):
            logger.debug(f"Detected Lumina model via checkpoint name: {checkpoint_name}")
            return True

        return False

    except Exception as e:
        logger.debug(f"Lumina detection failed: {e}")
        return False


def is_flux_model() -> bool:
    """Detect if Flux (Dev or Schnell) is the currently loaded model.

    Returns:
        True if Flux is loaded, False otherwise
    """
    try:
        shared = _get_shared_module()
        if shared is None or not hasattr(shared, 'sd_model'):
            return False

        model = shared.sd_model

        # Check 1: Model class name contains 'Flux'
        class_name = _get_model_class_name(model)
        if class_name and 'Flux' in class_name:
            return True

        # Check 2: Checkpoint name contains 'flux'
        checkpoint_name = _get_checkpoint_name(shared)
        if checkpoint_name and 'flux' in checkpoint_name:
            return True

        return False

    except Exception as e:
        logger.debug(f"Flux detection failed: {e}")
        return False


def is_zimage_model() -> bool:
    """Detect if Z-Image-Turbo is the currently loaded model.

    Returns:
        True if Z-Image-Turbo is loaded, False otherwise
    """
    try:
        shared = _get_shared_module()
        if shared is None or not hasattr(shared, 'sd_model'):
            return False

        model = shared.sd_model

        # Check 1: Model class name contains 'SD3' (Z-Image is based on SD3)
        class_name = _get_model_class_name(model)
        if class_name and 'SD3' in class_name:
            # Need to disambiguate from actual SD3 models
            checkpoint_name = _get_checkpoint_name(shared)
            if checkpoint_name:
                checkpoint_lower = checkpoint_name.lower()
                # Z-Image specific patterns
                if any(pattern in checkpoint_lower for pattern in ['z-image', 'zimage', 'zit', 'tongyi']):
                    logger.debug(f"Detected Z-Image model via checkpoint name: {checkpoint_name}")
                    return True
                # Check path contains Z-Image directory
                if hasattr(shared.sd_model, 'sd_checkpoint_info'):
                    full_path = getattr(shared.sd_model.sd_checkpoint_info, 'filename', '')
                    if 'z-image' in full_path.lower():
                        logger.debug(f"Detected Z-Image model via path: {full_path}")
                        return True

        # Check 2: Checkpoint name contains z-image patterns
        checkpoint_name = _get_checkpoint_name(shared)
        if checkpoint_name:
            checkpoint_lower = checkpoint_name.lower()
            if any(pattern in checkpoint_lower for pattern in ['z-image', 'zimage', 'zit', 'tongyi']):
                logger.debug(f"Detected Z-Image model via checkpoint name: {checkpoint_name}")
                return True

        # Check 3: Full path contains Z-Image directory
        if hasattr(shared.sd_model, 'sd_checkpoint_info'):
            full_path = getattr(shared.sd_model.sd_checkpoint_info, 'filename', '')
            if 'z-image' in full_path.lower():
                logger.debug(f"Detected Z-Image model via path: {full_path}")
                return True

        return False

    except Exception as e:
        logger.debug(f"Z-Image detection failed: {e}")
        return False


def _detect_flux_variant(checkpoint_name: Optional[str]) -> str:
    """Detect Flux variant (Dev or Schnell) from checkpoint name.

    Args:
        checkpoint_name: Lowercase checkpoint filename

    Returns:
        "Flux Schnell", "Flux Dev", or "Flux"
    """
    if checkpoint_name and 'schnell' in checkpoint_name:
        return "Flux Schnell"
    return "Flux Dev" if checkpoint_name else "Flux"


def get_model_name() -> str:
    """Get friendly name of currently loaded model.

    Returns:
        Model name string ("Lumina 2.0", "Flux Dev", "Flux Schnell", "Z-Image-Turbo", "Unknown")
    """
    if is_lumina_model():
        return LUMINA_CONFIG.name

    if is_flux_model():
        shared = _get_shared_module()
        checkpoint_name = _get_checkpoint_name(shared) if shared else None
        return _detect_flux_variant(checkpoint_name)

    if is_zimage_model():
        return "Z-Image-Turbo"

    return DEFAULT_CONFIG.name


def get_recommended_cfg_scale() -> tuple[float, float]:
    """Get recommended CFG scale range for current model.

    Returns:
        Tuple of (min_cfg, max_cfg)
    """
    if is_lumina_model():
        return LUMINA_CONFIG.cfg_range
    if is_flux_model():
        return FLUX_CONFIG.cfg_range
    return DEFAULT_CONFIG.cfg_range


def get_recommended_steps() -> int:
    """Get recommended step count for current model.

    Returns:
        Recommended number of sampling steps
    """
    if is_lumina_model():
        return LUMINA_CONFIG.recommended_steps
    if is_flux_model():
        return FLUX_CONFIG.recommended_steps
    return DEFAULT_CONFIG.recommended_steps
