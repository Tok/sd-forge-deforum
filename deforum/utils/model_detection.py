"""Model Detection Utilities

Detect which diffusion model is currently loaded in Forge.
Used for model-specific tuning and compatibility handling.
"""

from typing import Any, Optional
from dataclasses import dataclass

from deforum.utils.system.logging import get_logger

logger = get_logger()

# Cache for model detection results (keyed by checkpoint filename)
_detection_cache = {
    'checkpoint': None,
    'is_flux': None,
    'is_lumina': None,
    'is_zimage': None,
    'is_sdxl': None,
    'model_name': None,
}


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

FLUX2_KLEIN_CONFIG = ModelConfig(
    name="Flux 2 Klein",
    cfg_range=(1.0, 5.0),  # Klein supports wider CFG range
    recommended_steps=20,
)

ZIMAGE_CONFIG = ModelConfig(
    name="Z-Image-Turbo",
    cfg_range=(0.0, 0.0),  # Z-Image requires CFG=0.0 (distilled model)
    recommended_steps=9,
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


def _invalidate_cache_if_model_changed(checkpoint_name: str) -> bool:
    """Check if model changed and invalidate cache if needed.

    Args:
        checkpoint_name: Current checkpoint filename

    Returns:
        True if cache was invalidated (model changed), False otherwise
    """
    global _detection_cache

    if _detection_cache['checkpoint'] != checkpoint_name:
        # Model changed - invalidate all cached detection results
        _detection_cache = {
            'checkpoint': checkpoint_name,
            'is_flux': None,
            'is_lumina': None,
            'is_zimage': None,
            'is_sdxl': None,
            'model_name': None,
        }
        logger.debug(f"Model detection cache invalidated for new checkpoint: {checkpoint_name}")
        return True
    return False


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


def is_sdxl_model() -> bool:
    """Detect if SDXL is the currently loaded model.

    Returns:
        True if SDXL is loaded, False otherwise
    """
    try:
        shared = _get_shared_module()
        if shared is None or not hasattr(shared, 'sd_model'):
            return False

        model = shared.sd_model

        # Get detection data
        class_name = _get_model_class_name(model)
        checkpoint_name = _get_checkpoint_name(shared)
        full_path = getattr(shared.sd_model.sd_checkpoint_info, 'filename', '') if hasattr(shared.sd_model, 'sd_checkpoint_info') else ''

        # Check 1: Model class name contains 'SDXL'
        if class_name and 'SDXL' in class_name:
            logger.debug(f"Detected SDXL model via class name: {class_name}")
            return True

        # Check 2: Full path or checkpoint name contains sdxl patterns
        sdxl_patterns = ['sdxl', 'sd_xl', 'sd-xl', 'stable-diffusion-xl']

        if full_path:
            full_path_lower = full_path.lower()
            if any(pattern in full_path_lower for pattern in sdxl_patterns):
                logger.debug(f"Detected SDXL model via path: {full_path}")
                return True

        if checkpoint_name:
            checkpoint_lower = checkpoint_name.lower()
            if any(pattern in checkpoint_lower for pattern in sdxl_patterns):
                logger.debug(f"Detected SDXL model via checkpoint name: {checkpoint_name}")
                return True

        return False

    except Exception as e:
        logger.debug(f"SDXL detection failed: {e}")
        return False


def is_zimage_model() -> bool:
    """Detect if Z-Image-Turbo is the currently loaded model.

    Uses caching to avoid repeated detection calls during rendering.

    Returns:
        True if Z-Image-Turbo is loaded, False otherwise
    """
    global _detection_cache

    try:
        shared = _get_shared_module()
        if shared is None or not hasattr(shared, 'sd_model'):
            logger.info("Z-Image detection: shared module or sd_model not available")
            return False

        model = shared.sd_model
        checkpoint_name = _get_checkpoint_name(shared)

        # Check cache first
        _invalidate_cache_if_model_changed(checkpoint_name)
        if _detection_cache['is_zimage'] is not None:
            return _detection_cache['is_zimage']

        # Get all detection data upfront for logging
        class_name = _get_model_class_name(model)
        full_path = getattr(shared.sd_model.sd_checkpoint_info, 'filename', '') if hasattr(shared.sd_model, 'sd_checkpoint_info') else ''

        logger.debug(f"Z-Image detection attempt:")
        logger.debug(f"  - Class name: {class_name}")
        logger.debug(f"  - Checkpoint name: {checkpoint_name}")
        logger.debug(f"  - Full path: {full_path}")

        # Check 1: Model class name is exactly 'ZImage' (most reliable!)
        if class_name and class_name == 'ZImage':
            logger.debug(f"✓ Detected Z-Image model via class name: {class_name}")
            _detection_cache['is_zimage'] = True
            return True

        # Check 2: Full path or checkpoint name contains z-image/z_image patterns
        z_image_patterns = ['z-image', 'zimage', 'z_image', 'zit', 'tongyi']

        if full_path:
            full_path_lower = full_path.lower()
            if any(pattern in full_path_lower for pattern in z_image_patterns):
                logger.debug(f"✓ Detected Z-Image model via path: {full_path}")
                _detection_cache['is_zimage'] = True
                return True

        if checkpoint_name:
            checkpoint_lower = checkpoint_name.lower()
            if any(pattern in checkpoint_lower for pattern in z_image_patterns):
                logger.debug(f"✓ Detected Z-Image model via checkpoint name: {checkpoint_name}")
                _detection_cache['is_zimage'] = True
                return True

        logger.debug("✗ Z-Image model not detected")
        _detection_cache['is_zimage'] = False
        return False

    except Exception as e:
        logger.warning(f"Z-Image detection failed with error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        _detection_cache['is_zimage'] = False
        return False


def is_flux2_klein_model() -> bool:
    """Detect if Flux 2 Klein is the currently loaded model.

    Returns:
        True if Flux 2 Klein is loaded, False otherwise
    """
    try:
        shared = _get_shared_module()
        if shared is None or not hasattr(shared, 'sd_model'):
            return False

        model = shared.sd_model

        # Check 1: Model class name contains 'Flux2'
        class_name = _get_model_class_name(model)
        if class_name and 'Flux2' in class_name:
            logger.debug(f"Detected Flux 2 Klein via class name: {class_name}")
            return True

        # Check 2: Checkpoint name contains 'klein' or 'flux-2' or 'flux2'
        checkpoint_name = _get_checkpoint_name(shared)
        if checkpoint_name:
            klein_patterns = ['klein', 'flux-2', 'flux2', 'flux_2']
            if any(pattern in checkpoint_name for pattern in klein_patterns):
                logger.debug(f"Detected Flux 2 Klein via checkpoint name: {checkpoint_name}")
                return True

        return False

    except Exception as e:
        logger.debug(f"Flux 2 Klein detection failed: {e}")
        return False


def _detect_flux_variant(checkpoint_name: Optional[str]) -> str:
    """Detect Flux variant (Klein, Dev, or Schnell) from checkpoint name.

    Args:
        checkpoint_name: Lowercase checkpoint filename

    Returns:
        "Flux 2 Klein", "Flux Schnell", "Flux Dev", or "Flux"
    """
    if checkpoint_name:
        # Check for Klein first (Flux 2)
        klein_patterns = ['klein', 'flux-2', 'flux2', 'flux_2']
        if any(pattern in checkpoint_name for pattern in klein_patterns):
            # Determine size variant
            if '4b' in checkpoint_name or '4-b' in checkpoint_name:
                return "Flux 2 Klein 4B"
            elif '9b' in checkpoint_name or '9-b' in checkpoint_name:
                return "Flux 2 Klein 9B"
            return "Flux 2 Klein"

        # Check for Schnell (Flux 1)
        if 'schnell' in checkpoint_name:
            return "Flux Schnell"

    return "Flux Dev" if checkpoint_name else "Flux"


def get_model_name() -> str:
    """Get friendly name of currently loaded model.

    Returns:
        Model name string ("Flux 2 Klein", "Lumina 2.0", "Flux Dev", "Flux Schnell", "SDXL", "Z-Image-Turbo", "Unknown")
    """
    if is_lumina_model():
        return LUMINA_CONFIG.name

    # Check Flux 2 Klein before general Flux
    if is_flux2_klein_model():
        shared = _get_shared_module()
        checkpoint_name = _get_checkpoint_name(shared) if shared else None
        return _detect_flux_variant(checkpoint_name)

    if is_flux_model():
        shared = _get_shared_module()
        checkpoint_name = _get_checkpoint_name(shared) if shared else None
        return _detect_flux_variant(checkpoint_name)

    if is_zimage_model():
        return "Z-Image-Turbo"

    if is_sdxl_model():
        return "SDXL"

    return DEFAULT_CONFIG.name


def get_recommended_cfg_scale() -> tuple[float, float]:
    """Get recommended CFG scale range for current model.

    Returns:
        Tuple of (min_cfg, max_cfg)
    """
    if is_lumina_model():
        return LUMINA_CONFIG.cfg_range
    if is_zimage_model():
        return ZIMAGE_CONFIG.cfg_range
    if is_flux2_klein_model():
        return FLUX2_KLEIN_CONFIG.cfg_range
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
    if is_zimage_model():
        return ZIMAGE_CONFIG.recommended_steps
    if is_flux2_klein_model():
        return FLUX2_KLEIN_CONFIG.recommended_steps
    if is_flux_model():
        return FLUX_CONFIG.recommended_steps
    return DEFAULT_CONFIG.recommended_steps


def validate_zimage_prompt(prompt: str) -> tuple[bool, str]:
    """Validate Z-Image prompt meets minimum requirements.

    Z-Image-Turbo requires long, detailed prompts (80-250 words) to work properly.
    Short prompts will be ignored or produce poor results.

    Args:
        prompt: Prompt text to validate

    Returns:
        Tuple of (is_valid, warning_message)
    """
    if not is_zimage_model():
        return True, ""

    word_count = len(prompt.split())

    if word_count < 20:
        return False, (
            f"⚠️ Z-Image prompt too short ({word_count} words). "
            f"Z-Image-Turbo requires detailed prompts (80-250 words) including: "
            f"camera angle, lighting, environment, style, and explicit constraints. "
            f"Short prompts will be ignored. "
            f"See: https://gist.github.com/illuminatianon/c42f8e57f1e3ebf037dd58043da9de32"
        )

    if word_count < 80:
        return True, (
            f"⚠️ Z-Image prompt short ({word_count} words). "
            f"Optimal range is 80-250 words for best results. "
            f"Consider adding: camera/cinematography details, lighting, environment, style."
        )

    return True, ""
