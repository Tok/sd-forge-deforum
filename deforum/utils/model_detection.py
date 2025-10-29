"""Model Detection Utilities

Detect which diffusion model is currently loaded in Forge.
Used for model-specific tuning and compatibility handling.
"""

from typing import Optional
from deforum.utils.system.logging import get_logger

logger = get_logger()


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
        import modules.shared as shared

        if not hasattr(shared, 'sd_model'):
            return False

        model = shared.sd_model

        # Check 1: Model class name
        if hasattr(model, '__class__'):
            class_name = model.__class__.__name__
            if 'Lumina' in class_name:
                logger.debug(f"Detected Lumina model via class name: {class_name}")
                return True

        # Check 2: Check diffusion engine type
        if hasattr(model, 'forge_objects'):
            unet = model.forge_objects.unet
            if hasattr(unet, 'model') and hasattr(unet.model, 'diffusion_model'):
                diff_model = unet.model.diffusion_model
                if diff_model.__class__.__name__ == 'Lumina2NextDiT':
                    logger.debug("Detected Lumina model via diffusion_model class")
                    return True

        # Check 3: Checkpoint filename contains 'lumina'
        if hasattr(shared, 'opts') and hasattr(shared.opts, 'sd_model_checkpoint'):
            checkpoint_name = shared.opts.sd_model_checkpoint.lower()
            if 'lumina' in checkpoint_name or 'neta' in checkpoint_name:
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
        import modules.shared as shared

        if not hasattr(shared, 'sd_model'):
            return False

        model = shared.sd_model

        # Check model class name
        if hasattr(model, '__class__'):
            class_name = model.__class__.__name__
            if 'Flux' in class_name:
                return True

        # Check checkpoint name
        if hasattr(shared, 'opts') and hasattr(shared.opts, 'sd_model_checkpoint'):
            checkpoint_name = shared.opts.sd_model_checkpoint.lower()
            if 'flux' in checkpoint_name:
                return True

        return False

    except Exception as e:
        logger.debug(f"Flux detection failed: {e}")
        return False


def get_model_name() -> str:
    """Get friendly name of currently loaded model.

    Returns:
        Model name string ("Lumina 2.0", "Flux Dev", "Flux Schnell", "SDXL", "Unknown")
    """
    if is_lumina_model():
        return "Lumina 2.0"
    elif is_flux_model():
        # Try to detect Dev vs Schnell
        try:
            import modules.shared as shared
            if hasattr(shared, 'opts') and hasattr(shared.opts, 'sd_model_checkpoint'):
                checkpoint_name = shared.opts.sd_model_checkpoint.lower()
                if 'schnell' in checkpoint_name:
                    return "Flux Schnell"
                else:
                    return "Flux Dev"
        except:
            pass
        return "Flux"
    else:
        return "Unknown"


def get_recommended_cfg_scale() -> tuple[float, float]:
    """Get recommended CFG scale range for current model.

    Returns:
        Tuple of (min_cfg, max_cfg)
    """
    if is_lumina_model():
        return (4.0, 5.5)  # Lumina official recommendation
    elif is_flux_model():
        return (1.0, 3.5)  # Flux typical range
    else:
        return (7.0, 12.0)  # SDXL/SD15 typical range


def get_recommended_steps() -> int:
    """Get recommended step count for current model.

    Returns:
        Recommended number of sampling steps
    """
    if is_lumina_model():
        return 30  # Lumina official recommendation
    elif is_flux_model():
        return 20  # Flux typical
    else:
        return 20  # SDXL/SD15 typical
