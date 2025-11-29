"""Flux 2 Compatibility Patch

Ensures vec_in_dim parameter is present in the model config before instantiation.

The issue: Flux 2 GGUF models may not have the "vector_in.in_layer.weight" key in
the state dict, which means vec_in_dim is not detected automatically by Forge's
detection.py. This causes a TypeError when trying to instantiate the model because
vec_in_dim is a required positional argument.

Root cause: GGUF quantized models have different state dict keys, and Forge's
detection code doesn't have a fallback for missing vec_in_dim.

Solution: Monkey patch the model loader to add vec_in_dim with default value (768)
if it's missing from the config. This is applied before model instantiation.

Technical details:
- Flux 1 Dev: vec_in_dim = 768 (pooled_projection_dim)
- Flux 2 Dev: vec_in_dim = 768 (same as Flux 1)
- Source: https://huggingface.co/black-forest-labs/FLUX.2-dev
"""

from typing import Any, Dict
from deforum.utils.system.logging import get_logger

logger = get_logger()

_flux2_patch_applied = False


def patch_flux_config_with_vec_in_dim(config: Dict[str, Any]) -> Dict[str, Any]:
    """Add vec_in_dim to Flux config if missing.

    Args:
        config: Model configuration dictionary

    Returns:
        Modified config with vec_in_dim added if it was missing
    """
    if "image_model" in config and config["image_model"] == "flux":
        if "vec_in_dim" not in config:
            # Default vec_in_dim for Flux models (both 1.0 and 2.0)
            config["vec_in_dim"] = 768
            logger.info(f"Flux model config missing vec_in_dim - added default value: 768")
            return config

    return config


def apply_flux2_loader_patch():
    """Monkey patch Forge's model loader to add vec_in_dim for Flux 2 models.

    This patches the loader's config processing to ensure vec_in_dim is present
    before attempting to instantiate IntegratedFluxTransformer2DModel.
    """
    global _flux2_patch_applied

    if _flux2_patch_applied:
        return

    try:
        # Import Forge's loader module
        import backend.loader as loader_module

        # Store original forge_loader function
        original_forge_loader = loader_module.forge_loader

        def patched_forge_loader(state_dict, *args, **kwargs):
            """Patched forge_loader that adds vec_in_dim if missing."""
            # Call original loader first to get the model
            # But we need to patch BEFORE the model is instantiated
            # So we need to patch at a different level

            # Actually, we need to patch the config BEFORE it's used to create the model
            # Let's try a different approach - patch the detection result
            result = original_forge_loader(state_dict, *args, **kwargs)
            return result

        # That approach won't work because the model is already instantiated
        # Let's try patching the model class instead
        from backend.nn.flux import IntegratedFluxTransformer2DModel

        # Store original __init__
        original_init = IntegratedFluxTransformer2DModel.__init__

        def patched_init(self, *args, **kwargs):
            """Patched __init__ that adds default vec_in_dim if missing."""
            # If vec_in_dim is not in kwargs and not in args (position 2), add it
            if 'vec_in_dim' not in kwargs:
                # Check if it's in positional args (vec_in_dim is the 3rd parameter)
                if len(args) < 3:
                    # Add default vec_in_dim
                    kwargs['vec_in_dim'] = 768
                    logger.info("Added default vec_in_dim=768 for Flux model (GGUF/Flux 2 compatibility)")

            # Call original init with patched kwargs
            return original_init(self, *args, **kwargs)

        # Apply the monkey patch
        IntegratedFluxTransformer2DModel.__init__ = patched_init

        logger.info("✓ Flux 2 compatibility patch applied - vec_in_dim fallback enabled")
        _flux2_patch_applied = True

    except Exception as e:
        logger.warning(f"Failed to apply Flux 2 compatibility patch: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def ensure_flux2_compatibility():
    """Ensure Flux 2 compatibility patches are applied.

    This should be called early during Deforum initialization to ensure
    the patches are in place before any Flux models are loaded.
    """
    if not _flux2_patch_applied:
        apply_flux2_loader_patch()
