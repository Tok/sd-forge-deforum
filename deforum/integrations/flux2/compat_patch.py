"""Flux 2 Compatibility Patch

Ensures all Flux 2-specific parameters are correctly set before model instantiation.

The issues:
1. Flux 2 GGUF models may not have "vector_in.in_layer.weight" key → missing vec_in_dim
2. Flux 2 uses different architecture: in_channels=64, patch_size=1 vs Flux 1's 16/2
3. Matrix shape mismatch during sampling if wrong parameters used

Root cause: GGUF quantized models have different state dict keys, and Forge's
detection code doesn't distinguish between Flux 1 and Flux 2 architectures.

Solution: Monkey patch the model __init__ to:
1. Detect Flux 2 vs Flux 1 by transformer block counts (depth/depth_single_blocks)
2. Add missing vec_in_dim parameter (768 for both)
3. Adjust in_channels and patch_size for Flux 2

Technical details:
Flux 1 Dev:
- depth=19, depth_single_blocks=38
- in_channels=16, patch_size=2
- vec_in_dim=768

Flux 2 Dev:
- depth=8, depth_single_blocks=48
- in_channels=64, patch_size=1
- vec_in_dim=768

Sources:
- https://huggingface.co/black-forest-labs/FLUX.2-dev
- https://huggingface.co/blog/flux-2
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


def is_flux2_model(depth: int, depth_single_blocks: int) -> bool:
    """Detect if this is a Flux 2 model based on transformer block counts.

    Args:
        depth: Number of double-stream blocks
        depth_single_blocks: Number of single-stream blocks

    Returns:
        True if Flux 2, False if Flux 1
    """
    # Flux 2: 8 double + 48 single blocks
    # Flux 1: 19 double + 38 single blocks
    if depth == 8 and depth_single_blocks == 48:
        return True
    return False


def apply_flux2_loader_patch():
    """Monkey patch Forge's model loader to add Flux 2 compatibility.

    This patches the model __init__ to:
    1. Detect Flux 2 vs Flux 1 by transformer block counts
    2. Add missing vec_in_dim parameter (768 for both)
    3. Adjust in_channels and patch_size for Flux 2
    """
    global _flux2_patch_applied

    if _flux2_patch_applied:
        return

    try:
        from backend.nn.flux import IntegratedFluxTransformer2DModel

        # Store original __init__
        original_init = IntegratedFluxTransformer2DModel.__init__

        def patched_init(self, *args, **kwargs):
            """Patched __init__ that detects Flux 2 and adjusts parameters."""
            # Forge loader passes config as kwargs via: IntegratedFluxTransformer2DModel(**c)
            # So we extract depth from kwargs to detect Flux version

            # Get depth parameters to detect Flux version
            depth = kwargs.get('depth', 19)  # Default to Flux 1
            depth_single_blocks = kwargs.get('depth_single_blocks', 38)

            # Detect Flux 2
            is_flux2 = is_flux2_model(depth, depth_single_blocks)

            # Add missing vec_in_dim (required for both Flux 1 and 2)
            if 'vec_in_dim' not in kwargs:
                kwargs['vec_in_dim'] = 768
                logger.info(f"Added default vec_in_dim=768 for Flux model")

            # Adjust in_channels and patch_size for Flux 2
            if is_flux2:
                logger.info("🔍 Flux 2 model detected (depth=8, depth_single_blocks=48)")

                # Adjust in_channels
                current_in_channels = kwargs.get('in_channels', 16)
                if current_in_channels != 64:
                    kwargs['in_channels'] = 64
                    logger.info(f"  → Corrected in_channels: {current_in_channels} → 64 for Flux 2")

                # Adjust patch_size
                current_patch_size = kwargs.get('patch_size', 2)
                if current_patch_size != 1:
                    kwargs['patch_size'] = 1
                    logger.info(f"  → Corrected patch_size: {current_patch_size} → 1 for Flux 2")
            else:
                logger.debug(f"Flux 1 model detected (depth={depth}, depth_single_blocks={depth_single_blocks})")

            # Call original init with patched kwargs
            return original_init(self, *args, **kwargs)

        # Apply the monkey patch
        IntegratedFluxTransformer2DModel.__init__ = patched_init

        logger.info("✓ Flux 2 compatibility patch applied - auto-detection enabled")
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
