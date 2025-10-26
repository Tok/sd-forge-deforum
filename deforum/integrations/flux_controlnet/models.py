"""Flux ControlNet model loading and management for Deforum.

Handles loading and caching of Flux ControlNet models from HuggingFace.
Supports Canny and Depth ControlNet models.
"""

import torch
from diffusers import FluxControlNetModel
from typing import Optional, Dict
import os
from contextlib import contextmanager
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()



@contextmanager
def temporarily_unpatch_hf_download():
    """Temporarily restore original HuggingFace download to avoid etag parameter conflict.

    Forge patches huggingface_hub download functions with a signature that doesn't
    support the 'etag' parameter that newer diffusers/transformers uses.
    This context manager temporarily restores the original during model loading.
    """
    patched_fn = None
    restored = False

    try:
        from huggingface_hub import file_download

        # Save current (patched) function
        patched_fn = file_download._download_to_tmp_and_move

        # Get the original function from the closure
        # Forge's patch wraps the original: original_download_to_tmp_and_move
        if hasattr(patched_fn, '__code__') and patched_fn.__code__.co_freevars:
            # Try to extract original from closure
            for cell in patched_fn.__closure__ or []:
                try:
                    obj = cell.cell_contents
                    if callable(obj) and obj != patched_fn:
                        # Found the original function
                        file_download._download_to_tmp_and_move = obj
                        restored = True
                        logger.info("  Temporarily using original HF download (avoiding Forge patch)")
                        break
                except (ValueError, AttributeError):
                    continue

        if not restored:
            logger.info("  Using patched HF download (couldn't restore original, may fail)")

    except Exception as e:
        logger.info(f"  Warning during HF download unpatch setup: {e}")

    try:
        yield
    finally:
        # Always restore the patch if we changed it
        if restored and patched_fn is not None:
            try:
                from huggingface_hub import file_download
                file_download._download_to_tmp_and_move = patched_fn
                logger.info("  Restored Forge HF download patch")
            except Exception as e:
                logger.error(f"  Could not restore HF patch: {e}")


# Available Flux ControlNet models
FLUX_CONTROLNET_MODELS = {
    "canny": {
        "instantx": "InstantX/FLUX.1-dev-Controlnet-Canny",
        "xlabs": "XLabs-AI/flux-controlnet-canny-diffusers",
        "bfl": "black-forest-labs/FLUX.1-Canny-dev",
    },
    "depth": {
        "shakker": "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
        "instantx": "InstantX/FLUX.1-dev-Controlnet-Depth",
        "xlabs": "XLabs-AI/flux-controlnet-depth-diffusers",
        "bfl": "black-forest-labs/FLUX.1-Depth-dev",
    }
}

# Model cache (only for ControlNet models, not full pipelines)
_model_cache: Dict[str, FluxControlNetModel] = {}


def get_available_models(control_type: str) -> Dict[str, str]:
    """Get available models for a control type.

    Args:
        control_type: "canny" or "depth"

    Returns:
        Dictionary of model name -> HuggingFace repo ID
    """
    return FLUX_CONTROLNET_MODELS.get(control_type, {})


def load_flux_controlnet_model(
    control_type: str,
    model_name: str = "instantx",
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda"
) -> FluxControlNetModel:
    """Load a Flux ControlNet model.

    Args:
        control_type: "canny" or "depth"
        model_name: Model provider name (e.g., "instantx", "xlabs", "bfl")
        torch_dtype: Torch data type for model weights
        device: Device to load model on

    Returns:
        Loaded FluxControlNetModel

    Raises:
        ValueError: If control_type or model_name is invalid
    """
    # Check if model is already cached
    cache_key = f"{control_type}_{model_name}"
    if cache_key in _model_cache:
        logger.info(f"Using cached Flux {control_type.title()} ControlNet model: {model_name}")
        return _model_cache[cache_key]

    # Get model repo ID
    if control_type not in FLUX_CONTROLNET_MODELS:
        raise ValueError(f"Invalid control type: {control_type}. Use 'canny' or 'depth'.")

    models = FLUX_CONTROLNET_MODELS[control_type]
    if model_name not in models:
        raise ValueError(f"Invalid model name '{model_name}' for {control_type}. "
                        f"Available: {list(models.keys())}")

    model_id = models[model_name]

    logger.info(f"Loading Flux {control_type.title()} ControlNet model: {model_id}")
    logger.info(f"This may take a while on first load (downloading from HuggingFace)...")

    # Load model with temporarily unpatched HF download to avoid etag parameter conflict
    try:
        with temporarily_unpatch_hf_download():
            controlnet = FluxControlNetModel.from_pretrained(
                model_id,
                torch_dtype=torch_dtype
            )

        # Cache the model
        _model_cache[cache_key] = controlnet

        logger.info(f"✓ Flux {control_type.title()} ControlNet model loaded successfully")
        return controlnet

    except Exception as e:
        logger.info(f"Error loading Flux ControlNet model {model_id}: {e}")
        raise


def unload_controlnet_models():
    """Clear cached ControlNet models to free VRAM.

    Note: This only clears ControlNet models (~3.6GB each), not full pipelines.
    We use Forge's already-loaded Flux model, so no pipeline caching needed.
    """
    global _model_cache

    _model_cache.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Flux ControlNet models unloaded from cache")


def get_model_info(control_type: str, model_name: str) -> str:
    """Get human-readable info about a model.

    Args:
        control_type: "canny" or "depth"
        model_name: Model provider name

    Returns:
        Model information string
    """
    models = FLUX_CONTROLNET_MODELS.get(control_type, {})
    repo_id = models.get(model_name, "Unknown")

    provider_names = {
        "instantx": "InstantX",
        "xlabs": "XLabs-AI",
        "bfl": "Black Forest Labs (Official)",
        "shakker": "Shakker Labs"
    }

    provider = provider_names.get(model_name, model_name)

    return f"{control_type.title()} ControlNet by {provider} ({repo_id})"
