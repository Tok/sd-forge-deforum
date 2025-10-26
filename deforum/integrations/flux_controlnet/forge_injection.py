"""Flux ControlNet V2 - Global Control Sample Storage

Stores pre-computed Flux ControlNet control samples globally so they can be
accessed by the patched KModel.apply_model during sampling.
"""

import torch
from typing import Optional, Tuple
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


# Global storage for control samples (accessed by both generation code and patches)
_current_controlnet_samples: Optional[Tuple] = None


def store_control_samples(
    controlnet_block_samples: Tuple[torch.Tensor, ...],
    controlnet_single_block_samples: Tuple[torch.Tensor, ...]
):
    """Store control samples globally for the current generation.

    These will be picked up by the patched KModel.apply_model during sampling.

    Args:
        controlnet_block_samples: Control samples for double_blocks
        controlnet_single_block_samples: Control samples for single_blocks
    """
    global _current_controlnet_samples
    _current_controlnet_samples = (controlnet_block_samples, controlnet_single_block_samples)
    logger.info(f"🌐 Stored Flux ControlNet samples for generation")
    logger.info(f"   Block samples: {len(controlnet_block_samples) if controlnet_block_samples is not None else 0} tensors")
    logger.info(f"   Single block samples: {len(controlnet_single_block_samples) if controlnet_single_block_samples is not None else 0} tensors")


def get_stored_control_samples() -> Optional[Tuple]:
    """Get currently stored control samples.

    Called by patched KModel.apply_model to retrieve control samples.

    Returns:
        Tuple of (controlnet_block_samples, controlnet_single_block_samples) or None
    """
    global _current_controlnet_samples
    return _current_controlnet_samples


def clear_control_samples():
    """Clear stored control samples after generation."""
    global _current_controlnet_samples
    _current_controlnet_samples = None
