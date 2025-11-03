"""Noise generation and application for frame rendering.

This module contains impure noise functions that have side effects
(generator state manipulation) and external dependencies.
"""

import torch
from torch.nn.functional import interpolate
import numpy as np
import cv2
from PIL import Image

# Import pure functions from refactored utils module
from deforum.utils.generation.noise import (
    deforum_noise_gen,
    round_to_multiple,
    normalize_perlin,
    condition_noise_mask,
    rand_perlin_2d_octaves,
)

try:
    from modules.shared import opts
    DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False) if opts else False
except ImportError:
    DEBUG_MODE = False


def add_noise(
    sample: np.ndarray,
    noise_amt: float,
    seed: int,
    noise_type: str,
    noise_args: tuple,
    noise_mask: Image.Image | None = None,
    invert_mask: bool = False
) -> np.ndarray:
    """Add noise to image sample (white or Perlin) with optional mask.
    
    Args:
        sample: Input image as numpy array
        noise_amt: Amount of noise to add (0.0-1.0)
        seed: Random seed for noise generation
        noise_type: Type of noise ('white' or 'perlin')
        noise_args: Tuple of noise parameters (octaves, persistence, etc.)
        noise_mask: Optional PIL Image mask to constrain noise application
        invert_mask: Whether to invert the noise mask
        
    Returns:
        Image with noise added as numpy array
        
    Note:
        This is an impure function with side effects (generator state).
    """
    try:
        from deforum.animation.animation import sample_to_cv2
    except ImportError:
        from animation import sample_to_cv2

    deforum_noise_gen.manual_seed(seed)

    perlin_w = round_to_multiple(sample.shape[0], 64)
    perlin_h = round_to_multiple(sample.shape[1], 64)
    sample2dshape = (perlin_w, perlin_h)

    noise = torch.randn((sample.shape[2], perlin_w, perlin_h), generator=deforum_noise_gen)

    if noise_type == 'perlin':
        perlin = rand_perlin_2d_octaves(
            sample2dshape,
            (int(noise_args[0]), int(noise_args[1])),
            octaves=noise_args[2],
            persistence=noise_args[3]
        )
        noise = noise * normalize_perlin(perlin)

    # Always interpolate noise to match sample dimensions
    # This fixes size mismatch errors in cv2.addWeighted
    if noise.shape[1] != sample.shape[0] or noise.shape[2] != sample.shape[1]:
        noise = interpolate(
            noise.unsqueeze(1),
            size=(sample.shape[0], sample.shape[1])
        ).squeeze(1)

    if noise_mask is not None:
        mask = condition_noise_mask(noise_mask, invert_mask)
        noise_to_add = sample_to_cv2(noise * mask)
    else:
        noise_to_add = sample_to_cv2(noise)

    return cv2.addWeighted(sample, 1 - noise_amt, noise_to_add, noise_amt, 0)


__all__ = ['add_noise']
