"""
Monkey patch for Forge's img2img step calculation to enable fractional precision.

This patches modules.sd_samplers_common.setup_img2img_steps() to support
fractional t_enc values instead of integer rounding, giving true 1% strength precision.
"""

import torch
from modules import sd_samplers_common, shared


# Store original function for fallback
_original_setup_img2img_steps = None


def fractional_setup_img2img_steps(p, steps=None):
    """
    Patched version of setup_img2img_steps with fractional t_enc support.

    Original behavior (discrete):
        t_enc = int(denoising_strength * steps)  # Rounds down to integer

    Fractional behavior (smooth):
        t_enc = denoising_strength * steps       # Keeps fractional value

    Args:
        p: Processing object with denoising_strength
        steps: Optional step count override

    Returns:
        Tuple of (steps, t_enc) where t_enc can be fractional
    """
    from deforum.utils.system.logging import get_logger, emoji as emoji_utils
    logger = get_logger()
    magnifying_glass = emoji_utils.magnifying_glass()

    opts = shared.opts

    if opts.img2img_fix_steps or steps is not None:
        requested_steps = steps or p.steps
        steps = int(requested_steps / min(p.denoising_strength, 0.999)) if p.denoising_strength > 0 else 0
        t_enc = requested_steps - 1
        logger.debug(f"{magnifying_glass} fractional_setup (fix_steps): steps={steps}, t_enc={t_enc}")
    else:
        steps = p.steps
        # FRACTIONAL CHANGE: Remove int() to keep fractional precision
        t_enc = min(p.denoising_strength, 0.999) * steps
        discrete = int(t_enc)
        logger.debug(f"{magnifying_glass} FRACTIONAL SETUP: denoising={p.denoising_strength:.4f}, steps={steps}")
        logger.debug(f"   t_enc = {t_enc:.4f} (discrete would be {discrete})")

    return steps, t_enc


def apply_fractional_img2img_patch():
    """
    Apply monkey patch to enable fractional t_enc in img2img.

    This replaces modules.sd_samplers_common.setup_img2img_steps with our
    fractional version. The patch is applied once at extension load time.

    Returns:
        bool: True if patch applied successfully
    """
    global _original_setup_img2img_steps

    try:
        # Store original function if not already stored
        if _original_setup_img2img_steps is None:
            _original_setup_img2img_steps = sd_samplers_common.setup_img2img_steps

        # Replace with fractional version
        sd_samplers_common.setup_img2img_steps = fractional_setup_img2img_steps

        from deforum.utils.system.logging import get_logger
        logger = get_logger()
        logger.info("Applied fractional img2img patch - 1% strength precision enabled")

        return True

    except Exception as e:
        from deforum.utils.system.logging import get_logger
        logger = get_logger()
        logger.error(f"Failed to apply fractional img2img patch: {e}")
        return False


def remove_fractional_img2img_patch():
    """
    Remove monkey patch and restore original Forge behavior.

    This is useful for debugging or if the patch causes issues.

    Returns:
        bool: True if patch removed successfully
    """
    global _original_setup_img2img_steps

    try:
        if _original_setup_img2img_steps is not None:
            sd_samplers_common.setup_img2img_steps = _original_setup_img2img_steps

            from deforum.utils.system.logging import get_logger
            logger = get_logger()
            logger.info("Removed fractional img2img patch - reverted to discrete steps")

        return True

    except Exception as e:
        from deforum.utils.system.logging import get_logger
        logger = get_logger()
        logger.error(f"Failed to remove fractional img2img patch: {e}")
        return False


def fractional_sigma_slice(sigmas: torch.Tensor, steps: int, t_enc: float) -> torch.Tensor:
    """
    Slice sigma schedule using fractional t_enc value.

    Original (discrete):
        sigma_sched = sigmas[steps - int(t_enc) - 1 :]

    Fractional (interpolated):
        If t_enc=4.3, interpolate between sigmas[steps-5] and sigmas[steps-4]

    Args:
        sigmas: Full sigma schedule tensor
        steps: Total step count
        t_enc: Fractional denoising step count

    Returns:
        Sliced sigma schedule starting at fractional position
    """
    # Calculate fractional index
    frac_idx = steps - t_enc - 1

    # If t_enc is already integer, use normal slicing
    if frac_idx == int(frac_idx):
        return sigmas[int(frac_idx):]

    # Fractional case: interpolate starting sigma
    low_idx = int(torch.floor(torch.tensor(frac_idx)))
    high_idx = int(torch.ceil(torch.tensor(frac_idx)))
    weight = frac_idx - low_idx

    # Interpolate first sigma
    if high_idx < len(sigmas):
        first_sigma = (1 - weight) * sigmas[low_idx] + weight * sigmas[high_idx]
    else:
        first_sigma = sigmas[low_idx]

    # Concatenate interpolated first sigma with rest of schedule
    remaining_sigmas = sigmas[high_idx:]
    result = torch.cat([first_sigma.unsqueeze(0), remaining_sigmas])

    return result
