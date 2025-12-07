"""DA3-3DGS quality optimization helpers."""

import torch
from typing import Tuple
from deforum.utils.system.logging import get_logger

logger = get_logger()


# VRAM requirements per densification factor (empirically measured)
# Format: {factor: (min_vram_gb, recommended_vram_gb)}
DENSIFICATION_VRAM_REQUIREMENTS = {
    1: (1.5, 2.0),   # 705k splats - minimal quality
    2: (2.5, 3.0),   # 1.4M splats - good quality
    3: (3.5, 4.0),   # 2.1M splats - high quality
    4: (4.5, 5.0),   # 2.8M splats - very high quality
    5: (6.0, 7.0),   # 3.5M splats - excellent quality
    6: (8.0, 9.0),   # 4.2M splats - ultra quality
    7: (10.0, 11.0), # 4.9M splats - extreme quality
    8: (13.0, 14.0), # 5.6M splats - maximum quality
}


def get_available_vram_gb() -> float:
    """Get available VRAM in GB.

    Returns:
        Available VRAM in gigabytes, or 0.0 if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024 ** 3)
        return free_gb
    except Exception as e:
        logger.warning(f"Failed to get VRAM info: {e}")
        return 0.0


def get_optimal_densification_factor(
    available_vram_gb: float = None,
    safety_margin_gb: float = 2.0
) -> Tuple[int, str]:
    """Determine optimal densification factor based on available VRAM.

    Args:
        available_vram_gb: Available VRAM in GB (auto-detected if None)
        safety_margin_gb: Reserve this much VRAM for other operations

    Returns:
        Tuple of (optimal_factor, reason_message)

    Strategy:
        - Reserve safety_margin_gb for model, scene building, and overhead
        - Select highest densification factor that fits in remaining VRAM
        - Prefer recommended_vram over min_vram for better stability
    """
    if available_vram_gb is None:
        available_vram_gb = get_available_vram_gb()

    if available_vram_gb == 0.0:
        logger.warning("No CUDA device found, defaulting to densification factor 1")
        return 1, "No CUDA device (CPU fallback)"

    # Calculate usable VRAM (total - safety margin)
    usable_vram = max(0.0, available_vram_gb - safety_margin_gb)

    # Find highest factor that fits
    optimal_factor = 1
    reason = None  # Initialize to avoid UnboundLocalError

    for factor in sorted(DENSIFICATION_VRAM_REQUIREMENTS.keys(), reverse=True):
        min_req, recommended_req = DENSIFICATION_VRAM_REQUIREMENTS[factor]

        # Prefer recommended requirement, fall back to minimum if needed
        if usable_vram >= recommended_req:
            optimal_factor = factor
            reason = (
                f"Auto-selected factor {factor} ({recommended_req:.1f}GB recommended, "
                f"{usable_vram:.1f}GB available after {safety_margin_gb}GB safety margin)"
            )
            break
        elif usable_vram >= min_req and factor > optimal_factor:
            # Can fit but tight - only use if better than current
            optimal_factor = factor
            reason = (
                f"Auto-selected factor {factor} ({min_req:.1f}GB minimum, "
                f"{usable_vram:.1f}GB available, slightly tight)"
            )

    # Build reason message if not already set
    if reason is None:
        if optimal_factor == 1 and usable_vram < DENSIFICATION_VRAM_REQUIREMENTS[1][0]:
            reason = (
                f"Low VRAM: {usable_vram:.1f}GB available after safety margin, "
                f"using minimum factor 1 ({DENSIFICATION_VRAM_REQUIREMENTS[1][0]:.1f}GB required)"
            )
        elif optimal_factor not in DENSIFICATION_VRAM_REQUIREMENTS:
            # Fallback to factor 3 if something went wrong
            optimal_factor = 3
            reason = f"Fallback to default factor 3 ({usable_vram:.1f}GB available)"
        else:
            # Fallback message if somehow reason is still None
            reason = f"Selected factor {optimal_factor} ({usable_vram:.1f}GB available)"

    logger.info(f"VRAM-based quality selection: {reason}")
    return optimal_factor, reason


def parse_densification_factor(factor_input: str | int) -> int:
    """Parse densification factor from UI input.

    Args:
        factor_input: Either "Auto (Max Quality for VRAM)" or numeric string/int

    Returns:
        Integer densification factor (1-8)
    """
    # Handle Auto mode
    if isinstance(factor_input, str) and "Auto" in factor_input:
        optimal_factor, reason = get_optimal_densification_factor()
        logger.info(f"   AUTO MODE: {reason}")
        return optimal_factor

    # Handle manual numeric input
    try:
        factor = int(factor_input)
        if factor < 1 or factor > 8:
            logger.warning(f"Densification factor {factor} out of range (1-8), clamping")
            factor = max(1, min(8, factor))
        return factor
    except (ValueError, TypeError):
        logger.warning(f"Invalid densification factor '{factor_input}', defaulting to 3")
        return 3


def log_vram_usage_estimate(densification_factor: int, resolution: Tuple[int, int] = (1024, 1024)):
    """Log estimated VRAM usage for given settings.

    Args:
        densification_factor: Gaussian densification factor (1-8)
        resolution: Image resolution (width, height)
    """
    if densification_factor not in DENSIFICATION_VRAM_REQUIREMENTS:
        logger.warning(f"Unknown densification factor {densification_factor}")
        return

    min_vram, rec_vram = DENSIFICATION_VRAM_REQUIREMENTS[densification_factor]
    splat_count = 705000 * densification_factor  # Base 705k splats

    # Resolution multiplier (rough estimate)
    base_resolution = 1024 * 1024
    current_resolution = resolution[0] * resolution[1]
    resolution_mult = current_resolution / base_resolution

    estimated_min = min_vram * resolution_mult
    estimated_rec = rec_vram * resolution_mult

    logger.info(f"   Estimated VRAM usage:")
    logger.info(f"      Densification factor: {densification_factor}")
    logger.info(f"      Gaussian splats: {splat_count:,}")
    logger.info(f"      Resolution: {resolution[0]}x{resolution[1]}")
    logger.info(f"      Minimum VRAM: {estimated_min:.1f}GB")
    logger.info(f"      Recommended VRAM: {estimated_rec:.1f}GB")

    # Check current availability
    available = get_available_vram_gb()
    if available > 0:
        logger.info(f"      Available VRAM: {available:.1f}GB")
        if available < estimated_min:
            logger.warning(f"      ⚠️  VRAM may be insufficient! ({available:.1f}GB < {estimated_min:.1f}GB minimum)")
        elif available < estimated_rec:
            logger.warning(f"      ⚠️  VRAM tight, may cause instability ({available:.1f}GB < {estimated_rec:.1f}GB recommended)")
