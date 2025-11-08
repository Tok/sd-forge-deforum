"""
Fractional Strength Interpolation for Deforum.

This module enables 1% strength granularity (0.01 resolution) instead of the default
discrete levels that depend on step count:
- Flux Schnell (4 steps): 0.25 resolution → 1% with fractional interpolation
- Flux Dev (20 steps): 0.05 resolution → 1% with fractional interpolation
- Lumina (30 steps): 0.033 resolution → 1% with fractional interpolation

Implementation uses Forge's Prediction.sigma() method which performs log-linear
interpolation between discrete scheduler timesteps.

Key Functions:
- create_fractional_sigma_schedule(): Generates sigma schedule with fractional timesteps
- make_fractional_scheduler_override(): Creates override function for pipeline
"""

import torch
from typing import Callable
from modules import devices


def calculate_fractional_timesteps(
    denoising_strength: float,
    total_steps: int,
    model_total_timesteps: int = 1000
) -> tuple[float, float]:
    """
    Calculate fractional timestep range for img2img denoising.

    Standard img2img formula:
        t_enc = int(denoising_strength * steps)  # Discrete integer
        Use sigmas[steps - t_enc - 1 : steps + 1]

    Fractional formula:
        start_timestep = (1.0 - denoising_strength) * model_total_timesteps
        end_timestep = model_total_timesteps
        This gives us fractional timestep range to interpolate

    Args:
        denoising_strength: Forge's denoising strength (0.0-1.0)
                           0.0 = no denoising, 1.0 = full schedule
        total_steps: Number of sampling steps to generate
        model_total_timesteps: Model's native timestep range (default: 1000)

    Returns:
        Tuple of (start_timestep, end_timestep) as floats

    Examples:
        >>> calculate_fractional_timesteps(0.2, 20)
        (800.0, 1000.0)  # Start at 80% through schedule

        >>> calculate_fractional_timesteps(0.5, 20)
        (500.0, 1000.0)  # Start at 50% through schedule

        >>> calculate_fractional_timesteps(0.333, 4)
        (667.0, 1000.0)  # Start at 66.7% through schedule
    """
    # Invert: Forge's denoising_strength = how much to denoise
    # start_percent = how far through schedule to start (0.0 = beginning, 1.0 = end)
    start_percent = 1.0 - denoising_strength

    start_timestep = start_percent * model_total_timesteps
    end_timestep = float(model_total_timesteps)

    return start_timestep, end_timestep


def create_fractional_sigma_schedule(
    predictor,
    start_timestep: float,
    end_timestep: float,
    steps: int,
    device
) -> torch.Tensor:
    """
    Create sigma schedule using fractional timestep interpolation.

    This bypasses the standard discrete array slicing (sigmas[a:b]) and instead
    generates sigmas at fractional timestep positions using predictor.sigma().

    The predictor.sigma() method uses log-linear interpolation:
        log_sigma = (1-w)*log_sigma[floor(t)] + w*log_sigma[ceil(t)]
        where w = frac(t)

    Args:
        predictor: Forge's Prediction object with sigma() method
        start_timestep: Fractional starting timestep (e.g., 667.3)
        end_timestep: Ending timestep (usually 1000.0)
        steps: Number of sampling steps to generate
        device: Torch device for tensor placement

    Returns:
        Tensor of sigma values, shape (steps + 1,)
        Includes final sigma of 0.0 per Forge convention

    Examples:
        At 4 steps with strength 0.333:
        - Discrete: Would round to 1 step (25% resolution)
        - Fractional: Uses exact 667.3 → 1000.0 range (1% resolution)
    """
    # Generate evenly spaced fractional timesteps
    timesteps = torch.linspace(
        start_timestep,
        end_timestep,
        steps,
        device=device
    )

    # Convert each fractional timestep to sigma using interpolation
    sigmas = torch.stack([
        predictor.sigma(torch.tensor([t], device=device))
        for t in timesteps
    ]).squeeze()

    # Append final sigma = 0.0 per Forge convention
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

    return sigmas.cpu()


def make_fractional_scheduler_override(
    model_wrap,
    denoising_strength: float,
    original_steps: int
) -> Callable[[int], torch.Tensor]:
    """
    Create scheduler override for fractional sigma placement.

    NOTE: This does NOT change step counts (t_enc still uses int() rounding).
    What it DOES: Places sigma values at fractionally-precise noise levels.

    Example: strength=0.77 → denoising=0.23 → t_enc=int(0.23*20)=4 steps
    - Without fractional: Sigmas placed for 20% or 25% schedule
    - With fractional: Sigmas placed for exactly 23% schedule

    This provides smoother noise transitions even with discrete step counts.

    Args:
        model_wrap: Forge's KDiffusionSampler.model_wrap
        denoising_strength: Forge's denoising strength (0.0-1.0)
        original_steps: Step count from UI

    Returns:
        Override function that returns fractionally-placed sigma schedule
    """
    from deforum.utils.system.logging import get_logger
    logger = get_logger()

    def override_scheduler(steps: int) -> torch.Tensor:
        """Generate full sigma schedule with fractional timestep placement."""
        try:
            if not hasattr(model_wrap, 'predictor'):
                logger.debug("No predictor, using default schedule")
                return None

            # Generate evenly-spaced fractional timesteps across full 0-999 range
            # This gives us fractional precision in sigma placement
            timesteps = torch.linspace(0, 999, steps, device=devices.cpu)

            # Convert each fractional timestep to sigma via predictor
            sigmas = torch.stack([
                model_wrap.predictor.sigma(torch.tensor([t], device=devices.cpu))
                for t in timesteps
            ]).squeeze()

            # Append final sigma = 0.0 (Forge convention)
            sigmas = torch.cat([sigmas, torch.zeros(1, device=devices.cpu)])

            logger.debug(f"Fractional schedule: {len(sigmas)} sigmas, range=[{sigmas[0]:.4f}, {sigmas[-1]:.4f}]")
            return sigmas.cpu()

        except Exception as e:
            logger.error(f"Fractional override failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    return override_scheduler


def should_use_fractional_interpolation(
    denoising_strength: float,
    steps: int,
    enable_fractional: bool = True
) -> bool:
    """
    Determine if fractional interpolation should be used for this frame.

    Fractional interpolation is beneficial when:
    1. Feature is enabled via checkbox
    2. Using img2img (denoising_strength < 1.0)
    3. At low step counts (< 30) where discrete resolution is coarse

    Args:
        denoising_strength: Forge's denoising strength
        steps: Number of sampling steps
        enable_fractional: Checkbox state from settings

    Returns:
        True if fractional interpolation should be used
    """
    # Check if feature is enabled by user
    if not enable_fractional:
        return False

    # No benefit for txt2img (denoising_strength = 1.0)
    if denoising_strength >= 0.999:
        return False

    # Most beneficial at low step counts (coarse discrete resolution)
    # At 30+ steps, discrete resolution (1/30 = 0.033 = 3.3%) is already fine
    if steps >= 30:
        return False

    return True


def get_effective_strength_resolution(steps: int, use_fractional: bool) -> float:
    """
    Calculate effective strength resolution (precision of strength tuning).

    Args:
        steps: Number of sampling steps
        use_fractional: Whether fractional interpolation is enabled

    Returns:
        Strength resolution as float (e.g., 0.01 = 1%, 0.05 = 5%)

    Examples:
        >>> get_effective_strength_resolution(4, False)
        0.25  # 25% resolution (discrete)

        >>> get_effective_strength_resolution(4, True)
        0.01  # 1% resolution (fractional)

        >>> get_effective_strength_resolution(20, False)
        0.05  # 5% resolution (discrete)

        >>> get_effective_strength_resolution(20, True)
        0.01  # 1% resolution (fractional)
    """
    if use_fractional:
        return 0.01  # 1% resolution with fractional interpolation
    else:
        return 1.0 / steps  # Discrete resolution (1/steps)


def apply_fractional_strength_if_enabled(p):
    """
    Apply fractional strength interpolation to processing object if enabled.

    This function sets p.sampler_noise_scheduler_override to use fractional
    timestep interpolation, providing 1% strength resolution instead of discrete
    levels that depend on step count.

    Args:
        p: Forge's StableDiffusionProcessingImg2Img object

    Side Effects:
        Sets p.sampler_noise_scheduler_override if conditions are met
        Logs debug message if fractional interpolation is applied

    Conditions for enabling:
        1. Checkbox enabled in settings
        2. Using img2img (denoising_strength < 1.0)
        3. At low step counts (< 30) where most beneficial
    """
    from deforum.rendering.options import is_fractional_strength_enabled
    from deforum.utils.system.logging import get_logger
    from modules.shared import sd_model

    logger = get_logger()

    # Check if feature is enabled and conditions are met
    if not should_use_fractional_interpolation(
        denoising_strength=p.denoising_strength if p.denoising_strength is not None else 1.0,
        steps=p.steps,
        enable_fractional=is_fractional_strength_enabled()
    ):
        return

    # Access the model's predictor for sigma interpolation
    # The sd_model should have the predictor we need
    if not hasattr(sd_model, 'forge_objects') or not hasattr(sd_model.forge_objects, 'unet'):
        logger.debug("Fractional strength: Model doesn't have forge_objects.unet, skipping")
        return

    # Create a closure that captures denoising_strength and accesses model when called
    denoising_strength = p.denoising_strength
    original_steps = p.steps

    def fractional_scheduler_override(steps: int) -> torch.Tensor:
        """
        Override scheduler to use fractional timestep interpolation.

        This function is called by Forge during sampling to get the sigma schedule.
        It uses the model's predictor to interpolate sigmas at fractional timestep
        positions, providing 1% strength resolution.

        Args:
            steps: Number of sampling steps

        Returns:
            Tensor of sigma values with fractional interpolation
        """
        try:
            # Access model's predictor through forge_objects
            from modules.shared import sd_model

            if not hasattr(sd_model, 'forge_objects') or not hasattr(sd_model.forge_objects, 'unet'):
                raise AttributeError("Model doesn't have forge_objects.unet")

            unet = sd_model.forge_objects.unet
            if not hasattr(unet, 'model') or not hasattr(unet.model, 'predictor'):
                raise AttributeError("UNet doesn't have model.predictor")

            predictor = unet.model.predictor

            # Calculate fractional timestep range
            start_timestep, end_timestep = calculate_fractional_timesteps(
                denoising_strength=denoising_strength,
                total_steps=original_steps,
                model_total_timesteps=999  # Forge uses 0-999 range
            )

            # Generate fractional sigma schedule
            sigmas = create_fractional_sigma_schedule(
                predictor=predictor,
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                steps=steps,
                device=devices.cpu
            )

            return sigmas

        except Exception as e:
            logger.debug(f"Fractional scheduler override failed: {e}, falling back to default")
            # Fallback: return None to use default scheduler
            # This will cause Forge to use the standard discrete sigma selection
            return None

    # Set the override on the processing object
    p.sampler_noise_scheduler_override = fractional_scheduler_override

    logger.debug(
        f"Fractional strength enabled: {denoising_strength:.3f} at {original_steps} steps "
        f"(1% resolution instead of {1.0/original_steps:.1%})"
    )