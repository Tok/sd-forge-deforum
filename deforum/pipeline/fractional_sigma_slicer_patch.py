"""
Monkey patch for Forge's sigma slicing to handle fractional t_enc values.

This patches the KDiffusionSampler.sample_img2img method to slice sigma schedules
using fractional interpolation when t_enc is not an integer.
"""

import torch
import inspect


def fractional_sigma_slice(sigmas: torch.Tensor, steps: int, t_enc: float) -> torch.Tensor:
    """
    Slice sigma schedule using fractional t_enc with interpolation.

    Standard (discrete):
        sigma_sched = sigmas[steps - int(t_enc) - 1 :]

    Fractional (interpolated):
        If t_enc=4.3, interpolate between positions to get smooth transition

    Args:
        sigmas: Full sigma schedule
        steps: Total step count
        t_enc: Fractional denoising step count (can be float)

    Returns:
        Sliced sigma schedule with fractional starting point
    """
    from deforum.utils.system.logging import get_logger, emoji as emoji_utils
    logger = get_logger()
    magnifying_glass = emoji_utils.magnifying_glass()
    check = emoji_utils.maybe_check()
    warning = emoji_utils.maybe_warning()

    logger.trace(f"{magnifying_glass} FRACTIONAL SIGMA: sigmas.shape={sigmas.shape}, steps={steps}, t_enc={t_enc:.4f}")

    # Calculate fractional index
    frac_idx = steps - t_enc - 1
    logger.debug(f"   frac_idx = {frac_idx:.4f}")

    # If already integer, use normal slicing
    if isinstance(t_enc, int) or frac_idx == int(frac_idx):
        result = sigmas[int(frac_idx):]
        logger.trace(f"   {check} INTEGER case: result.shape={result.shape}")
        return result

    # Fractional case: interpolate starting sigma
    low_idx = int(torch.floor(torch.tensor(frac_idx)).item())
    high_idx = int(torch.ceil(torch.tensor(frac_idx)).item())
    weight = frac_idx - low_idx

    logger.trace(f"   FRACTIONAL: low={low_idx}, high={high_idx}, weight={weight:.4f}")

    # Bound check
    if high_idx >= len(sigmas):
        logger.warning(f"   {warning} Clamping high_idx {high_idx} → {len(sigmas)-1}")
        high_idx = len(sigmas) - 1
    if low_idx < 0:
        logger.warning(f"   {warning} Clamping low_idx {low_idx} → 0")
        low_idx = 0

    # Interpolate first sigma
    first_sigma = (1 - weight) * sigmas[low_idx] + weight * sigmas[high_idx]
    logger.debug(f"   first_sigma = {first_sigma:.6f}")

    # Concatenate with remaining schedule
    remaining_sigmas = sigmas[high_idx:]
    result = torch.cat([first_sigma.unsqueeze(0), remaining_sigmas])
    logger.debug(f"   {check} RESULT: shape={result.shape}, range=[{result[0]:.6f}, {result[-1]:.6f}]")

    return result


# Store original methods
_original_sample_img2img = {}


def patch_kdiffusion_sampler_class():
    """
    Patch KDiffusionSampler.sample_img2img to handle fractional t_enc.

    Replaces the line:
        sigma_sched = sigmas[steps - t_enc - 1 :]
    With:
        sigma_sched = fractional_sigma_slice(sigmas, steps, t_enc)
    """
    try:
        from modules import sd_samplers_kdiffusion
        from modules.sd_samplers_kdiffusion import KDiffusionSampler

        # Store original method
        if 'KDiffusionSampler' not in _original_sample_img2img:
            _original_sample_img2img['KDiffusionSampler'] = KDiffusionSampler.sample_img2img

        # Create patched version
        original_method = _original_sample_img2img['KDiffusionSampler']

        def patched_sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
            """Patched sample_img2img with fractional sigma slicing."""
            from modules import sd_samplers_common, devices
            from modules.sd_samplers_kdiffusion import sampling_prepare

            unet_patcher = self.model_wrap.inner_model.forge_objects.unet
            sampling_prepare(self.model_wrap.inner_model.forge_objects.unet, x=x)

            steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)

            sigmas = self.get_sigmas(p, steps).to(x.device)

            # FRACTIONAL CHANGE: Use fractional_sigma_slice instead of array slicing
            sigma_sched = fractional_sigma_slice(sigmas, steps, t_enc)

            x = x.to(noise)

            xi = self.model_wrap.predictor.noise_scaling(sigma_sched[0], noise, x, max_denoise=False)

            # Continue with rest of original method
            from modules import shared
            opts = shared.opts
            if opts.img2img_extra_noise > 0:
                from modules.sd_samplers_kdiffusion import ExtraNoiseParams, extra_noise_callback
                p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
                extra_noise_params = ExtraNoiseParams(noise, x, xi)
                extra_noise_callback(extra_noise_params)
                noise = extra_noise_params.noise
                xi += noise * opts.img2img_extra_noise

            extra_params_kwargs = self.initialize(p)
            parameters = inspect.signature(self.func).parameters

            if 'sigma_min' in parameters:
                extra_params_kwargs['sigma_min'] = sigma_sched[-2]
            if 'sigma_max' in parameters:
                extra_params_kwargs['sigma_max'] = sigma_sched[0]
            if 'n' in parameters:
                extra_params_kwargs['n'] = len(sigma_sched) - 1
            if 'sigma_sched' in parameters:
                extra_params_kwargs['sigma_sched'] = sigma_sched
            if 'sigmas' in parameters:
                extra_params_kwargs['sigmas'] = sigma_sched

            self.model_wrap_cfg.init_latent = x
            self.last_latent = x

            samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, xi, extra_args={
                'cond': conditioning,
                'image_cond': image_conditioning,
                'uncond': unconditional_conditioning,
                'cond_scale': p.cfg_scale,
                's_min_uncond': self.s_min_uncond
            }, disable=False, callback=self.callback_state, **extra_params_kwargs))

            sampling_prepare(unet_patcher, x=x)

            return samples

        # Apply patch
        KDiffusionSampler.sample_img2img = patched_sample_img2img

        from deforum.utils.system.logging import get_logger
        logger = get_logger()
        logger.info("Applied fractional sigma slicer patch - fractional t_enc now supported")

        return True

    except Exception as e:
        from deforum.utils.system.logging import get_logger
        logger = get_logger()
        logger.error(f"Failed to apply sigma slicer patch: {e}")
        import traceback
        traceback.print_exc()
        return False


def remove_kdiffusion_sampler_patch():
    """Remove sigma slicer patch and restore original."""
    try:
        from modules.sd_samplers_kdiffusion import KDiffusionSampler

        if 'KDiffusionSampler' in _original_sample_img2img:
            KDiffusionSampler.sample_img2img = _original_sample_img2img['KDiffusionSampler']

            from deforum.utils.system.logging import get_logger
            logger = get_logger()
            logger.info("Removed fractional sigma slicer patch")

        return True

    except Exception as e:
        from deforum.utils.system.logging import get_logger
        logger = get_logger()
        logger.error(f"Failed to remove sigma slicer patch: {e}")
        return False
