#!/usr/bin/env python3
"""
Diffusers Compatibility Patch for Forge + Wan 2.2
Fixes compatibility issues between diffusers git main (required for WanPipeline) and Forge's Flux backend
"""

from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()


def patch_flow_match_scheduler():
    """
    Patch FlowMatchEulerDiscreteScheduler.time_shift to handle None self parameter

    Forge's backend calls: FlowMatchEulerDiscreteScheduler.time_shift(None, mu, 1.0, sigmas)
    But diffusers git main expects time_shift to be an instance method with self.config

    This patch makes it work both ways.
    """
    try:
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        import torch

        # Save original method
        original_time_shift = FlowMatchEulerDiscreteScheduler.time_shift

        @classmethod
        def patched_time_shift(cls, self_or_mu, mu_or_sigma=None, sigma_or_t=None, t=None):
            """
            Patched version that works when called as class method with None self OR as instance method

            Forge calls: FlowMatchEulerDiscreteScheduler.time_shift(None, mu, 1.0, sigmas)
            -> receives (cls, None, mu, 1.0, sigmas)

            New diffusers calls: instance.time_shift(mu, 1.0, sigmas)
            -> receives (cls, self, mu, 1.0, sigmas)

            Args:
                cls: The class
                self_or_mu: Either self (instance) or mu (if called with None as first arg)
                mu_or_sigma: Either mu or sigma depending on call pattern
                sigma_or_t: Either sigma or t depending on call pattern
                t: Time values (only present in Forge's call pattern)
            """
            # Determine call pattern
            if t is not None:
                # Forge pattern: time_shift(None, mu, sigma, t)
                # self_or_mu = None, mu_or_sigma = mu, sigma_or_t = sigma, t = t
                mu = mu_or_sigma
                sigma = sigma_or_t
                # Use static implementation
            elif hasattr(self_or_mu, 'config'):
                # Instance method pattern: self.time_shift(mu, sigma, t)
                # self_or_mu = self, mu_or_sigma = mu, sigma_or_t = sigma, t = None -> but wait, t is the 4th arg
                # Actually this won't happen because we're replacing the whole method
                # Let me just use original for this case
                return original_time_shift(self_or_mu, mu_or_sigma, sigma_or_t)
            else:
                # Assume Forge pattern with positional args
                mu = mu_or_sigma
                sigma = sigma_or_t

            # Static version for Forge compatibility (default "exponential" behavior)
            time_val = t if t is not None else sigma_or_t

            # Handle mixed scalar/tensor inputs - convert to match time_val's type
            if isinstance(time_val, torch.Tensor):
                # Convert mu and sigma to tensors if they aren't already
                if not isinstance(mu, torch.Tensor):
                    mu = torch.tensor(mu, dtype=time_val.dtype, device=time_val.device)
                if not isinstance(sigma, torch.Tensor):
                    sigma = torch.tensor(sigma, dtype=time_val.dtype, device=time_val.device)
                return torch.exp(mu) / (torch.exp(mu) + (1 / time_val - 1) ** sigma)
            else:
                # Use numpy for scalar inputs
                import numpy as np
                # Convert any tensors to numpy
                if isinstance(mu, torch.Tensor):
                    mu = mu.cpu().numpy()
                if isinstance(sigma, torch.Tensor):
                    sigma = sigma.cpu().numpy()
                if isinstance(time_val, torch.Tensor):
                    time_val = time_val.cpu().numpy()
                return np.exp(mu) / (np.exp(mu) + (1 / time_val - 1) ** sigma)

        # Replace the method
        FlowMatchEulerDiscreteScheduler.time_shift = patched_time_shift

        logger.info(f"{emoji_if_enabled('✅')} FlowMatchEulerDiscreteScheduler.time_shift patched")
        return True

    except Exception as e:
        logger.error(f"⚠️ Failed to apply diffusers compatibility patch: {e}")
        return False


def patch_torch_rmsnorm():
    """
    Add RMSNorm to torch.nn if not present (PyTorch < 2.4.0)

    Wan 2.2 models require RMSNorm which was added in PyTorch 2.4.0
    WebUI Forge uses PyTorch 2.3.1, so we need to provide a compatible implementation
    """
    import torch
    import torch.nn as nn

    if hasattr(nn, 'RMSNorm'):
        logger.debug(f"{emoji_if_enabled('✅')} torch.nn.RMSNorm already available (PyTorch 2.4.0+)")
        return True

    try:
        logger.debug("Adding RMSNorm compatibility for PyTorch < 2.4.0...", emoji='wrench')

        class RMSNorm(nn.Module):
            """
            Root Mean Square Layer Normalization
            Compatible implementation for PyTorch < 2.4.0
            """
            def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=False, device=None, dtype=None):
                super().__init__()
                factory_kwargs = {'device': device, 'dtype': dtype}

                if isinstance(normalized_shape, int):
                    normalized_shape = (normalized_shape,)
                self.normalized_shape = tuple(normalized_shape)
                self.eps = eps
                self.elementwise_affine = elementwise_affine

                if self.elementwise_affine:
                    self.weight = nn.Parameter(torch.ones(normalized_shape, **factory_kwargs))
                    if bias:
                        self.bias = nn.Parameter(torch.zeros(normalized_shape, **factory_kwargs))
                    else:
                        self.register_parameter('bias', None)
                else:
                    self.register_parameter('weight', None)
                    self.register_parameter('bias', None)

            def forward(self, input):
                # Compute RMS
                variance = input.pow(2).mean(-1, keepdim=True)
                input = input * torch.rsqrt(variance + self.eps)

                # Apply weight and bias if enabled
                if self.elementwise_affine:
                    input = input * self.weight
                    if self.bias is not None:
                        input = input + self.bias

                return input

            def extra_repr(self):
                return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

        # Add to torch.nn module
        nn.RMSNorm = RMSNorm
        torch.nn.RMSNorm = RMSNorm

        logger.info(f"{emoji_if_enabled('✅')} torch.nn.RMSNorm added (PyTorch 2.3.1 compatibility)")
        return True

    except Exception as e:
        logger.error(f"⚠️ Failed to apply RMSNorm patch: {e}")
        return False


def patch_diffusers_attention():
    """
    Patch PyTorch's scaled_dot_product_attention to filter out enable_gqa parameter

    Diffusers git main uses enable_gqa parameter in scaled_dot_product_attention,
    but this parameter was added in PyTorch 2.4.0. Forge uses PyTorch 2.3.1.

    Nuclear option: Patch torch.nn.functional directly to filter unsupported parameters.
    """
    try:
        import torch
        import inspect

        # Check PyTorch version
        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if torch_version >= (2, 4):
            logger.debug(f"{emoji_if_enabled('✅')} PyTorch 2.4.0+ detected - enable_gqa parameter supported")
            return True

        logger.debug(f"PyTorch {torch.__version__} detected - patching scaled_dot_product_attention...", emoji='wrench')

        # Save original PyTorch function
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        # PyTorch 2.3.1 supports these parameters (hardcoded since it's a C++ builtin)
        supported_params_231 = {'query', 'key', 'value', 'attn_mask', 'dropout_p', 'is_causal', 'scale'}
        logger.debug(f"   PyTorch 2.3.1 SDPA parameters: {supported_params_231}")

        def patched_scaled_dot_product_attention(*args, **kwargs):
            """Wrapper that filters out unsupported parameters like enable_gqa"""
            # Remove unsupported parameters
            if 'enable_gqa' in kwargs:
                enable_gqa_value = kwargs.pop('enable_gqa')
                # Only warn if it's True (False is default anyway)
                if enable_gqa_value:
                    logger.warning("   ⚠️ enable_gqa=True requested but not supported in PyTorch 2.3.1, ignoring")

            # Remove any other parameters not supported in PyTorch 2.3.1
            unsupported = [k for k in list(kwargs.keys()) if k not in supported_params_231]
            for param in unsupported:
                logger.warning(f"   ⚠️ Removing unsupported parameter: {param}={kwargs[param]}")
                kwargs.pop(param)

            # Call original function with filtered parameters
            return original_sdpa(*args, **kwargs)

        # Replace PyTorch's function globally
        torch.nn.functional.scaled_dot_product_attention = patched_scaled_dot_product_attention

        logger.info(f"{emoji_if_enabled('✅')} scaled_dot_product_attention patched (enable_gqa filter)")
        return True

    except Exception as e:
        logger.error(f"⚠️ Failed to apply attention patch: {e}")
        import traceback
        traceback.print_exc()
        return False


def patch_forge_flux_controlnet():
    """
    Patch Forge's Flux transformer to support Flux ControlNet

    Adds controlnet_block_samples and controlnet_single_block_samples parameters
    to inner_forward() method and injects control after each block.

    This matches diffusers' FluxTransformer2DModel ControlNet implementation.
    """
    try:
        import sys
        import torch

        # Import Forge's Flux transformer
        sys.path.insert(0, '/home/zirteq/workspace/stable-diffusion-webui-forge')
        from backend.nn.flux import IntegratedFluxTransformer2DModel

        # Save original inner_forward method
        original_inner_forward = IntegratedFluxTransformer2DModel.inner_forward

        def patched_inner_forward(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None,
                                 controlnet_block_samples=None, controlnet_single_block_samples=None):
            """
            Patched Flux.inner_forward with ControlNet support

            Adds control sample injection after each double_block and single_block,
            matching diffusers' FluxTransformer2DModel implementation.
            """
            # Import here to avoid circular deps
            from backend.nn.flux import timestep_embedding

            if img.ndim != 3 or txt.ndim != 3:
                raise ValueError("Input img and txt tensors must have 3 dimensions.")

            img = self.img_in(img)
            vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))

            if self.guidance_embed:
                if guidance is None:
                    raise ValueError("Didn't get guidance strength for guidance distilled model.")
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

            vec = vec + self.vector_in(y)
            txt = self.txt_in(txt)
            del y, guidance

            ids = torch.cat((txt_ids, img_ids), dim=1)
            del txt_ids, img_ids

            pe = self.pe_embedder(ids)
            del ids

            # Process double_blocks with ControlNet injection
            for index_block, block in enumerate(self.double_blocks):
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

                # Inject ControlNet control samples (Flux ControlNet support)
                if controlnet_block_samples is not None and index_block < len(controlnet_block_samples):
                    # Add control sample to image hidden states
                    # This matches diffusers' FluxTransformer2DModel implementation
                    control_sample = controlnet_block_samples[index_block]
                    if index_block == 0:
                        print(f"      [Flux Block {index_block}] img: {img.shape} [{img.min():.4f}, {img.max():.4f}], "
                              f"control: {control_sample.shape} [{control_sample.min():.4f}, {control_sample.max():.4f}]")
                    img = img + control_sample

            img = torch.cat((txt, img), 1)

            # Process single_blocks with ControlNet injection
            for index_block, block in enumerate(self.single_blocks):
                img = block(img, vec=vec, pe=pe)

                # Inject ControlNet control samples (Flux ControlNet support)
                if controlnet_single_block_samples is not None and index_block < len(controlnet_single_block_samples):
                    # Add control sample to hidden states
                    # This matches diffusers' FluxTransformer2DModel implementation
                    img = img + controlnet_single_block_samples[index_block]

            del pe
            img = img[:, txt.shape[1]:, ...]
            del txt
            img = self.final_layer(img, vec)
            del vec
            return img

        # Also patch forward() to accept and pass ControlNet parameters
        original_forward = IntegratedFluxTransformer2DModel.forward

        def patched_forward(self, x, timestep, context, y, guidance=None,
                          controlnet_block_samples=None, controlnet_single_block_samples=None, **kwargs):
            """Patched Flux.forward with ControlNet support"""
            bs, c, h, w = x.shape
            input_device = x.device
            input_dtype = x.dtype
            patch_size = 2
            pad_h = (patch_size - x.shape[-2] % patch_size) % patch_size
            pad_w = (patch_size - x.shape[-1] % patch_size) % patch_size
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")

            from einops import rearrange, repeat
            img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            del x, pad_h, pad_w

            h_len = ((h + (patch_size // 2)) // patch_size)
            w_len = ((w + (patch_size // 2)) // patch_size)

            img_ids = torch.zeros((h_len, w_len, 3), device=input_device, dtype=input_dtype)
            img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=input_device, dtype=input_dtype)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=input_device, dtype=input_dtype)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

            txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)
            del input_device, input_dtype

            # Call inner_forward with ControlNet parameters
            out = self.inner_forward(img, img_ids, context, txt_ids, timestep, y, guidance,
                                    controlnet_block_samples=controlnet_block_samples,
                                    controlnet_single_block_samples=controlnet_single_block_samples)

            del img, img_ids, txt_ids, timestep, context

            out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:, :, :h, :w]
            del h_len, w_len, bs
            return out

        # Replace both methods
        IntegratedFluxTransformer2DModel.inner_forward = patched_inner_forward
        IntegratedFluxTransformer2DModel.forward = patched_forward

        logger.info(f"{emoji_if_enabled('✅')} IntegratedFluxTransformer2DModel patched (Flux ControlNet V2)")
        return True

    except Exception as e:
        logger.error(f"⚠️ Failed to apply Forge Flux ControlNet patch: {e}")
        import traceback
        traceback.print_exc()
        return False


def patch_forge_kmodel_for_controlnet():
    """
    Patch Forge's KModel.apply_model to compute and pass Flux ControlNet samples

    This allows control samples to be computed on-the-fly during each denoising step
    and passed to the patched Flux transformer.
    """
    try:
        import sys
        sys.path.insert(0, '/home/zirteq/workspace/stable-diffusion-webui-forge')
        from backend.modules.k_model import KModel

        # Save original apply_model
        original_apply_model = KModel.apply_model

        def patched_apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
            """
            Patched apply_model that passes Flux ControlNet samples to transformer

            Retrieves pre-computed control samples from global storage and passes them
            to the Flux transformer via kwargs.
            """
            # Import here to avoid issues during patch application
            try:
                # Get control samples from global storage
                from deforum.integrations.flux_controlnet.forge_injection import get_stored_control_samples

                control_samples = get_stored_control_samples()

                if control_samples is not None:
                    controlnet_block_samples, controlnet_single_block_samples = control_samples

                    # Pass control samples to Flux via extra_conds
                    kwargs['controlnet_block_samples'] = controlnet_block_samples
                    kwargs['controlnet_single_block_samples'] = controlnet_single_block_samples

                    # Debug print once per generation
                    if not hasattr(self, '_flux_cn_logged'):
                        logger.debug(f"🌐 Passing Flux ControlNet samples to transformer")
                        logger.debug(f"   Block samples: {len(controlnet_block_samples) if controlnet_block_samples is not None else 0} tensors")
                        logger.debug(f"   Single block samples: {len(controlnet_single_block_samples) if controlnet_single_block_samples is not None else 0} tensors")
                        self._flux_cn_logged = True
            except Exception as e:
                # Silently fail if control samples not available (not all generations use ControlNet)
                pass

            # Call original apply_model with potentially added controlnet kwargs
            return original_apply_model(self, x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

        # Replace the method
        KModel.apply_model = patched_apply_model

        logger.info(f"{emoji_if_enabled('✅')} KModel.apply_model patched (ControlNet sample injection)")
        return True

    except Exception as e:
        logger.error(f"⚠️ Failed to apply KModel ControlNet patch: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_all_patches():
    """Apply all compatibility patches"""
    logger.info("Applying compatibility patches: RMSNorm, FlowMatch, SDPA, Flux ControlNet, KModel", emoji='wrench')
    patch_torch_rmsnorm()
    patch_flow_match_scheduler()
    patch_diffusers_attention()
    patch_forge_flux_controlnet()
    patch_forge_kmodel_for_controlnet()
