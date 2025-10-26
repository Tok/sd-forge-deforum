"""Flux ControlNet v2 - Forge-native integration.

Loads ONLY FluxControlNetModel (~3.6GB) and computes control samples directly,
then injects into Forge's already-loaded Flux transformer (no VRAM duplication).

This replaces the v1 approach which loaded full FluxControlNetPipeline (~24GB total).
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Union, Tuple
from diffusers import FluxControlNetModel, AutoencoderKL

from .models import (
    load_flux_controlnet_model,
    get_model_info,
    temporarily_unpatch_hf_download
)
from .preprocessors import (
    preprocess_image_for_controlnet,
    numpy_to_pil
)
from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()



def _pack_latents(latents: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int) -> torch.Tensor:
    """Patchify latents into 2x2 patches for Flux transformer.

    Converts (B, C, H, W) → (B, (H//2)*(W//2), C*4)

    Args:
        latents: Latent tensor from VAE encoding
        batch_size: Batch size
        num_channels: Number of latent channels (16 for Flux)
        height: Latent height (image_height // 16)
        width: Latent width (image_width // 16)

    Returns:
        Patchified latents ready for Flux transformer
    """
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    return latents


class FluxControlNetV2Manager:
    """V2 Manager for Flux ControlNet - Forge-native integration.

    This version:
    - Loads ONLY FluxControlNetModel (~3.6GB)
    - Computes control samples via forward() call
    - Returns samples to be injected into Forge's Flux transformer
    - No pipeline, no duplicate Flux model
    """

    def __init__(
        self,
        control_type: str = "canny",
        model_name: str = "instantx",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda"
    ):
        """Initialize Flux ControlNet V2 manager.

        Args:
            control_type: "canny" or "depth"
            model_name: Model provider ("instantx", "xlabs", "bfl", "shakker")
            torch_dtype: Torch data type for weights
            device: Device to load model on
        """
        self.control_type = control_type
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device = device

        self.controlnet = None
        self.vae = None
        self.is_loaded = False

        logger.info(f"🌐 Initialized Flux {control_type.title()} ControlNet V2 (Forge-native)")
        logger.info(f"   Model: {get_model_info(control_type, model_name)}")

    def load_model(self):
        """Load ControlNet model and Flux VAE."""
        if self.is_loaded:
            logger.info("   ControlNet model already loaded")
            return

        logger.info("🌐 Loading Flux ControlNet model (v2 - model only)...")

        # Load ControlNet model (~3.6GB)
        self.controlnet = load_flux_controlnet_model(
            control_type=self.control_type,
            model_name=self.model_name,
            torch_dtype=self.torch_dtype,
            device=self.device
        )

        self.controlnet = self.controlnet.to(self.device)

        # Use Forge's already-loaded Flux VAE for control image encoding
        logger.info("🌐 Accessing Forge's Flux VAE for control image encoding...")
        try:
            from modules.shared import sd_model

            # Forge's Flux model already has VAE loaded
            if hasattr(sd_model, 'forge_objects') and hasattr(sd_model.forge_objects, 'vae'):
                forge_vae = sd_model.forge_objects.vae
                logger.info(f"   {emoji_if_enabled("✓")} Using Forge's loaded Flux VAE")

                # Forge's VAE is backend.patcher.vae.VAE, we need to extract the actual model
                if hasattr(forge_vae, 'first_stage_model'):
                    self.vae = forge_vae.first_stage_model
                    logger.info(f"   {emoji_if_enabled("✓")} Extracted VAE model from Forge wrapper")
                else:
                    self.vae = forge_vae

                # Make sure it's on the right device
                self.vae = self.vae.to(self.device)
                logger.info(f"{emoji_if_enabled("✓")} Forge's Flux VAE ready for use")
            else:
                logger.error("   ⚠️ Could not access Forge's VAE")
                self.vae = None

        except Exception as e:
            logger.error(f"⚠️ Could not access Forge's Flux VAE: {e}")
            import traceback
            traceback.print_exc()
            logger.info("   Will proceed without VAE (control will not work)")
            self.vae = None

        self.is_loaded = True

        logger.info(f"{emoji_if_enabled("✓")} Flux ControlNet model loaded ({self.control_type}, ~3.6GB)")
        logger.info("   No pipeline created - will use Forge's Flux transformer")

    def compute_control_samples(
        self,
        hidden_states: torch.Tensor,
        control_image: Union[np.ndarray, Image.Image],
        conditioning_scale: float = 0.7,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        preprocess_control: bool = True,
        canny_low: int = 100,
        canny_high: int = 200,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Compute control samples from preprocessed control image.

        This method calls FluxControlNetModel.forward() directly with parameters
        from Forge's processing pipeline.

        Args:
            hidden_states: Latent image tensor from Forge
            control_image: Control image (preprocessed or raw)
            conditioning_scale: ControlNet strength (0.0-1.0)
            encoder_hidden_states: Text embeddings from Forge's CLIP
            pooled_projections: Pooled text embeddings
            timestep: Current denoising timestep
            img_ids: Image position IDs
            txt_ids: Text position IDs
            guidance: Guidance scale tensor
            preprocess_control: Whether to preprocess control image
            canny_low: Canny low threshold (for Canny mode)
            canny_high: Canny high threshold (for Canny mode)

        Returns:
            Tuple of (controlnet_block_samples, controlnet_single_block_samples)
            to be passed to Forge's Flux transformer
        """
        if not self.is_loaded:
            self.load_model()

        # Preprocess control image if needed
        if preprocess_control:
            if isinstance(control_image, Image.Image):
                control_image = np.array(control_image)

            control_image = preprocess_image_for_controlnet(
                control_image,
                self.control_type,
                canny_low=canny_low,
                canny_high=canny_high
            )

        # Convert to PIL if numpy
        if isinstance(control_image, np.ndarray):
            control_image = numpy_to_pil(control_image)

        # Convert PIL to tensor
        # Format: (batch, channels, height, width) in range [0, 1]
        # NOTE: Forge's VAE expects [0, 1] and does its own [-1, 1] normalization internally
        control_rgb = torch.from_numpy(np.array(control_image)).float() / 255.0
        control_rgb = control_rgb.permute(2, 0, 1).unsqueeze(0)  # (B, C, H, W)
        control_rgb = control_rgb.to(device=self.device, dtype=torch.float32)  # VAE needs float32

        logger.info(f"🌐 Computing ControlNet control samples...")
        logger.info(f"   Control RGB shape: {control_rgb.shape}")
        logger.info(f"   Control RGB range: [{control_rgb.min():.3f}, {control_rgb.max():.3f}]")

        # VAE encode and patchify control image to match hidden_states format
        if self.vae is not None:
            logger.info(f"   VAE encoding control image with Forge's Flux VAE...")
            try:
                with torch.inference_mode():
                    # Forge's VAE model - use encode
                    # Check if it's Forge's backend VAE or diffusers VAE
                    if hasattr(self.vae, 'encode') and hasattr(self.vae, 'quant_conv'):
                        # Forge's backend.nn.vae.AutoencodingEngine
                        # Normalize to [-1, 1] for Forge's VAE
                        control_rgb_normalized = control_rgb * 2.0 - 1.0

                        # Convert to VAE's dtype (likely bfloat16)
                        # Get dtype from VAE's first conv layer
                        vae_dtype = next(self.vae.parameters()).dtype
                        control_rgb_normalized = control_rgb_normalized.to(dtype=vae_dtype)

                        # Call encode without regulation (will use posterior.sample())
                        control_latent = self.vae.encode(control_rgb_normalized)
                        logger.info(f"   Using Forge's backend VAE encoder (dtype: {vae_dtype})")
                    else:
                        # Fallback to diffusers-style (shouldn't happen now)
                        control_latent = self.vae.encode(control_rgb).latent_dist.sample()
                        logger.info(f"   Using diffusers-style VAE encoder")

                    # Apply VAE scaling (Flux VAE config)
                    # shift_factor and scaling_factor from Flux VAE
                    shift_factor = 0.1159  # Flux VAE default
                    scaling_factor = 0.3611  # Flux VAE default
                    control_latent = (control_latent - shift_factor) * scaling_factor

                    logger.info(f"   Control latent shape: {control_latent.shape}")

                    # Patchify latent to match transformer input
                    batch_size, num_channels, latent_h, latent_w = control_latent.shape
                    control_tensor = _pack_latents(control_latent, batch_size, num_channels, latent_h, latent_w)
                    control_tensor = control_tensor.to(dtype=self.torch_dtype)

                    logger.info(f"   Control patchified shape: {control_tensor.shape}")

                    # Create dummy hidden_states matching control_tensor shape if not provided
                    if hidden_states is None:
                        hidden_states = torch.zeros_like(control_tensor)
                        logger.info(f"   Created dummy hidden_states matching control shape: {hidden_states.shape}")

                    # Create dummy parameters if not provided (required by ControlNet)
                    if timestep is None:
                        # Use middle timestep (500 out of 1000)
                        timestep = torch.tensor([500.0], device=self.device, dtype=torch.float32)
                        logger.info(f"   Created dummy timestep: {timestep}")

                    if encoder_hidden_states is None:
                        # Create dummy text embeddings (Flux uses 4096 dim)
                        encoder_hidden_states = torch.zeros((1, 77, 4096), device=self.device, dtype=self.torch_dtype)
                        logger.info(f"   Created dummy encoder_hidden_states: {encoder_hidden_states.shape}")

                    if pooled_projections is None:
                        # Create dummy pooled projections (Flux uses 768 dim)
                        pooled_projections = torch.zeros((1, 768), device=self.device, dtype=self.torch_dtype)
                        logger.info(f"   Created dummy pooled_projections: {pooled_projections.shape}")

                    if img_ids is None:
                        # Create position IDs for image tokens
                        batch_size, seq_len, _ = hidden_states.shape
                        img_ids = torch.zeros((batch_size, seq_len, 3), device=self.device, dtype=torch.float32)
                        logger.info(f"   Created dummy img_ids: {img_ids.shape}")

                    if txt_ids is None:
                        # Create position IDs for text tokens (matches encoder_hidden_states seq len)
                        txt_ids = torch.zeros((1, 77, 3), device=self.device, dtype=torch.float32)
                        logger.info(f"   Created dummy txt_ids: {txt_ids.shape}")

                    if guidance is None:
                        # Create dummy guidance (scalar guidance scale, typically 3.5 for Flux)
                        guidance = torch.tensor([3.5], device=self.device, dtype=torch.float32)
                        logger.info(f"   Created dummy guidance: {guidance}")
            except Exception as e:
                logger.error(f"   ⚠️ VAE encoding failed: {e}")
                import traceback
                traceback.print_exc()
                logger.info(f"   Falling back to RGB image (may not work)")
                control_tensor = control_rgb.to(dtype=self.torch_dtype)
        else:
            # No VAE available - control will not work
            logger.warning(f"   ⚠️ No VAE available - control will not work correctly")
            logger.warning(f"   ⚠️ Make sure Flux model is loaded in Forge")
            control_tensor = control_rgb.to(dtype=self.torch_dtype)

        logger.info(f"   Hidden states shape: {hidden_states.shape if hidden_states is not None else 'None'}")
        logger.info(f"   Conditioning scale: {conditioning_scale}")

        # Call FluxControlNetModel.forward() directly
        with torch.inference_mode():
            controlnet_output = self.controlnet(
                hidden_states=hidden_states,
                controlnet_cond=control_tensor,
                conditioning_scale=conditioning_scale,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=True
            )

        # Extract control samples
        controlnet_block_samples = controlnet_output.controlnet_block_samples
        controlnet_single_block_samples = controlnet_output.controlnet_single_block_samples

        logger.info(f"{emoji_if_enabled("✓")} ControlNet samples computed:")
        logger.info(f"   Block samples: {len(controlnet_block_samples) if controlnet_block_samples is not None else 0} tensors")
        logger.info(f"   Single block samples: {len(controlnet_single_block_samples) if controlnet_single_block_samples is not None else 0} tensors")

        return controlnet_block_samples, controlnet_single_block_samples

    def unload(self):
        """Unload models to free VRAM."""
        self.controlnet = None
        self.vae = None
        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("🌐 Flux ControlNet V2 models unloaded")
