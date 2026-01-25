"""LTX-2 Pipeline Integration for Deforum.

Handles audio-conditioned video generation using LTX-2 model.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
from PIL import Image

from deforum.utils.system.logging import get_logger

logger = get_logger()


class LTX2Pipeline:
    """LTX-2 Audio-Video Generation Pipeline.

    Integration approach:
    1. Load LTX-2 model with appropriate variant (NF4/Full/HD)
    2. For each keyframe segment:
       - Extract audio chunk for time range
       - Generate video from start keyframe with audio conditioning
       - Chain segments together
    """

    def __init__(self, device: str = 'cuda', variant: str = 'LTX-2-4K-NF4'):
        """Initialize LTX-2 pipeline.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            variant: Model variant to use:
                - 'LTX-2-4K-NF4': 4-bit quantized, 12GB VRAM (recommended)
                - 'LTX-2-4K': Full precision, 24GB+ VRAM
                - 'LTX-2-HD': HD variant, 18GB VRAM
        """
        self.device = device
        self.variant = variant
        self.pipeline = None

    def load_model(self):
        """Load LTX-2 model from HuggingFace."""
        try:
            from diffusers import LTXPipeline
        except ImportError:
            logger.error("Failed to import LTX-2 dependencies", emoji='x')
            raise ImportError(
                "LTX-2 requires diffusers with LTX support. "
                "Install with: pip install 'diffusers>=0.32.0'"
            )

        logger.info(f"Loading LTX-2 model: {self.variant}...", emoji='download')

        # Fix transformers lazy loading issue - import T5Tokenizer explicitly
        # This resolves the _LazyModule Placeholder error
        from transformers import T5Tokenizer
        from deforum.integrations.ltx2.ltx2_model_discovery import LTX2ModelDiscovery

        # Get correct model ID based on variant
        discovery = LTX2ModelDiscovery()
        variant_info = discovery.MODEL_VARIANTS.get(self.variant)

        if variant_info is None:
            raise ValueError(f"Unknown LTX-2 variant: {self.variant}")

        model_id = variant_info['huggingface_id']
        logger.info(f"Using model: {model_id} (VRAM requirement: {variant_info['vram_gb']}GB)", emoji='info')

        # Inform about auto-download
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")

        if not os.path.exists(model_cache):
            logger.info(f"First run: Auto-downloading from HuggingFace ({model_id})", emoji='info')
            logger.info(f"Download size: ~19GB (caches to {cache_dir})", emoji='download')
            logger.info(f"This may take 5-15 minutes depending on connection speed...", emoji='hourglass')
        else:
            logger.info(f"Using cached model from {cache_dir}", emoji='check')

        # Load with appropriate precision
        if self.variant == 'LTX-2-4K-NF4':
            # Load with 4-bit quantization
            logger.info("Using 4-bit NF4 quantization (saves ~75% VRAM)", emoji='zap')
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

                self.pipeline = LTXPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    quantization_config=quantization_config,
                )
            except ImportError:
                logger.warning("BitsAndBytes not available, falling back to fp16", emoji='warning')
                self.pipeline = LTXPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                )
        else:
            # Full precision or fp16
            dtype = torch.float16 if self.device == 'cuda' else torch.float32
            self.pipeline = LTXPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )

        # Move to device
        self.pipeline.to(self.device)

        # Enable optimizations
        if self.device == 'cuda':
            try:
                self.pipeline.enable_model_cpu_offload()
                logger.info("Enabled CPU offload for memory efficiency", emoji='check')
            except:
                pass

        logger.info(f"LTX-2 pipeline loaded successfully", emoji='check')

    def generate_segment(
        self,
        start_image: Image.Image,
        audio_path: str,
        audio_start_sec: float,
        audio_duration_sec: float,
        prompt: str = "",
        negative_prompt: str = "blurry, low quality, distorted",
        num_frames: int = 121,
        fps: int = 24,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Generate video segment with audio conditioning.

        Args:
            start_image: Starting keyframe image
            audio_path: Path to audio file
            audio_start_sec: Start time in audio (seconds)
            audio_duration_sec: Duration of audio chunk (seconds)
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            num_frames: Number of frames to generate (should be 4n+1)
            fps: Target FPS
            guidance_scale: Guidance scale (3.0-7.0 recommended)
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility

        Returns:
            List of generated PIL Images
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")

        # Extract audio chunk
        audio_tensor = self._extract_audio_segment(
            audio_path, audio_start_sec, audio_duration_sec
        )

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.debug(f"Generating {num_frames} frames with LTX-2 (audio: {audio_start_sec:.1f}s-{audio_start_sec+audio_duration_sec:.1f}s)")

        # Generate video
        output = self.pipeline(
            image=start_image,
            audio=audio_tensor,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        # Extract frames
        frames = output.frames[0]

        return frames

    def _extract_audio_segment(
        self,
        audio_path: str,
        start_sec: float,
        duration_sec: float
    ) -> torch.Tensor:
        """Extract audio segment and prepare for LTX-2.

        Args:
            audio_path: Path to audio file
            start_sec: Start time in seconds
            duration_sec: Duration in seconds

        Returns:
            Audio tensor prepared for LTX-2
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required for audio processing. Install with: pip install librosa")

        # Load audio segment
        audio, sr = librosa.load(
            audio_path,
            sr=16000,
            mono=True,
            offset=start_sec,
            duration=duration_sec
        )

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).to(self.device)

        # Ensure correct shape
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        return audio_tensor

    def cleanup(self):
        """Clean up pipeline to free VRAM."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("LTX-2 pipeline cleaned up", emoji='wastebasket')
