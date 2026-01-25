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
        import os

        # Enable optimized CUDA kernels for GGUF (10% speedup if available)
        os.environ["DIFFUSERS_GGUF_CUDA_KERNELS"] = "true"

        try:
            from diffusers import LTX2ImageToVideoPipeline, GGUFQuantizationConfig
            from diffusers.models.transformers import LTX2VideoTransformer3DModel
        except ImportError:
            logger.error("Failed to import LTX-2 dependencies", emoji='x')
            raise ImportError(
                "LTX-2 requires diffusers with LTX-2 and GGUF support. "
                "Install with: pip install 'diffusers>=0.32.0' gguf kernels"
            )

        logger.info(f"Loading LTX-2 model: {self.variant}...", emoji='download')

        from deforum.integrations.ltx2.ltx2_model_discovery import LTX2ModelDiscovery

        # Get correct model ID based on variant
        discovery = LTX2ModelDiscovery()
        variant_info = discovery.MODEL_VARIANTS.get(self.variant)

        if variant_info is None:
            raise ValueError(f"Unknown LTX-2 variant: {self.variant}")

        model_id = variant_info['huggingface_id']
        logger.info(f"Using model: {model_id} (VRAM requirement: {variant_info['vram_gb']}GB)", emoji='info')

        # Check if this is a GGUF variant
        is_gguf = variant_info['quantization'] and variant_info['quantization'].startswith('gguf-')

        # Note: LTX-2 uses Gemma tokenizer, not T5
        # Pre-loading is not needed - pipeline handles tokenizer loading
        tokenizer = None

        # Inform about auto-download
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

        # Check if model is cached (works for both GGUF and full models)
        if is_gguf:
            # Check for GGUF file
            gguf_cache = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
            if not os.path.exists(gguf_cache):
                logger.info(f"First run: Auto-downloading GGUF from HuggingFace ({model_id})", emoji='info')
                logger.info(f"Download size: ~{variant_info['vram_gb']}GB (caches to {cache_dir})", emoji='download')
                logger.info(f"This may take 5-15 minutes depending on connection speed...", emoji='hourglass')
            else:
                logger.info(f"Using cached GGUF model from {cache_dir}", emoji='check')
        else:
            # Check for full model (Lightricks/LTX-2)
            base_model_id = "Lightricks/LTX-2"
            model_cache = os.path.join(cache_dir, f"models--{base_model_id.replace('/', '--')}")
            if not os.path.exists(model_cache):
                logger.info(f"First run: Auto-downloading from HuggingFace ({base_model_id})", emoji='info')
                logger.info(f"Download size: ~19GB (caches to {cache_dir})", emoji='download')
                logger.info(f"This may take 5-15 minutes depending on connection speed...", emoji='hourglass')
            else:
                logger.info(f"Using cached model from {cache_dir}", emoji='check')

        # Load with appropriate quantization method
        if is_gguf:
            # Load GGUF quantized model (recommended - better quality and VRAM efficiency)
            gguf_filename = variant_info['gguf_filename']

            logger.info(f"Using GGUF quantization: {variant_info['quantization']}", emoji='zap')
            logger.info(f"Downloading GGUF model: {gguf_filename} (~{variant_info['vram_gb']}GB)...", emoji='download')

            try:
                from huggingface_hub import hf_hub_download

                # Download GGUF transformer
                logger.info(f"Downloading GGUF transformer: {gguf_filename}", emoji='download')
                transformer_gguf_path = hf_hub_download(
                    repo_id=model_id,
                    filename=gguf_filename,
                )

                # Download GGUF text encoder (Gemma-3-12B Q2_K to save VRAM)
                logger.info(f"Downloading GGUF text encoder: gemma-3-12b-it-Q2_K.gguf (~4.4GB)", emoji='download')
                text_encoder_gguf_path = hf_hub_download(
                    repo_id="unsloth/gemma-3-12b-it-GGUF",
                    filename="gemma-3-12b-it-Q2_K.gguf",
                )

                # Load transformer with GGUF quantization (use LTX2VideoTransformer3DModel for LTX-2!)
                logger.info(f"Loading GGUF transformer...", emoji='robot')
                transformer = LTX2VideoTransformer3DModel.from_single_file(
                    transformer_gguf_path,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    config="Lightricks/LTX-2",  # Use base model config (not GGUF repo)
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                )

                # Load text encoder with 4-bit quantization (GGUF doesn't work with transformers)
                logger.info(f"Attempting to load text encoder with 4-bit quantization...", emoji='robot')
                text_encoder = None
                quantized_text_encoder_worked = False

                try:
                    import os
                    # Disable warmup to prevent OOM during quantization
                    os.environ["DISABLE_WARMUP"] = "1"

                    # Try 8-bit quantization (more stable than 4-bit)
                    logger.info(f"Attempting 8-bit quantization (saves ~2.5GB, more stable than NF4)...", emoji='zap')
                    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

                    # Create 8-bit quantization config (more compatible than 4-bit)
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                    )

                    logger.info(f"Loading text encoder with 8-bit quantization...", emoji='robot')
                    text_encoder = AutoModelForCausalLM.from_pretrained(
                        "Lightricks/LTX-2",
                        subfolder="text_encoder",
                        quantization_config=bnb_config,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )

                    logger.info(f"Text encoder quantized to 8-bit! (~2.5GB vs 5GB full)", emoji='check')
                    quantized_text_encoder_worked = True

                except Exception as text_enc_error:
                    logger.warning(f"8-bit quantization failed: {text_enc_error}", emoji='warning')
                    logger.warning(f"Using full precision text encoder (5GB) - may OOM on <17GB VRAM", emoji='warning')
                    text_encoder = None

                # Load rest of pipeline with quantized transformer
                # Only pass text_encoder if we successfully loaded and quantized it
                pipeline_kwargs = {
                    "transformer": transformer,
                    "torch_dtype": torch.bfloat16,
                }
                if quantized_text_encoder_worked:
                    pipeline_kwargs["text_encoder"] = text_encoder
                    logger.info(f"Using quantized text encoder in pipeline", emoji='check')

                # Check VRAM before loading pipeline (pipeline loads all components including text_encoder)
                free_before_pipeline = torch.cuda.mem_get_info()[0] / 1024**3
                transformer_vram = torch.cuda.memory_allocated() / 1024**3
                logger.debug(f"Free VRAM before pipeline load: {free_before_pipeline:.2f}GB (transformer: {transformer_vram:.2f}GB)")

                # Pipeline will load: text_encoder (5GB), VAE (2GB), + other components (~1GB)
                # Total needed: ~8GB
                if free_before_pipeline < 7.0 and not quantized_text_encoder_worked:
                    logger.error(f"Insufficient VRAM to load pipeline components", emoji='x')
                    logger.error(f"  Transformer already loaded: {transformer_vram:.2f}GB", emoji='info')
                    logger.error(f"  Free VRAM: {free_before_pipeline:.2f}GB", emoji='info')
                    logger.error(f"  Pipeline needs: ~8GB (text_encoder 5GB + VAE 2GB + overhead 1GB)", emoji='info')
                    logger.error(f"  Shortfall: {7.0 - free_before_pipeline:.2f}GB", emoji='x')
                    raise torch.OutOfMemoryError(
                        f"Cannot load pipeline: need 7GB, have {free_before_pipeline:.2f}GB"
                    )

                self.pipeline = LTX2ImageToVideoPipeline.from_pretrained(
                    "Lightricks/LTX-2",
                    **pipeline_kwargs
                )

                # CRITICAL: Cannot use CPU offload with GGUF due to metadata loss
                # GGUF tensors have quant_type metadata that becomes None when moved to meta device
                # This causes KeyError in GGML_QUANT_SIZES lookup
                # Must keep everything on GPU - requires smaller GGUF variant to fit in 14GB VRAM

                # Explicitly move all components to GPU (GGUF requires all on GPU)
                try:
                    self.pipeline.transformer.to('cuda')
                    transformer_vram = torch.cuda.memory_allocated() / 1024**3
                    logger.debug(f"Transformer on GPU: {transformer_vram:.2f}GB")

                    # Check VRAM before trying to load text encoder
                    free_before_text_enc = torch.cuda.mem_get_info()[0] / 1024**3
                    logger.debug(f"Free VRAM before text encoder: {free_before_text_enc:.2f}GB")

                    # Estimate text encoder size
                    text_enc_size = 2.5 if quantized_text_encoder_worked else 5.0
                    logger.debug(f"Text encoder needs ~{text_enc_size:.1f}GB, have {free_before_text_enc:.2f}GB")

                    if free_before_text_enc < text_enc_size:
                        raise torch.OutOfMemoryError(
                            f"Insufficient VRAM for text encoder: need {text_enc_size:.1f}GB, have {free_before_text_enc:.2f}GB"
                        )

                    self.pipeline.text_encoder.to('cuda')
                    logger.debug(f"Text encoder on GPU: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

                    self.pipeline.vae.to('cuda')
                    logger.debug(f"VAE on GPU: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

                    # Log VRAM usage to verify components fit
                    if torch.cuda.is_available():
                        allocated_gb = torch.cuda.memory_allocated() / 1024**3
                        reserved_gb = torch.cuda.memory_reserved() / 1024**3
                        logger.info(f"VRAM usage: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved", emoji='info')

                    logger.info("GGUF model loaded successfully (no CPU offload - incompatible with GGUF)", emoji='check')
                    logger.info("All pipeline components moved to GPU", emoji='zap')

                except torch.OutOfMemoryError as e:
                    # Cleanup failed components
                    if hasattr(self, 'pipeline') and self.pipeline is not None:
                        logger.debug("Cleaning up failed GGUF components...")
                        del self.pipeline
                        self.pipeline = None
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    # Calculate VRAM requirement based on variant and text encoder status
                    variant_name = self.variant
                    text_enc_size = "2GB (NF4)" if quantized_text_encoder_worked else "5GB (full precision)"

                    # Get transformer size from variant
                    transformer_sizes = {
                        "LTX-2-Q2_K-GGUF": "8GB",
                        "LTX-2-Q3_K_M-GGUF": "10GB",
                        "LTX-2-Q4_K_M-GGUF": "13GB",
                    }
                    transformer_size = transformer_sizes.get(variant_name, "8-13GB")

                    logger.error(f"Insufficient VRAM to load {variant_name} on GPU", emoji='x')
                    logger.error(f"  Transformer ({variant_name}): {transformer_size}", emoji='x')
                    logger.error(f"  Text encoder: {text_enc_size}", emoji='x')
                    logger.error(f"  VAE: 2GB", emoji='x')
                    logger.error(f"Available VRAM: {torch.cuda.mem_get_info()[0] / 1024**3:.1f}GB", emoji='x')
                    logger.error(f"GGUF cannot use CPU offload due to metadata incompatibility", emoji='x')

                    if not quantized_text_encoder_worked:
                        logger.error(f"Text encoder NF4 quantization failed - using full 5GB instead of 2GB", emoji='x')
                        logger.error(f"This requires 3GB more VRAM than expected!", emoji='warning')

                    logger.error(f"Recommended: Use Wan FLF2V instead", emoji='info')

                    # Re-raise as-is to prevent NF4 fallback (won't work either)
                    raise

            except Exception as e:
                import traceback

                # Don't fallback to NF4 if it's an OOM error (won't work either)
                if isinstance(e, torch.OutOfMemoryError):
                    logger.error(f"GGUF OOM - NF4 fallback skipped (would also OOM)", emoji='x')
                    raise  # Re-raise OOM to abort

                logger.error(f"GGUF loading failed: {e}", emoji='x')
                logger.debug(f"GGUF error traceback: {traceback.format_exc()}")

                # CRITICAL: Cleanup failed GGUF pipeline before NF4 fallback
                if hasattr(self, 'pipeline') and self.pipeline is not None:
                    logger.debug("Cleaning up failed GGUF pipeline components...")
                    del self.pipeline
                    self.pipeline = None

                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                vram_after_cleanup = torch.cuda.mem_get_info()[0] / 1024**3
                logger.debug(f"VRAM after GGUF cleanup: {vram_after_cleanup:.2f}GB available")

                logger.info("Falling back to BitsAndBytes NF4 quantization...", emoji='warning')
                # Fall through to NF4 loading below
                is_gguf = False  # Trigger fallback
                model_id = "Lightricks/LTX-2"  # Reset to base model for fallback

        if not is_gguf and self.variant == 'LTX-2-4K-NF4':
            # Load with 4-bit quantization for transformer AND text encoder
            logger.info("Using 4-bit NF4 quantization for transformer and text encoder (saves ~75% VRAM)", emoji='zap')
            try:
                from transformers import BitsAndBytesConfig
                from diffusers import PipelineQuantizationConfig

                # Disable warmup to prevent OOM during loading
                os.environ["DISABLE_WARMUP"] = "1"

                # Create BitsAndBytes config (use bfloat16 as recommended for LTX-2)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=None,  # Don't skip any modules
                )

                # Wrap in PipelineQuantizationConfig with quant_mapping
                # Quantize both transformer (video) and text_encoder (T5-XXL ~11GB)
                quantization_config = PipelineQuantizationConfig(
                    quant_mapping={
                        "transformer": bnb_config,
                        "text_encoder": bnb_config,  # Quantize T5-XXL too (11GB → ~3GB)
                    }
                )

                # Use low_cpu_mem_usage to reduce memory spikes during loading
                self.pipeline = LTX2ImageToVideoPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    quantization_config=quantization_config,
                    device_map="auto",  # Let accelerate handle device placement
                    low_cpu_mem_usage=True,  # Reduce memory during load
                    max_memory={0: "13GB"},  # Limit GPU memory usage
                )

            except (ImportError, Exception) as e:
                logger.error(f"Quantization failed with insufficient VRAM", emoji='x')
                logger.debug(f"Quantization error: {e}")
                logger.error(f"LTX-2-4K-NF4 requires successful quantization to fit in 14GB VRAM", emoji='x')
                logger.error(f"The full bfloat16 model requires 24GB+ VRAM", emoji='x')
                raise RuntimeError(
                    f"Insufficient VRAM for LTX-2. Quantization failed and full model won't fit. "
                    f"Available: {torch.cuda.mem_get_info()[0] / 1024**3:.1f}GB, Required: 24GB+ (or 10GB with working quantization). "
                    f"Try closing other programs to free VRAM, or use a different interpolation method (Wan FLF2V)."
                )
        elif not is_gguf:
            # Full precision or bfloat16 (LTX-2 recommends bfloat16, not fp16)
            # These variants require 24GB+ VRAM
            dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
            self.pipeline = LTX2ImageToVideoPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="cuda",  # Load to CUDA device
            )

        # Enable additional memory optimizations
        if self.device == 'cuda':
            # Enable VAE tiling to reduce memory usage during decode
            try:
                self.pipeline.vae.enable_tiling()
                logger.info("Enabled VAE tiling for memory efficiency", emoji='check')
            except:
                pass

        logger.info(f"LTX-2 pipeline loaded successfully", emoji='check')

    def generate_segment(
        self,
        start_image: Image.Image,
        audio_path: str = None,  # NOTE: LTX-2 generates audio, doesn't consume it
        audio_start_sec: float = 0.0,  # Kept for API compatibility
        audio_duration_sec: float = 0.0,  # Kept for API compatibility
        prompt: str = "",
        negative_prompt: str = "blurry, low quality, distorted",
        num_frames: int = 121,
        fps: float = 24.0,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 40,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Generate video segment using LTX-2 image-to-video.

        NOTE: LTX-2 GENERATES audio automatically - it doesn't use audio as input!
        The audio_* parameters are kept for API compatibility but are not used.

        Args:
            start_image: Starting keyframe image
            audio_path: NOT USED (LTX-2 generates audio)
            audio_start_sec: NOT USED (kept for compatibility)
            audio_duration_sec: NOT USED (kept for compatibility)
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            num_frames: Number of frames to generate (default 121)
            fps: Target FPS (float, default 24.0)
            guidance_scale: Guidance scale (default 4.0 for LTX-2)
            num_inference_steps: Number of denoising steps (default 40)
            seed: Random seed for reproducibility

        Returns:
            List of generated PIL Images (audio is discarded)
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")

        # Validate and adjust resolution
        width, height = start_image.size

        # LTX-2 supported resolutions (must be multiples of 32)
        # Common: 768x512, 1024x576, 1280x720, 1920x1080
        if width % 32 != 0 or height % 32 != 0:
            # Round to nearest multiple of 32
            width = (width // 32) * 32
            height = (height // 32) * 32
            logger.warning(f"Resizing image to {width}x{height} (LTX-2 requires multiples of 32)", emoji='warning')
            start_image = start_image.resize((width, height), Image.Resampling.LANCZOS)

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.debug(f"Generating {num_frames} frames with LTX-2 I2V")

        # Generate video (LTX-2 also generates audio, but we discard it)
        video, generated_audio = self.pipeline(
            image=start_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=fps,  # LTX-2 uses 'frame_rate' parameter
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            return_dict=False,  # Returns tuple (video, audio)
        )

        # Video is already a list of PIL Images
        return video

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
