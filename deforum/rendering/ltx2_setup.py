"""LTX-2 Pipeline Setup and Validation

Extracted from keyframe_interp.py for better separation of concerns.
Handles LTX-2-specific initialization, VRAM checking, and validation.
"""

from typing import Tuple
from deforum.utils.system.logging import get_logger

logger = get_logger()


def setup_ltx2_pipeline(args, video_args, wan_args, keyframes):
    """Initialize and validate LTX-2 pipeline with appropriate settings.

    Args:
        args: DeforumArgs (resolution, etc.)
        video_args: DeforumOutputArgs (audio settings)
        wan_args: DeforumWanArgs (ltx2_model_variant, ltx2_audio_mode)
        keyframes: List of keyframe numbers

    Returns:
        Tuple of (ltx2_pipeline, ltx2_variant, ltx2_audio_mode)

    Raises:
        RuntimeError: If VRAM insufficient or audio track missing
        ValueError: If resolution invalid for LTX-2
    """
    logger.info(f"Initializing LTX-2 Audio-Video pipeline...", emoji='video_camera')

    # Get LTX-2 model variant from args
    ltx2_variant = getattr(wan_args, 'ltx2_model_variant', 'Auto')
    ltx2_audio_mode = getattr(wan_args, 'ltx2_audio_mode', 'condition_only')

    logger.info(f"LTX-2 Configuration:", emoji='gear')
    logger.info(f"  Model Variant: {ltx2_variant}")
    logger.info(f"  Audio Mode: {ltx2_audio_mode}")

    # Check VRAM requirements
    import torch
    if torch.cuda.is_available():
        free_vram_gb = torch.cuda.mem_get_info()[0] / 1024**3
        logger.info(f"  Current free VRAM: {free_vram_gb:.1f}GB")

        # VRAM requirements for different variants
        # NEW: Text encoder (Gemma-3-12B, 24GB) offloaded to system RAM via CPU offload
        # Only transformer + VAE use VRAM
        vram_requirements = {
            'LTX-2-Q2_K-GGUF': 10.0,   # 8GB transformer + 0GB text_encoder (CPU offload) + 2GB VAE
            'LTX-2-Q3_K_M-GGUF': 12.0, # 10GB transformer + 0GB text_encoder (CPU offload) + 2GB VAE
            'LTX-2-Q4_K_M-GGUF': 15.0, # 13GB transformer + 0GB text_encoder (CPU offload) + 2GB VAE
            'LTX-2-4K': 24.0,           # Full precision (no offload)
        }

        # Auto-select variant based on VRAM if Auto
        # NOTE: Thresholds include 3-4GB headroom for intermediate tensors during generation
        if ltx2_variant == 'Auto':
            if free_vram_gb >= 24.0:
                ltx2_variant = 'LTX-2-4K'
                logger.info(f"  Auto-selected: LTX-2-4K (24GB+ VRAM available, full precision)")
            elif free_vram_gb >= 18.0:
                # 18GB+ - use Q4_K_M (needs 15GB + 3GB headroom)
                ltx2_variant = 'LTX-2-Q4_K_M-GGUF'
                logger.info(f"  Auto-selected: LTX-2-Q4_K_M-GGUF (18GB+ VRAM available)")
                logger.info(f"  Needs: 13GB transformer + 2GB VAE + 3GB headroom = 18GB VRAM")
                logger.info(f"  Text encoder (24GB) offloaded to system RAM ✓")
            elif free_vram_gb >= 14.0:
                # 14-18GB - use Q3_K_M (needs 12GB + 2-3GB headroom)
                ltx2_variant = 'LTX-2-Q3_K_M-GGUF'
                logger.info(f"  Auto-selected: LTX-2-Q3_K_M-GGUF (14GB+ VRAM available)")
                logger.info(f"  Needs: 10GB transformer + 2GB VAE + 2-3GB headroom = 14-15GB VRAM")
                logger.info(f"  Text encoder (24GB) offloaded to system RAM ✓")
            elif free_vram_gb >= 10.0:
                # 10-14GB VRAM - use Q2_K for maximum compatibility (includes 16GB cards)
                ltx2_variant = 'LTX-2-Q2_K-GGUF'
                logger.info(f"  Auto-selected: LTX-2-Q2_K-GGUF ({free_vram_gb:.1f}GB available)")
                logger.info(f"  Needs: 8GB transformer + 2GB VAE + 2-3GB headroom = 12-13GB VRAM")
                logger.info(f"  Text encoder (24GB) offloaded to system RAM ✓")
                logger.info(f"  Note: More conservative quantization ensures stable generation on 16GB cards")
            else:
                logger.error(f"Insufficient VRAM for LTX-2! Minimum 10GB required, found {free_vram_gb:.1f}GB", emoji='x')
                logger.error(f"Recommended: Use Wan FLF2V instead (works with 10GB+ VRAM)", emoji='info')
                raise RuntimeError(
                    f"Insufficient VRAM for LTX-2. Minimum 10GB required for Q2_K GGUF. "
                    f"Found {free_vram_gb:.1f}GB. Use Wan FLF2V instead."
                )

        # Check if selected variant fits in VRAM
        required_vram = vram_requirements.get(ltx2_variant, 24.0)
        if free_vram_gb < required_vram:
            logger.warning(f"Low VRAM: {free_vram_gb:.1f}GB free, {ltx2_variant} needs ~{required_vram:.0f}GB", emoji='warning')
            logger.warning(f"   Generation may fail or be very slow!")
            logger.warning(f"   Consider selecting a smaller variant or using Wan FLF2V instead")

    # Check for audio track (required for LTX-2)
    if not video_args.add_soundtrack or not video_args.soundtrack_path:
        raise RuntimeError(
            "LTX-2 requires an audio track for conditioning. "
            "Please enable 'Add Soundtrack' in the Output tab and provide an audio file."
        )

    # NOTE: Resolution validation happens before Phase 1 in keyframe_interp.py
    # If we reach here, resolution is already validated and fixed (if needed)

    # Cleanup any previous failed LTX-2 attempts (critical for resume mode)
    import torch
    import gc
    if torch.cuda.is_available():
        # Force Python garbage collection to release any LTX-2 pipeline references
        gc.collect()
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        vram_before_ltx = torch.cuda.mem_get_info()[0] / 1024**3
        logger.debug(f"VRAM available before LTX-2 load: {vram_before_ltx:.2f}GB")

    # Initialize LTX-2 pipeline
    from deforum.integrations.ltx2 import LTX2Pipeline

    ltx2_pipeline = LTX2Pipeline(device='cuda', variant=ltx2_variant)
    ltx2_pipeline.load_model()

    logger.info(f"LTX-2 pipeline ready", emoji='check')
    logger.info(f"  Resolution: {args.W}x{args.H} ✓", emoji='info')
    logger.info(f"  Video segments: {len(keyframes) - 1} (keyframes - 1)", emoji='info')
    logger.info(f"  Audio mode: {ltx2_audio_mode} (audio drives motion)", emoji='sound')

    return ltx2_pipeline, ltx2_variant, ltx2_audio_mode
