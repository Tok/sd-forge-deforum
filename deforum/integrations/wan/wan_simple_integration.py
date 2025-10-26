#!/usr/bin/env python3
"""
Wan Simple Integration with Styled Progress
Updated to use render core styling for progress indicators
"""

from pathlib import Path
from typing import List, Dict, Optional
import torch  # type: ignore
import os
import numpy as np
import time
import sys
import json
from decimal import Decimal

# Import our new styled progress utilities
from .utils.wan_progress_utils import (
    WanModelLoadingContext, WanGenerationContext,
    print_wan_info, print_wan_success, print_wan_warning, print_wan_error, print_wan_progress,
    create_wan_clip_progress, create_wan_frame_progress, create_wan_inference_progress
)

# Import Deforum utilities for settings and audio handling
from deforum.media.video_audio_utilities import download_audio
from deforum.media.subtitle_handler import init_srt_file, write_frame_subtitle, calculate_frame_duration
from deforum.config.settings import save_settings_from_animation_run
from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()


class WanSimpleIntegration:
    """Simplified Wan integration with auto-discovery and proper progress styling"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.models = []
        self.pipeline = None
        self.model_size = None
        self.optimal_width = 1280  # Wan 2.2 TI2V default: 720p landscape (divisible by 32)
        self.optimal_height = 736  # Aligned to 32 for Wan 2.2 (VAE scale=16 * patch=2)
        self.flash_attention_mode = "auto"  # auto, enabled, disabled
        print_wan_info(f"Simple Integration initialized on {self.device}")
    
    def discover_models(self) -> List[Dict]:
        """Discover available Wan models with styled progress"""
        models = []
        search_paths = [
            Path("models/Deforum/wan"),
            Path("models/Wan"),
            Path("models"),
            Path("../models/Deforum/wan"),
            Path("../models/Wan"),
        ]
        
        print_wan_progress("Discovering Wan models...")
        
        for search_path in search_paths:
            if search_path.exists():
                print_wan_info(f"🔍 Searching: {search_path}")
                
                for model_dir in search_path.iterdir():
                    # Check for Wan 2.2, Wan 2.1, or generic wan directories
                    if model_dir.is_dir() and model_dir.name.lower().startswith(('wan2.2', 'wan2.1', 'wan-', 'wan_')):
                        model_info = self._analyze_model_directory(model_dir)
                        if model_info:
                            models.append(model_info)
                            print_wan_success(f"Found: {model_info['name']} ({model_info['type']}, {model_info['size']})")
        
        if not models:
            print_wan_warning("No Wan models found in search paths")
        else:
            print_wan_success(f"Discovery complete - found {len(models)} model(s)")
            
        return models
    
    def _analyze_model_directory(self, model_dir: Path) -> Optional[Dict]:
        """Analyze a model directory and return model info if valid"""
        if not model_dir.is_dir():
            return None
            
        # Check if this looks like a Wan model
        model_name = model_dir.name.lower()
        if 'wan' not in model_name and not any(file.name.startswith('wan') for file in model_dir.rglob('*') if file.is_file()):
            return None
        
        # Check for required model files
        if not self._has_required_files(model_dir):
            return None
        
        # Determine model type and size (Wan 2.2 first, then 2.1)
        model_type = "Unknown"
        model_size = "Unknown"

        # Detect type (prioritize Wan 2.2 models and FLF2V)
        if 'flf2v' in model_name:
            model_type = "FLF2V"  # First-Last-Frame-to-Video (interpolation)
        elif 'ti2v' in model_name:
            model_type = "TI2V"  # Text+Image-to-Video (cannot do FLF2V!)
        elif 's2v' in model_name:
            model_type = "S2V"
        elif 'animate' in model_name:
            model_type = "Animate"
        elif 't2v' in model_name:
            model_type = "T2V"
        elif 'i2v' in model_name:
            model_type = "I2V"

        # Detect size (Wan 2.2 and 2.1)
        if '5b' in model_name or '5_b' in model_name:
            model_size = "5B"
        elif 'a14b' in model_name or 'a_14b' in model_name:
            model_size = "A14B"
        elif '1.3b' in model_name:
            model_size = "1.3B"
        elif '14b' in model_name:
            model_size = "14B"

        # Detect quantization
        quantization = "FP16"  # Default
        if 'fp8' in model_name or 'e4m3fn' in model_name or 'e5m2' in model_name:
            quantization = "FP8"
        elif 'gguf' in model_name or 'q8' in model_name or 'q5' in model_name or 'q4' in model_name:
            quantization = "GGUF"
        elif 'bf16' in model_name:
            quantization = "BF16"

        return {
            'name': model_dir.name,
            'path': str(model_dir.absolute()),
            'type': model_type,
            'size': model_size,
            'quantization': quantization,
            'directory': model_dir
        }
    
    def _has_required_files(self, model_dir: Path) -> bool:
        """Check if model directory has required files"""
        required_files = [
            "config.json",
            "model_index.json"
        ]
        
        # Check for model weight files
        has_weights = any(
            file.suffix in ['.safetensors', '.bin', '.pt', '.pth']
            for file in model_dir.rglob('*')
            if file.is_file()
        )
        
        has_config = any(
            (model_dir / req_file).exists()
            for req_file in required_files
        )
        
        return has_weights and has_config
    
    
    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model for T2V/I2V generation (excludes FLF2V models)"""
        if not self.models:
            self.models = self.discover_models()

        if not self.models:
            return None

        # Filter out FLF2V models - they can ONLY interpolate, NOT generate from text/image
        usable_models = [m for m in self.models if m['type'] != 'FLF2V']

        if not usable_models:
            print_wan_warning("Only FLF2V models found - these cannot do T2V/I2V generation!")
            print_wan_info("FLF2V models are for interpolation only. Please download a T2V/I2V/TI2V model.")
            return None

        # Priority: TI2V > T2V > I2V (Wan 2.2 preferred), 5B > 1.3B > 14B > A14B, FP8 > GGUF > FP16 (VRAM efficient)
        def model_priority(model):
            type_priority = {'TI2V': 0, 'T2V': 1, 'I2V': 2, 'S2V': 3, 'Animate': 4, 'Unknown': 5}
            size_priority = {'5B': 0, '1.3B': 1, '14B': 2, 'A14B': 3, 'Unknown': 4}
            quant_priority = {'FP8': 0, 'GGUF': 1, 'FP16': 2, 'BF16': 3}  # FP8 preferred for VRAM efficiency
            return (
                type_priority.get(model['type'], 5),
                size_priority.get(model['size'], 4),
                quant_priority.get(model.get('quantization', 'FP16'), 2)
            )

        best_model = min(usable_models, key=model_priority)
        quantization_info = best_model.get('quantization', 'Unknown')
        logger.info(f"🎯 Best model selected: {best_model['name']} ({best_model['type']}, {best_model['size']}, {quantization_info})")
        return best_model
    
    def load_simple_wan_pipeline(self, model_info: Dict, wan_args=None) -> bool:
        """Load Wan pipeline with styled progress indicators"""
        model_name = model_info['name']

        with WanModelLoadingContext(model_name) as progress:
            try:
                progress.update(10, "Initializing...")
                progress.update(30, "Loading model...")
                success = self._load_standard_wan_model(model_info, wan_args)

                if success:
                    progress.update(80, "Configuring...")
                    # Configure Flash Attention if requested
                    if wan_args and hasattr(wan_args, 'wan_flash_attention'):
                        self.flash_attention_mode = wan_args.wan_flash_attention
                        print_wan_info(f"Flash Attention mode: {self.flash_attention_mode}")

                    progress.update(100, "Complete!")
                    return True
                else:
                    print_wan_error(f"Failed to load pipeline for {model_name}")
                    return False

            except Exception as e:
                print_wan_error(f"Model loading failed: {e}")
                return False
    
    def _load_standard_wan_model(self, model_info: Dict, wan_args=None) -> bool:
        """Load standard T2V/I2V Wan model"""
        try:
            # Strategy 1: Try official Wan implementation
            extension_root = Path(__file__).parent.parent.parent.parent
            wan_repo_path = extension_root / "Wan2.1"
            
            if wan_repo_path.exists() and (wan_repo_path / "wan").exists():
                logger.info(f"Trying official Wan implementation from: {wan_repo_path}", emoji='wrench')
                
                if str(wan_repo_path) not in sys.path:
                    sys.path.insert(0, str(wan_repo_path))
                
                try:
                    import wan  # type: ignore
                    from wan.text2video import WanT2V  # type: ignore
                    
                    # Apply Flash Attention patches AFTER Wan modules are imported
                    try:
                        from .wan_flash_attention_patch import apply_flash_attention_patch, update_patched_flash_attention_mode
                        
                        # Update mode if stored
                        if hasattr(self, 'flash_attention_mode'):
                            update_patched_flash_attention_mode(self.flash_attention_mode)
                        
                        success = apply_flash_attention_patch()
                        if success:
                            logger.info(f"{emoji_if_enabled('✅')} Flash Attention monkey patch applied successfully")
                        else:
                            logger.error("⚠️ Flash Attention patch could not be applied - may be already patched")
                    except Exception as patch_e:
                        logger.error(f"⚠️ Flash Attention patch failed: {patch_e}")
                        logger.info("Continuing without patches...", emoji='refresh')
                    
                    logger.info("🚀 Loading with official Wan T2V...")
                    
                    # Create minimal config
                    class MinimalConfig:
                        def __init__(self):
                            self.model = type('obj', (object,), {
                                'num_attention_heads': 32,
                                'attention_head_dim': 128,
                                'in_channels': 4,
                                'out_channels': 4,
                                'num_layers': 28,
                                'sample_size': 32,
                                'patch_size': 2,
                            })
                    
                    config = MinimalConfig()
                    
                    t2v_model = WanT2V(
                        config=config,
                        checkpoint_dir=model_info['path'],
                        device_id=0,
                        rank=0,
                        dit_fsdp=False,
                        t5_fsdp=False
                    )
                    
                    # Create wrapper
                    class WanWrapper:
                        def __init__(self, t2v_model):
                            self.t2v_model = t2v_model
                        
                        def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                            # Ensure dimensions are aligned
                            aligned_width = ((width + 15) // 16) * 16
                            aligned_height = ((height + 15) // 16) * 16
                            
                            if aligned_width != width or aligned_height != height:
                                logger.info(f"Dimension alignment: {width}x{height} -> {aligned_width}x{aligned_height}", emoji='wrench')
                            
                            return self.t2v_model.generate(
                                input_prompt=prompt,
                                size=(aligned_width, aligned_height),
                                frame_num=num_frames,
                                sampling_steps=num_inference_steps,
                                guide_scale=guidance_scale,
                                shift=5.0,
                                sample_solver='unipc',
                                offload_model=True,
                                **kwargs
                            )
                        
                        def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                            # For I2V, enhance prompt for continuity
                            enhanced_prompt = f"Starting from the given image, {prompt}. Maintain visual continuity."
                            return self.__call__(enhanced_prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs)
                    
                    self.pipeline = WanWrapper(t2v_model)
                    logger.info(f"{emoji_if_enabled('✅')} Official Wan model loaded successfully")
                    return True
                    
                except Exception as wan_e:
                    logger.error(f"⚠️ Official Wan loading failed: {wan_e}")
                    logger.info("Trying diffusers fallback...", emoji='refresh')
            
            # Strategy 2: Try diffusers fallback
            try:
                # Check if this is a Wan 2.2 Diffusers model
                is_wan22_diffusers = (
                    'TI2V' in model_info['type'] or
                    '5B' in model_info['size'] or
                    'A14B' in model_info['size'] or
                    'wan2.2' in model_info['name'].lower()
                )

                # Initialize flags for memory optimization
                use_aggressive_offload = False

                # Initialize variables before conditionals (prevents UnboundLocalError for Wan 2.1)
                has_i2v_pipeline = False
                WanImageToVideoPipeline = None
                is_5b_model = False
                is_flf2v_model = False
                model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                vae_dtype = model_dtype
                vae = None

                if is_wan22_diffusers:
                    logger.info("Loading Wan 2.2 Diffusers pipeline...", emoji='refresh')

                    # Apply compatibility patches BEFORE importing diffusers
                    # This ensures patches are active even if diffusers was imported elsewhere
                    try:
                        from deforum.integrations.flux_controlnet.diffusers_compat import apply_all_patches
                        logger.info("Applying diffusers compatibility patches before pipeline load...", emoji='wrench')
                        apply_all_patches()
                    except Exception as patch_e:
                        logger.error(f"Compatibility patches failed: {patch_e}")
                        import traceback
                        traceback.print_exc()

                    from diffusers import WanPipeline, AutoencoderKLWan  # type: ignore

                    # Also try to import I2V pipeline for chaining support
                    try:
                        from diffusers import WanImageToVideoPipeline  # type: ignore
                        has_i2v_pipeline = True
                        print_wan_info("✅ WanImageToVideoPipeline available for I2V chaining")
                    except ImportError:
                        WanImageToVideoPipeline = None
                        has_i2v_pipeline = False
                        print_wan_info("ℹ️ WanImageToVideoPipeline not available, will use latent-based I2V")

                    # Enable aggressive offload for TI2V-5B models (16GB VRAM optimization)
                    # A14B models are too large and need >24GB VRAM anyway
                    is_5b_model = '5B' in model_info['size'] or '5b' in model_info['name'].lower()

                    if is_5b_model:
                        print_wan_info("🔍 TI2V-5B model detected - enabling aggressive VRAM optimizations for 16GB GPUs")
                        model_dtype = torch.float16
                        vae_dtype = torch.float16  # FP16 VAE for lower VRAM
                        use_aggressive_offload = True
                    else:
                        print_wan_info("🔍 Large model detected (A14B) - standard precision")
                        model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                        vae_dtype = torch.float32  # Standard VAE needs float32 for stability
                        use_aggressive_offload = False

                    # Load VAE separately for better compatibility (keep on CPU initially)
                    print_wan_info(f"Loading VAE with dtype={vae_dtype}")
                    vae = AutoencoderKLWan.from_pretrained(
                        model_info['path'],
                        subfolder="vae",
                        torch_dtype=vae_dtype,
                        low_cpu_mem_usage=True  # Reduce CPU RAM usage during load
                    )

                    # Load main pipeline (keep on CPU initially)
                    print_wan_info(f"Loading pipeline with dtype={model_dtype}")
                    pipeline = WanPipeline.from_pretrained(
                        model_info['path'],
                        vae=vae,
                        torch_dtype=model_dtype,
                        low_cpu_mem_usage=True,  # Reduce CPU RAM usage during load
                        use_safetensors=True
                    )

                    if is_5b_model:
                        print_wan_success("✅ Loaded Wan 2.2 TI2V-5B pipeline with VRAM optimizations")
                    else:
                        print_wan_success("✅ Loaded Wan 2.2 pipeline")
                else:
                    # Wan 2.1 models - check if it's FLF2V
                    is_flf2v_model = 'FLF2V' in model_info['type'] or 'flf2v' in model_info['name'].lower()

                    if is_flf2v_model:
                        # FLF2V models MUST use WanImageToVideoPipeline
                        logger.info("Loading Wan 2.1 FLF2V model with WanImageToVideoPipeline...", emoji='refresh')

                        try:
                            from diffusers import WanImageToVideoPipeline as _WanImageToVideoPipeline  # type: ignore
                            has_i2v_pipeline = True
                            WanImageToVideoPipeline = _WanImageToVideoPipeline  # Store class for later checks

                            pipeline = _WanImageToVideoPipeline.from_pretrained(
                                model_info['path'],
                                torch_dtype=model_dtype,
                                low_cpu_mem_usage=True,
                                use_safetensors=True
                            )
                            print_wan_success("✅ Loaded Wan 2.1 FLF2V pipeline (WanImageToVideoPipeline)")
                        except ImportError:
                            raise RuntimeError("WanImageToVideoPipeline not available in diffusers. Update diffusers to support FLF2V models.")
                    else:
                        # Fallback to generic DiffusionPipeline for other Wan 2.1 models
                        logger.info("Loading with generic DiffusionPipeline...", emoji='refresh')
                        from diffusers import DiffusionPipeline  # type: ignore

                        pipeline = DiffusionPipeline.from_pretrained(
                            model_info['path'],
                            torch_dtype=model_dtype,
                            use_safetensors=True
                        )

                # Enable memory optimizations based on model type
                if torch.cuda.is_available():
                    if is_wan22_diffusers and use_aggressive_offload:
                        print_wan_info("🔧 Enabling aggressive VRAM optimizations for 16GB VRAM...")

                        # Sequential CPU offload (more aggressive than model CPU offload)
                        try:
                            pipeline.enable_sequential_cpu_offload()
                            print_wan_success("✅ Sequential CPU offload enabled")
                        except:
                            # Fallback to model CPU offload
                            pipeline.enable_model_cpu_offload()
                            print_wan_success("✅ Model CPU offload enabled (fallback)")

                        # Enable attention slicing for lower VRAM
                        try:
                            pipeline.enable_attention_slicing(slice_size="auto")
                            print_wan_success("✅ Attention slicing enabled")
                        except Exception as e:
                            print_wan_warning(f"Attention slicing not available: {e}")

                        # Enable VAE tiling for lower VRAM
                        try:
                            if hasattr(pipeline, 'enable_vae_tiling'):
                                pipeline.enable_vae_tiling()
                                print_wan_success("✅ VAE tiling enabled")
                        except Exception as e:
                            print_wan_warning(f"VAE tiling not available: {e}")

                        # Enable VAE slicing
                        try:
                            if hasattr(pipeline, 'enable_vae_slicing'):
                                pipeline.enable_vae_slicing()
                                print_wan_success("✅ VAE slicing enabled")
                        except Exception as e:
                            print_wan_warning(f"VAE slicing not available: {e}")

                        print_wan_success("✅ All VRAM optimizations enabled - should work with <16GB VRAM")

                    elif is_flf2v_model:
                        # FLF2V 14B models need aggressive VRAM optimizations
                        print_wan_info("🔧 Enabling aggressive VRAM optimizations for FLF2V 14B model...")

                        # Sequential CPU offload
                        try:
                            pipeline.enable_sequential_cpu_offload()
                            print_wan_success("✅ Sequential CPU offload enabled")
                        except:
                            pipeline.enable_model_cpu_offload()
                            print_wan_success("✅ Model CPU offload enabled (fallback)")

                        # Enable attention slicing
                        try:
                            pipeline.enable_attention_slicing(slice_size="auto")
                            print_wan_success("✅ Attention slicing enabled")
                        except Exception as e:
                            print_wan_warning(f"Attention slicing not available: {e}")

                        # Enable VAE tiling for lower VRAM
                        try:
                            if hasattr(pipeline, 'enable_vae_tiling'):
                                pipeline.enable_vae_tiling()
                                print_wan_success("✅ VAE tiling enabled")
                        except Exception as e:
                            print_wan_warning(f"VAE tiling not available: {e}")

                        # Enable VAE slicing
                        try:
                            if hasattr(pipeline, 'enable_vae_slicing'):
                                pipeline.enable_vae_slicing()
                                print_wan_success("✅ VAE slicing enabled")
                        except Exception as e:
                            print_wan_warning(f"VAE slicing not available: {e}")

                        print_wan_success("✅ All VRAM optimizations enabled for FLF2V")
                    else:
                        # Standard CPU offload for other models
                        print_wan_info("🔧 Enabling model CPU offload to save VRAM...")
                        pipeline.enable_model_cpu_offload()
                        print_wan_success("✅ CPU offload enabled - model will stream to GPU during inference")

                    # User-controlled VRAM optimizations (from UI settings)
                    if wan_args:
                        # T5 CPU Offload
                        if hasattr(wan_args, 'wan_t5_cpu_offload') and wan_args.wan_t5_cpu_offload:
                            try:
                                if hasattr(pipeline, 'text_encoder'):
                                    print_wan_info("🔧 Enabling T5 CPU offload (user setting)...")
                                    pipeline.text_encoder.to('cpu')
                                    print_wan_success("✅ T5 text encoder moved to CPU - saves ~3-4GB VRAM")
                                elif hasattr(pipeline, 'text_encoder_2'):
                                    print_wan_info("🔧 Enabling T5 CPU offload for text_encoder_2...")
                                    pipeline.text_encoder_2.to('cpu')
                                    print_wan_success("✅ T5 text encoder moved to CPU - saves ~3-4GB VRAM")
                            except Exception as e:
                                print_wan_warning(f"T5 CPU offload failed: {e}")

                        # Gradient Checkpointing
                        if hasattr(wan_args, 'wan_gradient_checkpointing') and wan_args.wan_gradient_checkpointing:
                            try:
                                print_wan_info("🔧 Enabling gradient checkpointing (user setting)...")
                                if hasattr(pipeline, 'transformer') and hasattr(pipeline.transformer, 'enable_gradient_checkpointing'):
                                    pipeline.transformer.enable_gradient_checkpointing()
                                    print_wan_success("✅ Gradient checkpointing enabled - saves ~2-3GB VRAM (~15-20% slower)")
                                elif hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'enable_gradient_checkpointing'):
                                    pipeline.unet.enable_gradient_checkpointing()
                                    print_wan_success("✅ Gradient checkpointing enabled - saves ~2-3GB VRAM (~15-20% slower)")
                                else:
                                    print_wan_warning("Gradient checkpointing not supported by this model architecture")
                            except Exception as e:
                                print_wan_warning(f"Gradient checkpointing failed: {e}")

                # Load I2V pipeline if available and model supports I2V
                # Check if model has I2V support by checking for transformer config or model type
                model_supports_i2v = False
                try:
                    import json
                    config_path = Path(model_info['path']) / "model_index.json"
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            model_config = json.load(f)
                            # Check if the model lists an image-to-video capable transformer
                            # TI2V models should support both T2V and I2V
                            model_type = model_config.get('_class_name', '')
                            if 'ImageToVideo' in model_type or 'TI2V' in model_info['type']:
                                model_supports_i2v = True
                                print_wan_info(f"✅ Model supports I2V: {model_type}")
                            else:
                                print_wan_info(f"ℹ️ Model is T2V only: {model_type}")
                except Exception as e:
                    # If we can't determine, assume TI2V models support I2V
                    if 'TI2V' in model_info['type'] or 'I2V' in model_info['type']:
                        model_supports_i2v = True
                        print_wan_info(f"ℹ️ Model type suggests I2V support: {model_info['type']}")
                
                i2v_pipeline = None

                # For FLF2V models, the pipeline itself IS the I2V pipeline
                if is_flf2v_model and has_i2v_pipeline:
                    i2v_pipeline = pipeline
                    print_wan_info("✅ FLF2V model: Using main pipeline as I2V pipeline (same instance)")
                    model_supports_i2v = True  # FLF2V models support I2V by definition
                elif has_i2v_pipeline and WanImageToVideoPipeline is not None and model_supports_i2v:
                    try:
                        print_wan_info("🔧 Loading WanImageToVideoPipeline for I2V chaining...")
                        # Build kwargs conditionally - only pass vae if it was loaded
                        i2v_kwargs = {
                            'torch_dtype': model_dtype,
                            'low_cpu_mem_usage': True,
                            'use_safetensors': True
                        }
                        if vae is not None:
                            i2v_kwargs['vae'] = vae

                        i2v_pipeline = WanImageToVideoPipeline.from_pretrained(
                            model_info['path'],
                            **i2v_kwargs
                        )
                        
                        # Apply same memory optimizations to I2V pipeline
                        if torch.cuda.is_available():
                            if is_wan22_diffusers and use_aggressive_offload:
                                try:
                                    i2v_pipeline.enable_sequential_cpu_offload()
                                    print_wan_success("✅ I2V pipeline: Sequential CPU offload enabled")
                                except:
                                    i2v_pipeline.enable_model_cpu_offload()
                                    print_wan_success("✅ I2V pipeline: Model CPU offload enabled")
                                
                                try:
                                    i2v_pipeline.enable_attention_slicing(slice_size="auto")
                                    print_wan_success("✅ I2V pipeline: Attention slicing enabled")
                                except:
                                    pass
                                
                                try:
                                    if hasattr(i2v_pipeline, 'enable_vae_tiling'):
                                        i2v_pipeline.enable_vae_tiling()
                                        print_wan_success("✅ I2V pipeline: VAE tiling enabled")
                                except:
                                    pass
                                
                                try:
                                    if hasattr(i2v_pipeline, 'enable_vae_slicing'):
                                        i2v_pipeline.enable_vae_slicing()
                                        print_wan_success("✅ I2V pipeline: VAE slicing enabled")
                                except:
                                    pass
                            else:
                                i2v_pipeline.enable_model_cpu_offload()
                                print_wan_success("✅ I2V pipeline: CPU offload enabled")

                            # User-controlled VRAM optimizations for I2V pipeline
                            if wan_args:
                                # T5 CPU Offload
                                if hasattr(wan_args, 'wan_t5_cpu_offload') and wan_args.wan_t5_cpu_offload:
                                    try:
                                        if hasattr(i2v_pipeline, 'text_encoder'):
                                            print_wan_info("🔧 I2V pipeline: Enabling T5 CPU offload...")
                                            i2v_pipeline.text_encoder.to('cpu')
                                            print_wan_success("✅ I2V pipeline: T5 text encoder on CPU")
                                        elif hasattr(i2v_pipeline, 'text_encoder_2'):
                                            i2v_pipeline.text_encoder_2.to('cpu')
                                            print_wan_success("✅ I2V pipeline: T5 text encoder on CPU")
                                    except Exception as e:
                                        print_wan_warning(f"I2V pipeline T5 CPU offload failed: {e}")

                                # Gradient Checkpointing
                                if hasattr(wan_args, 'wan_gradient_checkpointing') and wan_args.wan_gradient_checkpointing:
                                    try:
                                        print_wan_info("🔧 I2V pipeline: Enabling gradient checkpointing...")
                                        if hasattr(i2v_pipeline, 'transformer') and hasattr(i2v_pipeline.transformer, 'enable_gradient_checkpointing'):
                                            i2v_pipeline.transformer.enable_gradient_checkpointing()
                                            print_wan_success("✅ I2V pipeline: Gradient checkpointing enabled")
                                        elif hasattr(i2v_pipeline, 'unet') and hasattr(i2v_pipeline.unet, 'enable_gradient_checkpointing'):
                                            i2v_pipeline.unet.enable_gradient_checkpointing()
                                            print_wan_success("✅ I2V pipeline: Gradient checkpointing enabled")
                                    except Exception as e:
                                        print_wan_warning(f"I2V pipeline gradient checkpointing failed: {e}")

                        print_wan_success("✅ WanImageToVideoPipeline loaded successfully for I2V chaining")
                    except Exception as i2v_e:
                        print_wan_warning(f"⚠️ Failed to load I2V pipeline: {i2v_e}")
                        print_wan_warning("⚠️ Will fall back to latent-based I2V conditioning")
                        i2v_pipeline = None
                
                # Create wrapper for diffusers pipeline
                class DiffusersWrapper:
                    def __init__(self, t2v_pipeline, i2v_pipeline=None):
                        self.pipeline = t2v_pipeline
                        self.i2v_pipeline = i2v_pipeline
                        self.vae = t2v_pipeline.vae if hasattr(t2v_pipeline, 'vae') else None
                    
                    def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                        # Wan 2.2 requires dimensions divisible by 32 (VAE spatial_scale=16 * transformer patch_size=2)
                        aligned_width = ((width + 31) // 32) * 32
                        aligned_height = ((height + 31) // 32) * 32

                        # WanPipeline requires specific parameter names
                        import inspect
                        pipeline_signature = inspect.signature(self.pipeline.__call__)

                        print_wan_info(f"🔍 Pipeline parameters available: {list(pipeline_signature.parameters.keys())}")

                        generation_kwargs = {
                            "prompt": prompt,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                        }

                        # Add video-specific parameters based on pipeline signature
                        if 'height' in pipeline_signature.parameters:
                            generation_kwargs['height'] = aligned_height
                        if 'width' in pipeline_signature.parameters:
                            generation_kwargs['width'] = aligned_width
                        if 'num_frames' in pipeline_signature.parameters:
                            generation_kwargs['num_frames'] = num_frames
                        elif 'video_length' in pipeline_signature.parameters:
                            generation_kwargs['video_length'] = num_frames
                        elif 'num_video_frames' in pipeline_signature.parameters:
                            generation_kwargs['num_video_frames'] = num_frames

                        print_wan_info(f"🎬 Generation parameters:")
                        print_wan_info(f"   Resolution: {aligned_width}x{aligned_height}")
                        print_wan_info(f"   Frames: {num_frames}")
                        print_wan_info(f"   Steps: {num_inference_steps}")
                        print_wan_info(f"   Guidance: {guidance_scale}")
                        print_wan_info(f"   Full kwargs: {generation_kwargs}")

                        with torch.no_grad():
                            return self.pipeline(**generation_kwargs)
                    
                    def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, strength=0.8, **kwargs):
                        """
                        Image-to-Video generation with proper image conditioning

                        Args:
                            image: PIL Image to use as first frame/conditioning
                            prompt: Text prompt for video generation
                            strength: Image conditioning strength (0.0-1.0)
                                     1.0 = maximum continuity from image
                                     0.0 = ignore image, pure T2V
                        
                        Note on prompts:
                            When we have image conditioning (via image parameter or latents),
                            we use the ORIGINAL prompt without modifications. The image provides
                            the visual continuity, while the prompt guides the evolution.
                            Only add continuity language when we have NO image conditioning.
                        """
                        # Wan 2.2 requires dimensions divisible by 32
                        aligned_width = ((width + 31) // 32) * 32
                        aligned_height = ((height + 31) // 32) * 32

                        # Use dedicated I2V pipeline if available
                        if self.i2v_pipeline is not None:
                            print_wan_info("✅ Using dedicated WanImageToVideoPipeline for I2V")
                            import inspect
                            pipeline_signature = inspect.signature(self.i2v_pipeline.__call__)
                            print_wan_info(f"🔍 I2V Pipeline parameters: {list(pipeline_signature.parameters.keys())}")
                            
                            # Use original prompt - image provides continuity, prompt guides evolution
                            generation_kwargs = {
                                "prompt": prompt,  # Original prompt, no modifications
                                "num_inference_steps": num_inference_steps,
                                "guidance_scale": guidance_scale,
                            }
                            
                            # Add image parameter (standard for I2V pipelines)
                            if 'image' in pipeline_signature.parameters:
                                generation_kwargs['image'] = image
                                print_wan_info(f"✅ I2V conditioning: Using 'image' parameter")
                                print_wan_info(f"📝 Using original prompt (image handles continuity, prompt guides changes)")
                            
                            # Check if strength parameter is supported and if we should use I2V
                            use_i2v_pipeline = True
                            
                            if 'strength' not in pipeline_signature.parameters:
                                # I2V pipeline doesn't support strength parameter
                                if strength >= 0.70:
                                    # High strength - I2V pipeline is naturally good at continuity
                                    print_wan_info(f"ℹ️ I2V pipeline lacks strength control, but strength is high ({strength:.2f})")
                                    print_wan_info(f"   → Using I2V as-is (naturally high continuity)")
                                else:
                                    # Low/medium strength - I2V is too sticky, use T2V with latents instead
                                    print_wan_info(f"⚠️ I2V pipeline too sticky for strength {strength:.2f}, falling back to T2V with latent control")
                                    use_i2v_pipeline = False
                            else:
                                # Pipeline supports strength - use it!
                                generation_kwargs['strength'] = strength
                                print_wan_info(f"✅ Strength parameter supported: {strength:.2f}")
                            
                            # Only use I2V pipeline if we decided to
                            if use_i2v_pipeline:
                                # Add resolution parameters
                                if 'height' in pipeline_signature.parameters:
                                    generation_kwargs['height'] = aligned_height
                                if 'width' in pipeline_signature.parameters:
                                    generation_kwargs['width'] = aligned_width
                                if 'num_frames' in pipeline_signature.parameters:
                                    generation_kwargs['num_frames'] = num_frames
                                elif 'video_length' in pipeline_signature.parameters:
                                    generation_kwargs['video_length'] = num_frames
                                
                                print_wan_info(f"🎬 I2V Generation with dedicated pipeline:")
                                print_wan_info(f"   Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
                                print_wan_info(f"   Resolution: {aligned_width}x{aligned_height}")
                                print_wan_info(f"   Frames: {num_frames}")
                                print_wan_info(f"   I2V Strength: {strength:.2f}")
                                print_wan_info(f"   CFG: {guidance_scale}")
                                
                                with torch.no_grad():
                                    return self.i2v_pipeline(**generation_kwargs)
                        
                        # Fall back to T2V pipeline with I2V conditioning
                        print_wan_info("ℹ️ Using T2V pipeline with I2V conditioning fallback")
                        import inspect
                        pipeline_signature = inspect.signature(self.pipeline.__call__)

                        # Log available parameters for debugging
                        print_wan_info(f"🔍 T2V Pipeline parameters: {list(pipeline_signature.parameters.keys())}")

                        # Start with original prompt - we'll only enhance it if NO image conditioning works
                        generation_kwargs = {
                            "prompt": prompt,  # Start with original prompt
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                        }

                        # Try different I2V parameter names
                        i2v_param_added = False

                        # Check for common I2V parameter names
                        if 'image' in pipeline_signature.parameters:
                            generation_kwargs['image'] = image
                            print_wan_info(f"✅ I2V conditioning: Using 'image' parameter with strength {strength:.2f}")
                            print_wan_info(f"📝 Using original prompt (image handles continuity)")
                            i2v_param_added = True
                        elif 'init_image' in pipeline_signature.parameters:
                            generation_kwargs['init_image'] = image
                            print_wan_info(f"✅ I2V conditioning: Using 'init_image' parameter with strength {strength:.2f}")
                            print_wan_info(f"📝 Using original prompt (image handles continuity)")
                            i2v_param_added = True
                        elif 'input_image' in pipeline_signature.parameters:
                            generation_kwargs['input_image'] = image
                            print_wan_info(f"✅ I2V conditioning: Using 'input_image' parameter with strength {strength:.2f}")
                            print_wan_info(f"📝 Using original prompt (image handles continuity)")
                            i2v_param_added = True
                        elif 'conditioning_image' in pipeline_signature.parameters:
                            generation_kwargs['conditioning_image'] = image
                            print_wan_info(f"✅ I2V conditioning: Using 'conditioning_image' parameter with strength {strength:.2f}")
                            print_wan_info(f"📝 Using original prompt (image handles continuity)")
                            i2v_param_added = True
                        elif 'latents' in pipeline_signature.parameters:
                            # Use latents for I2V conditioning with proper strength control
                            print_wan_info("🔧 Using latent-based I2V conditioning (encoding image to latents)")
                            try:
                                # Encode image to latents and add noise based on strength
                                # Lower strength = more noise = less influence from image = more prompt freedom
                                init_latents = self._encode_image_to_latents_with_noise(
                                    image, aligned_width, aligned_height, strength, num_frames
                                )
                                if init_latents is not None:
                                    generation_kwargs['latents'] = init_latents
                                    print_wan_info(f"✅ I2V conditioning: Latent initialization with strength {strength:.2f}")
                                    print_wan_info(f"   → Noise level: {1.0 - strength:.2f} (lower strength = more prompt freedom)")
                                    print_wan_info(f"📝 Using original prompt (latents handle continuity)")
                                    i2v_param_added = True
                                else:
                                    print_wan_warning("⚠️ Failed to encode image to latents, falling back to prompt-only")
                            except Exception as e:
                                print_wan_warning(f"⚠️ Latent encoding failed: {e}, falling back to prompt-only")

                        if not i2v_param_added:
                            # No image conditioning available - enhance prompt for continuity
                            enhanced_prompt = f"{prompt}. Smooth continuation, maintaining consistent style and subject."
                            generation_kwargs['prompt'] = enhanced_prompt
                            print_wan_warning("⚠️ Pipeline does not support any known I2V image parameters")
                            print_wan_warning(f"⚠️ Available parameters: {list(pipeline_signature.parameters.keys())}")
                            print_wan_warning("⚠️ Using enhanced prompt for continuity (limited effectiveness)")
                            print_wan_info(f"📝 Enhanced prompt: {enhanced_prompt[:80]}...")

                        # Add strength parameter if supported (some I2V pipelines use this)
                        if 'strength' in pipeline_signature.parameters and not i2v_param_added:
                            generation_kwargs['strength'] = strength

                        # Add image_guidance_scale if supported (alternative to strength in some pipelines)
                        if 'image_guidance_scale' in pipeline_signature.parameters:
                            # Convert strength to guidance scale (typically 1.0-10.0 range)
                            image_guidance = 1.0 + (strength * 9.0)  # Map 0-1 to 1-10
                            generation_kwargs['image_guidance_scale'] = image_guidance
                            print_wan_info(f"🎯 Image guidance scale: {image_guidance:.1f}")

                        # Add resolution parameters
                        if 'height' in pipeline_signature.parameters:
                            generation_kwargs['height'] = aligned_height
                        if 'width' in pipeline_signature.parameters:
                            generation_kwargs['width'] = aligned_width
                        if 'num_frames' in pipeline_signature.parameters:
                            generation_kwargs['num_frames'] = num_frames
                        elif 'video_length' in pipeline_signature.parameters:
                            generation_kwargs['video_length'] = num_frames
                        elif 'num_video_frames' in pipeline_signature.parameters:
                            generation_kwargs['num_video_frames'] = num_frames

                        # Log generation parameters
                        actual_prompt = generation_kwargs['prompt']
                        print_wan_info(f"🎬 I2V Generation:")
                        print_wan_info(f"   Prompt: {actual_prompt[:60]}{'...' if len(actual_prompt) > 60 else ''}")
                        print_wan_info(f"   Resolution: {aligned_width}x{aligned_height}")
                        print_wan_info(f"   Frames: {num_frames}")
                        print_wan_info(f"   I2V Strength: {strength:.2f}")
                        print_wan_info(f"   CFG: {guidance_scale}")
                        print_wan_info(f"   Image conditioning: {'Yes' if i2v_param_added else 'No (prompt-only)'}")

                        with torch.no_grad():
                            return self.pipeline(**generation_kwargs)
                
                    def _add_noise_to_image(self, image, strength):
                        """
                        Add subtle noise to image in pixel space for strength control
                        
                        Args:
                            image: PIL Image
                            strength: 0.0-1.0, where 1.0 = no noise (full image), 0.0 = full noise
                        
                        Returns:
                            PIL Image with noise applied
                        
                        Note: Uses very subtle noise to avoid artifacts. The noise level is
                        intentionally much lower than (1-strength) to work well with I2V models.
                        """
                        import numpy as np
                        from PIL import Image
                        
                        # Convert to numpy array [0, 1]
                        img_array = np.array(image).astype(np.float32) / 255.0
                        
                        # Calculate noise amount - linear scale for more predictable control
                        # I2V pipeline is very sticky, so we need more aggressive noise
                        noise_amount = 1.0 - strength  # Linear, not squared
                        noise_scale = noise_amount * 0.5  # Scale to max 50% noise at strength 0.0
                        
                        # Generate Gaussian noise [-1, 1] range
                        noise = np.random.randn(*img_array.shape).astype(np.float32) * 0.5
                        
                        # Add noise (not blend!) - this preserves more of the original image
                        noisy = img_array + (noise * noise_scale)
                        
                        # Clip to valid range and convert back
                        noisy = np.clip(noisy, 0, 1)
                        noisy = (noisy * 255).astype(np.uint8)
                        
                        return Image.fromarray(noisy)
                    
                    def _encode_image_to_latents_with_noise(self, image, width, height, strength, num_frames):
                        """
                        Encode image to latents with noise-based strength control for I2V
                        
                        Args:
                            image: PIL Image to encode
                            width: Target width
                            height: Target height
                            strength: Strength parameter (0.0-1.0)
                                     1.0 = full influence from image (no noise added)
                                     0.0 = no influence (pure noise)
                            num_frames: Number of frames to generate
                        
                        Returns:
                            Tensor of noisy latents or None if encoding fails
                        """
                        try:
                            from PIL import Image
                            import torchvision.transforms as T  # type: ignore
                            
                            # Resize image to match target resolution
                            if image.size != (width, height):
                                image = image.resize((width, height), Image.Resampling.LANCZOS)
                            
                            # Convert PIL Image to tensor
                            transform = T.Compose([
                                T.ToTensor(),
                                T.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
                            ])
                            
                            image_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
                            
                            # Move to pipeline device
                            if hasattr(self.pipeline, 'vae'):
                                vae = self.pipeline.vae
                                device = vae.device if hasattr(vae, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                dtype = vae.dtype if hasattr(vae, 'dtype') else torch.float16
                                
                                image_tensor = image_tensor.to(device=device, dtype=dtype)
                                
                                # WanVAE expects 5D tensor: (B, C, F, H, W) for video
                                # Add temporal dimension: (1, C, H, W) -> (1, C, 1, H, W)
                                image_tensor = image_tensor.unsqueeze(2)
                                print_wan_info(f"🔍 Image tensor shape for VAE: {image_tensor.shape}")
                                
                                # Encode image to latents
                                with torch.no_grad():
                                    latent_dist = vae.encode(image_tensor).latent_dist
                                    latents = latent_dist.mode()  # Use mode for determinism
                                    
                                    # Scale latents (VAE uses scaling factor)
                                    if hasattr(vae.config, 'scaling_factor'):
                                        latents = latents * vae.config.scaling_factor
                                    else:
                                        latents = latents * 0.18215  # Default scaling factor
                                    
                                    # latents shape should be: (1, C, 1, H//8, W//8) from single-frame encoding
                                    # For video generation, we need: (1, C, F, H//8, W//8)
                                    print_wan_info(f"🔍 Encoded latent shape: {latents.shape}")
                                    
                                    # If latents are already (1, C, F, H//8, W//8), adjust F to match num_frames
                                    if len(latents.shape) == 5:
                                        current_frames = latents.shape[2]
                                        if current_frames == 1 and num_frames > 1:
                                            # Replicate single frame across temporal dimension
                                            latents = latents.repeat(1, 1, num_frames, 1, 1)
                                            print_wan_info(f"🔍 Replicated to {num_frames} frames: {latents.shape}")
                                        elif current_frames != num_frames:
                                            # Interpolate to match num_frames
                                            print_wan_warning(f"⚠️ Latent frames ({current_frames}) != requested frames ({num_frames})")
                                            print_wan_warning(f"⚠️ Using available frames as-is")
                                    else:
                                        # Unexpected shape, try to handle it
                                        print_wan_warning(f"⚠️ Unexpected latent shape: {latents.shape}")
                                    
                                    # Add noise based on strength
                                    # strength = 1.0: no noise (full image influence)
                                    # strength = 0.0: full noise (no image influence)
                                    noise_scale = 1.0 - strength
                                    
                                    if noise_scale > 0.01:  # Only add noise if strength < 0.99
                                        noise = torch.randn_like(latents)
                                        # Blend: latents * strength + noise * (1 - strength)
                                        latents = latents * strength + noise * noise_scale
                                        print_wan_info(f"🔍 Added noise: {noise_scale:.2f} scale (strength={strength:.2f})")
                                    else:
                                        print_wan_info(f"🔍 No noise added (strength={strength:.2f} is very high)")
                                    
                                    print_wan_info(f"🔍 Final latent shape: {latents.shape}")
                                    
                                    return latents
                            else:
                                print_wan_warning("⚠️ Pipeline has no VAE - cannot encode image to latents")
                                return None
                                
                        except Exception as e:
                            print_wan_error(f"Failed to encode image to latents with noise: {e}")
                            import traceback
                            traceback.print_exc()
                            return None
                    
                    def _encode_image_to_latents(self, image, width, height, strength):
                        """
                        Simple image encoding to latents (legacy method, kept for compatibility)
                        
                        Args:
                            image: PIL Image to encode
                            width: Target width
                            height: Target height
                            strength: Strength parameter (not used in simple encoding)
                        
                        Returns:
                            Tensor of latents or None if encoding fails
                        """
                        try:
                            from PIL import Image
                            import torchvision.transforms as T  # type: ignore
                            
                            # Resize image to match target resolution
                            if image.size != (width, height):
                                image = image.resize((width, height), Image.Resampling.LANCZOS)
                            
                            # Convert PIL Image to tensor
                            transform = T.Compose([
                                T.ToTensor(),
                                T.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
                            ])
                            
                            image_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
                            
                            # Move to pipeline device
                            if hasattr(self.pipeline, 'vae'):
                                vae = self.pipeline.vae
                                device = vae.device if hasattr(vae, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                dtype = vae.dtype if hasattr(vae, 'dtype') else torch.float16
                                
                                image_tensor = image_tensor.to(device=device, dtype=dtype)
                                
                                # Encode image to latents
                                with torch.no_grad():
                                    latent_dist = vae.encode(image_tensor).latent_dist
                                    latents = latent_dist.mode()  # Use mode instead of sample for determinism
                                    
                                    # Scale latents (VAE uses scaling factor)
                                    if hasattr(vae.config, 'scaling_factor'):
                                        latents = latents * vae.config.scaling_factor
                                    else:
                                        latents = latents * 0.18215  # Default scaling factor
                                    
                                    # For video generation, we need to replicate the latent across the temporal dimension
                                    # The latent should be shape (B, C, F, H//8, W//8) for video models
                                    # We'll create initial noise and blend it with the image latent based on strength
                                    
                                    print_wan_info(f"🔍 Encoded latent shape: {latents.shape}")
                                    
                                    # Note: For proper I2V, we would need to know the expected latent shape
                                    # including the frame dimension. Since we don't have direct access to this,
                                    # we'll return the encoded latent and let the pipeline handle the temporal expansion.
                                    # The strength parameter will control the noise level added during generation.
                                    
                                    return latents
                            else:
                                print_wan_warning("⚠️ Pipeline has no VAE - cannot encode image to latents")
                                return None
                                
                        except Exception as e:
                            print_wan_error(f"Failed to encode image to latents: {e}")
                            import traceback
                            traceback.print_exc()
                            return None

                    def generate_flf2v(self, first_frame, last_frame, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                        """
                        First-Last-Frame-to-Video (FLF2V) generation

                        Interpolates smooth video between two keyframes.
                        Perfect for Flux keyframe → Wan fill-in workflow!

                        Args:
                            first_frame: PIL Image - starting keyframe
                            last_frame: PIL Image - ending keyframe
                            prompt: Text prompt for interpolation guidance
                            num_frames: Number of frames to generate between keyframes

                        Note: Uses WanImageToVideoPipeline's 'last_image' parameter for FLF2V mode.
                        This should use less VRAM than T2V since both frames already exist.
                        """
                        # Wan 2.2 requires dimensions divisible by 32
                        aligned_width = ((width + 31) // 32) * 32
                        aligned_height = ((height + 31) // 32) * 32

                        # FLF2V requires I2V pipeline with last_image support
                        if self.i2v_pipeline is None:
                            raise RuntimeError("FLF2V requires WanImageToVideoPipeline - not available for this model")

                        import inspect
                        pipeline_signature = inspect.signature(self.i2v_pipeline.__call__)

                        # Check if pipeline supports last_image parameter (FLF2V mode)
                        if 'last_image' not in pipeline_signature.parameters:
                            raise RuntimeError("FLF2V mode requires 'last_image' parameter - not supported by this I2V pipeline")

                        print_wan_info("🎬 FLF2V Mode: Interpolating between two keyframes")
                        print_wan_info(f"   First frame: {first_frame.size}")
                        print_wan_info(f"   Last frame: {last_frame.size}")
                        print_wan_info(f"   Frames to generate: {num_frames}")
                        print_wan_info(f"   Resolution: {aligned_width}x{aligned_height}")
                        
                        # Verify images are different
                        import numpy as np
                        first_array = np.array(first_frame)
                        last_array = np.array(last_frame)
                        images_identical = np.array_equal(first_array, last_array)
                        if images_identical:
                            print_wan_error("⚠️  WARNING: First and last frames are IDENTICAL!")
                            print_wan_error("   FLF2V will not interpolate properly.")
                        else:
                            pixel_diff = np.abs(first_array - last_array).mean()
                            print_wan_info(f"   ✓ Images are different (avg pixel diff: {pixel_diff:.2f})")

                        generation_kwargs = {
                            "image": first_frame,           # Start keyframe
                            "last_image": last_frame,       # End keyframe
                            "prompt": prompt,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "height": aligned_height,
                            "width": aligned_width,
                            "num_frames": num_frames,
                        }

                        print_wan_success("✅ Generating FLF2V interpolation...")
                        print_wan_info(f"   Prompt: '{prompt}' (empty = pure interpolation)")
                        print_wan_info(f"   Guidance scale: {guidance_scale}")
                        print_wan_info(f"   Inference steps: {num_inference_steps}")

                        with torch.no_grad():
                            return self.i2v_pipeline(**generation_kwargs)

                self.pipeline = DiffusersWrapper(pipeline, i2v_pipeline)
                logger.info(f"{emoji_if_enabled('✅')} Diffusers model loaded successfully")
                
                # Provide clear feedback about I2V support
                if i2v_pipeline is not None:
                    print_wan_success("✅ I2V chaining is now fully supported with WanImageToVideoPipeline!")
                    print_wan_success("   → Clips will be seamlessly connected using image conditioning")
                    print_wan_success("   → First clip: T2V generation")
                    print_wan_success("   → Subsequent clips: I2V from last frame")
                elif model_supports_i2v:
                    print_wan_info("ℹ️ Model supports I2V but WanImageToVideoPipeline is not available")
                    print_wan_info("   → I2V chaining will use fallback: latent conditioning")
                else:
                    print_wan_info("ℹ️ Model is T2V only - no native I2V support")
                    print_wan_info("   → I2V chaining will use fallback: enhanced prompts for continuity")
                
                return True
                
            except Exception as diffusers_e:
                logger.error(f"Diffusers loading failed: {diffusers_e}", emoji='off')

                # Determine model version for error message (2.1 for FLF2V, 2.2 for TI2V)
                model_version = "2.1" if model_info.get('type') == 'FLF2V' else "2.2"

                raise RuntimeError(f"""
❌ CRITICAL: Could not load Wan {model_version} model!

🔧 TROUBLESHOOTING:
1. 📦 Dependencies upgraded automatically by extension
2. 💾 Verify model download is complete: {model_info['path']}
3. 🔄 Restart WebUI to apply dependency upgrades
4. 💬 Check console for detailed error messages

❌ Model loading failed!
Error: {diffusers_e}
""")
        
        except Exception as e:
            logger.error(f"Standard model loading failed: {e}", emoji='off')
            return False
    
    def generate_video_with_i2v_chaining(self, clips, model_info, output_dir, wan_args=None, **kwargs):
        """Generate video with I2V chaining using styled progress indicators"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded")
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestring for consistent file naming
            timestring = kwargs.pop('timestring', str(int(time.time())))

            # Handle audio download early if provided
            audio_url = kwargs.get('soundtrack_path') or kwargs.get('audio_url')
            cached_audio_path = None
            if audio_url:
                cached_audio_path = self.download_and_cache_audio(audio_url, output_dir, timestring)
                kwargs['cached_audio_path'] = cached_audio_path

            # Save settings file before generation (timestring removed from kwargs to avoid duplicate)
            settings_file = self.save_wan_settings_and_metadata(
                output_dir, timestring, clips, model_info, wan_args, **kwargs
            )
            
            # Create SRT subtitle file
            fps = kwargs.get('fps', 8.0)
            srt_file = self.create_wan_srt_file(output_dir, timestring, clips, fps)
            
            with WanGenerationContext(len(clips)) as gen_context:
                all_frame_paths = []
                last_frame_path = None
                total_frame_idx = 0
                
                # Get generation parameters
                steps = kwargs.get('num_inference_steps', 50)
                guidance_scale = kwargs.get('guidance_scale', 7.5)
                height = kwargs.get('height', self.optimal_height)
                width = kwargs.get('width', self.optimal_width)
                
                print_wan_info(f"Model: {model_info['name']} ({model_info['type']}, {model_info['size']})")
                print_wan_info(f"Output: {output_dir}")
                if settings_file:
                    print_wan_info(f"Settings: {os.path.basename(settings_file)}")
                if srt_file:
                    print_wan_info(f"Subtitles: {os.path.basename(srt_file)}")
                if cached_audio_path:
                    print_wan_info(f"Audio: {os.path.basename(cached_audio_path)}")
                
                # Generate each clip with progress tracking
                for clip_idx, clip in enumerate(clips):
                    try:
                        # Create frame progress bar for this clip
                        with create_wan_frame_progress(clip['num_frames'], clip_idx) as frame_progress:
                            gen_context.update_clip(clip_idx, clip['prompt'][:30])
                            
                            print_wan_progress(f"Generating clip {clip_idx + 1}/{len(clips)}")
                            print_wan_info(f"Prompt: {clip['prompt'][:50]}...")
                            print_wan_info(f"Frames: {clip['num_frames']}")
                            
                            # Create inference progress bar
                            with create_wan_inference_progress(steps) as inference_progress:
                                # Generate based on chaining mode
                                if clip_idx == 0 or last_frame_path is None:
                                    # First clip: T2V generation
                                    print_wan_info("Using T2V generation for first clip")
                                    result = self.pipeline(
                                        prompt=clip['prompt'],
                                        height=height,
                                        width=width,
                                        num_frames=clip['num_frames'],
                                        num_inference_steps=steps,
                                        guidance_scale=guidance_scale,
                                    )
                                    inference_progress.update(steps)
                                else:
                                    # Subsequent clips: I2V chaining
                                    print_wan_info(f"I2V chaining from: {os.path.basename(last_frame_path)}")

                                    # Get I2V strength from wan_args
                                    i2v_strength = 0.85  # Default: strong continuity (matches UI default)
                                    if wan_args and hasattr(wan_args, 'wan_strength_override') and wan_args.wan_strength_override:
                                        if hasattr(wan_args, 'wan_fixed_strength'):
                                            i2v_strength = float(wan_args.wan_fixed_strength)
                                            print_wan_info(f"Using fixed I2V strength from settings: {i2v_strength:.2f}")
                                    else:
                                        print_wan_info(f"Using default I2V strength: {i2v_strength:.2f}")

                                    if hasattr(self.pipeline, 'generate_image2video'):
                                        # Custom I2V pipeline with image conditioning
                                        from PIL import Image
                                        last_frame_image = Image.open(last_frame_path)

                                        result = self.pipeline.generate_image2video(
                                            image=last_frame_image,
                                            prompt=clip['prompt'],
                                            height=height,
                                            width=width,
                                            num_frames=clip['num_frames'],
                                            num_inference_steps=steps,
                                            guidance_scale=guidance_scale,
                                            strength=i2v_strength,  # Control image conditioning strength
                                        )
                                        inference_progress.update(steps)

                                        # Free the image immediately after generation
                                        del last_frame_image
                                    else:
                                        # Fallback to T2V with enhanced prompt
                                        print_wan_warning("⚠️ Pipeline does not support I2V - using enhanced T2V")
                                        enhanced_prompt = f"Continuing seamlessly from previous scene, {clip['prompt']}. Maintain visual continuity, smooth transition."
                                        result = self.pipeline(
                                            prompt=enhanced_prompt,
                                            height=height,
                                            width=width,
                                            num_frames=clip['num_frames'],
                                            num_inference_steps=steps,
                                            guidance_scale=guidance_scale,
                                        )
                                        inference_progress.update(steps)
                            
                            # Process and save frames with progress
                            clip_frames = self._process_and_save_frames(
                                result, clip_idx, output_dir, timestring, 
                                total_frame_idx, frame_progress
                            )
                            
                            if clip_frames:
                                all_frame_paths.extend(clip_frames)
                                total_frame_idx += len(clip_frames)
                                last_frame_path = clip_frames[-1]  # Update for next clip
                                print_wan_success(f"Generated {len(clip_frames)} frames for clip {clip_idx + 1}")
                            else:
                                raise RuntimeError(f"No frames generated for clip {clip_idx + 1}")

                            # Aggressive VRAM cleanup after each clip to prevent OOM on long generations
                            import gc
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            gc.collect()
                            print_wan_info(f"🧹 Cleaned up VRAM after clip {clip_idx + 1}")

                    except Exception as e:
                        print_wan_error(f"Clip {clip_idx + 1} generation failed: {e}")
                        raise
                
                print_wan_success(f"All clips generated! Total frames: {len(all_frame_paths)}")
                print_wan_info(f"Frames saved to: {output_dir}")
                
                # Return comprehensive results
                return {
                    'output_dir': output_dir,
                    'frame_paths': all_frame_paths,
                    'settings_file': settings_file,
                    'srt_file': srt_file,
                    'audio_file': cached_audio_path,
                    'timestring': timestring,
                    'total_frames': len(all_frame_paths),
                    'total_clips': len(clips)
                }
                
        except Exception as e:
            print_wan_error(f"I2V chained video generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"I2V chained video generation failed: {e}")
        finally:
            # Clean up if needed
            pass
    
    def _process_and_save_frames(self, result, clip_idx, output_dir, timestring, start_frame_idx, frame_progress=None):
        """Process generation result and save frames with progress tracking"""
        try:
            from PIL import Image
            import numpy as np

            frames = []

            # Debug: Log result type
            print_wan_info(f"🔍 Result type: {type(result)}")
            print_wan_info(f"🔍 Result attributes: {dir(result)[:10]}...")  # First 10 attributes

            # Handle different result formats
            if isinstance(result, tuple):
                print_wan_info("Result is tuple, extracting first element")
                frames_data = result[0]
            elif hasattr(result, 'frames'):
                print_wan_info("Result has .frames attribute")
                frames_data = result.frames
            elif hasattr(result, 'images'):
                print_wan_info("Result has .images attribute")
                frames_data = result.images
            elif hasattr(result, 'videos'):
                print_wan_info("Result has .videos attribute")
                frames_data = result.videos
            else:
                print_wan_info("Using result directly as frames_data")
                frames_data = result

            print_wan_info(f"🔍 Frames data type: {type(frames_data)}")

            # Check if frames_data is None or empty
            if frames_data is None:
                print_wan_error("❌ Frames data is None!")
                return []

            # Convert to individual frames
            if hasattr(frames_data, 'cpu'):  # Tensor
                frames_tensor = frames_data.cpu()
                print_wan_info(f"✅ Processing tensor: {frames_tensor.shape}")
                
                # Handle different tensor formats
                if len(frames_tensor.shape) == 5:  # (B, C, F, H, W)
                    frames_tensor = frames_tensor.squeeze(0)  # Remove batch dimension
                
                if len(frames_tensor.shape) == 4:  # (C, F, H, W)
                    for frame_idx in range(frames_tensor.shape[1]):
                        frame_tensor = frames_tensor[:, frame_idx, :, :]  # (C, H, W)
                        frame_np = frame_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
                        
                        # Models may output in [-1, 1] range, convert to [0, 255]
                        if frame_np.min() < -0.5:  # Likely [-1, 1] range
                            # Convert from [-1, 1] to [0, 1] then to [0, 255]
                            frame_np = (frame_np + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                            frame_np = np.clip(frame_np, 0, 1)  # Ensure valid range
                            frame_np = (frame_np * 255).astype(np.uint8)
                        elif frame_np.max() <= 1.0:  # [0, 1] range
                            frame_np = (frame_np * 255).astype(np.uint8)
                        else:  # Already in [0, 255] range
                            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                        
                        frames.append(frame_np)
                        
                        # Update progress if available
                        if frame_progress:
                            frame_progress.update(1)
                else:
                    print_wan_warning(f"Unexpected tensor shape: {frames_tensor.shape}")
                    # Try to treat as single frame
                    if len(frames_tensor.shape) == 3:  # (C, H, W)
                        frame_np = frames_tensor.permute(1, 2, 0).numpy()
                        
                        # Apply same normalization fix
                        if frame_np.min() < -0.5:  # Likely [-1, 1] range
                            frame_np = (frame_np + 1.0) / 2.0
                            frame_np = np.clip(frame_np, 0, 1)
                            frame_np = (frame_np * 255).astype(np.uint8)
                        elif frame_np.max() <= 1.0:
                            frame_np = (frame_np * 255).astype(np.uint8)
                        else:
                            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                            
                        frames.append(frame_np)
                        if frame_progress:
                            frame_progress.update(1)
            
            elif isinstance(frames_data, list):  # List of PIL Images or arrays
                print_wan_info(f"✅ Processing list with {len(frames_data)} items")
                for frame in frames_data:
                    if hasattr(frame, 'save'):  # PIL Image
                        frames.append(np.array(frame))
                    else:
                        frames.append(frame)

                    if frame_progress:
                        frame_progress.update(1)

            elif isinstance(frames_data, np.ndarray):  # Numpy array (WanPipelineOutput.frames)
                print_wan_info(f"✅ Processing numpy array: {frames_data.shape}")

                # WanPipeline returns frames in shape (B, F, H, W, C) or (B, C, F, H, W)
                if len(frames_data.shape) == 5:
                    # Check if channels are first (B, C, F, H, W) or last (B, F, H, W, C)
                    if frames_data.shape[1] == 3 or frames_data.shape[1] == 4:
                        # (B, C, F, H, W) format
                        print_wan_info("Detected (B, C, F, H, W) format, converting...")
                        frames_np = frames_data[0]  # Remove batch dim: (C, F, H, W)
                        num_frames_in_clip = frames_np.shape[1]

                        for frame_idx in range(num_frames_in_clip):
                            frame = frames_np[:, frame_idx, :, :]  # (C, H, W)
                            frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)

                            # Normalize to [0, 255]
                            if frame.min() < -0.5:  # [-1, 1] range
                                frame = (frame + 1.0) / 2.0
                                frame = np.clip(frame, 0, 1)
                                frame = (frame * 255).astype(np.uint8)
                            elif frame.max() <= 1.0:  # [0, 1] range
                                frame = (frame * 255).astype(np.uint8)
                            else:  # Already [0, 255]
                                frame = np.clip(frame, 0, 255).astype(np.uint8)

                            frames.append(frame)
                            if frame_progress:
                                frame_progress.update(1)
                    else:
                        # (B, F, H, W, C) format
                        print_wan_info("Detected (B, F, H, W, C) format, converting...")
                        frames_np = frames_data[0]  # Remove batch dim: (F, H, W, C)

                        for frame_idx in range(frames_np.shape[0]):
                            frame = frames_np[frame_idx]  # (H, W, C)

                            # Normalize to [0, 255]
                            if frame.min() < -0.5:  # [-1, 1] range
                                frame = (frame + 1.0) / 2.0
                                frame = np.clip(frame, 0, 1)
                                frame = (frame * 255).astype(np.uint8)
                            elif frame.max() <= 1.0:  # [0, 1] range
                                frame = (frame * 255).astype(np.uint8)
                            else:  # Already [0, 255]
                                frame = np.clip(frame, 0, 255).astype(np.uint8)

                            frames.append(frame)
                            if frame_progress:
                                frame_progress.update(1)

                elif len(frames_data.shape) == 4:
                    # (F, H, W, C) format (already removed batch)
                    print_wan_info("Detected (F, H, W, C) format, converting...")
                    for frame_idx in range(frames_data.shape[0]):
                        frame = frames_data[frame_idx]  # (H, W, C)

                        # Normalize to [0, 255]
                        if frame.min() < -0.5:
                            frame = (frame + 1.0) / 2.0
                            frame = np.clip(frame, 0, 1)
                            frame = (frame * 255).astype(np.uint8)
                        elif frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)

                        frames.append(frame)
                        if frame_progress:
                            frame_progress.update(1)
                else:
                    print_wan_error(f"❌ Unexpected numpy array shape: {frames_data.shape}")
                    return []

            else:
                # Unknown format - try to debug it
                print_wan_error(f"❌ Unknown frames_data format!")
                print_wan_error(f"   Type: {type(frames_data)}")
                print_wan_error(f"   Has __len__: {hasattr(frames_data, '__len__')}")
                print_wan_error(f"   Has __iter__: {hasattr(frames_data, '__iter__')}")
                if hasattr(frames_data, '__len__'):
                    try:
                        print_wan_error(f"   Length: {len(frames_data)}")
                    except:
                        pass
                return []

            # Check if we extracted any frames
            if not frames:
                print_wan_error(f"❌ No frames extracted from result!")
                print_wan_error(f"   Result type: {type(result)}")
                print_wan_error(f"   Frames data type: {type(frames_data)}")
                return []

            print_wan_info(f"✅ Extracted {len(frames)} frames, proceeding to save...")

            # Save frames as PNG files
            saved_paths = []
            for i, frame_np in enumerate(frames):
                frame_filename = f"{start_frame_idx + i:09d}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                
                try:
                    pil_image = Image.fromarray(frame_np)
                    pil_image.save(frame_path)
                    saved_paths.append(frame_path)
                except Exception as save_e:
                    print_wan_warning(f"Failed to save frame {i}: {save_e}")
                    continue
            
            print_wan_success(f"Saved {len(saved_paths)} frames for clip {clip_idx + 1}")
            return saved_paths
            
        except Exception as e:
            print_wan_error(f"Frame processing failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def test_wan_setup(self) -> bool:
        """Test if Wan setup is working with styled output"""
        try:
            print_wan_progress("Testing Wan setup...")
            
            # Test model discovery
            models = self.discover_models()
            if not models:
                print_wan_error("No models found")
                return False
            
            # Test best model selection
            best_model = self.get_best_model()
            if not best_model:
                print_wan_error("No suitable model found")
                return False
            
            print_wan_success(f"Setup test passed - found {len(models)} models")
            return True
            
        except Exception as e:
            print_wan_error(f"Setup test failed: {e}")
            return False
    
    def unload_model(self):
        """Unload the model to free memory"""
        if self.pipeline:
            try:
                if hasattr(self.pipeline, 'to'):
                    self.pipeline.to('cpu')
                del self.pipeline
                self.pipeline = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                print_wan_success("Model unloaded, memory freed")
            except Exception as e:
                print_wan_warning(f"Error unloading model: {e}")
    
    def save_wan_settings_and_metadata(self, output_dir: str, timestring: str, clips: List[Dict], 
                                       model_info: Dict, wan_args=None, **kwargs) -> str:
        """Save Wan generation settings to match normal Deforum format"""
        try:
            settings_filename = os.path.join(output_dir, f"{timestring}_settings.txt")
            
            # Create comprehensive settings dictionary
            settings = {
                # Wan-specific settings
                "wan_model_name": model_info['name'],
                "wan_model_type": model_info['type'],
                "wan_model_size": model_info['size'],
                "wan_model_path": model_info['path'],
                "wan_flash_attention_mode": self.flash_attention_mode,
                
                # Generation parameters
                "width": kwargs.get('width', self.optimal_width),
                "height": kwargs.get('height', self.optimal_height),
                "num_inference_steps": kwargs.get('num_inference_steps', 50),
                "guidance_scale": kwargs.get('guidance_scale', 7.5),
                "fps": kwargs.get('fps', 8),
                
                # Clip information
                "total_clips": len(clips),
                "total_frames": sum(clip['num_frames'] for clip in clips),
                "clips": clips,
                
                # Metadata
                "generation_mode": "wan_i2v_chaining",
                "timestring": timestring,
                "output_directory": output_dir,
                "generation_timestamp": time.time(),
                "device": str(self.device),
                
                # Version info
                "wan_integration_version": "1.0.0",
                "deforum_git_commit_id": self._get_deforum_version(),
            }
            
            # Add wan_args if provided
            if wan_args:
                settings.update({f"wan_{k}": v for k, v in vars(wan_args).items()})
            
            # Save settings file
            with open(settings_filename, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
            
            print_wan_success(f"Settings saved: {os.path.basename(settings_filename)}")
            return settings_filename
            
        except Exception as e:
            print_wan_error(f"Failed to save settings: {e}")
            return None
    
    def create_wan_srt_file(self, output_dir: str, timestring: str, clips: List[Dict], 
                           fps: float = 8.0) -> str:
        """Create SRT subtitle file for Wan generation"""
        try:
            srt_filename = os.path.join(output_dir, f"{timestring}.srt")
            
            # Initialize SRT file
            frame_duration = init_srt_file(srt_filename, fps)
            
            # Write subtitle entries for each clip
            current_frame = 0
            for clip_idx, clip in enumerate(clips):
                clip_frames = clip['num_frames']
                
                # Create subtitle text for this clip
                subtitle_text = f"Clip {clip_idx + 1}/{len(clips)}: {clip['prompt'][:80]}"
                if len(clip['prompt']) > 80:
                    subtitle_text += "..."
                
                # Calculate timing
                start_time = Decimal(current_frame) * frame_duration
                end_time = Decimal(current_frame + clip_frames) * frame_duration
                
                # Write subtitle entry
                with open(srt_filename, "a", encoding="utf-8") as f:
                    f.write(f"{clip_idx + 1}\n")
                    f.write(f"{self._time_to_srt_format(start_time)} --> {self._time_to_srt_format(end_time)}\n")
                    f.write(f"{subtitle_text}\n\n")
                
                current_frame += clip_frames
            
            print_wan_success(f"SRT file created: {os.path.basename(srt_filename)}")
            return srt_filename
            
        except Exception as e:
            print_wan_error(f"Failed to create SRT file: {e}")
            return None
    
    def download_and_cache_audio(self, audio_url: str, output_dir: str, timestring: str) -> str:
        """Download and cache audio file in output directory"""
        try:
            if not audio_url or not audio_url.startswith(('http://', 'https://')):
                return audio_url  # Return as-is if not a URL
            
            print_wan_progress(f"Downloading audio: {audio_url}")
            
            # Download audio using Deforum's utility
            temp_audio_path = download_audio(audio_url)
            
            # Determine file extension
            _, ext = os.path.splitext(audio_url)
            if not ext:
                ext = '.mp3'  # Default to MP3
            
            # Create cached audio path in output directory
            cached_audio_path = os.path.join(output_dir, f"{timestring}_soundtrack{ext}")
            
            # Copy to output directory
            import shutil
            shutil.copy2(temp_audio_path, cached_audio_path)
            
            # Clean up temp file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            
            print_wan_success(f"Audio cached: {os.path.basename(cached_audio_path)}")
            return cached_audio_path
            
        except Exception as e:
            print_wan_error(f"Failed to download/cache audio: {e}")
            return audio_url  # Return original URL as fallback
    
    def _time_to_srt_format(self, seconds: Decimal) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours, remainder = divmod(float(seconds), 3600)
        minutes, remainder = divmod(remainder, 60)
        seconds_int, milliseconds = divmod(remainder, 1)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds_int):02},{int(milliseconds * 1000):03}"
    
    def _get_deforum_version(self) -> str:
        """Get Deforum version/commit ID"""
        try:
            from deforum.config.settings import get_deforum_version
            return get_deforum_version()
        except:
            return "unknown"

# Global instance for easy access
wan_integration = WanSimpleIntegration()

def wan_generate_video_main(*args, **kwargs):
    """Main entry point for Wan video generation with styled output"""
    try:
        print_wan_progress("Generate video main called")
        
        # Test setup first
        if not wan_integration.test_wan_setup():
            return "❌ Wan setup test failed - check model installation"
        
        # Get best model
        model_info = wan_integration.get_best_model()
        if not model_info:
            return "❌ No suitable Wan models found"
        
        # For now, return a simple success message
        return f"✅ Wan setup verified - ready to generate with {model_info['name']}"
        
    except Exception as e:
        print_wan_error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Wan generation error: {e}"

if __name__ == "__main__":
    # Test the integration
    integration = WanSimpleIntegration()
    integration.test_wan_setup() 