#!/usr/bin/env python3
"""
Wan Simple Integration for Deforum
Provides simple Wan video generation capabilities with auto-discovery and fallback support
"""

# Apply compatibility patches FIRST, before any other imports
try:
    import sys
    import os
    from pathlib import Path
    
    # Import wan_flash_attention_patch to trigger its immediate patching logic.
    # The patch applies itself automatically when the module is first loaded.
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path: # Ensure the directory is in path for the import
        sys.path.insert(0, str(current_dir))
    import wan_flash_attention_patch # This import triggers the patch
    
    print("✅ Early Flash Attention patch module imported (patch should have been applied automatically).")
    # Optionally, could call wan_flash_attention_patch.apply_wan_compatibility_patches() here
    # if we wanted to be absolutely sure or get a return status, but it should be redundant.

except Exception as early_patch_e:
    print(f"⚠️ Early Flash Attention patch import/application FAILED: {early_patch_e}")
    import traceback
    traceback.print_exc() # Print full traceback for early patch failures

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import tempfile
import shutil
from pathlib import Path
import sys
import os

class WanSimpleIntegration:
    """Simple, robust Wan integration that directly loads models"""
    
    def __init__(self):
        self.extension_root = Path(__file__).parent.parent.parent
        self.discovered_models = []
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def discover_models(self) -> List[Dict]:
        """Discover Wan models using our discovery system"""
        try:
            from .wan_model_discovery import WanModelDiscovery
        except ImportError:
            # Handle running as standalone script
            import sys
            sys.path.append(str(Path(__file__).parent))
            from wan_model_discovery import WanModelDiscovery
        
        discovery = WanModelDiscovery()
        self.discovered_models = discovery.discover_models()
        return self.discovered_models
    
    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model"""
        if not self.discovered_models:
            self.discover_models()
        return self.discovered_models[0] if self.discovered_models else None
    
    def test_wan_setup(self) -> bool:
        """Test if Wan setup is working properly"""
        try:
            print("🧪 Testing Wan setup...")
            
            # Check if models are available
            models = self.discover_models()
            if not models:
                print("❌ No Wan models found")
                return False
            
            print(f"✅ Found {len(models)} Wan models")
            
            # Try to load a model
            best_model = models[0]
            print(f"🔧 Testing model loading: {best_model['name']}")
            
            # Test model validation
            if not self._validate_wan_model(best_model):
                print("❌ Model validation failed")
                return False
            
            print("✅ Model validation passed")
            
            # Test pipeline creation (but don't actually load the heavy model)
            try:
                pipeline = self._create_custom_wan_pipeline(best_model)
                print("✅ Pipeline creation test passed")
                return True
            except Exception as e:
                print(f"❌ Pipeline creation failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Wan setup test failed: {e}")
            return False
    
    def load_simple_wan_pipeline(self, model_info: Dict) -> bool:
        """Load a simple Wan pipeline using the discovered model"""
        try:
            print(f"🚀 Loading Wan pipeline: {model_info['name']}")
            print(f"   📁 Path: {model_info['path']}")
            print(f"   🏷️ Type: {model_info['type']}")
            print(f"   📏 Size: {model_info['size']}")
            
            # Validate model files first
            if not self._validate_wan_model(model_info):
                raise RuntimeError("Model validation failed")
            
            # Create custom pipeline based on model type
            self.pipeline = self._create_custom_wan_pipeline(model_info)
            
            if self.pipeline:
                print(f"✅ Wan pipeline loaded successfully")
                return True
            else:
                print(f"❌ Failed to create Wan pipeline")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load Wan pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_custom_wan_pipeline(self, model_info: Dict):
        """Create a custom Wan pipeline that handles the model correctly"""
        
        class CustomWanPipeline:
            def __init__(self, model_path: str, device: str):
                self.model_path = model_path
                self.device = device
                self.model = None
                self.loaded = False
                
                print(f"🔧 Initializing custom Wan pipeline for {model_path}")
                
                # Try to load the actual Wan model
                self._load_wan_model()
            
            def _load_wan_model(self):
                """Load the actual Wan model with multiple fallback strategies"""
                try:
                    # Strategy 1: Try to import and use official Wan
                    print("🔄 Attempting to load official Wan model...")
                    
                    # Add Wan2.1 to path if it exists
                    import sys
                    from pathlib import Path
                    
                    # Look for Wan2.1 directory in the extension root
                    extension_root = Path(__file__).parent.parent.parent
                    wan_repo_path = extension_root / "Wan2.1"
                    
                    print(f"🔍 Looking for Wan repository at: {wan_repo_path}")
                    
                    if wan_repo_path.exists() and (wan_repo_path / "wan").exists():
                        # Add to Python path
                        if str(wan_repo_path) not in sys.path:
                            sys.path.insert(0, str(wan_repo_path))
                            print(f"✅ Added Wan repo to path: {wan_repo_path}")
                        
                        # Apply Flash Attention compatibility patches
                        try:
                            from scripts.deforum_helpers.wan_flash_attention_patch import apply_wan_compatibility_patches
                            apply_wan_compatibility_patches()
                        except Exception as patch_e:
                            print(f"⚠️ Could not apply compatibility patches: {patch_e}")
                        
                        # Verify the wan module can be imported
                        try:
                            import wan  # type: ignore
                            print("✅ Successfully imported wan module from local repository")
                            
                            # Import specific components
                            from wan.text2video import WanT2V  # type: ignore
                            from wan.image2video import WanI2V  # type: ignore
                            print("✅ Successfully imported WanT2V and WanI2V classes")
                            
                            # Try to load configs
                            try:
                                # Check if config files exist
                                config_dir = wan_repo_path / "wan" / "configs"
                                if config_dir.exists():
                                    print(f"✅ Found config directory: {config_dir}")
                                    
                                    # Try to import configs
                                    try:
                                        from wan.configs.wan_t2v_14B import t2v_14B as t2v_config
                                        from wan.configs.wan_i2v_14B import i2v_14B as i2v_config
                                        print("✅ Loaded official Wan configs")
                                    except ImportError as config_e:
                                        print(f"⚠️ Config import failed: {config_e}, will use minimal configs")
                                        # Create minimal config structure
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
                                                    'num_vector_embeds': None,
                                                    'activation_fn': "geglu",
                                                    'num_embeds_ada_norm': 1000,
                                                    'norm_elementwise_affine': False,
                                                    'norm_eps': 1e-6,
                                                    'attention_bias': True,
                                                    'caption_channels': 4096
                                                })
                                                
                                        t2v_config = MinimalConfig()
                                        i2v_config = MinimalConfig()
                                else:
                                    print("⚠️ Config directory not found, creating minimal configs")
                                    # Create minimal config structure
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
                                                'num_vector_embeds': None,
                                                'activation_fn': "geglu",
                                                'num_embeds_ada_norm': 1000,
                                                'norm_elementwise_affine': False,
                                                'norm_eps': 1e-6,
                                                'attention_bias': True,
                                                'caption_channels': 4096
                                            })
                                            
                                    t2v_config = MinimalConfig()
                                    i2v_config = MinimalConfig()
                                    
                            except Exception as config_e:
                                print(f"⚠️ Config loading failed: {config_e}, using minimal configs")
                                # Create minimal config structure
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
                                            'num_vector_embeds': None,
                                            'activation_fn': "geglu",
                                            'num_embeds_ada_norm': 1000,
                                            'norm_elementwise_affine': False,
                                            'norm_eps': 1e-6,
                                            'attention_bias': True,
                                            'caption_channels': 4096
                                        })
                                        
                                t2v_config = MinimalConfig()
                                i2v_config = MinimalConfig()
                            
                            # Initialize T2V model with correct parameters
                            print(f"🚀 Initializing WanT2V with checkpoint dir: {self.model_path}")
                            self.t2v_model = WanT2V(
                                config=t2v_config,
                                checkpoint_dir=self.model_path,
                                device_id=0,
                                rank=0,
                                dit_fsdp=False,
                                t5_fsdp=False
                            )
                            
                            # Try to initialize I2V model if available
                            try:
                                print(f"🚀 Initializing WanI2V with checkpoint dir: {self.model_path}")
                                self.i2v_model = WanI2V(
                                    config=i2v_config,
                                    checkpoint_dir=self.model_path,
                                    device_id=0,
                                    rank=0,
                                    dit_fsdp=False,
                                    t5_fsdp=False
                                )
                                print("✅ I2V model loaded for chaining support")
                            except Exception as e:
                                print(f"⚠️ I2V model not available: {e}")
                                self.i2v_model = None
                            
                            self.loaded = True
                            print("✅ Official Wan models loaded successfully")
                            return
                            
                        except ImportError as import_e:
                            print(f"❌ Failed to import wan module: {import_e}")
                            # Continue to fallback methods
                    else:
                        print(f"❌ Wan repository not found at: {wan_repo_path}")
                        print("💡 Expected structure: Wan2.1/wan/ directory with Python modules")
                    
                    # If we get here, the official Wan import failed
                    print("⚠️ Official Wan import failed, trying diffusers fallback...")
                    
                    # Strategy 2: Try diffusers-based loading as fallback
                    try:
                        from diffusers import DiffusionPipeline
                        
                        print("🔄 Attempting diffusers-based loading...")
                        self.model = DiffusionPipeline.from_pretrained(
                            self.model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        
                        if torch.cuda.is_available():
                            self.model = self.model.to(self.device)
                        
                        self.loaded = True
                        print("✅ Diffusers-based model loaded successfully (fallback)")
                        return
                        
                    except Exception as diffusers_e:
                        print(f"❌ Diffusers loading also failed: {diffusers_e}")
                    
                    # Both methods failed, provide comprehensive error
                    raise RuntimeError(f"""
❌ SETUP REQUIRED: Wan 2.1 Official Repository Setup Issue!

🔧 QUICK SETUP:

📥 The Wan2.1 directory exists but import failed. Try:

1. 📦 Install Wan dependencies:
   cd Wan2.1
   pip install -e .

2. 📂 Download models to correct location:
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan

3. 🔄 Restart WebUI completely

💡 The Wan2.1 repository is already present at: {wan_repo_path}
   But the Python modules couldn't be imported properly.

🌐 If issues persist, check: https://github.com/Wan-Video/Wan2.1#readme
""")
                
                except Exception as e:
                    print(f"❌ All model loading strategies failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"""
❌ CRITICAL: Wan model loading completely failed!

🔧 COMPLETE SETUP GUIDE:
1. 📥 Ensure Wan2.1 repository is properly set up
2. 📦 Install Wan dependencies: cd Wan2.1 && pip install -e .
3. 📂 Download models to: models/wan/
4. 🔄 Restart WebUI completely

❌ NO FALLBACKS AVAILABLE - Real Wan implementation required!
Error: {str(e)}
""")
            
            def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                """Generate video frames"""
                if not self.loaded:
                    raise RuntimeError("Wan model not loaded")
                
                print(f"🎬 Generating {num_frames} frames with Wan...")
                print(f"   📝 Prompt: {prompt[:50]}...")
                print(f"   📐 Size: {width}x{height}")
                print(f"   🔧 Steps: {num_inference_steps}")
                print(f"   📏 Guidance: {guidance_scale}")
                
                try:
                    # Use official Wan T2V if available
                    if hasattr(self, 't2v_model') and self.t2v_model:
                        print("🚀 Using official Wan T2V model")
                        
                        result = self.t2v_model.generate(
                            input_prompt=prompt,
                            size=(width, height),
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps,
                            guide_scale=guidance_scale,
                            shift=5.0,
                            sample_solver='unipc',
                            offload_model=True,
                            **kwargs
                        )
                        
                        return result
                    
                    # Use diffusers model if available
                    elif hasattr(self, 'model') and self.model:
                        print("🚀 Using diffusers-based model")
                        
                        generation_kwargs = {
                            "prompt": prompt,
                            "height": height,
                            "width": width,
                            "num_frames": num_frames,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                        }
                        
                        # Add optional parameters if supported
                        import inspect
                        if hasattr(self.model, '__call__'):
                            sig = inspect.signature(self.model.__call__)
                            
                            if 'output_type' in sig.parameters:
                                generation_kwargs['output_type'] = 'pil'
                            if 'return_dict' in sig.parameters:
                                generation_kwargs['return_dict'] = False
                        
                        with torch.no_grad():
                            result = self.model(**generation_kwargs)
                        
                        return result
                    
                    # No valid model available
                    else:
                        raise RuntimeError("""
❌ CRITICAL: No valid Wan model loaded!

🔧 SETUP REQUIRED:
1. Install official Wan repository
2. Download Wan models
3. Restart WebUI

💡 No fallbacks available - real Wan implementation required.
""")
                
                except Exception as e:
                    print(f"❌ Generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    raise RuntimeError(f"""
❌ CRITICAL: Wan video generation failed!

🔧 TROUBLESHOOTING:
1. ✅ Verify Wan models are properly installed
2. 🔄 Check CUDA/GPU availability: {torch.cuda.is_available()}
3. 💾 Check available VRAM/memory
4. 📦 Verify all dependencies are installed

❌ NO FALLBACKS - Real Wan implementation required!
Error: {e}
""")
            
            def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                """Generate video from image (I2V)"""
                if not self.loaded:
                    raise RuntimeError("Wan model not loaded")
                
                print(f"🎬 Generating I2V {num_frames} frames with Wan...")
                
                try:
                    # Use official Wan I2V if available
                    if hasattr(self, 'i2v_model') and self.i2v_model:
                        print("🚀 Using official Wan I2V model")
                        
                        result = self.i2v_model.generate(
                            input_prompt=prompt,
                            img=image,
                            max_area=height * width,
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps,
                            guide_scale=guidance_scale,
                            shift=5.0,
                            sample_solver='unipc',
                            offload_model=True,
                            **kwargs
                        )
                        
                        return result
                    
                    # Fallback to T2V with enhanced prompt if I2V not available
                    else:
                        print("⚠️ I2V model not available, using enhanced T2V")
                        enhanced_prompt = f"Starting from the given image, {prompt}. Maintain visual continuity."
                        
                        return self.__call__(
                            enhanced_prompt, height, width, num_frames, 
                            num_inference_steps, guidance_scale, **kwargs
                        )
                
                except Exception as e:
                    print(f"❌ I2V generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    raise RuntimeError(f"""
❌ CRITICAL: Wan I2V generation failed!

🔧 TROUBLESHOOTING:
1. ✅ Verify I2V model is properly loaded
2. 🖼️ Check input image format and size
3. 💾 Check available VRAM/memory
4. 📦 Verify Wan I2V dependencies

❌ NO FALLBACKS - Real Wan I2V implementation required!
Error: {e}
""")
        
        # Create and return the custom pipeline
        return CustomWanPipeline(model_info['path'], self.device)
    
    def _validate_wan_model(self, model_info: Dict) -> bool:
        """Validate Wan model has required files"""
        model_path = Path(model_info['path'])
        
        # Check for required files
        required_files = [
            "config.json"
        ]
        
        # Check for VAE and T5 files (different naming conventions)
        vae_files = ["Wan2.1_VAE.pth", "vae.pth", "vae.safetensors"]
        t5_files = ["models_t5_umt5-xxl-enc-bf16.pth", "t5.pth", "t5.safetensors"]
        
        missing_files = []
        
        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
        # Check for diffusion model files (single file OR multi-part)
        single_diffusion_file = model_path / "diffusion_pytorch_model.safetensors"
        multi_part_index = model_path / "diffusion_pytorch_model.safetensors.index.json"
        
        if single_diffusion_file.exists():
            print("✅ Found single diffusion model file")
        elif multi_part_index.exists():
            print("✅ Found multi-part diffusion model (14B style)")
            # Verify at least some multi-part files exist
            multi_part_files = list(model_path.glob("diffusion_pytorch_model-*-of-*.safetensors"))
            if not multi_part_files:
                missing_files.append("multi-part diffusion model files (diffusion_pytorch_model-*-of-*.safetensors)")
            else:
                print(f"✅ Found {len(multi_part_files)} multi-part diffusion files")
        else:
            missing_files.append("diffusion model (diffusion_pytorch_model.safetensors OR multi-part files)")
        
        # Check for at least one VAE file
        if not any((model_path / vae).exists() for vae in vae_files):
            missing_files.append("VAE file (Wan2.1_VAE.pth, vae.pth, or vae.safetensors)")
            
        # Check for at least one T5 file  
        if not any((model_path / t5).exists() for t5 in t5_files):
            missing_files.append("T5 file (models_t5_umt5-xxl-enc-bf16.pth, t5.pth, or t5.safetensors)")
        
        if missing_files:
            print(f"❌ Missing required model files: {missing_files}")
            return False
            
        print("✅ All required Wan model files found")
        return True
    
    def _generate_with_wan_pipeline(self, 
                                  prompt: str,
                                  width: int,
                                  height: int,
                                  num_frames: int,
                                  steps: int,
                                  guidance_scale: float,
                                  seed: int,
                                  output_path: str) -> bool:
        """Generate video using the loaded Wan pipeline"""
        try:
            if not self.pipeline:
                raise RuntimeError("Wan pipeline not loaded")
            
            print(f"🎬 Running Wan inference...")
            print(f"   📝 Prompt: {prompt[:50]}...")
            print(f"   📐 Size: {width}x{height}")
            print(f"   🎬 Frames: {num_frames}")
            print(f"   🔧 Steps: {steps}")
            print(f"   📏 Guidance: {guidance_scale}")
            
            # Set seed for reproducibility
            if seed > 0:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Prepare generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
            }
            
            # Add additional parameters if the pipeline supports them
            if hasattr(self.pipeline, '__call__'):
                import inspect
                sig = inspect.signature(self.pipeline.__call__)
                
                # Add seed if supported
                if 'generator' in sig.parameters and seed > 0:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(seed)
                    generation_kwargs['generator'] = generator
                
                # Add other common parameters
                if 'output_type' in sig.parameters:
                    generation_kwargs['output_type'] = 'pil'
                if 'return_dict' in sig.parameters:
                    generation_kwargs['return_dict'] = False
            
            print("🚀 Starting Wan model inference...")
            
            # Generate the video
            with torch.no_grad():
                result = self.pipeline(**generation_kwargs)
            
            # Handle different result formats
            if isinstance(result, tuple):
                frames = result[0]
            elif hasattr(result, 'frames'):
                frames = result.frames
            elif isinstance(result, list):
                frames = result
            else:
                frames = result
            
            # Save frames as video
            self._save_frames_as_video(frames, output_path, fps=8)
            
            print(f"✅ Wan video saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Wan pipeline generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Wan pipeline generation failed: {e}")
    
    def _save_frames_as_video(self, frame_paths_or_frames, output_path: str, fps: int = 8):
        """Save frames as video file - handles both PNG paths and frame arrays"""
        try:
            import imageio
            import numpy as np
            from PIL import Image
            
            # Check if we have file paths or frame data
            if isinstance(frame_paths_or_frames, list) and len(frame_paths_or_frames) > 0:
                if isinstance(frame_paths_or_frames[0], str):
                    # We have PNG file paths
                    print(f"💾 Creating video from {len(frame_paths_or_frames)} PNG files...")
                    
                    processed_frames = []
                    for i, frame_path in enumerate(frame_paths_or_frames):
                        # Load PNG file
                        pil_image = Image.open(frame_path)
                        frame_np = np.array(pil_image)
                        
                        # Ensure RGB format
                        if len(frame_np.shape) == 2:  # Grayscale
                            frame_np = np.stack([frame_np, frame_np, frame_np], axis=2)
                        elif len(frame_np.shape) == 3 and frame_np.shape[2] == 4:  # RGBA
                            frame_np = frame_np[:, :, :3]  # Remove alpha
                        
                        processed_frames.append(frame_np)
                        
                        if i % 10 == 0:
                            print(f"  📖 Loaded frame {i+1}/{len(frame_paths_or_frames)}")
                    
                    print(f"🎬 Saving {len(processed_frames)} frames to {output_path}")
                    imageio.mimsave(output_path, processed_frames, fps=fps, format='mp4')
                    print(f"✅ Video saved successfully with {len(processed_frames)} frames at {fps} FPS")
                    return
            
            # Fallback to original frame array handling
            frames = frame_paths_or_frames
            print(f"💾 Saving video with {len(frames) if hasattr(frames, '__len__') else 'unknown'} frames...")
            
            # Handle tensor format conversion
            if isinstance(frames, torch.Tensor):
                print(f"🔄 Converting tensor with shape: {frames.shape}")
                frames_np = frames.cpu().numpy()
                
                # Handle different tensor formats
                if len(frames_np.shape) == 4:  # (C, F, H, W) or (F, H, W, C)
                    if frames_np.shape[0] == 3:  # (C, F, H, W) - channels first
                        print("🔄 Converting from (C, F, H, W) to (F, H, W, C)")
                        frames_np = frames_np.transpose(1, 2, 3, 0)  # (F, H, W, C)
                    # else assume (F, H, W, C) already
                
                processed_frames = []
                for i in range(frames_np.shape[0]):
                    frame = frames_np[i]  # (H, W, C)
                    
                    # Ensure 3 channels
                    if len(frame.shape) == 2:  # Grayscale
                        frame = np.stack([frame, frame, frame], axis=2)
                    elif len(frame.shape) == 3 and frame.shape[2] == 1:  # (H, W, 1)
                        frame = np.repeat(frame, 3, axis=2)
                    elif len(frame.shape) == 3 and frame.shape[2] > 3:  # Too many channels
                        frame = frame[:, :, :3]
                    
                    # Convert to uint8
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                    
                    processed_frames.append(frame)
            
            else:
                # Handle list of frames
                processed_frames = []
                for i, frame in enumerate(frames):
                    if hasattr(frame, 'cpu'):
                        frame_np = frame.cpu().numpy()
                    elif isinstance(frame, np.ndarray):
                        frame_np = frame
                    else:
                        frame_np = np.array(frame)
                    
                    # Ensure correct format (H, W, C)
                    if len(frame_np.shape) == 4:  # (B, H, W, C)
                        frame_np = frame_np[0]
                    if len(frame_np.shape) == 3 and frame_np.shape[0] == 3:  # (C, H, W)
                        frame_np = frame_np.transpose(1, 2, 0)
                    
                    # Ensure 3 channels
                    if len(frame_np.shape) == 2:  # Grayscale
                        frame_np = np.stack([frame_np, frame_np, frame_np], axis=2)
                    elif len(frame_np.shape) == 3 and frame_np.shape[2] == 1:  # (H, W, 1)
                        frame_np = np.repeat(frame_np, 3, axis=2)
                    elif len(frame_np.shape) == 3 and frame_np.shape[2] > 3:  # Too many channels
                        frame_np = frame_np[:, :, :3]
                    
                    # Convert to uint8
                    if frame_np.dtype != np.uint8:
                        if frame_np.max() <= 1.0:
                            frame_np = (frame_np * 255).astype(np.uint8)
                        else:
                            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                    
                    processed_frames.append(frame_np)
            
            print(f"🎬 Saving {len(processed_frames)} frames to {output_path}")
            
            # Save as video
            imageio.mimsave(output_path, processed_frames, fps=fps, format='mp4')
            print(f"✅ Video saved successfully with {len(processed_frames)} frames at {fps} FPS")
            
        except Exception as e:
            print(f"❌ Failed to save video: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to save video: {e}")
    
    def generate_video_simple(self, 
                            prompt: str,
                            model_info: Dict,
                            output_dir: str,
                            width: int = 1280,
                            height: int = 720,
                            num_frames: int = 81,
                            steps: int = 20,
                            guidance_scale: float = 7.5,
                            seed: int = -1,
                            **kwargs) -> Optional[str]:
        """Generate video using simple Wan integration with PNG frame output and I2V chaining"""
        
        print(f"🎬 Generating video using SIMPLE Wan integration...")
        print(f"   📝 Prompt: {prompt}")
        print(f"   📐 Size: {width}x{height}")
        print(f"   🎬 Frames: {num_frames}")
        print(f"   📁 Model: {model_info['name']} ({model_info['type']}, {model_info['size']})")
        
        # Calculate Wan frames (4n+1 rule) and show frame discarding info
        wan_frames = self._calculate_wan_frames(num_frames)
        discard_info = self._calculate_frame_discarding(num_frames, wan_frames)
        
        print(f"📊 Wan Frame Calculation:")
        print(f"   🎯 Requested frames: {num_frames}")
        print(f"   🔧 Wan frames (4n+1): {wan_frames}")
        print(f"   🗑️ Frames to discard: {discard_info['discard_count']}")
        if discard_info['discard_count'] > 0:
            print(f"   📍 Discarded frame range: {discard_info['discard_start']}-{discard_info['discard_end']}")
        
        # Validate model first
        if not self._validate_wan_model(model_info):
            raise RuntimeError("Wan model validation failed - missing required files")
        
        # Load the model if not loaded
        if not self.pipeline:
            self.load_simple_wan_pipeline(model_info)  # This will raise if it fails
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate PNG frames instead of video
            frames_dir = os.path.join(output_dir, "wan_frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            print("🎬 Generating PNG frames with Wan model...")
            
            # Generate frames using Wan
            frames = self._generate_wan_frames(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=wan_frames,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                frames_dir=frames_dir
            )
            
            if frames:
                # Apply frame discarding if needed
                final_frames = self._apply_frame_discarding(frames, discard_info)
                
                print(f"✅ Generated {len(final_frames)} final frames (discarded {len(frames) - len(final_frames)})")
                
                # Save frame paths for potential I2V chaining
                frame_paths = self._save_frame_paths(final_frames, frames_dir)
                
                # Create video from final frames
                output_filename = f"wan_video_{int(time.time())}.mp4"
                output_path = os.path.join(output_dir, output_filename)
                
                self._save_frames_as_video(final_frames, output_path, fps=8)
                
                print(f"✅ Wan video generated: {output_path}")
                return output_path
            else:
                raise RuntimeError("Wan frame generation returned no result")
                
        except Exception as e:
            print(f"❌ Wan video generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Wan video generation failed: {e}")
    
    def _calculate_wan_frames(self, requested_frames: int) -> int:
        """Calculate the number of frames Wan should generate (4n+1 rule)"""
        # Wan requires frames to be 4n+1
        # Find the smallest 4n+1 that is >= requested_frames
        if (requested_frames - 1) % 4 == 0:
            return requested_frames  # Already follows 4n+1
        else:
            n = (requested_frames - 1) // 4 + 1
            return 4 * n + 1
    
    def _calculate_frame_discarding(self, requested_frames: int, wan_frames: int) -> Dict:
        """Calculate which frames to discard to match requested count"""
        discard_count = wan_frames - requested_frames
        
        if discard_count <= 0:
            return {
                'discard_count': 0,
                'discard_start': None,
                'discard_end': None,
                'keep_indices': list(range(wan_frames))
            }
        
        # Discard from the middle to preserve start and end frames
        start_keep = wan_frames // 3  # Keep first third
        end_keep = wan_frames - (wan_frames // 3)  # Keep last third
        
        # Calculate actual discard range
        discard_start = start_keep
        discard_end = discard_start + discard_count
        
        # Ensure we don't go out of bounds
        if discard_end > end_keep:
            discard_end = end_keep
            discard_start = discard_end - discard_count
        
        # Create list of indices to keep
        keep_indices = list(range(discard_start)) + list(range(discard_end, wan_frames))
        
        return {
            'discard_count': discard_count,
            'discard_start': discard_start,
            'discard_end': discard_end - 1,  # Make it inclusive
            'keep_indices': keep_indices[:requested_frames]  # Ensure exact count
        }
    
    def _generate_wan_frames(self,
                           prompt: str,
                           width: int,
                           height: int,
                           num_frames: int,
                           steps: int,
                           guidance_scale: float,
                           seed: int,
                           frames_dir: str) -> List:
        """Generate frames using Wan and save as PNGs"""
        try:
            if not self.pipeline:
                raise RuntimeError("Wan pipeline not loaded")
            
            print(f"🎬 Running Wan inference for {num_frames} frames...")
            print(f"   📝 Prompt: {prompt[:50]}...")
            print(f"   📐 Size: {width}x{height}")
            print(f"   🔧 Steps: {steps}")
            print(f"   📏 Guidance: {guidance_scale}")
            
            # Set seed for reproducibility
            if seed > 0:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                print(f"   🎲 Seed: {seed}")
            
            # Prepare generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
            }
            
            # Add seed if supported
            if hasattr(self.pipeline, '__call__'):
                import inspect
                sig = inspect.signature(self.pipeline.__call__)
                
                if 'generator' in sig.parameters and seed > 0:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(seed)
                    generation_kwargs['generator'] = generator
                
                if 'output_type' in sig.parameters:
                    generation_kwargs['output_type'] = 'pil'
                if 'return_dict' in sig.parameters:
                    generation_kwargs['return_dict'] = False
            
            print("🚀 Starting Wan frame generation...")
            
            # Generate the frames
            with torch.no_grad():
                result = self.pipeline(**generation_kwargs)
            
            # Handle different result formats
            if isinstance(result, tuple):
                frames = result[0]
            elif hasattr(result, 'frames'):
                frames = result.frames
            elif isinstance(result, list):
                frames = result
            else:
                frames = result
            
            print(f"🎬 Wan generation completed, processing {len(frames) if hasattr(frames, '__len__') else 'unknown'} frames...")
            
            # Save frames as PNGs
            saved_frames = self._save_frames_as_pngs(frames, frames_dir)
            
            print(f"✅ Generated and saved {len(saved_frames)} PNG frames")
            return saved_frames
            
        except Exception as e:
            print(f"❌ Wan frame generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Wan frame generation failed: {e}")
    
    def _save_frames_as_pngs(self, frames, frames_dir: str) -> List[str]:
        """Save frames as individual PNG files with improved error handling"""
        try:
            import imageio
            import numpy as np
            from PIL import Image
            
            saved_paths = []
            
            print(f"💾 Saving frames as PNGs to {frames_dir}...")
            
            # Handle tensor format conversion
            if isinstance(frames, torch.Tensor):
                print(f"🔄 Converting tensor with shape: {frames.shape}")
                frames_np = frames.cpu().numpy()
                
                # Handle different tensor formats
                if len(frames_np.shape) == 4:  # (C, F, H, W) or (F, H, W, C)
                    if frames_np.shape[0] == 3:  # (C, F, H, W) - channels first
                        print("🔄 Converting from (C, F, H, W) to (F, H, W, C)")
                        frames_np = frames_np.transpose(1, 2, 3, 0)  # (F, H, W, C)
                    # else assume (F, H, W, C) already
                elif len(frames_np.shape) == 5:  # (B, F, C, H, W) or (B, F, H, W, C)
                    print(f"🔄 Converting 5D tensor: {frames_np.shape}")
                    frames_np = frames_np[0]  # Remove batch dimension
                    if frames_np.shape[1] == 3:  # (F, C, H, W)
                        frames_np = frames_np.transpose(0, 2, 3, 1)  # (F, H, W, C)
                
                # Debug: Check tensor value range and normalize if needed
                print(f"🎨 Tensor value range: min={frames_np.min():.3f}, max={frames_np.max():.3f}")
                print(f"🎨 Tensor dtype: {frames_np.dtype}")
                
                # Normalize to [0, 1] if needed (Wan often outputs in [-1, 1] range)
                if frames_np.min() < 0 and frames_np.max() <= 1:
                    print("🔄 Normalizing from [-1, 1] to [0, 1] range")
                    frames_np = (frames_np + 1.0) / 2.0
                    frames_np = np.clip(frames_np, 0, 1)
                
                processed_frames = []
                for i in range(frames_np.shape[0]):
                    frame = frames_np[i]  # (H, W, C)
                    
                    # Ensure 3 channels
                    if len(frame.shape) == 2:  # Grayscale
                        frame = np.stack([frame, frame, frame], axis=2)
                    elif len(frame.shape) == 3 and frame.shape[2] == 1:  # (H, W, 1)
                        frame = np.repeat(frame, 3, axis=2)
                    elif len(frame.shape) == 3 and frame.shape[2] > 3:  # Too many channels
                        frame = frame[:, :, :3]
                    
                    # Convert to uint8
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                    
                    # Wan outputs RGB format consistently for both T2V and I2V
                    # No BGR conversion needed
                    processed_frames.append(frame)
            
            else:
                # Handle list of frames or PIL images
                processed_frames = []
                for i, frame in enumerate(frames):
                    if hasattr(frame, 'cpu'):  # PyTorch tensor
                        frame_np = frame.cpu().numpy()
                    elif isinstance(frame, np.ndarray):
                        frame_np = frame
                    elif hasattr(frame, 'save'):  # PIL Image
                        frame_np = np.array(frame)
                    else:
                        frame_np = np.array(frame)
                    
                    # Ensure correct format (H, W, C)
                    if len(frame_np.shape) == 4:  # (B, H, W, C)
                        frame_np = frame_np[0]
                    if len(frame_np.shape) == 3 and frame_np.shape[0] == 3:  # (C, H, W)
                        frame_np = frame_np.transpose(1, 2, 0)
                    
                    # Ensure 3 channels
                    if len(frame_np.shape) == 2:  # Grayscale
                        frame_np = np.stack([frame_np, frame_np, frame_np], axis=2)
                    elif len(frame_np.shape) == 3 and frame_np.shape[2] == 1:  # (H, W, 1)
                        frame_np = np.repeat(frame_np, 3, axis=2)
                    elif len(frame_np.shape) == 3 and frame_np.shape[2] > 3:  # Too many channels
                        frame_np = frame_np[:, :, :3]
                    
                    # Convert to uint8
                    if frame_np.dtype != np.uint8:
                        if frame_np.max() <= 1.0:
                            frame_np = (frame_np * 255).astype(np.uint8)
                        else:
                            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                    
                    # Wan outputs RGB format consistently for both T2V and I2V
                    # No BGR conversion needed
                    processed_frames.append(frame_np)
            
            # Save each frame as PNG
            for i, frame in enumerate(processed_frames):
                frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
                
                # Convert to PIL Image and save
                pil_image = Image.fromarray(frame)
                pil_image.save(frame_path, "PNG")
                
                saved_paths.append(frame_path)
                
                if i % 10 == 0:
                    print(f"  💾 Saved frame {i+1}/{len(processed_frames)}")
            
            print(f"✅ All {len(saved_paths)} frames saved as PNGs")
            return saved_paths
            
        except Exception as e:
            print(f"❌ Failed to save frames as PNGs: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to save frames as PNGs: {e}")
    
    def _apply_frame_discarding(self, frame_paths: List[str], discard_info: Dict) -> List[str]:
        """Apply frame discarding based on the calculated discard info"""
        if discard_info['discard_count'] <= 0:
            return frame_paths
        
        keep_indices = discard_info['keep_indices']
        final_frames = [frame_paths[i] for i in keep_indices if i < len(frame_paths)]
        
        print(f"🗑️ Discarded {len(frame_paths) - len(final_frames)} frames to match requested count")
        return final_frames
    
    def _save_frame_paths(self, frame_paths: List[str], frames_dir: str) -> str:
        """Save frame paths to a text file for potential I2V chaining"""
        paths_file = os.path.join(frames_dir, "frame_paths.txt")
        
        with open(paths_file, 'w') as f:
            for path in frame_paths:
                f.write(f"{path}\n")
        
        print(f"📝 Frame paths saved to: {paths_file}")
        return paths_file
    
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
                    
                print("🧹 Wan model unloaded, memory freed")
            except Exception as e:
                print(f"⚠️ Error unloading model: {e}")
    
    def generate_video_with_i2v_chaining(self, 
                                       clips: List[Dict],
                                       model_info: Dict,
                                       output_dir: str,
                                       width: int = 1280,
                                       height: int = 720,
                                       steps: int = 20,
                                       guidance_scale: float = 7.5,
                                       seed: int = -1,
                                       anim_args=None,
                                       wan_args=None,
                                       **kwargs) -> Optional[str]:
        """Generate video using I2V chaining for better continuity between clips - with unified frame output and strength scheduling"""
        try:
            import shutil
            import os
            from datetime import datetime
            
            print(f"🎬 Starting I2V chained generation with {len(clips)} clips...")
            print(f"📁 Model: {model_info['name']} ({model_info['type']}, {model_info['size']})")
            
            # Check for strength override
            use_strength_override = False
            fixed_strength = 0.85  # Default fallback
            
            if wan_args and hasattr(wan_args, 'wan_strength_override') and wan_args.wan_strength_override:
                use_strength_override = True
                fixed_strength = getattr(wan_args, 'wan_fixed_strength', 1.0)
                print(f"🔒 Using strength override: {fixed_strength} (ignoring Deforum schedules)")
            
            # Parse strength schedule if not overridden
            strength_values = {}
            if not use_strength_override and anim_args and hasattr(anim_args, 'strength_schedule'):
                try:
                    # Parse the strength schedule (format: "0: (0.85), 60: (0.7)")
                    import re
                    strength_schedule = anim_args.strength_schedule
                    print(f"🎯 Using Deforum strength schedule: {strength_schedule}")
                    
                    # Extract frame:value pairs
                    matches = re.findall(r'(\d+):\s*\(([0-9.]+)\)', strength_schedule)
                    for frame_str, strength_str in matches:
                        frame_num = int(frame_str)
                        strength_val = float(strength_str)
                        strength_values[frame_num] = strength_val
                        
                    print(f"📊 Parsed strength schedule: {strength_values}")
                except Exception as e:
                    print(f"⚠️ Failed to parse strength schedule: {e}, using default strength")
            
            # Validate model first
            if not self._validate_wan_model(model_info):
                raise RuntimeError("Wan model validation failed - missing required files")
            
            # Resolve T2V and I2V model paths based on UI selections
            if wan_args:
                try:
                    model_paths = self.resolve_model_paths(wan_args)
                    t2v_path = model_paths['t2v']
                    i2v_path = model_paths['i2v']
                    
                    print(f"🎯 Using resolved model paths:")
                    print(f"   🎬 T2V: {t2v_path}")
                    print(f"   🔗 I2V: {i2v_path}")
                    
                    # Load models separately if needed
                    if not self.pipeline or (t2v_path != model_info['path']):
                        print("🔧 Loading models with new resolution system...")
                        if not self.load_wan_models_separately(t2v_path, i2v_path):
                            raise RuntimeError("Failed to load resolved models")
                    
                except Exception as resolve_e:
                    print(f"⚠️ Model resolution failed: {resolve_e}")
                    print("🔄 Falling back to original model loading...")
                    # Fallback to original model loading
                    if not self.pipeline:
                        print("🔧 Loading Wan pipeline for I2V chaining...")
                        if not self.load_simple_wan_pipeline(model_info):
                            raise RuntimeError("Failed to load Wan pipeline")
            else:
                # Load the model if not loaded (original behavior)
                if not self.pipeline:
                    print("🔧 Loading Wan pipeline for I2V chaining...")
                    if not self.load_simple_wan_pipeline(model_info):
                        raise RuntimeError("Failed to load Wan pipeline")
            
            # Get the timestring from the output directory name or create one
            timestring = os.path.basename(output_dir).split('_')[-1]
            if not timestring or len(timestring) != 14:
                # Fallback: extract from directory name or create new
                dir_parts = os.path.basename(output_dir).split('_')
                for part in dir_parts:
                    if len(part) == 14 and part.isdigit():
                        timestring = part
                        break
                else:
                    # Create new timestring if none found
                    timestring = datetime.now().strftime("%Y%m%d%H%M%S")
            
            print(f"📁 Output directory: {output_dir}")
            print(f"🕐 Using timestring: {timestring}")
            
            # Create unified frames directory
            unified_frames_dir = output_dir
            os.makedirs(unified_frames_dir, exist_ok=True)
            
            all_frame_paths = []
            total_frame_idx = 0
            last_frame_path = None
            
            for clip_idx, clip in enumerate(clips):
                print(f"\n🎬 Generating clip {clip_idx + 1}/{len(clips)}")
                print(f"   📝 Prompt: {clip['prompt'][:50]}...")
                print(f"   🎞️ Frames: {clip['num_frames']}")
                
                # Calculate strength for this clip
                if use_strength_override:
                    clip_strength = fixed_strength
                    print(f"   🔒 Using override strength: {clip_strength}")
                elif strength_values:
                    # Find the appropriate strength value for this clip's start frame
                    clip_start_frame = clip['start_frame']
                    
                    # Find the closest strength value at or before this frame
                    applicable_frames = [f for f in strength_values.keys() if f <= clip_start_frame]
                    if applicable_frames:
                        closest_frame = max(applicable_frames)
                        clip_strength = strength_values[closest_frame]
                        print(f"   💪 Using strength {clip_strength} (from frame {closest_frame} schedule)")
                    else:
                        # Use the first strength value if no earlier frame found
                        first_frame = min(strength_values.keys())
                        clip_strength = strength_values[first_frame]
                        print(f"   💪 Using strength {clip_strength} (from first scheduled frame {first_frame})")
                else:
                    clip_strength = 0.85  # Default strength
                    print(f"   💪 Using default strength {clip_strength}")
                
                # Create temporary directory for this clip
                temp_clip_dir = os.path.join(output_dir, f"_temp_clip_{clip_idx:03d}")
                os.makedirs(temp_clip_dir, exist_ok=True)
                
                # Calculate frame discarding for this clip
                wan_frames = self._calculate_wan_frames(clip['num_frames'])
                discard_info = self._calculate_frame_discarding(clip['num_frames'], wan_frames)
                
                print(f"🎯 Wan will generate {wan_frames} frames, targeting {clip['num_frames']} final frames")
                if discard_info['discard_count'] > 0:
                    print(f"🗑️ Will discard {discard_info['discard_count']} frames from the middle to preserve start/end frames")
                
                # Generate frames for this clip
                if clip_idx == 0 or last_frame_path is None:
                    # First clip: use T2V
                    print("🚀 Using T2V for first clip (T2V model will be loaded)")
                    clip_frames = self._generate_wan_frames(
                        prompt=clip['prompt'],
                        width=width,
                        height=height,
                        num_frames=wan_frames,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        seed=seed if seed > 0 else -1,
                        frames_dir=temp_clip_dir
                    )
                else:
                    # Subsequent clips: use I2V with last frame and enhanced continuity
                    print(f"🔗 Using I2V chaining from: {os.path.basename(last_frame_path)}")
                    print(f"🔄 Model will switch: T2V → I2V (unload T2V, load I2V)")
                    print(f"💪 I2V strength: {clip_strength} (controls influence of previous frame)")
                    
                    # Enhanced I2V with proper parameters for maximum continuity
                    clip_frames = self._generate_wan_i2v_frames_enhanced(
                        prompt=clip['prompt'],
                        init_image_path=last_frame_path,
                        width=width,
                        height=height,
                        num_frames=wan_frames,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        seed=seed if seed > 0 else -1,
                        frames_dir=temp_clip_dir,
                        strength=clip_strength,
                        use_enhanced_continuity=True
                    )
                
                if not clip_frames:
                    raise RuntimeError(f"Failed to generate frames for clip {clip_idx + 1}")
                
                print(f"✅ Generated {len(clip_frames)} I2V frames")
                
                # Apply frame discarding
                final_clip_frames = self._apply_frame_discarding(clip_frames, discard_info)
                
                print(f"✅ Generated {len(final_clip_frames)} final frames for clip {clip_idx + 1}")
                
                # Copy frames to unified directory with continuous numbering and proper naming
                import shutil
                for frame_idx, src_path in enumerate(final_clip_frames):
                    # Use Deforum's naming convention: timestring_000000000.png
                    dst_filename = f"{timestring}_{total_frame_idx:09d}.png"
                    dst_path = os.path.join(unified_frames_dir, dst_filename)
                    
                    # Copy the frame
                    shutil.copy2(src_path, dst_path)
                    all_frame_paths.append(dst_path)
                    
                    total_frame_idx += 1
                
                # Clean up temporary clip directory
                shutil.rmtree(temp_clip_dir, ignore_errors=True)
                
                # Update last frame for next clip - use the EXACT last frame for maximum continuity
                if final_clip_frames:
                    last_frame_path = all_frame_paths[-1]  # Use the copied frame path
                    print(f"🔗 Last frame for next clip: {last_frame_path}")
            
            print(f"\n✅ All clips generated! Total frames: {len(all_frame_paths)}")
            print(f"📁 All frames saved to: {unified_frames_dir}")
            
            if use_strength_override:
                print(f"🔒 Strength override ({fixed_strength}) was applied across {len(clips)} clips")
            elif strength_values:
                print(f"💪 Strength scheduling was applied across {len(clips)} clips")
            
            # No need to create video here - Deforum will handle it with ffmpeg
            # Just return the output directory
            return unified_frames_dir
            
        except Exception as e:
            print(f"❌ I2V chained video generation failed: {e}")
            raise RuntimeError(f"I2V chained video generation failed: {e}")
        finally:
            # Clean up loaded models
            if hasattr(self.pipeline, 'cleanup'):
                print("🧹 Cleaning up Wan models...")
                self.pipeline.cleanup()
    
    def _generate_wan_i2v_frames_enhanced(self,
                               prompt: str,
                               init_image_path: str,
                               width: int,
                               height: int,
                               num_frames: int,
                               steps: int,
                               guidance_scale: float,
                               seed: int,
                               frames_dir: str,
                               strength: float = 0.85,
                               use_enhanced_continuity: bool = True) -> List[str]:
        """Generate frames using Wan I2V (Image-to-Video) mode with strength control and enhanced continuity"""
        try:
            from PIL import Image
            
            print(f"🎬 Running Enhanced Wan I2V inference for {num_frames} frames...")
            print(f"   🖼️ Init image: {init_image_path}")
            print(f"   📝 Prompt: {prompt[:50]}...")
            print(f"   📐 Size: {width}x{height}")
            print(f"   💪 Strength: {strength} (influence of init image)")
            print(f"   🔗 Enhanced continuity: {use_enhanced_continuity}")
            
            # Load and prepare the initial image
            if not os.path.exists(init_image_path):
                raise RuntimeError(f"Init image not found: {init_image_path}")
                
            init_image = Image.open(init_image_path)
            
            # Resize to match target dimensions if needed
            if init_image.size != (width, height):
                init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)
                print(f"🔄 Resized init image from {init_image.size} to {width}x{height}")
            
            # Set seed for reproducibility
            if seed > 0:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                print(f"   🎲 Seed: {seed}")
            
            # Enhanced I2V parameters for maximum continuity
            frames = None
            
            # Method 1: Check if pipeline has dedicated I2V method
            if hasattr(self.pipeline, 'generate_image2video'):
                print("🚀 Using dedicated I2V pipeline method with enhanced continuity")
                
                generation_kwargs = {
                    "image": init_image,
                    "prompt": prompt,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                }
                
                # Enhanced continuity parameters based on official Wan documentation
                import inspect
                sig = inspect.signature(self.pipeline.generate_image2video)
                
                if use_enhanced_continuity:
                    # Use official Wan I2V parameters for maximum continuity
                    if 'strength' in sig.parameters:
                        generation_kwargs['strength'] = strength
                        print(f"✅ Added strength parameter: {strength}")
                    
                    if 'image_guidance_scale' in sig.parameters:
                        # For maximum continuity, use high image guidance scale
                        # This makes the model follow the input image more closely
                        image_guidance = guidance_scale * strength
                        generation_kwargs['image_guidance_scale'] = image_guidance
                        print(f"✅ Using image_guidance_scale: {image_guidance} (enhanced for continuity)")
                    
                    if 'sample_guide_scale' in sig.parameters:
                        # Official Wan parameter for controlling adherence to input
                        sample_guide_scale = guidance_scale * (0.5 + 0.5 * strength)  # Scale between 50-100% of guidance
                        generation_kwargs['sample_guide_scale'] = sample_guide_scale
                        print(f"✅ Using sample_guide_scale: {sample_guide_scale}")
                    
                    if 'sample_shift' in sig.parameters:
                        # Lower shift values for more stability and continuity
                        sample_shift = max(1, int(5 * (1.0 - strength)))  # Inverse relationship with strength
                        generation_kwargs['sample_shift'] = sample_shift
                        print(f"✅ Using sample_shift: {sample_shift} (lower for more continuity)")
                    
                    if 'fast_mode' in sig.parameters:
                        # Use balanced or off mode for better quality continuity
                        generation_kwargs['fast_mode'] = 'Off'
                        print(f"✅ Using fast_mode: Off (for maximum quality)")
                else:
                    # Standard I2V parameters
                    if 'strength' in sig.parameters:
                        generation_kwargs['strength'] = strength
                        print(f"✅ Added strength parameter: {strength}")
                    elif 'image_guidance_scale' in sig.parameters:
                        image_guidance = guidance_scale * (1.0 - strength)
                        generation_kwargs['image_guidance_scale'] = image_guidance
                        print(f"✅ Using image_guidance_scale: {image_guidance} (derived from strength {strength})")
                
                # Add seed if supported
                if seed > 0 and 'generator' in sig.parameters:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(seed)
                    generation_kwargs['generator'] = generator
                
                with torch.no_grad():
                    result = self.pipeline.generate_image2video(**generation_kwargs)
                
                # Handle result format
                if isinstance(result, tuple):
                    frames = result[0]
                elif hasattr(result, 'frames'):
                    frames = result.frames
                elif isinstance(result, list):
                    frames = result
                else:
                    frames = result
            
            # Method 2: Check if pipeline supports image conditioning in main call
            elif hasattr(self.pipeline, '__call__'):
                import inspect
                sig = inspect.signature(self.pipeline.__call__)
                
                if 'image' in sig.parameters or 'init_image' in sig.parameters:
                    print("🚀 Using main pipeline with image conditioning and enhanced continuity")
                    
                    generation_kwargs = {
                        "prompt": prompt,
                        "height": height,
                        "width": width,
                        "num_frames": num_frames,
                        "num_inference_steps": steps,
                        "guidance_scale": guidance_scale,
                    }
                    
                    # Add image parameter
                    if 'image' in sig.parameters:
                        generation_kwargs['image'] = init_image
                    elif 'init_image' in sig.parameters:
                        generation_kwargs['init_image'] = init_image
                    
                    # Enhanced continuity parameters
                    if use_enhanced_continuity:
                        if 'strength' in sig.parameters:
                            generation_kwargs['strength'] = strength
                            print(f"✅ Added strength parameter: {strength}")
                        
                        if 'image_guidance_scale' in sig.parameters:
                            image_guidance = guidance_scale * strength
                            generation_kwargs['image_guidance_scale'] = image_guidance
                            print(f"✅ Using image_guidance_scale: {image_guidance} (enhanced)")
                        
                        if 'sample_guide_scale' in sig.parameters:
                            sample_guide_scale = guidance_scale * (0.5 + 0.5 * strength)
                            generation_kwargs['sample_guide_scale'] = sample_guide_scale
                            print(f"✅ Using sample_guide_scale: {sample_guide_scale}")
                        
                        if 'sample_shift' in sig.parameters:
                            sample_shift = max(1, int(5 * (1.0 - strength)))
                            generation_kwargs['sample_shift'] = sample_shift
                            print(f"✅ Using sample_shift: {sample_shift}")
                    else:
                        # Standard parameters
                        if 'strength' in sig.parameters:
                            generation_kwargs['strength'] = strength
                            print(f"✅ Added strength parameter: {strength}")
                        elif 'image_guidance_scale' in sig.parameters:
                            image_guidance = guidance_scale * (1.0 - strength)
                            generation_kwargs['image_guidance_scale'] = image_guidance
                            print(f"✅ Using image_guidance_scale: {image_guidance}")
                    
                    # Add seed if supported
                    if 'generator' in sig.parameters and seed > 0:
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(seed)
                        generation_kwargs['generator'] = generator
                    
                    with torch.no_grad():
                        result = self.pipeline(**generation_kwargs)
                    
                    # Handle result format
                    if isinstance(result, tuple):
                        frames = result[0]
                    elif hasattr(result, 'frames'):
                        frames = result.frames
                    elif isinstance(result, list):
                        frames = result
                    else:
                        frames = result
                else:
                    print("⚠️ Pipeline doesn't support image conditioning, using enhanced T2V")
                    frames = None
            
            # Method 3: Fallback to enhanced T2V with image-aware prompt
            if frames is None:
                print("🔄 Using enhanced T2V with image-aware prompt as I2V fallback")
                
                # Create a more detailed prompt that references the starting image and strength
                if use_enhanced_continuity and strength > 0.7:
                    continuity_desc = "maintaining strong visual continuity and consistency from the previous scene, keeping the same style, lighting, and composition"
                elif strength > 0.4:
                    continuity_desc = "with moderate visual continuity from the previous scene, preserving key visual elements"
                else:
                    continuity_desc = "with subtle visual continuity from the previous scene"
                
                enhanced_prompt = f"Continuing seamlessly from the previous scene, {prompt}. {continuity_desc}."
                
                generation_kwargs = {
                    "prompt": enhanced_prompt,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                }
                
                # Add seed if supported
                if hasattr(self.pipeline, '__call__'):
                    import inspect
                    sig = inspect.signature(self.pipeline.__call__)
                    
                    if 'generator' in sig.parameters and seed > 0:
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(seed)
                        generation_kwargs['generator'] = generator
                    
                    if 'output_type' in sig.parameters:
                        generation_kwargs['output_type'] = 'pil'
                    if 'return_dict' in sig.parameters:
                        generation_kwargs['return_dict'] = False
                
                with torch.no_grad():
                    result = self.pipeline(**generation_kwargs)
                
                # Handle result format
                if isinstance(result, tuple):
                    frames = result[0]
                elif hasattr(result, 'frames'):
                    frames = result.frames
                elif isinstance(result, list):
                    frames = result
                else:
                    frames = result
            
            if frames is None:
                raise RuntimeError("All enhanced I2V methods failed to generate frames")
            
            print(f"🎬 Enhanced I2V generation completed, processing {len(frames) if hasattr(frames, '__len__') else 'unknown'} frames...")
            
            # Save frames as PNG files
            frame_paths = self._save_frames_as_pngs(frames, frames_dir)
            
            print(f"✅ Enhanced I2V saved {len(frame_paths)} frames to {frames_dir}")
            return frame_paths
            
        except Exception as e:
            print(f"❌ Enhanced I2V frame generation failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _generate_wan_i2v_frames(self,
                               prompt: str,
                               init_image_path: str,
                               width: int,
                               height: int,
                               num_frames: int,
                               steps: int,
                               guidance_scale: float,
                               seed: int,
                               frames_dir: str,
                               strength: float = 0.85) -> List[str]:
        """Compatibility wrapper for the original I2V method - calls enhanced version"""
        return self._generate_wan_i2v_frames_enhanced(
            prompt=prompt,
            init_image_path=init_image_path,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            frames_dir=frames_dir,
            strength=strength,
            use_enhanced_continuity=False  # Use standard parameters for compatibility
        )
    
    def resolve_model_paths(self, wan_args) -> Dict[str, str]:
        """Resolve T2V and I2V model paths based on UI selections"""
        try:
            from .wan_model_downloader import WanModelDownloader
            
            downloader = WanModelDownloader()
            results = {}
            
            # Get UI selections
            t2v_selection = getattr(wan_args, 'wan_t2v_model', 'Auto-Detect')
            i2v_selection = getattr(wan_args, 'wan_i2v_model', 'Use T2V Model')
            auto_download = getattr(wan_args, 'wan_auto_download', True)
            preferred_size = getattr(wan_args, 'wan_preferred_size', '1.3B (Recommended)')
            custom_path = getattr(wan_args, 'wan_model_path', 'models/wan')
            
            print(f"🎯 Resolving models: T2V={t2v_selection}, I2V={i2v_selection}")
            print(f"   📥 Auto-download: {auto_download}")
            print(f"   📏 Preferred size: {preferred_size}")
            
            # Resolve T2V model
            if t2v_selection == "Auto-Detect":
                # Try to find existing models first
                discovered_models = self.discover_models()
                if discovered_models:
                    # Use best available model
                    best_model = discovered_models[0]
                    results['t2v'] = best_model['path']
                    print(f"✅ Auto-detected T2V model: {best_model['name']}")
                elif auto_download:
                    # Download preferred model
                    size_key = "1.3B" if preferred_size.startswith("1.3B") else "14B"
                    model_key = f"{size_key} T2V"
                    print(f"📥 Auto-downloading {model_key}...")
                    if downloader.download_model(model_key):
                        results['t2v'] = downloader.get_model_path(model_key)
                        print(f"✅ Downloaded T2V model: {model_key}")
                    else:
                        raise RuntimeError(f"Failed to download {model_key}")
                else:
                    raise RuntimeError("No T2V models found and auto-download disabled")
                    
            elif t2v_selection == "Custom Path":
                results['t2v'] = custom_path
                print(f"✅ Using custom T2V path: {custom_path}")
                
            else:
                # Specific model selected (e.g., "1.3B T2V", "14B T2V")
                model_key = t2v_selection
                if downloader.is_model_downloaded(model_key):
                    results['t2v'] = downloader.get_model_path(model_key)
                    print(f"✅ Using existing {model_key}")
                elif auto_download:
                    print(f"📥 Downloading {model_key}...")
                    if downloader.download_model(model_key):
                        results['t2v'] = downloader.get_model_path(model_key)
                        print(f"✅ Downloaded {model_key}")
                    else:
                        raise RuntimeError(f"Failed to download {model_key}")
                else:
                    raise RuntimeError(f"{model_key} not found and auto-download disabled")
            
            # Resolve I2V model
            if i2v_selection == "Use T2V Model":
                results['i2v'] = results['t2v']
                print(f"✅ Using T2V model for I2V: {results['t2v']}")
                
            elif i2v_selection == "Auto-Detect":
                # Try to find I2V models, fallback to T2V
                discovered_models = self.discover_models()
                i2v_models = [m for m in discovered_models if 'I2V' in m.get('type', '')]
                if i2v_models:
                    results['i2v'] = i2v_models[0]['path']
                    print(f"✅ Auto-detected I2V model: {i2v_models[0]['name']}")
                else:
                    results['i2v'] = results['t2v']
                    print(f"✅ No I2V models found, using T2V for I2V")
                    
            elif i2v_selection == "Use T2V Model (No Continuity)":
                results['i2v'] = results['t2v']
                print(f"⚠️ Using T2V model for I2V - NO CONTINUITY between clips!")
                    
            elif i2v_selection == "Custom Path":
                results['i2v'] = custom_path
                print(f"✅ Using custom I2V path: {custom_path}")
                
            else:
                # Specific I2V model selection (14B I2V 720P, 14B I2V 480P)
                if "14B I2V 720P" in i2v_selection:
                    target_model = "14B I2V 720P"
                elif "14B I2V 480P" in i2v_selection:
                    target_model = "14B I2V 480P"
                elif "1.3B I2V" in i2v_selection:
                    # 1.3B I2V doesn't exist, fallback to T2V with warning
                    print("⚠️ 1.3B I2V model doesn't exist! Using T2V model instead.")
                    print("💡 Tip: Use '14B I2V 720P' or '14B I2V 480P' for real I2V functionality.")
                    results['i2v'] = results['t2v']
                    return results
                elif "14B I2V" in i2v_selection:
                    # Legacy support - default to 720P
                    target_model = "14B I2V 720P"
                else:
                    target_model = i2v_selection
                
                # Try to download if auto-download enabled
                if auto_download:
                    try:
                        from .wan_model_downloader import WanModelDownloader
                        downloader = WanModelDownloader()
                        
                        if downloader.download_model(target_model):
                            results['i2v'] = downloader.get_model_path(target_model)
                            print(f"✅ Downloaded {target_model}")
                        else:
                            print(f"❌ Failed to download {target_model}, using T2V for I2V")
                            results['i2v'] = results['t2v']
                    except Exception as e:
                        print(f"❌ Download failed for {target_model}: {e}, using T2V for I2V")
                        results['i2v'] = results['t2v']
                else:
                    print(f"❌ {target_model} not found and auto-download disabled, using T2V for I2V")
                    results['i2v'] = results['t2v']
            
            print(f"✅ Model resolution complete:")
            print(f"   🎬 T2V: {results['t2v']}")
            print(f"   🔗 I2V: {results['i2v']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Model resolution failed: {e}")
            # Fallback to discovery
            discovered_models = self.discover_models()
            if discovered_models:
                fallback_path = discovered_models[0]['path']
                print(f"🔄 Using fallback model: {fallback_path}")
                return {'t2v': fallback_path, 'i2v': fallback_path}
            else:
                raise RuntimeError(f"Model resolution failed and no fallback available: {e}")
    
    def load_wan_models_separately(self, t2v_path: str, i2v_path: str) -> bool:
        """Load separate T2V and I2V models"""
        try:
            print(f"🚀 Loading separate Wan models...")
            print(f"   🎬 T2V: {t2v_path}")
            print(f"   🔗 I2V: {i2v_path}")
            
            # Validate paths
            from pathlib import Path
            t2v_dir = Path(t2v_path)
            i2v_dir = Path(i2v_path)
            
            if not t2v_dir.exists():
                raise RuntimeError(f"T2V model path does not exist: {t2v_path}")
            if not i2v_dir.exists():
                raise RuntimeError(f"I2V model path does not exist: {i2v_path}")
            
            # Load T2V model
            t2v_model_info = {'path': t2v_path, 'name': f'T2V-{t2v_dir.name}', 'type': 'T2V'}
            if not self._validate_wan_model(t2v_model_info):
                raise RuntimeError(f"T2V model validation failed: {t2v_path}")
            
            # Load I2V model (if different from T2V)
            if i2v_path != t2v_path:
                i2v_model_info = {'path': i2v_path, 'name': f'I2V-{i2v_dir.name}', 'type': 'I2V'}
                if not self._validate_wan_model(i2v_model_info):
                    print(f"⚠️ I2V model validation failed, will use T2V model: {i2v_path}")
                    i2v_path = t2v_path
            
            # Create pipeline with separate models
            self.pipeline = self._create_custom_wan_pipeline_with_separate_models(t2v_path, i2v_path)
            
            if self.pipeline:
                print(f"✅ Separate Wan models loaded successfully")
                return True
            else:
                print(f"❌ Failed to create pipeline with separate models")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load separate Wan models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_custom_wan_pipeline_with_separate_models(self, t2v_path: str, i2v_path: str):
        """Create a custom Wan pipeline with separate T2V and I2V models"""
        
        class SeparateWanPipeline:
            def __init__(self, t2v_path: str, i2v_path: str, device: str):
                self.t2v_path = t2v_path
                self.i2v_path = i2v_path
                self.device = device
                self.current_model = None
                self.current_model_type = None
                self.loaded = False
                
                print(f"🔧 Initializing separate Wan pipeline with model switching")
                print(f"   🎬 T2V: {t2v_path}")
                print(f"   🔗 I2V: {i2v_path}")
                
                # Start by loading T2V model for first clip
                self._load_t2v_model()
            
            def _unload_current_model(self):
                """Unload the currently loaded model to free memory"""
                if self.current_model is not None:
                    print(f"🧹 Unloading {self.current_model_type} model to free memory...")
                    try:
                        # Move model to CPU and delete references
                        if hasattr(self.current_model, 'to'):
                            self.current_model.to('cpu')
                        del self.current_model
                        self.current_model = None
                        self.current_model_type = None
                        
                        # Force garbage collection and clear CUDA cache
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        print(f"✅ Model unloaded, memory freed")
                    except Exception as e:
                        print(f"⚠️ Warning during model unload: {e}")
            
            def _load_t2v_model(self):
                """Load T2V model"""
                try:
                    print("🔄 Loading T2V model...")
                    self._unload_current_model()  # Unload any existing model first
                    
                    # Import and setup Wan components
                    import sys
                    from pathlib import Path
                    
                    # Look for Wan2.1 directory in the extension root
                    extension_root = Path(__file__).parent.parent.parent
                    wan_repo_path = extension_root / "Wan2.1"
                    
                    if wan_repo_path.exists() and (wan_repo_path / "wan").exists():
                        if str(wan_repo_path) not in sys.path:
                            sys.path.insert(0, str(wan_repo_path))
                        
                        # Apply Flash Attention compatibility patches
                        try:
                            from scripts.deforum_helpers.wan_flash_attention_patch import apply_wan_compatibility_patches
                            apply_wan_compatibility_patches()
                        except Exception as patch_e:
                            print(f"⚠️ Could not apply compatibility patches: {patch_e}")
                        
                        # Import Wan components
                        import wan  # type: ignore
                        from wan.text2video import WanT2V  # type: ignore
                        
                        # Load configs
                        try:
                            from wan.configs.wan_t2v_14B import t2v_14B as t2v_config
                        except ImportError:
                            # Create minimal config structure
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
                                        'num_vector_embeds': None,
                                        'activation_fn': "geglu",
                                        'num_embeds_ada_norm': 1000,
                                        'norm_elementwise_affine': False,
                                        'norm_eps': 1e-6,
                                        'attention_bias': True,
                                        'caption_channels': 4096
                                    })
                            t2v_config = MinimalConfig()
                        
                        # Initialize T2V model
                        print(f"🚀 Initializing WanT2V with checkpoint dir: {self.t2v_path}")
                        self.current_model = WanT2V(
                            config=t2v_config,
                            checkpoint_dir=self.t2v_path,
                            device_id=0,
                            rank=0,
                            dit_fsdp=False,
                            t5_fsdp=False
                        )
                        
                        self.current_model_type = "T2V"
                        self.loaded = True
                        print("✅ T2V model loaded successfully")
                        return True
                    else:
                        raise RuntimeError("Wan repository not found")
                    
                except Exception as e:
                    print(f"❌ T2V model loading failed: {e}")
                    return False
            
            def _load_i2v_model(self):
                """Load I2V model"""
                try:
                    # If I2V path is same as T2V, just switch mode (don't reload)
                    if self.i2v_path == self.t2v_path:
                        print("✅ Using T2V model for I2V (same path)")
                        if self.current_model_type != "T2V":
                            return self._load_t2v_model()
                        return True
                    
                    print("🔄 Loading I2V model...")
                    self._unload_current_model()  # Unload T2V model first
                    
                    # Import and setup Wan components
                    import sys
                    from pathlib import Path
                    
                    # Look for Wan2.1 directory in the extension root
                    extension_root = Path(__file__).parent.parent.parent
                    wan_repo_path = extension_root / "Wan2.1"
                    
                    if wan_repo_path.exists() and (wan_repo_path / "wan").exists():
                        if str(wan_repo_path) not in sys.path:
                            sys.path.insert(0, str(wan_repo_path))
                        
                        # Apply Flash Attention compatibility patches
                        try:
                            from scripts.deforum_helpers.wan_flash_attention_patch import apply_wan_compatibility_patches
                            apply_wan_compatibility_patches()
                        except Exception as patch_e:
                            print(f"⚠️ Could not apply compatibility patches: {patch_e}")
                        
                        # Import Wan components
                        import wan  # type: ignore
                        from wan.image2video import WanI2V  # type: ignore
                        
                        # Load configs
                        try:
                            from wan.configs.wan_i2v_14B import i2v_14B as i2v_config
                        except ImportError:
                            # Create minimal config structure
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
                                        'num_vector_embeds': None,
                                        'activation_fn': "geglu",
                                        'num_embeds_ada_norm': 1000,
                                        'norm_elementwise_affine': False,
                                        'norm_eps': 1e-6,
                                        'attention_bias': True,
                                        'caption_channels': 4096
                                    })
                            i2v_config = MinimalConfig()
                        
                        # Initialize I2V model
                        print(f"🚀 Initializing WanI2V with checkpoint dir: {self.i2v_path}")
                        self.current_model = WanI2V(
                            config=i2v_config,
                            checkpoint_dir=self.i2v_path,
                            device_id=0,
                            rank=0,
                            dit_fsdp=False,
                            t5_fsdp=False
                        )
                        
                        self.current_model_type = "I2V"
                        self.loaded = True
                        print("✅ I2V model loaded successfully")
                        return True
                    else:
                        raise RuntimeError("Wan repository not found")
                    
                except Exception as e:
                    print(f"❌ I2V model loading failed: {e}")
                    # Fallback to T2V if I2V fails
                    print("🔄 Falling back to T2V model...")
                    return self._load_t2v_model()
            
            def _ensure_model_loaded(self, model_type: str):
                """Ensure the correct model is loaded for the operation"""
                if model_type == "T2V" and self.current_model_type != "T2V":
                    return self._load_t2v_model()
                elif model_type == "I2V" and self.current_model_type != "I2V":
                    return self._load_i2v_model()
                return True
            
            def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                """Generate video frames using T2V model"""
                if not self._ensure_model_loaded("T2V"):
                    raise RuntimeError("Failed to load T2V model")
                
                print(f"🎬 Generating {num_frames} frames with T2V model...")
                
                try:
                    result = self.current_model.generate(
                        input_prompt=prompt,
                        size=(width, height),
                        frame_num=num_frames,
                        sampling_steps=num_inference_steps,
                        guide_scale=guidance_scale,
                        shift=5.0,
                        sample_solver='unipc',
                        offload_model=True,
                        **kwargs
                    )
                    return result
                    
                except Exception as e:
                    print(f"❌ T2V generation failed: {e}")
                    raise RuntimeError(f"T2V generation failed: {e}")
            
            def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                """Generate video from image using I2V model (with model switching)"""
                print(f"🎬 Generating I2V {num_frames} frames...")
                
                # Try to load I2V model (will unload T2V if different)
                if not self._ensure_model_loaded("I2V"):
                    print("⚠️ Failed to load I2V model, falling back to T2V with enhanced prompt")
                    # Fallback to T2V with enhanced prompt
                    enhanced_prompt = f"Starting from the given image, {prompt}. Maintain visual continuity."
                    
                    if not self._ensure_model_loaded("T2V"):
                        raise RuntimeError("Failed to load any model")
                    
                    return self.__call__(
                        enhanced_prompt, height, width, num_frames, 
                        num_inference_steps, guidance_scale, **kwargs
                    )
                
                try:
                    # Use dedicated I2V model if successfully loaded
                    if self.current_model_type == "I2V":
                        print("🚀 Using dedicated I2V model")
                        result = self.current_model.generate(
                            input_prompt=prompt,
                            img=image,
                            max_area=height * width,
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps,
                            guide_scale=guidance_scale,
                            shift=5.0,
                            sample_solver='unipc',
                            offload_model=True,
                            **kwargs
                        )
                        return result
                    
                    # If we somehow don't have I2V, use T2V with enhanced prompt
                    else:
                        print("⚠️ Warning: Using T2V model for I2V (enhanced prompt)")
                        enhanced_prompt = f"Starting from the given image, {prompt}. Maintain visual continuity."
                        
                        return self.__call__(
                            enhanced_prompt, height, width, num_frames, 
                            num_inference_steps, guidance_scale, **kwargs
                        )
                
                except Exception as e:
                    print(f"❌ I2V generation failed: {e}")
                    raise RuntimeError(f"I2V generation failed: {e}")
            
            def cleanup(self):
                """Clean up and unload all models"""
                self._unload_current_model()
        
        # Create and return the separate pipeline
        return SeparateWanPipeline(t2v_path, i2v_path, self.device)

def generate_video_with_simple_wan(prompt: str, 
                                 output_dir: str,
                                 width: int = 1280,
                                 height: int = 720,
                                 num_frames: int = 81,
                                 steps: int = 20,
                                 guidance_scale: float = 7.5,
                                 seed: int = -1,
                                 **kwargs) -> str:
    """Convenience function to generate video using Wan - fail fast if no models"""
    
    integration = WanSimpleIntegration()
    
    # Find the best model
    best_model = integration.get_best_model()
    if not best_model:
        raise RuntimeError("""❌ No Wan models found!

💡 SOLUTION: Download a Wan model first:
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./models/wan

Then restart generation.""")
    
    # Generate video - let errors propagate
    try:
        result = integration.generate_video_simple(
            prompt=prompt,
            model_info=best_model,
            output_dir=output_dir,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            **kwargs
        )
        
        return result
        
    finally:
        # Always clean up
        integration.unload_model()

if __name__ == "__main__":
    # Test the simple integration
    print("🧪 Testing Wan Simple Integration...")
    
    integration = WanSimpleIntegration()
    
    # Test model discovery
    models = integration.discover_models()
    if models:
        print(f"✅ Found {len(models)} model(s)")
        best = integration.get_best_model()
        print(f"🏆 Best model: {best['name']} ({best['type']}, {best['size']})")
        
        # Test simple loading
        if integration.load_simple_wan_pipeline(best):
            print("✅ Simple Wan integration ready")
            
            # Test demo generation
            output = integration.generate_video_simple(
                prompt="A beautiful landscape",
                model_info=best,
                output_dir="./test_output",
                num_frames=10
            )
            
            if output:
                print(f"✅ Demo generation successful: {output}")
            else:
                print("❌ Demo generation failed")
        else:
            print("❌ Failed to load Wan models")
    else:
        print("❌ No models found")
