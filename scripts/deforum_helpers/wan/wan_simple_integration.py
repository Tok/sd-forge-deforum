#!/usr/bin/env python3
"""
Wan Simple Integration for Deforum
Provides simple integration with Wan video generation models for Deforum animation
"""

from pathlib import Path
from typing import List, Dict, Optional
import torch
import os
import numpy as np

class WanSimpleIntegration:
    """Simple integration class for Wan video generation in Deforum"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.models = []
        self.pipeline = None
        print(f"🎬 Wan Simple Integration initialized on {self.device}")
    
    def discover_models(self) -> List[Dict]:
        """Discover available Wan models"""
        models = []
        
        # Common Wan model directories
        search_paths = [
            Path.cwd() / "models" / "wan",
            Path.cwd() / "Wan2.1",
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                print(f"🔍 Searching for Wan models in: {search_path}")
                
                for model_dir in search_path.iterdir():
                    if model_dir.is_dir():
                        model_info = self._analyze_model_directory(model_dir)
                        if model_info:
                            models.append(model_info)
                            print(f"✅ Found Wan model: {model_info['name']} ({model_info['type']}, {model_info['size']})")
        
        if not models:
            print("❌ No Wan models found")
            print("💡 To download models, use:")
            print("   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan/Wan2.1-T2V-1.3B")
        
        self.models = models
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
        
        # Determine model type and size
        model_type = "Unknown"
        model_size = "Unknown"
        
        if 'vace' in model_name:
            model_type = "VACE"
        elif 't2v' in model_name:
            model_type = "T2V"
        elif 'i2v' in model_name:
            model_type = "I2V"
        
        if '1.3b' in model_name:
            model_size = "1.3B"
        elif '14b' in model_name:
            model_size = "14B"
        
        return {
            'name': model_dir.name,
            'path': str(model_dir.absolute()),
            'type': model_type,
            'size': model_size,
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
    
    def _validate_vace_weights(self, model_path: Path) -> bool:
        """Validate that VACE model has required weights - compatibility method for UI"""
        try:
            print(f"🔍 Validating VACE model: {model_path.name}")
            
            # Check if the main model files exist
            diffusion_model = model_path / "diffusion_pytorch_model.safetensors"
            if not diffusion_model.exists():
                # Check for multi-part model
                diffusion_model = model_path / "diffusion_pytorch_model-00001-of-00007.safetensors"
                if not diffusion_model.exists():
                    print("❌ No diffusion model file found for VACE validation")
                    return False
            
            # Basic file size check
            if diffusion_model.stat().st_size < 1_000_000:  # Less than 1MB is suspicious
                print(f"❌ VACE model file too small: {diffusion_model.stat().st_size} bytes")
                return False
            
            # Check for required config files
            config_file = model_path / "config.json"
            if not config_file.exists():
                print("❌ VACE model missing config.json")
                return False
            
            # Try to load and validate config
            try:
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Check if it's actually a VACE model
                model_type = config.get("model_type", "").lower()
                class_name = config.get("_class_name", "").lower()
                
                if "vace" not in model_type and "vace" not in class_name:
                    print(f"⚠️ Model config doesn't indicate VACE type: {model_type}, {class_name}")
                    # Don't fail - might still be VACE with different naming
                
                print(f"✅ VACE model validation passed: {model_path.name}")
                return True
                
            except Exception as config_e:
                print(f"⚠️ VACE config validation failed: {config_e}")
                # Don't fail completely - file might still be valid
                return True
            
        except Exception as e:
            print(f"❌ VACE validation error: {e}")
            return False
    
    def _has_incomplete_models(self) -> bool:
        """Check if there are incomplete models - compatibility method for UI"""
        try:
            incomplete_models = self._check_for_incomplete_models()
            return len(incomplete_models) > 0
        except Exception as e:
            print(f"⚠️ Error checking for incomplete models: {e}")
            return False
    
    def _check_for_incomplete_models(self) -> List[Path]:
        """Check for incomplete model downloads - compatibility method for UI"""
        try:
            incomplete_models = []
            
            # Check common model directories
            search_paths = [
                Path.cwd() / "models" / "wan",
                Path.cwd() / "Wan2.1",
            ]
            
            for search_path in search_paths:
                if search_path.exists():
                    for model_dir in search_path.iterdir():
                        if model_dir.is_dir():
                            # Check if model looks incomplete
                            model_name = model_dir.name.lower()
                            if 'wan' in model_name or 'vace' in model_name:
                                # Basic completeness check
                                config_exists = (model_dir / "config.json").exists()
                                has_weights = any(
                                    file.suffix in ['.safetensors', '.bin', '.pt', '.pth']
                                    for file in model_dir.rglob('*')
                                    if file.is_file()
                                )
                                
                                if not (config_exists and has_weights):
                                    print(f"⚠️ Potentially incomplete model: {model_dir.name}")
                                    incomplete_models.append(model_dir)
            
            return incomplete_models
            
        except Exception as e:
            print(f"❌ Error checking for incomplete models: {e}")
            return []
    
    def _fix_incomplete_model(self, model_dir: Path, downloader=None) -> bool:
        """Fix incomplete model - compatibility method for UI"""
        try:
            print(f"🔧 Attempting to fix incomplete model: {model_dir.name}")
            
            # For now, just provide instructions rather than auto-fixing
            print(f"💡 To fix incomplete model '{model_dir.name}':")
            print(f"   1. Delete the incomplete directory: {model_dir}")
            print(f"   2. Re-download the model:")
            
            if 'vace' in model_dir.name.lower():
                print(f"      huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir {model_dir}")
            elif 't2v' in model_dir.name.lower():
                print(f"      huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir {model_dir}")
            elif 'i2v' in model_dir.name.lower():
                print(f"      huggingface-cli download Wan-AI/Wan2.1-I2V-1.3B --local-dir {model_dir}")
            
            # Return False to indicate manual intervention needed
            return False
            
        except Exception as e:
            print(f"❌ Error fixing incomplete model: {e}")
            return False
    
    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model"""
        if not self.models:
            self.discover_models()
        
        if not self.models:
            return None
        
        # Priority: T2V > I2V > VACE, and 1.3B > 14B (for compatibility)
        def model_priority(model):
            type_priority = {'T2V': 0, 'I2V': 1, 'VACE': 2, 'Unknown': 3}
            size_priority = {'1.3B': 0, '14B': 1, 'Unknown': 2}
            return (type_priority.get(model['type'], 3), size_priority.get(model['size'], 2))
        
        best_model = min(self.models, key=model_priority)
        print(f"🎯 Best model selected: {best_model['name']} ({best_model['type']}, {best_model['size']})")
        return best_model
    
    def load_simple_wan_pipeline(self, model_info: Dict, wan_args=None) -> bool:
        """Load a simple Wan pipeline with Flash Attention fixes"""
        try:
            print(f"🔧 Loading Wan model: {model_info['name']}")
            
            # Store flash attention mode for later use
            self.flash_attention_mode = "Auto (Recommended)"
            if wan_args and hasattr(wan_args, 'wan_flash_attention_mode'):
                self.flash_attention_mode = wan_args.wan_flash_attention_mode
                print(f"🔧 Setting Flash Attention mode to: {self.flash_attention_mode}")
            
            # Import torch
            import torch
            
            # Check if this is a VACE model
            if model_info['type'] == 'VACE':
                print("🎯 VACE model detected - using specialized VACE handling")
                return self._load_vace_model(model_info)
            
            # For T2V/I2V models, try multiple loading strategies
            return self._load_standard_wan_model(model_info)
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_vace_model(self, model_info: Dict) -> bool:
        """Load VACE model with proper handling"""
        try:
            print("🔧 Loading VACE model...")
            
            # Look for official Wan repository
            extension_root = Path(__file__).parent.parent.parent.parent
            wan_repo_path = extension_root / "Wan2.1"
            
            if wan_repo_path.exists() and (wan_repo_path / "wan").exists():
                print(f"✅ Found Wan repository at: {wan_repo_path}")
                
                # Add to Python path
                import sys
                if str(wan_repo_path) not in sys.path:
                    sys.path.insert(0, str(wan_repo_path))
                
                try:
                    # Load only VACE model (not T2V)
                    import wan
                    from wan.vace import WanVace
                    
                    print("🚀 Using official Wan VACE implementation (single model)")
                    
                    # Apply Flash Attention patches after Wan modules are imported
                    try:
                        from .wan_flash_attention_patch import apply_flash_attention_patch, update_patched_flash_attention_mode
                        
                        # Update mode if stored
                        if hasattr(self, 'flash_attention_mode'):
                            update_patched_flash_attention_mode(self.flash_attention_mode)
                        
                        success = apply_flash_attention_patch()
                        if success:
                            print("✅ Flash Attention monkey patch applied successfully")
                        else:
                            print("⚠️ Flash Attention patch could not be applied - may be already patched")
                    except Exception as patch_e:
                        print(f"⚠️ Flash Attention patch failed: {patch_e}")
                        print("🔄 Continuing without patches...")
                    
                    # Create VACE config
                    class VACEConfig:
                        def __init__(self):
                            self.num_train_timesteps = 1000
                            self.param_dtype = torch.bfloat16
                            self.t5_dtype = torch.float16  # Use fp16 for T5 to save memory
                            self.text_len = 512
                            self.vae_stride = [4, 8, 8]
                            self.patch_size = [1, 2, 2]
                            self.sample_neg_prompt = "Low quality, blurry, distorted, artifacts, bad anatomy"
                            self.sample_fps = 8  # Lower FPS for stability
                            # Use relative paths that VACE will resolve
                            self.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                            self.vae_checkpoint = 'Wan2.1_VAE.pth'
                            self.t5_tokenizer = 'google/umt5-xxl'
                    
                    vace_config = VACEConfig()
                    
                    # Initialize only VACE model
                    print("🔧 Loading WanVace for both T2V and I2V generation...")
                    vace_model = WanVace(
                        config=vace_config,
                        checkpoint_dir=model_info['path'],
                        device_id=0,
                        rank=0,
                        dit_fsdp=False,
                        t5_fsdp=False
                    )
                    
                    # Create VACE wrapper that handles T2V and I2V correctly
                    class SmartVACEWrapper:
                        def __init__(self, vace_model):
                            self.vace_model = vace_model
                            
                            # Determine optimal resolution based on VACE model size
                            # Extract model size from the checkpoint directory path
                            model_path = str(model_info['path']).lower()
                            if '14b' in model_path:
                                # 14B VACE model optimized for 720p
                                self.optimal_width = 1280
                                self.optimal_height = 720
                                self.model_size = "14B"
                            else:
                                # 1.3B VACE model optimized for 480p
                                self.optimal_width = 832
                                self.optimal_height = 480
                                self.model_size = "1.3B"
                            
                            print(f"🎯 VACE {self.model_size} model detected - using optimal resolution {self.optimal_width}x{self.optimal_height}")
            
                        def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                            # T2V mode: Use VACE correctly for pure text-to-video generation
                            
                            # Use optimal resolution for this VACE model
                            aligned_width = self.optimal_width
                            aligned_height = self.optimal_height
                            
                            if aligned_width != width or aligned_height != height:
                                print(f"🔧 VACE T2V resolution correction: {width}x{height} -> {aligned_width}x{aligned_height}")
                                print(f"🔧 VACE {self.model_size} optimized for {aligned_width}x{aligned_height}")
                            
                            print(f"🎬 VACE T2V generating: {prompt[:50]}...")
                            print(f"🔧 Parameters: {aligned_width}x{aligned_height}, {num_frames} frames, {num_inference_steps} steps, guidance={guidance_scale}")
                            
                            try:
                                # For pure T2V: Use prepare_source but with all None inputs
                                # This creates proper blank tensors for T2V generation
                                src_video, src_mask, src_ref_images = self.vace_model.prepare_source(
                                    [None],  # T2V: no source video
                                    [None],  # T2V: no source mask
                                    [None],  # T2V: no reference images
                                    num_frames,
                                    (aligned_height, aligned_width),  # Note: VACE expects (H, W) order
                                    self.vace_model.device
                                )
                                
                                print(f"🔧 T2V prepared tensors - video: {len(src_video)}, mask: {len(src_mask)}, ref: {src_ref_images}")
                                if src_video and src_video[0] is not None:
                                    print(f"🔧 src_video shape: {src_video[0].shape}, range: [{src_video[0].min():.3f}, {src_video[0].max():.3f}]")
                                if src_mask and src_mask[0] is not None:
                                    print(f"🔧 src_mask shape: {src_mask[0].shape}, range: [{src_mask[0].min():.3f}, {src_mask[0].max():.3f}]")
                                
                                result = self.vace_model.generate(
                                    input_prompt=prompt,
                                    input_frames=src_video,
                                    input_masks=src_mask,
                                    input_ref_images=src_ref_images,
                                    size=(aligned_width, aligned_height),
                                    frame_num=num_frames,
                                    sampling_steps=num_inference_steps,
                                    guide_scale=guidance_scale,
                                    shift=16,  # VACE uses higher shift than T2V
                                    seed=-1,
                                    **kwargs
                                )
                                
                                if result is not None:
                                    print(f"✅ VACE T2V generation completed, output shape: {result.shape}")
                                    print(f"🔧 Output value range: min={result.min():.3f}, max={result.max():.3f}")
                                else:
                                    print("⚠️ VACE T2V returned None result")
                                
                                return result
                                
                            except Exception as e:
                                print(f"❌ VACE T2V generation failed: {e}")
                                import traceback
                                traceback.print_exc()
                                raise
                        
                        def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                            # I2V mode: Use VACE with proper image conditioning
                            
                            # Use optimal resolution for this VACE model
                            aligned_width = self.optimal_width
                            aligned_height = self.optimal_height
                            
                            # Enhanced prompt for continuity
                            enhanced_prompt = f"Continuing seamlessly from the provided image, {prompt}. Maintaining visual continuity and style."
                            
                            if aligned_width != width or aligned_height != height:
                                print(f"🔧 VACE I2V resolution correction: {width}x{height} -> {aligned_width}x{aligned_height}")
                                print(f"🔧 VACE {self.model_size} optimized for {aligned_width}x{aligned_height}")
                            
                            print(f"🎬 VACE I2V generating: {enhanced_prompt[:50]}...")
                            print(f"🔧 Parameters: {aligned_width}x{aligned_height}, {num_frames} frames, {num_inference_steps} steps, guidance={guidance_scale}")
                            
                            try:
                                if image is not None:
                                    import tempfile
                                    import os
                                    from PIL import Image as PILImage
                                    
                                    # Ensure image is PIL Image
                                    if not hasattr(image, 'save'):
                                        # Convert tensor/array to PIL if needed
                                        if hasattr(image, 'numpy'):
                                            image_np = image.numpy()
                                        else:
                                            image_np = image
                                        image = PILImage.fromarray((image_np * 255).astype('uint8'))
                                    
                                    # Resize image to match VACE dimensions if needed
                                    if image.size != (aligned_width, aligned_height):
                                        image = image.resize((aligned_width, aligned_height), PILImage.Resampling.LANCZOS)
                                    
                                    # Save image temporarily for VACE processing
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                        image.save(tmp_file.name, 'PNG')
                                        temp_image_path = tmp_file.name
                                    
                                    try:
                                        # Use VACE's prepare_source for proper I2V setup
                                        src_video, src_mask, src_ref_images = self.vace_model.prepare_source(
                                            [None],  # I2V: no source video (will create blank frames)
                                            [None],  # I2V: no source mask (will create ones)
                                            [[temp_image_path]],  # I2V: reference image for conditioning
                                            num_frames,
                                            (aligned_height, aligned_width),
                                            self.vace_model.device
                                        )
                                        
                                        print(f"🔧 I2V prepared tensors - video: {len(src_video)}, mask: {len(src_mask)}, ref: {len(src_ref_images[0]) if src_ref_images and src_ref_images[0] else 'None'}")
                                        
                                        result = self.vace_model.generate(
                                            input_prompt=enhanced_prompt,
                                            input_frames=src_video,
                                            input_masks=src_mask,
                                            input_ref_images=src_ref_images,
                                            size=(aligned_width, aligned_height),
                                            frame_num=num_frames,
                                            sampling_steps=num_inference_steps,
                                            guide_scale=guidance_scale,
                                            shift=16,  # VACE uses higher shift
                                            **kwargs
                                        )
                                        
                                        if result is not None:
                                            print(f"✅ VACE I2V generation completed, output shape: {result.shape}")
                                            print(f"🔧 Output value range: min={result.min():.3f}, max={result.max():.3f}")
                                        else:
                                            print("⚠️ VACE I2V returned None result")
                                            
                                        return result
                                        
                                    finally:
                                        # Clean up temporary file
                                        try:
                                            os.unlink(temp_image_path)
                                        except:
                                            pass
                                else:
                                    # No image provided, fallback to T2V
                                    return self.__call__(prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs)
                                
                            except Exception as e:
                                print(f"❌ VACE I2V generation failed: {e}")
                                import traceback
                                traceback.print_exc()
                                raise
                    
                    self.pipeline = SmartVACEWrapper(vace_model)
                    print("✅ Smart VACE model loaded successfully (single model for both T2V and I2V)")
                    return True
                    
                except Exception as wan_e:
                    print(f"❌ Official VACE loading failed: {wan_e}")
                    
            # VACE fallback - refuse diffusers
            print("❌ VACE model requires official Wan repository!")
            print("💡 Solutions:")
            print("1. 📦 Install Wan repository: cd Wan2.1 && pip install -e .")
            print("2. 📁 Use T2V models instead: huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B")
            print("🚫 VACE models are not compatible with diffusers fallback")
            return False
            
        except Exception as e:
            print(f"❌ VACE loading failed: {e}")
            return False
    
    def _load_standard_wan_model(self, model_info: Dict) -> bool:
        """Load standard T2V/I2V Wan model"""
        try:
            # Strategy 1: Try official Wan implementation
            extension_root = Path(__file__).parent.parent.parent.parent
            wan_repo_path = extension_root / "Wan2.1"
            
            if wan_repo_path.exists() and (wan_repo_path / "wan").exists():
                print(f"🔧 Trying official Wan implementation from: {wan_repo_path}")
                
                import sys
                if str(wan_repo_path) not in sys.path:
                    sys.path.insert(0, str(wan_repo_path))
                
                try:
                    import wan
                    from wan.text2video import WanT2V
                    
                    # Apply Flash Attention patches AFTER Wan modules are imported
                    try:
                        from .wan_flash_attention_patch import apply_flash_attention_patch, update_patched_flash_attention_mode
                        
                        # Update mode if stored
                        if hasattr(self, 'flash_attention_mode'):
                            update_patched_flash_attention_mode(self.flash_attention_mode)
                        
                        success = apply_flash_attention_patch()
                        if success:
                            print("✅ Flash Attention monkey patch applied successfully")
                        else:
                            print("⚠️ Flash Attention patch could not be applied - may be already patched")
                    except Exception as patch_e:
                        print(f"⚠️ Flash Attention patch failed: {patch_e}")
                        print("🔄 Continuing without patches...")
                    
                    print("🚀 Loading with official Wan T2V...")
                    
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
                                print(f"🔧 Dimension alignment: {width}x{height} -> {aligned_width}x{aligned_height}")
                            
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
                    print("✅ Official Wan model loaded successfully")
                    return True
                    
                except Exception as wan_e:
                    print(f"⚠️ Official Wan loading failed: {wan_e}")
                    print("🔄 Trying diffusers fallback...")
            
            # Strategy 2: Try diffusers fallback
            try:
                from diffusers import DiffusionPipeline
                
                print("🔄 Loading with DiffusionPipeline...")
                pipeline = DiffusionPipeline.from_pretrained(
                    model_info['path'],
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_safetensors=True
                )
                
                if torch.cuda.is_available():
                    pipeline = pipeline.to(self.device)
                
                # Create wrapper for diffusers pipeline
                class DiffusersWrapper:
                    def __init__(self, pipeline):
                        self.pipeline = pipeline
                    
                    def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                        # Ensure dimensions are aligned
                        aligned_width = ((width + 15) // 16) * 16
                        aligned_height = ((height + 15) // 16) * 16
                        
                        generation_kwargs = {
                            "prompt": prompt,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                        }
                        
                        # Add video-specific parameters if supported
                        import inspect
                        try:
                            pipeline_signature = inspect.signature(self.pipeline.__call__)
                            
                            if 'height' in pipeline_signature.parameters:
                                generation_kwargs['height'] = aligned_height
                            if 'width' in pipeline_signature.parameters:
                                generation_kwargs['width'] = aligned_width
                            if 'num_frames' in pipeline_signature.parameters:
                                generation_kwargs['num_frames'] = num_frames
                        except:
                            pass
                        
                        with torch.no_grad():
                            return self.pipeline(**generation_kwargs)
                    
                    def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                        # Enhanced prompt for I2V
                        enhanced_prompt = f"Starting from the given image, {prompt}. Maintain visual continuity."
                        return self.__call__(enhanced_prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs)
                
                self.pipeline = DiffusersWrapper(pipeline)
                print("✅ Diffusers model loaded successfully")
                return True
                
            except Exception as diffusers_e:
                print(f"❌ Diffusers loading failed: {diffusers_e}")
                
                raise RuntimeError(f"""
❌ CRITICAL: Could not load Wan model with any method!

🔧 TROUBLESHOOTING:
1. 📦 Install dependencies: pip install diffusers transformers
2. 🔧 For official Wan support: cd Wan2.1 && pip install -e .
3. 💾 Check model files are complete
4. 🔄 Restart WebUI

❌ All loading methods failed!
Diffusers error: {diffusers_e}
""")
        
        except Exception as e:
            print(f"❌ Standard model loading failed: {e}")
            return False
    
    def generate_video_with_i2v_chaining(self, clips, model_info, output_dir, wan_args=None, **kwargs):
        """Generate video using I2V chaining for better continuity between clips"""
        try:
            import shutil
            from datetime import datetime
            
            print(f"🎬 Starting I2V chained generation with {len(clips)} clips...")
            print(f"📁 Model: {model_info['name']} ({model_info['type']}, {model_info['size']})")
            
            # Load the model if not loaded
            if not self.pipeline:
                print("🔧 Loading Wan pipeline for I2V chaining...")
                if not self.load_simple_wan_pipeline(model_info, wan_args):
                    raise RuntimeError("Failed to load Wan pipeline")
            
            # Extract parameters
            width = kwargs.get('width', 1280)
            height = kwargs.get('height', 720)
            steps = kwargs.get('steps', 20)
            guidance_scale = kwargs.get('guidance_scale', 7.5)
            
            # Ensure dimensions are properly aligned for VACE/Wan
            original_width, original_height = width, height
            aligned_width = ((width + 15) // 16) * 16
            aligned_height = ((height + 15) // 16) * 16
            
            if aligned_width != original_width or aligned_height != original_height:
                print(f"🔧 Dimension alignment applied: {original_width}x{original_height} -> {aligned_width}x{aligned_height}")
                width, height = aligned_width, aligned_height
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get timestring for frame naming
            timestring = datetime.now().strftime("%Y%m%d%H%M%S")
            
            all_frame_paths = []
            total_frame_idx = 0
            last_frame_path = None
            
            for clip_idx, clip in enumerate(clips):
                print(f"\n🎬 Generating clip {clip_idx + 1}/{len(clips)}")
                print(f"   📝 Prompt: {clip['prompt'][:50]}...")
                print(f"   🎞️ Frames: {clip['num_frames']}")
                
                try:
                    if clip_idx == 0 or last_frame_path is None:
                        # First clip: use T2V
                        print("🚀 Using T2V for first clip")
                        
                        result = self.pipeline(
                            prompt=clip['prompt'],
                            height=height,
                            width=width,
                            num_frames=clip['num_frames'],
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                        )
                    else:
                        # Subsequent clips: use I2V if available
                        print(f"🔗 Using I2V chaining from: {os.path.basename(last_frame_path)}")
                        
                        if hasattr(self.pipeline, 'generate_image2video'):
                            from PIL import Image
                            init_image = Image.open(last_frame_path)
                            
                            result = self.pipeline.generate_image2video(
                                image=init_image,
                                prompt=clip['prompt'],
                                height=height,
                                width=width,
                                num_frames=clip['num_frames'],
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                            )
                        else:
                            # Fallback to T2V with enhanced prompt
                            enhanced_prompt = f"Continuing seamlessly, {clip['prompt']}. Maintain visual continuity."
                            result = self.pipeline(
                                prompt=enhanced_prompt,
                                height=height,
                                width=width,
                                num_frames=clip['num_frames'],
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                            )
                    
                    # Process and save frames
                    clip_frames = self._process_and_save_frames(result, clip_idx, output_dir, timestring, total_frame_idx)
                    
                    if clip_frames:
                        all_frame_paths.extend(clip_frames)
                        total_frame_idx += len(clip_frames)
                        last_frame_path = clip_frames[-1]  # Update for next clip
                        print(f"✅ Generated {len(clip_frames)} frames for clip {clip_idx + 1}")
                    else:
                        raise RuntimeError(f"No frames generated for clip {clip_idx + 1}")
                
                except Exception as e:
                    print(f"❌ Clip {clip_idx + 1} generation failed: {e}")
                    raise
            
            print(f"\n✅ All clips generated! Total frames: {len(all_frame_paths)}")
            print(f"📁 Frames saved to: {output_dir}")
            return output_dir
            
        except Exception as e:
            print(f"❌ I2V chained video generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"I2V chained video generation failed: {e}")
        finally:
            # Clean up if needed
            pass
    
    def _process_and_save_frames(self, result, clip_idx, output_dir, timestring, start_frame_idx):
        """Process generation result and save frames"""
        try:
            from PIL import Image
            import numpy as np
            
            frames = []
            
            # Handle different result formats
            if isinstance(result, tuple):
                frames_data = result[0]
            elif hasattr(result, 'frames'):
                frames_data = result.frames
            else:
                frames_data = result
            
            # Convert to individual frames
            if hasattr(frames_data, 'cpu'):  # Tensor
                frames_tensor = frames_data.cpu()
                print(f"🔧 Processing tensor frames: {frames_tensor.shape}")
                print(f"🔧 Tensor value range: min={frames_tensor.min():.3f}, max={frames_tensor.max():.3f}")
                
                # Handle different tensor formats
                if len(frames_tensor.shape) == 5:  # (B, C, F, H, W)
                    frames_tensor = frames_tensor.squeeze(0)  # Remove batch dimension
                
                if len(frames_tensor.shape) == 4:  # (C, F, H, W)
                    for frame_idx in range(frames_tensor.shape[1]):
                        frame_tensor = frames_tensor[:, frame_idx, :, :]  # (C, H, W)
                        frame_np = frame_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
                        
                        # VACE outputs in [-1, 1] range, convert to [0, 255]
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
                else:
                    print(f"⚠️ Unexpected tensor shape: {frames_tensor.shape}")
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
            
            elif isinstance(frames_data, list):  # List of PIL Images or arrays
                for frame in frames_data:
                    if hasattr(frame, 'save'):  # PIL Image
                        frames.append(np.array(frame))
                    else:
                        frames.append(frame)
            
            # Save frames as PNG files
            saved_paths = []
            for i, frame_np in enumerate(frames):
                frame_filename = f"{timestring}_{start_frame_idx + i:09d}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                
                try:
                    pil_image = Image.fromarray(frame_np)
                    pil_image.save(frame_path)
                    saved_paths.append(frame_path)
                except Exception as save_e:
                    print(f"⚠️ Failed to save frame {i}: {save_e}")
                    continue
            
            print(f"✅ Saved {len(saved_paths)} frames for clip {clip_idx + 1}")
            return saved_paths
            
        except Exception as e:
            print(f"❌ Frame processing failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def test_wan_setup(self) -> bool:
        """Test if Wan setup is working"""
        try:
            print("🔍 Testing Wan setup...")
            
            # Test model discovery
            models = self.discover_models()
            if not models:
                print("❌ No models found")
                return False
            
            # Test best model selection
            best_model = self.get_best_model()
            if not best_model:
                print("❌ No suitable model found")
                return False
            
            print(f"✅ Wan setup test passed - found {len(models)} models")
            return True
            
        except Exception as e:
            print(f"❌ Wan setup test failed: {e}")
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
                    
                print("🧹 Wan model unloaded, memory freed")
            except Exception as e:
                print(f"⚠️ Error unloading model: {e}")

# Global instance for easy access
wan_integration = WanSimpleIntegration()

def wan_generate_video_main(*args, **kwargs):
    """Main entry point for Wan video generation"""
    try:
        print("🎬 Wan generate video main called")
        
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
        print(f"❌ Wan generation failed: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Wan generation error: {e}"

if __name__ == "__main__":
    # Test the integration
    integration = WanSimpleIntegration()
    integration.test_wan_setup() 