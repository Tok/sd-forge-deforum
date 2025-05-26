#!/usr/bin/env python3
"""
WAN Real Implementation - Using Official WAN Repository with Compatibility Layer
Integrates with the actual WAN T2V implementation from the official repo
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import json
import tempfile
import imageio

# Import compatibility layer first
try:
    from .wan_compatibility import ensure_wan_compatibility
    print("✅ WAN compatibility layer imported")
except ImportError as e:
    print(f"❌ Failed to import WAN compatibility layer: {e}")
    print("💡 Ensure wan_compatibility.py is in the same directory")
    raise


class WanRealIntegration:
    """Real WAN integration using the official repository with compatibility layer"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.wan_model = None
        self.config = None
        self.model_path = None
        self._setup_wan_import()
    
    def _setup_wan_import(self):
        """Setup WAN import paths"""
        # Use Wan2.1 instead of wan_official_repo for consistency
        wan_repo_path = Path(__file__).parent.parent.parent / "Wan2.1"
        
        # Add to sys.path if not already there
        if str(wan_repo_path) not in sys.path:
            sys.path.insert(0, str(wan_repo_path))
        
        # Also try adding the project root
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
    
    def _check_wan_availability(self):
        """Check if WAN can be imported with compatibility layer"""
        try:
            # Ensure compatibility first
            ensure_wan_compatibility()
            
            # Now try importing WAN
            import wan
            return True, None
        except ImportError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Compatibility error: {e}"
    
    def load_pipeline(self, model_path: str) -> bool:
        """Load the real WAN pipeline with compatibility layer"""
        try:
            print(f"🚀 Loading REAL WAN pipeline from: {model_path}")
            
            # Ensure WAN compatibility before proceeding
            print("🔧 Ensuring WAN compatibility...")
            ensure_wan_compatibility()
            
            # Check if WAN is available
            wan_available, import_error = self._check_wan_availability()
            if not wan_available:
                print(f"❌ Cannot import WAN: {import_error}")
                print(f"💡 To fix this, install WAN package:")
                print(f"   cd Wan2.1 && pip install -e .")
                return False
            
            # Validate model files
            if not self._validate_model_files(model_path):
                return False
            
            self.model_path = model_path
            
            # Import WAN components
            import wan
            from wan.configs.wan_t2v_1_3B import t2v_1_3B
            from easydict import EasyDict
            
            # Create config - ensure it's an EasyDict
            if isinstance(t2v_1_3B, dict):
                self.config = EasyDict(t2v_1_3B.copy())
            else:
                self.config = t2v_1_3B.copy()
            
            print(f"📋 Using WAN T2V 1.3B configuration")
            print(f"   🔧 Dim: {self.config.dim}")
            print(f"   🔧 Layers: {self.config.num_layers}")
            print(f"   🔧 Heads: {self.config.num_heads}")
            print(f"   🔧 VAE Stride: {self.config.vae_stride}")
            
            # Initialize WAN model
            print(f"🔄 Initializing WAN T2V model...")
            self.wan_model = wan.WanT2V(
                config=self.config,
                checkpoint_dir=model_path,
                device_id=0 if self.device == "cuda" else -1,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=False
            )
            
            print(f"✅ Real WAN pipeline loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load real WAN pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _validate_model_files(self, model_path: str) -> bool:
        """Validate required model files exist"""
        path = Path(model_path)
        
        required_files = {
            "DiT": "diffusion_pytorch_model.safetensors",
            "VAE": "Wan2.1_VAE.pth", 
            "T5": "models_t5_umt5-xxl-enc-bf16.pth",
            "Config": "config.json"
        }
        
        missing = []
        
        for name, filename in required_files.items():
            file_path = path / filename
            if file_path.exists():
                size_gb = file_path.stat().st_size / (1024**3)
                print(f"✅ {name}: {filename} ({size_gb:.1f}GB)")
            else:
                missing.append(f"{name} ({filename})")
        
        if missing:
            print(f"❌ Missing files: {missing}")
            return False
        
        print("✅ All required WAN model files validated")
        return True
    
    def generate_video(self, 
                      prompt: str,
                      output_path: str,
                      width: int = 1280,
                      height: int = 720,
                      num_frames: int = 81,
                      num_inference_steps: int = 20,
                      guidance_scale: float = 7.5,
                      seed: int = -1,
                      **kwargs) -> bool:
        """Generate video using real WAN implementation"""
        
        if not self.wan_model:
            raise RuntimeError("WAN model not loaded. Call load_pipeline() first.")
        
        try:
            print(f"🎬 Generating video with REAL WAN T2V...")
            print(f"   📝 Prompt: {prompt}")
            print(f"   📐 Size: {width}x{height}")
            print(f"   🎬 Frames: {num_frames}")
            print(f"   🔧 Steps: {num_inference_steps}")
            print(f"   📏 Guidance: {guidance_scale}")
            print(f"   🎲 Seed: {seed}")
            
            # Ensure num_frames follows WAN's 4n+1 requirement
            if (num_frames - 1) % 4 != 0:
                adjusted_frames = ((num_frames - 1) // 4) * 4 + 1
                print(f"⚠️ Adjusting frames from {num_frames} to {adjusted_frames} (WAN requirement: 4n+1)")
                num_frames = adjusted_frames
            
            # Set negative prompt
            negative_prompt = kwargs.get('negative_prompt', '')
            if not negative_prompt:
                negative_prompt = self.config.sample_neg_prompt
            
            print(f"🚀 Starting real WAN inference...")
            
            # Generate video using official WAN implementation
            video_tensor = self.wan_model.generate(
                input_prompt=prompt,
                size=(width, height),
                frame_num=num_frames,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=num_inference_steps,
                guide_scale=guidance_scale,
                n_prompt=negative_prompt,
                seed=seed,
                offload_model=True
            )
            
            if video_tensor is None:
                raise RuntimeError("WAN generate() returned None")
            
            print(f"✅ Video generation completed!")
            print(f"   📊 Tensor shape: {video_tensor.shape}")
            print(f"   📊 Tensor dtype: {video_tensor.dtype}")
            print(f"   📊 Tensor device: {video_tensor.device}")
            
            # Save video to file
            self._save_video_tensor(video_tensor, output_path)
            
            print(f"✅ Video saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Real WAN video generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_video_tensor(self, video_tensor: torch.Tensor, output_path: str):
        """Save video tensor to file"""
        try:
            print(f"💾 Saving video tensor to {output_path}...")
            
            # Convert tensor to numpy
            if video_tensor.device != torch.device('cpu'):
                video_tensor = video_tensor.cpu()
            
            video_np = video_tensor.numpy()
            print(f"   📊 Numpy shape: {video_np.shape}")
            
            # Handle different tensor formats
            # WAN returns (C, F, H, W) format
            if len(video_np.shape) == 4:
                C, F, H, W = video_np.shape
                print(f"   🔄 Converting from (C={C}, F={F}, H={H}, W={W}) to frames")
                
                # Transpose to (F, H, W, C) for video saving
                frames = video_np.transpose(1, 2, 3, 0)  # (F, H, W, C)
            else:
                raise ValueError(f"Unexpected tensor shape: {video_np.shape}")
            
            # Convert to uint8
            if frames.dtype != np.uint8:
                # Normalize to [0, 1] range if needed
                if frames.min() < 0 or frames.max() > 1:
                    frames = np.clip(frames, 0, 1)
                
                # Convert to [0, 255]
                frames = (frames * 255).astype(np.uint8)
            
            print(f"   🎬 Saving {len(frames)} frames...")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as MP4 using imageio
            imageio.mimsave(
                output_path, 
                frames, 
                fps=16,  # WAN default FPS
                format='mp4',
                codec='libx264',
                quality=8
            )
            
            print(f"✅ Video saved successfully!")
            
        except Exception as e:
            print(f"❌ Failed to save video: {e}")
            raise
    
    def unload_pipeline(self):
        """Unload the WAN pipeline to free memory"""
        if self.wan_model:
            try:
                # The WAN model handles its own cleanup
                del self.wan_model
                self.wan_model = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print("🧹 Real WAN pipeline unloaded")
            except Exception as e:
                print(f"⚠️ Error unloading WAN pipeline: {e}")


def test_real_wan():
    """Test the real WAN implementation with compatibility layer"""
    print("🧪 Testing Real WAN Implementation with Compatibility Layer...")
    
    # Test paths - adjust as needed
    model_paths = [
        "C:/Users/Zirteq/Documents/workspace/webui-forge/webui/models/wan",
        "../../../models/wan",
        "models/wan"
    ]
    
    integration = WanRealIntegration()
    
    # Check WAN availability first
    wan_available, import_error = integration._check_wan_availability()
    if not wan_available:
        print(f"❌ WAN not available: {import_error}")
        print(f"💡 To install WAN:")
        print(f"   cd Wan2.1")
        print(f"   pip install -e .")
        return
    
    print("✅ WAN import successful with compatibility layer!")
    
    # Find a valid model path
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ No WAN model found. Tested paths:")
        for path in model_paths:
            print(f"   • {path}")
        return
    
    # Test loading
    if integration.load_pipeline(model_path):
        print("✅ Pipeline loaded successfully")
        
        # Test generation
        output_path = "test_output/wan_real_test.mp4"
        
        result = integration.generate_video(
            prompt="A cat walking in a garden",
            output_path=output_path,
            width=1280,
            height=720,
            num_frames=17,  # 4*4+1
            num_inference_steps=10,  # Fast test
            guidance_scale=7.5,
            seed=42
        )
        
        if result:
            print(f"✅ Test generation successful: {output_path}")
        else:
            print("❌ Test generation failed")
        
        # Cleanup
        integration.unload_pipeline()
    else:
        print("❌ Pipeline loading failed")


if __name__ == "__main__":
    test_real_wan()
