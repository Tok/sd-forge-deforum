#!/usr/bin/env python3
"""
Wan Simple Integration - Direct Model Loading Without Complex Dependencies
A simpler, more reliable approach that directly loads Wan models
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
from PIL import Image
import tempfile

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
    
    def load_simple_wan_pipeline(self, model_info: Dict) -> bool:
        """Load Wan models properly - create custom pipeline for Wan format"""
        try:
            print(f"🔧 Loading Wan model: {model_info['name']}")
            print(f"📁 Model path: {model_info['path']}")
            
            # Wan models are not standard Diffusers format, we need a custom loader
            print("🚀 Creating custom Wan pipeline...")
            
            # Create a custom Wan pipeline class
            pipeline = self._create_custom_wan_pipeline(model_info)
            
            self.pipeline = pipeline
            print("✅ Wan model loaded successfully with custom pipeline")
            return True
                
        except Exception as e:
            print(f"❌ Failed to load Wan model: {e}")
            raise RuntimeError(f"Wan model loading failed: {e}")
    
    def _create_custom_wan_pipeline(self, model_info: Dict):
        """Create a custom Wan pipeline that can handle the specific Wan model format"""
        
        class CustomWanPipeline:
            def __init__(self, model_path: str, device: str):
                self.model_path = Path(model_path)
                self.device = device
                
                # Load config
                config_path = self.model_path / "config.json"
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                
                print(f"📋 Wan Model config: {self.config}")
                
                # For now, this is a stub that will demonstrate proper Wan loading
                # In a full implementation, we would load the actual model components
                self.loaded = True
                
            def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                """Generate video using Wan model - FAIL FAST ON ARCHITECTURE ISSUES"""
                
                print(f"🎬 Generating video with Wan model...")
                print(f"🎬 Running Wan inference...")
                print(f"   📝 Prompt: {prompt[:50]}...")
                print(f"   📐 Size: {width}x{height}")
                print(f"   🎬 Frames: {num_frames}")
                print(f"   🔧 Steps: {num_inference_steps}")
                print(f"   📏 Guidance: {guidance_scale}")
                
                print(f"🚀 Starting Wan model inference...")
                
                                    # Use the REAL Wan implementation from the official repo
                try:
                    print(f"🔄 Trying REAL Wan implementation...")
                    
                    # Import the real implementation
                    import importlib
                    import sys
                    from pathlib import Path
                    
                    # Add the current directory to sys.path for absolute imports
                    current_dir = Path(__file__).parent
                    if str(current_dir) not in sys.path:
                        sys.path.insert(0, str(current_dir))
                    
                    try:
                        # Import the real Wan implementation
                        module = importlib.import_module("wan_real_implementation")
                    except ImportError:
                        try:
                            module = importlib.import_module(f".wan_real_implementation", package=__package__)
                        except (ImportError, ValueError):
                            raise ImportError(f"Could not import wan_real_implementation")
                    
                    integration_class = getattr(module, "WanRealIntegration")
                    
                    # Create integration instance
                    integration = integration_class()
                    
                    # Load the model
                    success = integration.load_pipeline(str(self.model_path))
                    if not success:
                        raise RuntimeError("Failed to load real Wan pipeline")
                    
                    # Generate video directly to the output
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                        temp_output = tmp_file.name
                    
                    # Generate video using the REAL Wan implementation
                    generation_success = integration.generate_video(
                        prompt=prompt,
                        output_path=temp_output,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed=kwargs.get('seed', -1)
                    )
                    
                    if generation_success and os.path.exists(temp_output):
                        print(f"✅ Wan video generated successfully with REAL Wan implementation!")
                        
                        # Load video frames to return
                        import imageio
                        frames = imageio.mimread(temp_output, format='mp4', memtest=False)
                        
                        # Convert to tensor format expected by pipeline
                        import torch
                        import numpy as np
                        
                        # Convert frames to tensor (C, F, H, W)
                        frames_array = np.stack(frames)  # (F, H, W, C)
                        frames_array = frames_array.transpose(3, 0, 1, 2)  # (C, F, H, W)
                        frames_tensor = torch.from_numpy(frames_array).float() / 255.0
                        
                        # Cleanup temp file
                        try:
                            os.unlink(temp_output)
                        except:
                            pass
                        
                        # Cleanup integration
                        integration.unload_pipeline()
                        
                        return frames_tensor
                    else:
                        raise RuntimeError("Real Wan implementation failed to generate video")
                        
                except Exception as e:
                    print(f"❌ Real Wan Implementation Failed: {e}")
                    print(f"❌ CRITICAL: Failed to use real Wan implementation from official repository!")
                    print(f"❌ This could be due to:")
                    print(f"   • Missing dependencies from wan_official_repo")
                    print(f"   • Import path issues")
                    print(f"   • Model loading problems")
                    print(f"❌ NO FALLBACKS - Real implementation required.")
                    
                    # Re-raise the exception to fail fast - NO FALLBACKS
                    raise RuntimeError(
                        f"Wan video generation failed: {e}\n\n"
                        "CRITICAL: Real Wan implementation failed. "
                        "Check that wan_official_repo is properly set up and all dependencies are installed. "
                        "No fallbacks available - real Wan implementation required."
                    )
        
        return CustomWanPipeline(model_info['path'], self.device)
    
    def _validate_wan_model(self, model_info: Dict) -> bool:
        """Validate Wan model has required files"""
        model_path = Path(model_info['path'])
        
        # Check for required files
        required_files = [
            "diffusion_pytorch_model.safetensors",
            "config.json"
        ]
        
        # Check for VAE and T5 files (different naming conventions)
        vae_files = ["Wan2.1_VAE.pth", "vae.pth", "vae.safetensors"]
        t5_files = ["models_t5_umt5-xxl-enc-bf16.pth", "t5.pth", "t5.safetensors"]
        
        missing_files = []
        
        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
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
    
    def _save_frames_as_video(self, frames, output_path: str, fps: int = 8):
        """Save frames as video file"""
        try:
            import imageio
            import numpy as np
            
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
                    
                    print(f"  Frame {i}: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
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
                    
                    print(f"  Frame {i}: shape={frame_np.shape}, dtype={frame_np.dtype}")
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
        """Generate video using simple Wan integration"""
        
        print(f"🎬 Generating video using SIMPLE Wan integration...")
        print(f"   📝 Prompt: {prompt}")
        print(f"   📐 Size: {width}x{height}")
        print(f"   🎬 Frames: {num_frames}")
        print(f"   📁 Model: {model_info['name']} ({model_info['type']}, {model_info['size']})")
        
        # Validate model first
        if not self._validate_wan_model(model_info):
            raise RuntimeError("Wan model validation failed - missing required files")
        
        # Load the model if not loaded
        if not self.pipeline:
            self.load_simple_wan_pipeline(model_info)  # This will raise if it fails
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename
            import time
            timestamp = int(time.time())
            output_filename = f"wan_video_{timestamp}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            print("🎬 Generating video with Wan model...")
            
            # Real Wan video generation
            result = self._generate_with_wan_pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                output_path=output_path
            )
            
            if result:
                print(f"✅ Wan video generated: {output_path}")
                return output_path
            else:
                raise RuntimeError("Wan video generation returned no result")
                
        except Exception as e:
            print(f"❌ Wan video generation failed: {e}")
            raise RuntimeError(f"Wan video generation failed: {e}")
    
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
