"""
WAN Unified Integration
Single entry point for all WAN video generation with intelligent fallbacks
"""

import os
import torch
from typing import Dict, List, Optional, Union
from pathlib import Path

from ..utils.model_discovery import WanModelDiscovery
from ..utils.video_utils import VideoProcessor
from ..pipelines.procedural_pipeline import WanProceduralPipeline


class WanUnifiedIntegration:
    """Unified WAN integration with smart pipeline selection and fallbacks"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_discovery = WanModelDiscovery()
        self.video_processor = VideoProcessor()
        self.pipeline = None
        self.pipeline_type = None
        self.model_info = None
        
    def discover_models(self) -> List[Dict]:
        """Discover available WAN models"""
        return self.model_discovery.discover_models()
    
    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model"""
        models = self.discover_models()
        return models[0] if models else None
    
    def load_pipeline(self, model_path: str = None, pipeline_type: str = "auto") -> bool:
        """Load appropriate WAN pipeline"""
        try:
            if model_path:
                # Try to load specific model
                print(f"🔄 Loading WAN pipeline from: {model_path}")
                return self._load_model_pipeline(model_path, pipeline_type)
            else:
                # Auto-discover and load best model
                print(f"🔍 Auto-discovering WAN models...")
                best_model = self.get_best_model()
                
                if best_model:
                    print(f"🎯 Found model: {best_model['name']}")
                    return self._load_model_pipeline(best_model['path'], pipeline_type)
                else:
                    print(f"⚠️ No WAN models found, using procedural pipeline")
                    return self._load_procedural_pipeline()
                    
        except Exception as e:
            print(f"❌ Failed to load WAN pipeline: {e}")
            print(f"🔄 Falling back to procedural pipeline...")
            return self._load_procedural_pipeline()
    
    def _load_model_pipeline(self, model_path: str, pipeline_type: str) -> bool:
        """Load model-based pipeline with fallbacks"""
        
        # Validate model files
        if not self._validate_model_files(model_path):
            print(f"❌ Model validation failed, using procedural pipeline")
            return self._load_procedural_pipeline()
        
        # Try different pipeline types in order of preference
        pipeline_attempts = []
        
        if pipeline_type == "auto":
            # Auto-detect best pipeline type
            model_config = self._load_model_config(model_path)
            if model_config.get("model_type") == "vace":
                pipeline_attempts = ["vace", "diffusers", "procedural"]
            else:
                pipeline_attempts = ["diffusers", "vace", "procedural"]
        else:
            pipeline_attempts = [pipeline_type, "procedural"]
        
        for attempt_type in pipeline_attempts:
            try:
                if attempt_type == "procedural":
                    return self._load_procedural_pipeline()
                elif attempt_type == "diffusers":
                    return self._load_diffusers_pipeline(model_path)
                elif attempt_type == "vace":
                    return self._load_vace_pipeline(model_path)
                    
            except Exception as e:
                print(f"❌ Failed to load {attempt_type} pipeline: {e}")
                continue
        
        # If all else fails, use procedural
        return self._load_procedural_pipeline()
    
    def _load_diffusers_pipeline(self, model_path: str) -> bool:
        """Load Diffusers-based pipeline"""
        try:
            # Import the old working implementation as diffusers pipeline
            from ...wan_real_implementation import WanRealIntegration
            
            print(f"🚀 Loading Diffusers-based WAN pipeline...")
            
            real_integration = WanRealIntegration()
            success = real_integration.load_pipeline(model_path)
            
            if success:
                self.pipeline = real_integration
                self.pipeline_type = "diffusers"
                self.model_info = {"path": model_path, "type": "diffusers"}
                print(f"✅ Diffusers WAN pipeline loaded")
                return True
            else:
                raise RuntimeError("Failed to load diffusers pipeline")
                
        except Exception as e:
            print(f"❌ Diffusers pipeline failed: {e}")
            raise
    
    def _load_vace_pipeline(self, model_path: str) -> bool:
        """Load VACE (Video and Content Editing) pipeline"""
        try:
            # Import the complete implementation as VACE pipeline
            from ...wan_complete_implementation import WanWorkingIntegration
            
            print(f"🚀 Loading VACE WAN pipeline...")
            
            complete_integration = WanWorkingIntegration()
            success = complete_integration.load_pipeline(model_path)
            
            if success:
                self.pipeline = complete_integration
                self.pipeline_type = "vace"
                self.model_info = {"path": model_path, "type": "vace"}
                print(f"✅ VACE WAN pipeline loaded")
                return True
            else:
                raise RuntimeError("Failed to load VACE pipeline")
                
        except Exception as e:
            print(f"❌ VACE pipeline failed: {e}")
            raise
    
    def _load_procedural_pipeline(self) -> bool:
        """Load procedural fallback pipeline"""
        try:
            print(f"🚀 Loading procedural WAN pipeline...")
            
            self.pipeline = WanProceduralPipeline(self.device)
            success = self.pipeline.load_components()
            
            if success:
                self.pipeline_type = "procedural"
                self.model_info = {"type": "procedural"}
                print(f"✅ Procedural WAN pipeline loaded")
                return True
            else:
                raise RuntimeError("Failed to load procedural pipeline")
                
        except Exception as e:
            print(f"❌ Procedural pipeline failed: {e}")
            raise
    
    def _validate_model_files(self, model_path: str) -> bool:
        """Validate required model files exist"""
        path = Path(model_path)
        
        required_files = [
            "diffusion_pytorch_model.safetensors",
            "config.json"
        ]
        
        # Check for at least one VAE file
        vae_files = ["Wan2.1_VAE.pth", "vae.pth", "vae.safetensors"]
        # Check for at least one T5 file  
        t5_files = ["models_t5_umt5-xxl-enc-bf16.pth", "t5.pth", "t5.safetensors"]
        
        missing_files = []
        
        for file in required_files:
            if not (path / file).exists():
                missing_files.append(file)
        
        if not any((path / vae).exists() for vae in vae_files):
            missing_files.append("VAE file")
            
        if not any((path / t5).exists() for t5 in t5_files):
            missing_files.append("T5 file")
        
        if missing_files:
            print(f"❌ Missing required model files: {missing_files}")
            return False
            
        print("✅ All required WAN model files found")
        return True
    
    def _load_model_config(self, model_path: str) -> Dict:
        """Load model configuration"""
        config_path = Path(model_path) / "config.json"
        
        if config_path.exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {}
    
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
        """Generate video using loaded pipeline"""
        
        if not self.pipeline:
            print(f"⚠️ No pipeline loaded, auto-loading...")
            if not self.load_pipeline():
                raise RuntimeError("Failed to load any WAN pipeline")
        
        print(f"🎬 Generating video with {self.pipeline_type} pipeline...")
        print(f"   📝 Prompt: {prompt[:50]}...")
        print(f"   📐 Size: {width}x{height}")
        print(f"   🎬 Frames: {num_frames}")
        
        try:
            # Different pipelines have different interfaces
            if self.pipeline_type == "procedural":
                return self.pipeline.generate_video(
                    prompt=prompt,
                    output_path=output_path,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    **kwargs
                )
            
            elif self.pipeline_type == "diffusers":
                return self.pipeline.generate_video(
                    prompt=prompt,
                    output_path=output_path,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    **kwargs
                )
            
            elif self.pipeline_type == "vace":
                return self.pipeline.generate_video(
                    prompt=prompt,
                    output_path=output_path,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    sampling_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    **kwargs
                )
            
            else:
                raise RuntimeError(f"Unknown pipeline type: {self.pipeline_type}")
                
        except Exception as e:
            print(f"❌ {self.pipeline_type} pipeline failed: {e}")
            
            # Try fallback to procedural if not already using it
            if self.pipeline_type != "procedural":
                print(f"🔄 Falling back to procedural pipeline...")
                if self._load_procedural_pipeline():
                    return self.generate_video(
                        prompt=prompt,
                        output_path=output_path,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                        **kwargs
                    )
            
            raise RuntimeError(f"All pipelines failed: {e}")
    
    def unload_pipeline(self):
        """Unload current pipeline"""
        if self.pipeline:
            try:
                if hasattr(self.pipeline, 'unload_pipeline'):
                    self.pipeline.unload_pipeline()
                elif hasattr(self.pipeline, 'unload_model'):
                    self.pipeline.unload_model()
                
                del self.pipeline
                self.pipeline = None
                self.pipeline_type = None
                self.model_info = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                print("🧹 WAN pipeline unloaded")
                
            except Exception as e:
                print(f"⚠️ Error unloading pipeline: {e}")
    
    def get_pipeline_info(self) -> Dict:
        """Get information about current pipeline"""
        return {
            "type": self.pipeline_type,
            "model_info": self.model_info,
            "loaded": self.pipeline is not None
        } 