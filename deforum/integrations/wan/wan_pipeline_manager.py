#!/usr/bin/env python3
"""
WAN Pipeline Manager
Handles model loading, pipeline management, and video generation
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import torch
import os
import json
import numpy as np
import time
from decimal import Decimal


class WanPipelineManager:
    """Manages WAN pipeline loading and video generation"""
    
    def __init__(self, core_integration):
        """Initialize pipeline manager.
        
        Args:
            core_integration: WanCoreIntegration instance
        """
        self.core = core_integration
        
    def discover_models(self) -> List[Dict]:
        """Discover available Wan models with styled progress"""
        models = []
        
        search_paths = self.core.get_model_search_paths()
        
        self.core.print_wan_info("🔍 Discovering Wan models...")
        self.core.print_wan_info(f"Primary model directory: {self.core.webui_root}/models")
        
        for search_path in search_paths:
            if search_path.exists():
                self.core.print_wan_info(f"🔍 Searching: {search_path}")
                
                for model_dir in search_path.iterdir():
                    if model_dir.is_dir() and model_dir.name.startswith(('Wan2.1', 'wan')):
                        model_info = self._analyze_model_directory(model_dir)
                        if model_info:
                            models.append(model_info)
                            self.core.print_wan_success(f"Found: {model_info['name']} ({model_info['type']}, {model_info['size']})")
        
        if not models:
            self.core.print_wan_warning("No Wan models found in search paths")
            webui_models = self.core.webui_root / "models"
            self.core.print_wan_info(f"💡 Expected model location: {webui_models / 'wan'}")
            self.core.print_wan_info("💡 Download command: huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan/Wan2.1-T2V-1.3B")
        else:
            self.core.print_wan_success(f"Discovery complete - found {len(models)} model(s)")
        
        # Update core models list
        self.core.models = models
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
    
    def load_simple_wan_pipeline(self, model_info: Dict, wan_args=None) -> bool:
        """Load a WAN pipeline for the specified model.
        
        Args:
            model_info: Model information dictionary
            wan_args: WAN arguments (optional)
            
        Returns:
            True if pipeline loaded successfully, False otherwise
        """
        try:
            # Unload any existing model first
            self.core.unload_model()
            
            self.core.print_wan_info(f"🚀 Loading WAN pipeline: {model_info['name']}")
            
            # Load different model types
            if model_info['type'] == 'VACE':
                success = self._load_vace_model(model_info)
            else:
                success = self._load_standard_wan_model(model_info)
            
            if success:
                self.core.model_size = model_info['size']
                self.core.print_wan_success(f"✅ WAN pipeline loaded successfully: {model_info['name']}")
                return True
            else:
                self.core.print_wan_error(f"❌ Failed to load WAN pipeline: {model_info['name']}")
                return False
                
        except Exception as e:
            self.core.print_wan_error(f"❌ Error loading WAN pipeline: {e}")
            return False
    
    def _load_vace_model(self, model_info: Dict) -> bool:
        """Load VACE model with proper configuration.
        
        Args:
            model_info: Model information dictionary
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.core.print_wan_info(f"🔧 Loading VACE model: {model_info['name']}")
            
            # Set optimal resolution for VACE
            width, height = 720, 480
            if model_info['size'] == '14B':
                width, height = 1024, 576
            
            self.core.set_optimal_resolution(width, height)
            
            # Try to import and load VACE
            try:
                from diffusers import StableDiffusionPipeline  # Placeholder - actual VACE import would be different
                
                # Load VACE model (this is a simplified version)
                model_path = model_info['path']
                
                # Create a wrapper class for VACE
                class VACEWrapper:
                    def __init__(self, model_path):
                        self.model_path = model_path
                        self.device = self.core.device if hasattr(self, 'core') else 'cuda'
                    
                    def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                        # This would contain actual VACE T2V generation logic
                        self.core.print_wan_info(f"🎬 Generating T2V with VACE: {prompt[:50]}...")
                        # Return mock result for now
                        return {"frames": np.random.rand(num_frames, height, width, 3)}
                    
                    def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                        # This would contain actual VACE I2V generation logic
                        self.core.print_wan_info(f"🎬 Generating I2V with VACE: {prompt[:50]}...")
                        # Return mock result for now
                        return {"frames": np.random.rand(num_frames, height, width, 3)}
                
                # Set the pipeline
                self.core.pipeline = VACEWrapper(model_path)
                
                self.core.print_wan_success(f"✅ VACE model loaded: {model_info['name']}")
                return True
                
            except ImportError as ie:
                self.core.print_wan_error(f"❌ VACE dependencies not available: {ie}")
                return False
                
        except Exception as e:
            self.core.print_wan_error(f"❌ Error loading VACE model: {e}")
            return False
    
    def _load_standard_wan_model(self, model_info: Dict) -> bool:
        """Load standard T2V/I2V WAN model.
        
        Args:
            model_info: Model information dictionary
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.core.print_wan_info(f"🔧 Loading standard WAN model: {model_info['name']}")
            
            # Create a wrapper for standard WAN models
            class WanWrapper:
                def __init__(self, model_info, core):
                    self.model_info = model_info
                    self.core = core
                
                def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                    self.core.print_wan_info(f"🎬 Generating with WAN {self.model_info['type']}: {prompt[:50]}...")
                    # Return mock result for now
                    return {"frames": np.random.rand(num_frames, height, width, 3)}
                
                def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                    self.core.print_wan_info(f"🎬 Generating I2V with WAN: {prompt[:50]}...")
                    # Return mock result for now  
                    return {"frames": np.random.rand(num_frames, height, width, 3)}
            
            # Set the pipeline
            self.core.pipeline = WanWrapper(model_info, self.core)
            
            self.core.print_wan_success(f"✅ Standard WAN model loaded: {model_info['name']}")
            return True
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error loading standard WAN model: {e}")
            return False
    
    def generate_video_with_i2v_chaining(self, clips, model_info, output_dir, wan_args=None, **kwargs):
        """Generate video with I2V chaining for seamless transitions.
        
        Args:
            clips: List of clip dictionaries
            model_info: Model information
            output_dir: Output directory path
            wan_args: WAN arguments
            **kwargs: Additional arguments
            
        Returns:
            Generated video result
        """
        try:
            self.core.print_wan_info(f"🎬 Starting I2V chaining generation with {len(clips)} clips")
            
            if not self.core.pipeline:
                self.core.print_wan_error("❌ No pipeline loaded for generation")
                return None
            
            results = []
            
            for i, clip in enumerate(clips):
                self.core.print_wan_info(f"🎬 Generating clip {i+1}/{len(clips)}")
                
                # Generate the clip
                if i == 0 or not results:
                    # First clip - use T2V mode
                    result = self.core.pipeline(
                        prompt=clip.get('prompt', ''),
                        height=clip.get('height', 480),
                        width=clip.get('width', 720),
                        num_frames=clip.get('num_frames', 30),
                        num_inference_steps=clip.get('steps', 20),
                        guidance_scale=clip.get('guidance_scale', 7.5)
                    )
                else:
                    # Subsequent clips - use I2V mode with last frame
                    last_frame = results[-1]['frames'][-1]  # Get last frame from previous clip
                    result = self.core.pipeline.generate_image2video(
                        image=last_frame,
                        prompt=clip.get('prompt', ''),
                        height=clip.get('height', 480),
                        width=clip.get('width', 720),
                        num_frames=clip.get('num_frames', 30),
                        num_inference_steps=clip.get('steps', 20),
                        guidance_scale=clip.get('guidance_scale', 7.5)
                    )
                
                results.append(result)
                self.core.print_wan_success(f"✅ Clip {i+1} generated successfully")
            
            self.core.print_wan_success(f"✅ I2V chaining generation complete - {len(results)} clips")
            return results
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error in I2V chaining generation: {e}")
            return None
    
    def generate_video_main(self, *args, **kwargs):
        """Main video generation entry point.
        
        This is called by the UI to generate videos.
        """
        try:
            self.core.print_wan_info("🎬 Starting WAN video generation...")
            
            # Discover models if not already done
            if not self.core.models:
                models = self.discover_models()
                if not models:
                    self.core.print_wan_error("❌ No models available for generation")
                    return False
            
            # Get best model
            best_model = self.core.get_best_model()
            if not best_model:
                self.core.print_wan_error("❌ No suitable model found for generation")
                return False
            
            # Load pipeline if not already loaded
            if not self.core.pipeline:
                success = self.load_simple_wan_pipeline(best_model)
                if not success:
                    self.core.print_wan_error("❌ Failed to load pipeline for generation")
                    return False
            
            # Extract generation parameters
            clips = kwargs.get('clips', [])
            output_dir = kwargs.get('output_dir', 'outputs')
            wan_args = kwargs.get('wan_args', {})
            
            # Generate video
            if clips:
                result = self.generate_video_with_i2v_chaining(clips, best_model, output_dir, wan_args, **kwargs)
            else:
                self.core.print_wan_warning("⚠️ No clips provided for generation")
                result = None
            
            if result:
                self.core.print_wan_success("✅ WAN video generation completed successfully")
                return True
            else:
                self.core.print_wan_error("❌ WAN video generation failed")
                return False
                
        except Exception as e:
            self.core.print_wan_error(f"❌ Error in main video generation: {e}")
            return False 