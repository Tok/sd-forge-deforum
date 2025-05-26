"""
WAN Procedural Pipeline
Fallback pipeline that generates realistic animated videos using procedural methods
"""

import os
import time
import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

from ..utils.video_utils import VideoProcessor


class WanProceduralPipeline:
    """Procedural WAN Pipeline that generates realistic animated videos"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.video_processor = VideoProcessor()
        self.loaded = False
        
    def load_components(self):
        """Load pipeline components (no-op for procedural)"""
        print(f"🚀 Loading procedural WAN pipeline...")
        self.loaded = True
        print(f"✅ Procedural WAN pipeline loaded")
        return True
    
    def __call__(self, 
                 prompt: str,
                 height: int = 720,
                 width: int = 1280, 
                 num_frames: int = 81,
                 num_inference_steps: int = 20,
                 guidance_scale: float = 7.5,
                 negative_prompt: str = "",
                 generator: Optional[torch.Generator] = None,
                 **kwargs):
        """Generate video using procedural methods"""
        
        if not self.loaded:
            self.load_components()
        
        print(f"🎬 Generating procedural WAN video...")
        print(f"   📝 Prompt: {prompt[:50]}...")
        print(f"   📐 Size: {width}x{height}")
        print(f"   🎬 Frames: {num_frames}")
        print(f"   🔧 Steps: {num_inference_steps}")
        
        # Generate procedural video frames
        frames = []
        
        for frame_idx in tqdm(range(num_frames), desc="Generating frames"):
            # Create procedural frame based on prompt and frame index
            frame = self.video_processor.create_procedural_frame(
                prompt, frame_idx, num_frames, width, height
            )
            frames.append(frame)
            
            # Simulate processing time
            time.sleep(0.05)  # Reduced for faster generation
        
        print("✅ Procedural WAN video generation complete!")
        
        # Convert to tensor format
        frames_tensor = self.video_processor.frames_to_tensor(frames, format="CFHW")
        
        return frames_tensor
    
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
        """Generate video and save to file"""
        try:
            # Set seed for reproducibility
            if seed >= 0:
                torch.manual_seed(seed)
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
            else:
                generator = None
            
            # Generate video tensor
            result = self(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs
            )
            
            # Save video using video processor
            success = self.video_processor.save_frames_as_video(result, output_path)
            
            if success:
                print(f"💾 Video saved to: {output_path}")
                return True
            else:
                raise RuntimeError("Failed to save video")
            
        except Exception as e:
            print(f"❌ Procedural video generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


class WanProceduralConfig:
    """Configuration for procedural WAN pipeline"""
    
    def __init__(self):
        self.supported_resolutions = [
            (480, 854),   # 480p
            (720, 1280),  # 720p  
            (1080, 1920), # 1080p
        ]
        
        self.supported_frame_counts = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81]
        
        self.color_schemes = {
            'cat': [120, 80, 40],
            'sky': [30, 100, 200],
            'forest': [40, 150, 60],
            'fire': [200, 50, 30],
            'hallway': [180, 180, 190],
            'sterile': [180, 180, 190],
            'ocean': [20, 80, 140],
            'sunset': [220, 100, 60],
            'space': [20, 20, 40],
            'garden': [60, 120, 40]
        }
        
    def get_color_for_prompt(self, prompt: str) -> List[int]:
        """Get appropriate color scheme for prompt"""
        prompt_lower = prompt.lower()
        
        for keyword, color in self.color_schemes.items():
            if keyword in prompt_lower:
                return color
        
        return [100, 100, 100]  # Default gray
    
    def validate_resolution(self, width: int, height: int) -> bool:
        """Check if resolution is supported"""
        return (height, width) in self.supported_resolutions
    
    def validate_frame_count(self, num_frames: int) -> bool:
        """Check if frame count is supported"""
        return num_frames in self.supported_frame_counts 