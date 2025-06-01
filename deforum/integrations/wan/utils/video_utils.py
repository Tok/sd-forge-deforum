"""
WAN Video Utilities
Common video processing functions for WAN pipelines
"""

import os
import numpy as np
import torch
from typing import List, Union, Optional
from pathlib import Path

try:
    import imageio
    from PIL import Image
    print("✅ Video utilities dependencies loaded")
except ImportError as e:
    print(f"❌ Missing video dependencies: {e}")


class VideoProcessor:
    """Video processing utilities for WAN pipelines"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        
    def save_frames_as_video(self, frames, output_path: str, fps: int = 16):
        """Save frames as video file with proper format handling"""
        try:
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
                    frame = self._process_frame(frame, i)
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
                    
                    frame_np = self._process_frame(frame_np, i)
                    processed_frames.append(frame_np)
            
            print(f"🎬 Saving {len(processed_frames)} frames to {output_path}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as video
            imageio.mimsave(output_path, processed_frames, fps=fps, format='mp4')
            print(f"✅ Video saved successfully with {len(processed_frames)} frames at {fps} FPS")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save video: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process a single frame to ensure correct format"""
        
        # Ensure 3 channels
        if len(frame.shape) == 2:  # Grayscale
            frame = np.stack([frame, frame, frame], axis=2)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:  # (H, W, 1)
            frame = np.repeat(frame, 3, axis=2)
        elif len(frame.shape) == 3 and frame.shape[2] > 3:  # Too many channels
            frame = frame[:, :, :3]
        elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
            frame = frame[:, :, :3]  # Remove alpha
        
        # Convert to uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Validate final frame
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"⚠️ Frame {frame_idx}: Invalid shape {frame.shape}, fixing...")
            if len(frame.shape) == 2:
                frame = np.stack([frame, frame, frame], axis=2)
            elif len(frame.shape) == 3:
                if frame.shape[2] == 1:
                    frame = np.repeat(frame, 3, axis=2)
                elif frame.shape[2] > 3:
                    frame = frame[:, :, :3]
        
        return frame
    
    def load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load frames from video file"""
        try:
            frames = imageio.mimread(video_path, format='mp4')
            processed_frames = []
            
            for i, frame in enumerate(frames):
                processed_frame = self._process_frame(frame, i)
                processed_frames.append(processed_frame)
            
            print(f"✅ Loaded {len(processed_frames)} frames from {video_path}")
            return processed_frames
            
        except Exception as e:
            print(f"❌ Failed to load video: {e}")
            return []
    
    def frames_to_tensor(self, frames: List[np.ndarray], format: str = "CFHW") -> torch.Tensor:
        """Convert frame list to tensor format"""
        
        frames_array = np.stack(frames)  # (F, H, W, C)
        
        if format == "CFHW":  # (C, F, H, W)
            frames_array = frames_array.transpose(3, 0, 1, 2)
        elif format == "FCHW":  # (F, C, H, W) 
            frames_array = frames_array.transpose(0, 3, 1, 2)
        # else keep as FHWC
        
        frames_tensor = torch.from_numpy(frames_array).float() / 255.0
        return frames_tensor
    
    def tensor_to_frames(self, tensor: torch.Tensor) -> List[np.ndarray]:
        """Convert tensor to frame list"""
        
        if isinstance(tensor, torch.Tensor):
            frames_np = tensor.cpu().numpy()
        else:
            frames_np = tensor
        
        # Handle different formats
        if len(frames_np.shape) == 4:
            if frames_np.shape[0] == 3:  # (C, F, H, W)
                frames_np = frames_np.transpose(1, 2, 3, 0)  # (F, H, W, C)
            elif frames_np.shape[1] == 3:  # (F, C, H, W)
                frames_np = frames_np.transpose(0, 2, 3, 1)  # (F, H, W, C)
            # else assume (F, H, W, C)
        
        frames = []
        for i in range(frames_np.shape[0]):
            frame = self._process_frame(frames_np[i], i)
            frames.append(frame)
        
        return frames
    
    def create_procedural_frame(self, prompt: str, frame_idx: int, total_frames: int, 
                               width: int, height: int) -> np.ndarray:
        """Create a procedural frame based on prompt analysis"""
        
        # Create base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate content based on prompt keywords
        prompt_lower = prompt.lower()
        
        # Color scheme based on prompt
        if 'cat' in prompt_lower:
            base_color = [120, 80, 40]  # Brown/orange
        elif 'sky' in prompt_lower or 'blue' in prompt_lower:
            base_color = [30, 100, 200]  # Blue
        elif 'forest' in prompt_lower or 'green' in prompt_lower:
            base_color = [40, 150, 60]  # Green
        elif 'fire' in prompt_lower or 'red' in prompt_lower:
            base_color = [200, 50, 30]  # Red
        elif 'hallway' in prompt_lower or 'sterile' in prompt_lower:
            base_color = [180, 180, 190]  # Cool gray
        else:
            base_color = [100, 100, 100]  # Gray
        
        # Animation progress
        progress = frame_idx / max(total_frames - 1, 1)
        
        # Create animated pattern
        import math
        for y in range(height):
            for x in range(width):
                # Create animated wave patterns
                wave_x = math.sin((x / width) * 4 * math.pi + progress * 2 * math.pi)
                wave_y = math.cos((y / height) * 4 * math.pi + progress * 2 * math.pi)
                
                # Blend colors
                intensity = (wave_x + wave_y + 2) / 4  # Normalize to 0-1
                
                for c in range(3):
                    color_val = int(base_color[c] * intensity + 50)
                    frame[y, x, c] = max(0, min(255, color_val))
        
        # Add some noise for texture
        noise = np.random.randint(-20, 20, (height, width, 3))
        frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def validate_video_path(self, path: str) -> bool:
        """Validate video file path"""
        path_obj = Path(path)
        
        if not path_obj.parent.exists():
            return False
            
        if path_obj.suffix.lower() not in self.supported_formats:
            return False
            
        return True


# Global video processor instance
video_processor = VideoProcessor() 