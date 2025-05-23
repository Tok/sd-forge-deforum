# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

"""
Wan 2.1 Integration Module for Deforum
Handles text-to-video and image-to-video generation using Wan 2.1
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Dict, Any
import json
import math

class WanVideoGenerator:
    """
    Main class for Wan 2.1 video generation integration
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.pipeline = None
        self.loaded = False
        
    def is_wan_available(self) -> bool:
        """Check if Wan 2.1 is available and can be imported"""
        try:
            # TODO: Import Wan 2.1 modules when available
            # import wan
            # return True
            return os.path.exists(self.model_path) if self.model_path else False
        except ImportError:
            return False
        
    def load_model(self):
        """Load Wan 2.1 model"""
        if self.loaded:
            return
            
        if not self.is_wan_available():
            raise RuntimeError(f"Wan 2.1 model not found at {self.model_path}")
        
        print(f"Loading Wan 2.1 model from {self.model_path}")
        
        try:
            # TODO: Implement actual Wan 2.1 model loading
            # self.pipeline = wan.load_pipeline(self.model_path, device=self.device)
            # self.model = self.pipeline.model
            
            # Placeholder for now
            self.loaded = True
            print("Wan 2.1 model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Wan 2.1 model: {e}")
        
    def generate_txt2video(self, 
                          prompt: str, 
                          duration: float = 4.0,
                          fps: int = 24,
                          resolution: str = "768x768",
                          steps: int = 50,
                          guidance_scale: float = 7.5,
                          seed: int = -1,
                          motion_strength: float = 1.0,
                          **kwargs) -> List[np.ndarray]:
        """
        Generate video from text prompt using Wan 2.1
        
        Args:
            prompt: Text prompt for video generation
            duration: Video duration in seconds
            fps: Frames per second
            resolution: Video resolution (e.g., "768x768")
            steps: Number of inference steps
            guidance_scale: Guidance scale for prompt adherence
            seed: Random seed (-1 for random)
            motion_strength: Strength of motion in the video
            
        Returns:
            List of video frames as numpy arrays
        """
        if not self.loaded:
            self.load_model()
            
        # Parse resolution
        width, height = map(int, resolution.split('x'))
        num_frames = self.calculate_frame_count(duration, fps)
        
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
        print(f"Generating txt2video: '{prompt}' ({num_frames} frames @ {fps}fps, {resolution})")
        
        try:
            # TODO: Implement actual Wan 2.1 text-to-video generation
            # frames = self.pipeline.txt2video(
            #     prompt=prompt,
            #     num_frames=num_frames,
            #     width=width,
            #     height=height,
            #     num_inference_steps=steps,
            #     guidance_scale=guidance_scale,
            #     generator=torch.Generator(device=self.device).manual_seed(seed),
            #     motion_strength=motion_strength,
            #     **kwargs
            # )
            
            # Placeholder: Generate dummy frames for testing
            frames = []
            for i in range(num_frames):
                # Create a simple gradient frame for testing
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :, 0] = int(255 * i / num_frames)  # Red gradient
                frame[:, :, 1] = 128  # Green constant
                frame[:, :, 2] = int(255 * (1 - i / num_frames))  # Blue gradient
                frames.append(frame)
            
            return frames
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate txt2video: {e}")
        
    def generate_img2video(self, 
                          init_image: np.ndarray,
                          prompt: str, 
                          duration: float = 4.0,
                          fps: int = 24,
                          resolution: str = "768x768",
                          steps: int = 50,
                          guidance_scale: float = 7.5,
                          seed: int = -1,
                          motion_strength: float = 1.0,
                          **kwargs) -> List[np.ndarray]:
        """
        Generate video from initial image and text prompt using Wan 2.1
        
        Args:
            init_image: Initial image as numpy array
            prompt: Text prompt for video generation
            duration: Video duration in seconds
            fps: Frames per second
            resolution: Video resolution (e.g., "768x768")
            steps: Number of inference steps
            guidance_scale: Guidance scale for prompt adherence
            seed: Random seed (-1 for random)
            motion_strength: Strength of motion in the video
            
        Returns:
            List of video frames as numpy arrays
        """
        if not self.loaded:
            self.load_model()
            
        # Parse resolution
        width, height = map(int, resolution.split('x'))
        num_frames = self.calculate_frame_count(duration, fps)
        
        # Resize init_image to target resolution
        init_image_resized = cv2.resize(init_image, (width, height))
        
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
        print(f"Generating img2video: '{prompt}' ({num_frames} frames @ {fps}fps, {resolution})")
        
        try:
            # TODO: Implement actual Wan 2.1 image-to-video generation
            # frames = self.pipeline.img2video(
            #     image=init_image_resized,
            #     prompt=prompt,
            #     num_frames=num_frames,
            #     width=width,
            #     height=height,
            #     num_inference_steps=steps,
            #     guidance_scale=guidance_scale,
            #     generator=torch.Generator(device=self.device).manual_seed(seed),
            #     motion_strength=motion_strength,
            #     **kwargs
            # )
            
            # Placeholder: Generate dummy frames based on init_image
            frames = []
            for i in range(num_frames):
                # Create frames that morph from the init_image
                alpha = 0.8 - (0.3 * i / num_frames)  # Fade effect
                frame = (init_image_resized * alpha).astype(np.uint8)
                
                # Add some motion effect
                motion_offset = int(10 * math.sin(i * 0.2))
                if motion_offset != 0:
                    frame = np.roll(frame, motion_offset, axis=1)
                    
                frames.append(frame)
            
            return frames
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate img2video: {e}")
        
    def calculate_frame_count(self, duration: float, fps: float) -> int:
        """Calculate number of frames for given duration and FPS"""
        return max(1, int(duration * fps))
        
    def extract_last_frame(self, video_frames: List[np.ndarray]) -> np.ndarray:
        """Extract the last frame from a video sequence"""
        if not video_frames:
            raise ValueError("Empty video frames list")
        return video_frames[-1].copy()
        
    def extract_first_frame(self, video_frames: List[np.ndarray]) -> np.ndarray:
        """Extract the first frame from a video sequence"""
        if not video_frames:
            raise ValueError("Empty video frames list")
        return video_frames[0].copy()
        
    def blend_frames(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend two frames together"""
        return cv2.addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0)
        
    def interpolate_between_frames(self, frame1: np.ndarray, frame2: np.ndarray, num_frames: int) -> List[np.ndarray]:
        """Create interpolated frames between two frames"""
        if num_frames <= 0:
            return []
            
        interpolated = []
        for i in range(num_frames):
            alpha = (i + 1) / (num_frames + 1)
            interpolated_frame = self.blend_frames(frame1, frame2, alpha)
            interpolated.append(interpolated_frame)
            
        return interpolated
        
    def validate_resolution(self, resolution: str) -> Tuple[int, int]:
        """Validate and parse resolution string"""
        try:
            width, height = map(int, resolution.split('x'))
            if width <= 0 or height <= 0:
                raise ValueError("Resolution dimensions must be positive")
            if width > 2048 or height > 2048:
                raise ValueError("Resolution too large (max 2048x2048)")
            return width, height
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid resolution format '{resolution}': {e}")
        
    def unload_model(self):
        """Free GPU memory"""
        if self.loaded:
            print("Unloading Wan 2.1 model...")
            del self.model
            del self.pipeline
            self.model = None
            self.pipeline = None
            self.loaded = False
            torch.cuda.empty_cache()
            print("Wan 2.1 model unloaded")


class WanPromptScheduler:
    """
    Handle prompt scheduling and timing calculations for Wan video generation
    """
    
    def __init__(self, animation_prompts: Dict[str, str], wan_args, video_args):
        self.animation_prompts = animation_prompts
        self.wan_args = wan_args
        self.video_args = video_args
        
    def parse_prompts_and_timing(self) -> List[Tuple[str, float, float]]:
        """
        Parse animation prompts and calculate timing for each clip
        
        Returns:
            List of tuples: (prompt, start_time, duration)
        """
        # Convert frame-based prompts to time-based clips
        frame_prompts = []
        
        # Parse and sort frame numbers
        for frame_str, prompt in self.animation_prompts.items():
            try:
                if isinstance(frame_str, str) and frame_str.isdigit():
                    frame_num = int(frame_str)
                elif isinstance(frame_str, (int, float)):
                    frame_num = int(frame_str)
                else:
                    # Handle expressions like "max_f-1"
                    continue
                    
                frame_prompts.append((frame_num, prompt))
            except ValueError:
                continue
                
        # Sort by frame number
        frame_prompts.sort(key=lambda x: x[0])
        
        if not frame_prompts:
            raise ValueError("No valid frame prompts found")
            
        # Calculate timing
        fps = self.wan_args.wan_fps
        default_duration = self.wan_args.wan_clip_duration
        
        clips = []
        for i, (frame_num, prompt) in enumerate(frame_prompts):
            start_time = frame_num / fps
            
            # Calculate duration until next prompt or use default
            if i < len(frame_prompts) - 1:
                next_frame = frame_prompts[i + 1][0]
                duration = (next_frame - frame_num) / fps
                # Ensure minimum duration
                duration = max(duration, 1.0)
            else:
                duration = default_duration
                
            # Limit maximum duration
            duration = min(duration, 30.0)
                
            clips.append((prompt, start_time, duration))
            
        return clips
        
    def synchronize_with_audio(self, clips: List[Tuple[str, float, float]], audio_path: str = None) -> List[Tuple[str, float, float]]:
        """
        Synchronize clips with audio timeline (placeholder for future implementation)
        
        Args:
            clips: List of (prompt, start_time, duration) tuples
            audio_path: Path to audio file for synchronization
            
        Returns:
            Synchronized clips
        """
        if not audio_path or not self.wan_args.wan_use_audio_sync:
            return clips
            
        # TODO: Implement audio synchronization
        # This would analyze the audio file and adjust clip timing
        # based on beat detection, voice activity, etc.
        
        return clips


def validate_wan_settings(wan_args) -> List[str]:
    """
    Validate Wan 2.1 settings and return list of validation errors
    
    Args:
        wan_args: Wan arguments namespace
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    if wan_args.wan_enabled:
        # Check model path
        if not wan_args.wan_model_path:
            errors.append("Wan model path is required when Wan is enabled")
        elif not os.path.exists(wan_args.wan_model_path):
            errors.append(f"Wan model path does not exist: {wan_args.wan_model_path}")
            
        # Validate resolution
        try:
            width, height = map(int, wan_args.wan_resolution.split('x'))
            if width <= 0 or height <= 0:
                errors.append("Invalid resolution: dimensions must be positive")
        except (ValueError, AttributeError):
            errors.append(f"Invalid resolution format: {wan_args.wan_resolution}")
            
        # Validate numeric ranges
        if wan_args.wan_clip_duration <= 0 or wan_args.wan_clip_duration > 30:
            errors.append("Clip duration must be between 0 and 30 seconds")
            
        if wan_args.wan_fps <= 0 or wan_args.wan_fps > 60:
            errors.append("FPS must be between 1 and 60")
            
        if wan_args.wan_inference_steps < 1 or wan_args.wan_inference_steps > 100:
            errors.append("Inference steps must be between 1 and 100")
            
        if wan_args.wan_guidance_scale < 1.0 or wan_args.wan_guidance_scale > 20.0:
            errors.append("Guidance scale must be between 1.0 and 20.0")
            
    return errors


def should_disable_setting_for_wan(setting_name: str, wan_enabled: bool) -> bool:
    """
    Determine if a setting should be disabled when Wan mode is active
    
    Args:
        setting_name: Name of the setting to check
        wan_enabled: Whether Wan mode is enabled
        
    Returns:
        True if setting should be disabled
    """
    if not wan_enabled:
        return False
        
    # Settings that conflict with Wan video generation
    disabled_settings = {
        # Camera movement
        'angle', 'zoom', 'translation_x', 'translation_y', 'translation_z',
        'transform_center_x', 'transform_center_y',
        'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z',
        
        # 3D depth warping
        'use_depth_warping', 'depth_algorithm', 'midas_weight',
        'padding_mode', 'sampling_mode', 'save_depth_maps',
        
        # Optical flow
        'optical_flow_cadence', 'optical_flow_redo_generation',
        'cadence_flow_factor_schedule', 'redo_flow_factor_schedule',
        
        # Hybrid video
        'hybrid_motion', 'hybrid_composite', 'hybrid_flow_method',
        
        # Traditional diffusion params
        'diffusion_cadence', 'strength_schedule', 'noise_schedule',
        'cfg_scale_schedule', 'steps_schedule',
        
        # Color coherence (might conflict)
        'color_coherence', 'color_force_grayscale',
        
        # Noise and anti-blur
        'noise_type', 'amount_schedule', 'kernel_schedule',
        
        # Perspective flip
        'enable_perspective_flip',
        
        # Camera shake
        'shake_name', 'shake_intensity', 'shake_speed',
    }
    
    return setting_name in disabled_settings
