"""
WAN 2.1 Integration Module for Deforum
Handles text-to-video and image-to-video generation using WAN 2.1 with isolated environment
WAN uses Flow Matching framework, not traditional diffusion
FAIL FAST - No fallbacks, proper error propagation, no placeholder generation
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import json
from pathlib import Path


class WanVideoGenerator:
    """
    Main class for WAN 2.1 video generation integration using isolated environment - FAIL FAST
    WAN uses Flow Matching framework with 3D causal VAE and T5 text encoder
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.loaded = False
        self.isolated_generator = None
        
    def is_wan_available(self) -> bool:
        """Check if WAN 2.1 can be made available through isolated environment - FAIL FAST"""
        if not self.model_path:
            raise ValueError("WAN model path is required")
            
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"WAN model path does not exist: {self.model_path}")
            
        # Look for any model files that we can work with
        model_files = os.listdir(self.model_path)
        
        # We can work with various formats in isolated mode
        has_model_files = any(
            f.endswith(('.safetensors', '.bin', '.ckpt', '.pth')) 
            for f in model_files
        )
        
        if not has_model_files:
            raise FileNotFoundError(f"No valid model files found in {self.model_path}. Expected formats: .safetensors, .bin, .ckpt, .pth")
            
        return True
        
    def load_model(self):
        """Load WAN 2.1 model using isolated environment approach - FAIL FAST"""
        if self.loaded:
            return
            
        print("🔄 Loading WAN model using isolated environment...")
        
        # Check availability first - will raise exception if not available
        self.is_wan_available()
        
        # Import the isolated environment manager
        try:
            from .wan_isolated_env import WanIsolatedGenerator
        except ImportError as e:
            raise ImportError(f"Failed to import WAN isolated environment: {e}")
        
        # Get extension root directory
        extension_root = Path(__file__).parent.parent.parent
        
        # Create isolated generator
        try:
            self.isolated_generator = WanIsolatedGenerator(self.model_path, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to create WAN isolated generator: {e}")
        
        # Set up the isolated environment - FAIL FAST on any error
        try:
            self.isolated_generator.setup(str(extension_root))
        except Exception as e:
            raise RuntimeError(f"Failed to setup WAN isolated environment: {e}")
        
        self.loaded = True
        print("✅ WAN 2.1 model loaded successfully")
    
    def generate_txt2video(self, 
                          prompt: str, 
                          duration: float = 4.0,
                          fps: int = 60,
                          resolution: str = "1280x720",
                          steps: int = 50,
                          guidance_scale: float = 7.5,
                          seed: int = -1,
                          motion_strength: float = 1.0,
                          **kwargs) -> List[np.ndarray]:
        """Generate video from text prompt using WAN 2.1 Flow Matching - FAIL FAST"""
        if not self.loaded:
            self.load_model()
            
        # Validate inputs - FAIL FAST
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        # Parse resolution
        try:
            width, height = map(int, resolution.split('x'))
            if width <= 0 or height <= 0:
                raise ValueError("Resolution dimensions must be positive")
        except ValueError as e:
            raise ValueError(f"Invalid resolution format '{resolution}': {e}")
            
        num_frames = self.calculate_frame_count(duration, fps)
        
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
        print(f"Generating txt2video: '{prompt}' ({num_frames} frames @ {fps}fps, {resolution})")
        
        # Use isolated environment - FAIL FAST on any error
        try:
            frames = self.isolated_generator.generate_video(
                prompt=prompt,
                num_frames=num_frames,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if torch.cuda.is_available() else None,
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"WAN video generation failed: {e}")
        
        if not frames:
            raise RuntimeError("WAN generation returned no frames")
        
        # Convert PIL Images to numpy arrays if needed
        numpy_frames = []
        for frame in frames:
            try:
                if isinstance(frame, Image.Image):
                    numpy_frames.append(np.array(frame))
                else:
                    numpy_frames.append(frame)
            except Exception as e:
                raise RuntimeError(f"Failed to convert frame to numpy array: {e}")
        
        print(f"✅ Successfully generated {len(numpy_frames)} frames using isolated environment")
        return numpy_frames
        
    def generate_img2video(self, 
                          init_image: np.ndarray,
                          prompt: str, 
                          duration: float = 4.0,
                          fps: int = 60,
                          resolution: str = "1280x720",
                          steps: int = 50,
                          guidance_scale: float = 7.5,
                          seed: int = -1,
                          motion_strength: float = 1.0,
                          **kwargs) -> List[np.ndarray]:
        """Generate video from initial image and text prompt using WAN 2.1 Flow Matching - FAIL FAST"""
        if not self.loaded:
            self.load_model()
            
        # Validate inputs - FAIL FAST
        if init_image is None:
            raise ValueError("Init image cannot be None")
            
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        # Parse resolution
        try:
            width, height = map(int, resolution.split('x'))
            if width <= 0 or height <= 0:
                raise ValueError("Resolution dimensions must be positive")
        except ValueError as e:
            raise ValueError(f"Invalid resolution format '{resolution}': {e}")
            
        num_frames = self.calculate_frame_count(duration, fps)
        
        # Resize init_image to target resolution
        try:
            pil_image = Image.fromarray(init_image)
            pil_resized = pil_image.resize((width, height))
        except Exception as e:
            raise RuntimeError(f"Failed to process init image: {e}")
        
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
        print(f"Generating img2video: '{prompt}' ({num_frames} frames @ {fps}fps, {resolution})")
        
        # Use isolated environment - FAIL FAST on any error
        try:
            frames = self.isolated_generator.generate_video(
                prompt=prompt,
                image=pil_resized,  # Pass PIL image
                num_frames=num_frames,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if torch.cuda.is_available() else None,
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"WAN img2video generation failed: {e}")
        
        if not frames:
            raise RuntimeError("WAN generation returned no frames")
        
        # Convert PIL Images to numpy arrays if needed
        numpy_frames = []
        for frame in frames:
            try:
                if isinstance(frame, Image.Image):
                    numpy_frames.append(np.array(frame))
                else:
                    numpy_frames.append(frame)
            except Exception as e:
                raise RuntimeError(f"Failed to convert frame to numpy array: {e}")
        
        print(f"✅ Successfully generated {len(numpy_frames)} frames from image using isolated environment")
        return numpy_frames
        
    def calculate_frame_count(self, duration: float, fps: float) -> int:
        """Calculate number of frames for given duration and FPS"""
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if fps <= 0:
            raise ValueError("FPS must be positive")
        return max(1, int(duration * fps))
        
    def extract_last_frame(self, video_frames: List) -> np.ndarray:
        """Extract the last frame from a video sequence - handles both PIL Images and numpy arrays"""
        if not video_frames:
            raise ValueError("Empty video frames list")
        
        last_frame = video_frames[-1]
        
        # Convert PIL Image to numpy array if needed
        if hasattr(last_frame, 'mode'):  # PIL Image
            return np.array(last_frame)
        elif isinstance(last_frame, np.ndarray):
            return last_frame.copy()
        else:
            raise ValueError(f"Unsupported frame type: {type(last_frame)}")
        
    def extract_first_frame(self, video_frames: List) -> np.ndarray:
        """Extract the first frame from a video sequence - handles both PIL Images and numpy arrays"""
        if not video_frames:
            raise ValueError("Empty video frames list")
        
        first_frame = video_frames[0]
        
        # Convert PIL Image to numpy array if needed
        if hasattr(first_frame, 'mode'):  # PIL Image
            return np.array(first_frame)
        elif isinstance(first_frame, np.ndarray):
            return first_frame.copy()
        else:
            raise ValueError(f"Unsupported frame type: {type(first_frame)}")
        
    def blend_frames(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend two frames together"""
        if frame1.shape != frame2.shape:
            raise ValueError("Frame shapes must match for blending")
        return ((1.0 - alpha) * frame1 + alpha * frame2).astype(np.uint8)
        
    def interpolate_between_frames(self, frame1: np.ndarray, frame2: np.ndarray, num_frames: int) -> List[np.ndarray]:
        """Create interpolated frames between two frames"""
        if num_frames <= 0:
            return []
            
        if frame1.shape != frame2.shape:
            raise ValueError("Frame shapes must match for interpolation")
            
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
        if self.isolated_generator:
            # Clean up isolated environment if needed
            if hasattr(self.isolated_generator, 'cleanup'):
                try:
                    self.isolated_generator.cleanup()
                except Exception as e:
                    print(f"Warning: Error during WAN cleanup: {e}")
            self.isolated_generator = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🗑️ WAN model unloaded and GPU memory freed")
        self.loaded = False


class WanPromptScheduler:
    """
    Handle prompt scheduling and timing calculations for Wan video generation
    """
    
    def __init__(self, animation_prompts: Dict[str, str], wan_args, video_args):
        if not animation_prompts:
            raise ValueError("Animation prompts cannot be empty")
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
                    # Skip expressions like "max_f-1" for now - FAIL FAST
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
        if fps <= 0:
            raise ValueError("WAN FPS must be positive")
            
        default_duration = self.wan_args.wan_clip_duration
        if default_duration <= 0:
            raise ValueError("WAN clip duration must be positive")
        
        clips = []
        for i, (frame_num, prompt) in enumerate(frame_prompts):
            start_time = frame_num / fps
            
            # Calculate duration until next prompt or use default
            if i < len(frame_prompts) - 1:
                next_frame = frame_prompts[i + 1][0]
                # Calculate exact frame count between keyframes
                frame_count = next_frame - frame_num
                duration = frame_count / fps
                # Don't apply artificial minimums - use exact frame counts
                # duration = max(duration, 0.5)  # REMOVED - use actual frame counts
            else:
                duration = default_duration
                
            # Only limit maximum duration for the last clip
            if i == len(frame_prompts) - 1:
                duration = min(duration, 8.0)  # Only apply max to last clip
                
            clips.append((prompt, start_time, duration))
            
        return clips


def validate_wan_settings(wan_args) -> List[str]:
    """
    Validate WAN 2.1 settings and return list of validation errors - FAIL FAST approach
    """
    errors = []
    
    if wan_args.wan_enabled:
        # Check model path - FAIL FAST
        if not wan_args.wan_model_path:
            errors.append("WAN model path is required when WAN is enabled")
        elif not os.path.exists(wan_args.wan_model_path):
            errors.append(f"WAN model path does not exist: {wan_args.wan_model_path}")
            
        # Validate resolution - FAIL FAST
        try:
            width, height = map(int, wan_args.wan_resolution.split('x'))
            if width <= 0 or height <= 0:
                errors.append("Invalid resolution: dimensions must be positive")
        except (ValueError, AttributeError):
            errors.append(f"Invalid resolution format: {wan_args.wan_resolution}")
            
        # Validate numeric ranges - FAIL FAST
        if wan_args.wan_clip_duration <= 0 or wan_args.wan_clip_duration > 30:
            errors.append("Clip duration must be between 0 and 30 seconds")
            
        if wan_args.wan_fps <= 0 or wan_args.wan_fps > 60:
            errors.append("FPS must be between 1 and 60")
            
        if wan_args.wan_inference_steps < 1 or wan_args.wan_inference_steps > 100:
            errors.append("Inference steps must be between 1 and 100")
            
        if wan_args.wan_guidance_scale < 1.0 or wan_args.wan_guidance_scale > 20.0:
            errors.append("Guidance scale must be between 1.0 and 20.0")
    
    # FAIL FAST - if there are any errors, raise immediately
    if errors:
        raise ValueError("WAN validation failed: " + "; ".join(errors))
    
    return []


def should_disable_setting_for_wan(setting_name: str, wan_enabled: bool) -> bool:
    """
    Determine if a setting should be disabled when Wan mode is active
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
        
        # Color coherence
        'color_coherence', 'color_force_grayscale',
        
        # Noise and anti-blur
        'noise_type', 'amount_schedule', 'kernel_schedule',
        
        # Perspective flip
        'enable_perspective_flip',
        
        # Camera shake
        'shake_name', 'shake_intensity', 'shake_speed',
    }
    
    return setting_name in disabled_settings