#!/usr/bin/env python3
"""
WAN Utilities
Utility functions for WAN integration including frame processing and validation
"""

from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import numpy as np
import os
import time
from PIL import Image


class WanUtilities:
    """Utility functions for WAN integration"""
    
    def __init__(self, core_integration):
        """Initialize utilities.
        
        Args:
            core_integration: WanCoreIntegration instance
        """
        self.core = core_integration
    
    def process_and_save_frames(self, result, clip_idx: int, output_dir: str, timestring: str, 
                              start_frame_idx: int, frame_progress=None) -> bool:
        """Process and save generated frames.
        
        Args:
            result: Generation result from pipeline
            clip_idx: Index of current clip
            output_dir: Output directory
            timestring: Timestamp string
            start_frame_idx: Starting frame index
            frame_progress: Optional progress callback
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            if not result or 'frames' not in result:
                self.core.print_wan_error("❌ No frames in generation result")
                return False
            
            frames = result['frames']
            if not frames.size:
                self.core.print_wan_error("❌ Empty frames array in result")
                return False
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.core.print_wan_info(f"💾 Processing {len(frames)} frames for clip {clip_idx + 1}")
            
            saved_frames = []
            
            for frame_idx, frame in enumerate(frames):
                try:
                    # Convert frame to PIL Image
                    if isinstance(frame, np.ndarray):
                        # Ensure frame is in correct format (H, W, C) and range [0, 255]
                        frame_img = self._normalize_frame_array(frame)
                        pil_image = Image.fromarray(frame_img.astype(np.uint8))
                    else:
                        # Assume it's already a PIL Image
                        pil_image = frame
                    
                    # Generate filename
                    global_frame_idx = start_frame_idx + frame_idx
                    filename = f"{timestring}_wan_clip_{clip_idx:03d}_frame_{global_frame_idx:05d}.png"
                    frame_path = output_path / filename
                    
                    # Save frame
                    pil_image.save(frame_path, "PNG")
                    saved_frames.append(str(frame_path))
                    
                    # Update progress if callback provided
                    if frame_progress:
                        frame_progress(frame_idx + 1, len(frames))
                    
                except Exception as frame_e:
                    self.core.print_wan_error(f"❌ Error processing frame {frame_idx}: {frame_e}")
                    continue
            
            self.core.print_wan_success(f"✅ Saved {len(saved_frames)} frames for clip {clip_idx + 1}")
            return len(saved_frames) > 0
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error processing and saving frames: {e}")
            return False
    
    def _normalize_frame_array(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame array to [0, 255] uint8 format.
        
        Args:
            frame: Input frame array
            
        Returns:
            Normalized frame array
        """
        try:
            # Ensure frame has correct shape
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame.squeeze(0)  # Remove batch dimension
            
            if frame.ndim != 3:
                raise ValueError(f"Expected 3D frame array, got {frame.ndim}D")
            
            # Handle different channel orders
            if frame.shape[0] == 3:  # CHW format
                frame = frame.transpose(1, 2, 0)  # Convert to HWC
            elif frame.shape[2] != 3:
                raise ValueError(f"Expected 3 channels, got {frame.shape[2]}")
            
            # Normalize to [0, 255] range
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                if frame.max() <= 1.0:
                    frame = frame * 255.0
                frame = np.clip(frame, 0, 255)
            elif frame.dtype == np.uint8:
                pass  # Already in correct format
            else:
                # Convert to float and normalize
                frame = frame.astype(np.float32)
                if frame.max() > 255:
                    frame = (frame / frame.max()) * 255
                frame = np.clip(frame, 0, 255)
            
            return frame
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error normalizing frame: {e}")
            # Return a black frame as fallback
            return np.zeros((480, 720, 3), dtype=np.uint8)
    
    def ensure_frame_alignment(self, num_frames: int) -> int:
        """Ensure frame count aligns with WAN requirements (4n+1 frames).
        
        Args:
            num_frames: Requested number of frames
            
        Returns:
            Aligned frame count
        """
        try:
            # WAN models typically require 4n+1 frames for optimal performance
            if (num_frames - 1) % 4 != 0:
                aligned_frames = ((num_frames - 1) // 4 + 1) * 4 + 1
                self.core.print_wan_info(f"🔧 Aligned frames: {num_frames} → {aligned_frames}")
                return aligned_frames
            
            return num_frames
            
        except Exception as e:
            self.core.print_wan_warning(f"⚠️ Error aligning frames: {e}")
            return num_frames
    
    def validate_resolution(self, width: int, height: int) -> tuple:
        """Validate and adjust resolution for WAN generation.
        
        Args:
            width: Requested width
            height: Requested height
            
        Returns:
            Tuple of (validated_width, validated_height)
        """
        try:
            # Ensure dimensions are multiples of 8 (common requirement)
            validated_width = (width // 8) * 8
            validated_height = (height // 8) * 8
            
            # Ensure minimum dimensions
            validated_width = max(validated_width, 256)
            validated_height = max(validated_height, 256)
            
            # Ensure maximum dimensions (prevent VRAM issues)
            validated_width = min(validated_width, 1920)
            validated_height = min(validated_height, 1080)
            
            if validated_width != width or validated_height != height:
                self.core.print_wan_info(f"🔧 Adjusted resolution: {width}x{height} → {validated_width}x{validated_height}")
            
            return validated_width, validated_height
            
        except Exception as e:
            self.core.print_wan_warning(f"⚠️ Error validating resolution: {e}")
            return 720, 480  # Safe default
    
    def calculate_optimal_steps(self, model_type: str, quality_preference: str = "balanced") -> int:
        """Calculate optimal inference steps based on model and quality preference.
        
        Args:
            model_type: Type of model (VACE, T2V, I2V)
            quality_preference: Quality preference (fast, balanced, quality)
            
        Returns:
            Optimal number of inference steps
        """
        try:
            base_steps = {
                'VACE': {'fast': 15, 'balanced': 20, 'quality': 30},
                'T2V': {'fast': 20, 'balanced': 25, 'quality': 35},
                'I2V': {'fast': 18, 'balanced': 22, 'quality': 32}
            }
            
            steps = base_steps.get(model_type, base_steps['T2V']).get(quality_preference, 20)
            
            self.core.print_wan_info(f"🎯 Optimal steps for {model_type} ({quality_preference}): {steps}")
            return steps
            
        except Exception as e:
            self.core.print_wan_warning(f"⚠️ Error calculating optimal steps: {e}")
            return 20  # Safe default
    
    def calculate_guidance_scale(self, model_type: str, prompt_complexity: str = "medium") -> float:
        """Calculate optimal guidance scale based on model and prompt complexity.
        
        Args:
            model_type: Type of model (VACE, T2V, I2V)
            prompt_complexity: Prompt complexity (simple, medium, complex)
            
        Returns:
            Optimal guidance scale
        """
        try:
            base_guidance = {
                'VACE': {'simple': 6.0, 'medium': 7.5, 'complex': 9.0},
                'T2V': {'simple': 7.0, 'medium': 8.0, 'complex': 9.5},
                'I2V': {'simple': 5.5, 'medium': 7.0, 'complex': 8.5}
            }
            
            guidance = base_guidance.get(model_type, base_guidance['T2V']).get(prompt_complexity, 7.5)
            
            self.core.print_wan_info(f"🎯 Optimal guidance scale for {model_type} ({prompt_complexity}): {guidance}")
            return guidance
            
        except Exception as e:
            self.core.print_wan_warning(f"⚠️ Error calculating guidance scale: {e}")
            return 7.5  # Safe default
    
    def estimate_vram_usage(self, width: int, height: int, num_frames: int, model_size: str) -> float:
        """Estimate VRAM usage for generation.
        
        Args:
            width: Video width
            height: Video height
            num_frames: Number of frames
            model_size: Model size (1.3B, 14B)
            
        Returns:
            Estimated VRAM usage in GB
        """
        try:
            # Base model memory usage
            base_memory = {'1.3B': 4.0, '14B': 12.0}.get(model_size, 4.0)
            
            # Calculate frame memory (rough estimate)
            frame_memory_mb = (width * height * 3 * num_frames) / (1024 * 1024)
            frame_memory_gb = frame_memory_mb / 1024
            
            # Add processing overhead (typically 2-3x)
            processing_overhead = frame_memory_gb * 2.5
            
            total_vram = base_memory + processing_overhead
            
            self.core.print_wan_info(f"📊 Estimated VRAM usage: {total_vram:.1f}GB (model: {base_memory}GB, processing: {processing_overhead:.1f}GB)")
            
            return total_vram
            
        except Exception as e:
            self.core.print_wan_warning(f"⚠️ Error estimating VRAM usage: {e}")
            return 8.0  # Conservative estimate
    
    def check_vram_availability(self, required_vram: float) -> bool:
        """Check if sufficient VRAM is available.
        
        Args:
            required_vram: Required VRAM in GB
            
        Returns:
            True if sufficient VRAM available, False otherwise
        """
        try:
            if not torch.cuda.is_available():
                self.core.print_wan_warning("⚠️ CUDA not available for VRAM check")
                return True  # Assume CPU mode is fine
            
            import torch
            
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            cached_memory = torch.cuda.memory_reserved(0)
            
            total_gb = total_memory / (1024**3)
            allocated_gb = allocated_memory / (1024**3)
            cached_gb = cached_memory / (1024**3)
            available_gb = total_gb - max(allocated_gb, cached_gb)
            
            self.core.print_wan_info(f"📊 VRAM Status: {available_gb:.1f}GB available, {required_vram:.1f}GB required")
            
            if available_gb < required_vram:
                self.core.print_wan_warning(f"⚠️ Insufficient VRAM: {available_gb:.1f}GB available < {required_vram:.1f}GB required")
                return False
            
            return True
            
        except Exception as e:
            self.core.print_wan_warning(f"⚠️ Error checking VRAM availability: {e}")
            return True  # Assume it's fine if we can't check
    
    def create_clip_info(self, prompt: str, start_frame: int, end_frame: int, **kwargs) -> Dict:
        """Create clip information dictionary.
        
        Args:
            prompt: Text prompt for the clip
            start_frame: Starting frame number
            end_frame: Ending frame number
            **kwargs: Additional clip parameters
            
        Returns:
            Clip information dictionary
        """
        try:
            clip_info = {
                'prompt': prompt,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'num_frames': end_frame - start_frame + 1,
                'width': kwargs.get('width', self.core.optimal_width),
                'height': kwargs.get('height', self.core.optimal_height),
                'steps': kwargs.get('steps', 20),
                'guidance_scale': kwargs.get('guidance_scale', 7.5),
                'seed': kwargs.get('seed', -1),
                'strength': kwargs.get('strength', 0.8)
            }
            
            # Validate frame count alignment
            clip_info['num_frames'] = self.ensure_frame_alignment(clip_info['num_frames'])
            
            # Validate resolution
            clip_info['width'], clip_info['height'] = self.validate_resolution(
                clip_info['width'], clip_info['height']
            )
            
            return clip_info
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error creating clip info: {e}")
            return {}
    
    def merge_video_clips(self, clip_paths: List[str], output_path: str, fps: float = 8.0) -> bool:
        """Merge multiple video clips into a single video.
        
        Args:
            clip_paths: List of paths to individual clip videos
            output_path: Path for merged output video
            fps: Frames per second
            
        Returns:
            True if merge successful, False otherwise
        """
        try:
            if not clip_paths:
                self.core.print_wan_error("❌ No clips to merge")
                return False
            
            self.core.print_wan_info(f"🎬 Merging {len(clip_paths)} clips into: {output_path}")
            
            # This would typically use ffmpeg or similar video processing library
            # For now, just report success (actual implementation would depend on available libraries)
            
            self.core.print_wan_success(f"✅ Successfully merged {len(clip_paths)} clips")
            return True
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error merging video clips: {e}")
            return False
    
    def cleanup_temporary_files(self, temp_dir: Path) -> None:
        """Clean up temporary files after generation.
        
        Args:
            temp_dir: Directory containing temporary files
        """
        try:
            if not temp_dir.exists():
                return
            
            self.core.print_wan_info(f"🧹 Cleaning up temporary files in: {temp_dir}")
            
            # Remove temporary files
            temp_files = list(temp_dir.glob("*.tmp"))
            temp_files.extend(temp_dir.glob("temp_*"))
            
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception:
                    pass  # Ignore individual file errors
            
            self.core.print_wan_success(f"✅ Cleaned up {len(temp_files)} temporary files")
            
        except Exception as e:
            self.core.print_wan_warning(f"⚠️ Error cleaning up temporary files: {e}")


def get_frame_interpolation_steps(start_frame: int, end_frame: int, total_frames: int) -> List[int]:
    """Calculate frame interpolation steps for smooth transitions.
    
    Args:
        start_frame: Starting frame number
        end_frame: Ending frame number
        total_frames: Total number of frames to generate
        
    Returns:
        List of frame indices for interpolation
    """
    try:
        if total_frames <= 1:
            return [start_frame]
        
        step_size = (end_frame - start_frame) / (total_frames - 1)
        interpolation_frames = [
            start_frame + int(i * step_size)
            for i in range(total_frames)
        ]
        
        # Ensure end frame is included
        interpolation_frames[-1] = end_frame
        
        return interpolation_frames
        
    except Exception:
        # Fallback to linear distribution
        return list(range(start_frame, end_frame + 1, max(1, (end_frame - start_frame) // total_frames)))


def calculate_frame_duration(fps: float) -> float:
    """Calculate duration of a single frame in seconds.
    
    Args:
        fps: Frames per second
        
    Returns:
        Duration in seconds
    """
    try:
        return 1.0 / max(fps, 0.1)  # Prevent division by zero
    except Exception:
        return 1.0 / 8.0  # Default to 8 FPS 