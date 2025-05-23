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
Wan Video Rendering Module
Handles the main rendering loop for Wan video generation
"""

import os
import json
import time
import math
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
import modules.shared as shared

from .wan_integration import WanVideoGenerator, WanPromptScheduler
from .video_audio_utilities import ffmpeg_stitch_video, get_ffmpeg_params, get_ffmpeg_paths
from .save_images import save_image
from .rendering.util.log_utils import YELLOW, GREEN, RED, RESET_COLOR


def render_wan_animation(args, anim_args, video_args, wan_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, root):
    """
    Main rendering function for Wan video generation
    
    Process:
    1. Parse prompts and calculate timing
    2. Generate first clip using txt2video
    3. For subsequent clips, use img2video with last frame as init
    4. Handle frame overlaps and transitions
    5. Stitch clips together with FFmpeg
    """
    
    print(f"{YELLOW}=== Starting Wan 2.1 Video Generation ==={RESET_COLOR}")
    print(f"Model Path: {wan_args.wan_model_path}")
    print(f"Resolution: {wan_args.wan_resolution}")
    print(f"FPS: {wan_args.wan_fps}")
    print(f"Clip Duration: {wan_args.wan_clip_duration}s")
    print(f"Inference Steps: {wan_args.wan_inference_steps}")
    print(f"Guidance Scale: {wan_args.wan_guidance_scale}")
    
    # Initialize Wan generator
    wan_generator = WanVideoGenerator(wan_args.wan_model_path, root.device)
    
    try:
        wan_generator.load_model()
        
        # Parse animation prompts and calculate timing
        prompt_scheduler = WanPromptScheduler(root.animation_prompts, wan_args, video_args)
        prompt_schedule = prompt_scheduler.parse_prompts_and_timing()
        
        if wan_args.wan_use_audio_sync and video_args.add_soundtrack != "None":
            audio_path = video_args.soundtrack_path if video_args.add_soundtrack == "File" else None
            prompt_schedule = prompt_scheduler.synchronize_with_audio(prompt_schedule, audio_path)
        
        print(f"Generated {len(prompt_schedule)} video clips:")
        for i, (prompt, start_time, duration) in enumerate(prompt_schedule):
            print(f"  Clip {i+1}: '{prompt[:50]}...' (start: {start_time:.1f}s, duration: {duration:.1f}s)")
        
        # Generate video clips
        all_clips = []
        all_frames = []
        previous_frame = None
        total_frames_generated = 0
        
        for i, (prompt, start_time, duration) in enumerate(prompt_schedule):
            if shared.state.interrupted:
                print(f"{RED}Generation interrupted by user{RESET_COLOR}")
                break
                
            print(f"\n{YELLOW}Generating clip {i+1}/{len(prompt_schedule)}: {prompt[:50]}...{RESET_COLOR}")
            
            # Calculate seed for this clip
            clip_seed = wan_args.wan_seed if wan_args.wan_seed != -1 else -1
            
            try:
                if i == 0:
                    # First clip: text-to-video
                    print(f"Mode: Text-to-Video")
                    frames = wan_generator.generate_txt2video(
                        prompt=prompt,
                        duration=duration,
                        fps=wan_args.wan_fps,
                        resolution=wan_args.wan_resolution,
                        steps=wan_args.wan_inference_steps,
                        guidance_scale=wan_args.wan_guidance_scale,
                        seed=clip_seed,
                        motion_strength=wan_args.wan_motion_strength
                    )
                else:
                    # Subsequent clips: image-to-video
                    print(f"Mode: Image-to-Video (using previous frame as init)")
                    frames = wan_generator.generate_img2video(
                        init_image=previous_frame,
                        prompt=prompt,
                        duration=duration,
                        fps=wan_args.wan_fps,
                        resolution=wan_args.wan_resolution,
                        steps=wan_args.wan_inference_steps,
                        guidance_scale=wan_args.wan_guidance_scale,
                        seed=clip_seed,
                        motion_strength=wan_args.wan_motion_strength
                    )
                
                if not frames:
                    print(f"{RED}ERROR: No frames generated for clip {i+1}{RESET_COLOR}")
                    continue
                
                print(f"{GREEN}Generated {len(frames)} frames for clip {i+1}{RESET_COLOR}")
                
                # Handle frame overlap and save frames
                processed_frames = handle_frame_overlap(frames, previous_frame, wan_args.wan_frame_overlap, i > 0)
                
                # Save frames to disk
                saved_frame_count = save_clip_frames(processed_frames, args.outdir, root.timestring, i, total_frames_generated)
                total_frames_generated += saved_frame_count
                
                # Add frames to the all_frames list for video stitching
                all_frames.extend(processed_frames)
                all_clips.append(processed_frames)
                
                # Extract last frame for next clip initialization
                previous_frame = wan_generator.extract_last_frame(frames)
                
                # Update progress (this integrates with Deforum's progress system)
                shared.state.job = f"Wan clip {i+1}/{len(prompt_schedule)}"
                shared.state.job_no = i + 1
                shared.state.job_count = len(prompt_schedule)
                
                print(f"{GREEN}Clip {i+1} completed successfully{RESET_COLOR}")
                
            except Exception as e:
                print(f"{RED}ERROR generating clip {i+1}: {e}{RESET_COLOR}")
                # Continue with next clip instead of failing completely
                continue
        
        # Update anim_args.max_frames to match the total number of frames generated
        anim_args.max_frames = total_frames_generated
        print(f"\n{GREEN}Total frames generated: {total_frames_generated}{RESET_COLOR}")
        
        # Set first frame for Deforum's processed output
        if all_frames:
            root.first_frame = Image.fromarray(all_frames[0])
        
        # Create a summary of the generation
        generation_info = create_generation_summary(prompt_schedule, wan_args, total_frames_generated)
        root.initial_info = generation_info
        
        print(f"{GREEN}=== Wan Video Generation Complete ==={RESET_COLOR}")
        print(f"Generated {len(all_clips)} clips with {total_frames_generated} total frames")
        print(f"Output directory: {args.outdir}")
        
    except Exception as e:
        print(f"{RED}FATAL ERROR in Wan video generation: {e}{RESET_COLOR}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        # Always unload the model to free GPU memory
        wan_generator.unload_model()
        print(f"Wan model unloaded, GPU memory freed")


def handle_frame_overlap(frames: List[np.ndarray], previous_frame: np.ndarray, overlap_count: int, is_continuation: bool) -> List[np.ndarray]:
    """
    Handle frame overlapping between clips for smooth transitions
    
    Args:
        frames: List of frames from the current clip
        previous_frame: Last frame from the previous clip (None for first clip)
        overlap_count: Number of frames to overlap
        is_continuation: Whether this is a continuation clip (not the first)
    
    Returns:
        List of processed frames with overlaps handled
    """
    if not frames:
        return frames
    
    if not is_continuation or previous_frame is None or overlap_count <= 0:
        # No overlap needed for first clip or when overlap is disabled
        return frames
    
    if overlap_count >= len(frames):
        # Overlap count is larger than clip length, reduce it
        overlap_count = max(1, len(frames) // 2)
        print(f"Reducing overlap to {overlap_count} frames (clip too short)")
    
    processed_frames = []
    
    # Add interpolated frames between previous clip and current clip
    if overlap_count > 0:
        first_frame = frames[0]
        
        # Create smooth transitions by blending frames
        for i in range(overlap_count):
            alpha = (i + 1) / (overlap_count + 1)
            blended_frame = cv2.addWeighted(previous_frame, 1.0 - alpha, first_frame, alpha, 0)
            processed_frames.append(blended_frame)
    
    # Add the original frames (skip first frame if we have overlap)
    start_idx = 1 if overlap_count > 0 else 0
    processed_frames.extend(frames[start_idx:])
    
    return processed_frames


def save_clip_frames(frames: List[np.ndarray], outdir: str, timestring: str, clip_index: int, start_frame_number: int) -> int:
    """
    Save frames from a clip to disk using Deforum's save system
    
    Args:
        frames: List of frames to save
        outdir: Output directory
        timestring: Deforum timestring for filename
        clip_index: Index of the current clip
        start_frame_number: Starting frame number for continuous numbering
    
    Returns:
        Number of frames saved
    """
    saved_count = 0
    
    for i, frame in enumerate(frames):
        try:
            # Convert numpy array to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(frame)
            
            # Calculate global frame number for continuous sequence
            global_frame_number = start_frame_number + i
            
            # Create filename using Deforum's convention
            filename = f"{timestring}_{global_frame_number:09d}.png"
            filepath = os.path.join(outdir, filename)
            
            # Save using PIL (simpler than Deforum's save_image for our use case)
            pil_image.save(filepath, "PNG")
            saved_count += 1
            
            # Progress feedback
            if i % 10 == 0 or i == len(frames) - 1:
                print(f"  Saved frame {global_frame_number} (clip {clip_index+1}, frame {i+1}/{len(frames)})")
            
        except Exception as e:
            print(f"{RED}Error saving frame {i} of clip {clip_index}: {e}{RESET_COLOR}")
            continue
    
    print(f"{GREEN}Saved {saved_count} frames from clip {clip_index+1}{RESET_COLOR}")
    return saved_count


def create_generation_summary(prompt_schedule: List[Tuple[str, float, float]], wan_args, total_frames: int) -> str:
    """
    Create a summary of the generation process for Deforum's info display
    
    Args:
        prompt_schedule: List of (prompt, start_time, duration) tuples
        wan_args: Wan arguments
        total_frames: Total number of frames generated
    
    Returns:
        Formatted summary string
    """
    total_duration = sum(duration for _, _, duration in prompt_schedule)
    
    summary = f"""
=== Wan 2.1 Video Generation Summary ===
Model: {wan_args.wan_model_path}
Resolution: {wan_args.wan_resolution}
FPS: {wan_args.wan_fps}
Total Duration: {total_duration:.1f} seconds
Total Frames: {total_frames}
Clips Generated: {len(prompt_schedule)}

Inference Settings:
- Steps: {wan_args.wan_inference_steps}
- Guidance Scale: {wan_args.wan_guidance_scale}
- Motion Strength: {wan_args.wan_motion_strength}
- Frame Overlap: {wan_args.wan_frame_overlap}

Clip Breakdown:"""
    
    for i, (prompt, start_time, duration) in enumerate(prompt_schedule):
        frames_in_clip = int(duration * wan_args.wan_fps)
        summary += f"\n  Clip {i+1}: {frames_in_clip} frames ({duration:.1f}s) - '{prompt[:50]}...'"
    
    return summary


def validate_wan_animation_args(args, anim_args, video_args, wan_args):
    """
    Validate arguments specifically for Wan animation mode
    
    Args:
        args: Deforum args
        anim_args: Animation args
        video_args: Video args
        wan_args: Wan args
    
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Check that animation mode is set correctly
    if anim_args.animation_mode != 'Wan Video':
        errors.append("Animation mode must be set to 'Wan Video'")
    
    # Check Wan-specific validations
    from .wan_integration import validate_wan_settings
    wan_errors = validate_wan_settings(wan_args)
    errors.extend(wan_errors)
    
    # Check for conflicting settings
    if anim_args.use_depth_warping:
        errors.append("Depth warping is not compatible with Wan video generation")
    
    if anim_args.optical_flow_cadence != "None":
        errors.append("Optical flow is not compatible with Wan video generation")
    
    if anim_args.hybrid_motion != "None":
        errors.append("Hybrid motion is not compatible with Wan video generation")
    
    # Check prompts
    if not hasattr(args, 'animation_prompts') or not args.animation_prompts:
        errors.append("Animation prompts are required for Wan video generation")
    
    return errors


def estimate_wan_generation_time(prompt_schedule: List[Tuple[str, float, float]], wan_args) -> float:
    """
    Estimate the total generation time for Wan video
    
    Args:
        prompt_schedule: List of (prompt, start_time, duration) tuples
        wan_args: Wan arguments
    
    Returns:
        Estimated time in seconds
    """
    # Base time per frame (rough estimate based on inference steps and resolution)
    width, height = map(int, wan_args.wan_resolution.split('x'))
    pixels = width * height
    
    # Time estimation formula (very rough)
    base_time_per_frame = (wan_args.wan_inference_steps / 50.0) * (pixels / (512 * 512)) * 2.0
    
    total_frames = sum(int(duration * wan_args.wan_fps) for _, _, duration in prompt_schedule)
    estimated_time = total_frames * base_time_per_frame
    
    # Add overhead for model loading, frame processing, etc.
    overhead = 60.0  # 1 minute overhead
    
    return estimated_time + overhead


def get_wan_memory_requirements(wan_args) -> Dict[str, float]:
    """
    Estimate memory requirements for Wan generation
    
    Args:
        wan_args: Wan arguments
    
    Returns:
        Dictionary with memory estimates in GB
    """
    width, height = map(int, wan_args.wan_resolution.split('x'))
    
    # Base memory requirements (rough estimates)
    model_memory = 8.0  # GB for Wan 2.1 model
    
    # Video memory scales with resolution and clip length
    frame_memory = (width * height * 3 * 4) / (1024**3)  # 4 bytes per pixel (float32), 3 channels
    max_clip_frames = int(wan_args.wan_clip_duration * wan_args.wan_fps)
    video_memory = frame_memory * max_clip_frames * 2  # 2x for processing overhead
    
    return {
        'model': model_memory,
        'video_processing': video_memory,
        'total_minimum': model_memory + video_memory,
        'recommended': (model_memory + video_memory) * 1.5  # 50% safety margin
    }
