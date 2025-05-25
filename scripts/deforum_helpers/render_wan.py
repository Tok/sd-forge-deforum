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
Wan Video Rendering Module - STRICT MODE
Uses the new unified WAN integration with fail-fast behavior
"""

import os
import json
import time
import math
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
import modules.shared as shared

from .video_audio_utilities import ffmpeg_stitch_video, get_ffmpeg_params, get_ffmpeg_paths
from .save_images import save_image
from .rendering.util.log_utils import YELLOW, GREEN, RED, RESET_COLOR


def render_wan_animation(args, anim_args, video_args, wan_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, root):
    """
    Main rendering function for WAN video generation - STRICT MODE
    Uses the unified WAN integration with fail-fast behavior
    """
    
    # Import from the NEW unified integration (not the old one)
    from .wan_integration_unified import WanVideoGenerator, WanPromptScheduler, validate_wan_settings
    
    print(f"{YELLOW}=== Starting WAN Video Generation (STRICT MODE) ==={RESET_COLOR}")
    print(f"Model Path: {wan_args.wan_model_path}")
    print(f"Resolution: {wan_args.wan_resolution}")
    print(f"FPS: {wan_args.wan_fps}")
    print(f"Clip Duration: {wan_args.wan_clip_duration}s")
    print(f"Inference Steps: {wan_args.wan_inference_steps}")
    print(f"Guidance Scale: {wan_args.wan_guidance_scale}")
    
    # STRICT validation - will raise exception if invalid
    try:
        validate_wan_settings(wan_args)
        print(f"{GREEN}✅ WAN settings validated{RESET_COLOR}")
    except ValueError as e:
        error_msg = f"{RED}VALIDATION ERROR: {e}{RESET_COLOR}"
        print(error_msg)
        raise RuntimeError(f"WAN validation failed: {e}")
    
    # Validate that we have prompts
    if not hasattr(root, 'animation_prompts') or not root.animation_prompts:
        raise RuntimeError("❌ No animation prompts provided - WAN requires prompts to generate video")
    
    # Initialize WAN generator - STRICT MODE
    try:
        print(f"{YELLOW}🔧 Initializing WAN generator (STRICT MODE)...{RESET_COLOR}")
        wan_generator = WanVideoGenerator(wan_args.wan_model_path, shared.device)
        
        # Load model - will raise exception if components missing
        wan_generator.load_model()
        print(f"{GREEN}✅ WAN models loaded successfully{RESET_COLOR}")
            
    except Exception as e:
        error_msg = f"{RED}❌ WAN MODEL LOADING FAILED: {e}{RESET_COLOR}"
        print(error_msg)
        raise RuntimeError(f"WAN generator initialization failed: {e}")
    
    try:
        # Parse prompts and timing
        print(f"{YELLOW}📋 Parsing animation prompts...{RESET_COLOR}")
        prompt_scheduler = WanPromptScheduler(root.animation_prompts, wan_args, video_args)
        prompt_schedule = prompt_scheduler.parse_prompts_and_timing()
        
        print(f"Found {len(prompt_schedule)} clips to generate:")
        for i, (prompt, start_time, duration) in enumerate(prompt_schedule):
            frame_count = int(duration * wan_args.wan_fps)
            # Adjust for WAN's 4n+1 requirement
            if (frame_count - 1) % 4 != 0:
                frame_count = ((frame_count - 1) // 4) * 4 + 1
            print(f"  Clip {i+1}: {frame_count} frames ({duration:.1f}s) - '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        # Generate video clips using WAN
        all_frames = []
        total_frames_generated = 0
        previous_frame = None
        
        for clip_index, (prompt, start_time, duration) in enumerate(prompt_schedule):
            print(f"\n{YELLOW}🎬 Generating Clip {clip_index + 1}/{len(prompt_schedule)} with WAN{RESET_COLOR}")
            print(f"Prompt: {prompt}")
            print(f"Duration: {duration}s")
            
            try:
                # Prepare generation parameters
                generation_kwargs = {
                    'prompt': prompt,
                    'duration': duration,
                    'fps': wan_args.wan_fps,
                    'resolution': wan_args.wan_resolution,
                    'steps': wan_args.wan_inference_steps,
                    'guidance_scale': wan_args.wan_guidance_scale,
                    'seed': args.seed if hasattr(args, 'seed') else -1,
                    'motion_strength': wan_args.wan_motion_strength
                }
                
                # Determine generation type
                is_continuation = (clip_index > 0 and previous_frame is not None and wan_args.wan_frame_overlap > 0)
                
                if is_continuation:
                    print(f"🔗 Generating image-to-video continuation with WAN I2V")
                    clip_frames = wan_generator.generate_img2video(
                        init_image=previous_frame,
                        **generation_kwargs
                    )
                else:
                    print(f"🎨 Generating text-to-video with WAN T2V")
                    clip_frames = wan_generator.generate_txt2video(**generation_kwargs)
                
                if not clip_frames:
                    raise RuntimeError(f"❌ WAN generated no frames for clip {clip_index + 1}")
                
                print(f"✅ WAN generated {len(clip_frames)} frames for clip {clip_index + 1}")
                
                # Handle frame overlapping
                if clip_index > 0 and wan_args.wan_frame_overlap > 0:
                    processed_frames = handle_frame_overlap(
                        frames=clip_frames,
                        previous_frame=previous_frame,
                        overlap_count=wan_args.wan_frame_overlap,
                        is_continuation=True
                    )
                else:
                    processed_frames = clip_frames
                
                # Save frames
                frames_saved = save_clip_frames(
                    frames=processed_frames,
                    outdir=args.outdir,
                    timestring=root.timestring,
                    clip_index=clip_index,
                    start_frame_number=total_frames_generated,
                    args=args,
                    video_args=video_args,
                    root=root
                )
                
                all_frames.extend(processed_frames)
                total_frames_generated += frames_saved
                
                # Store last frame for continuation
                if processed_frames:
                    previous_frame = wan_generator.extract_last_frame(processed_frames)
                
                print(f"{GREEN}✅ Clip {clip_index + 1} completed: {frames_saved} frames saved{RESET_COLOR}")
                
            except Exception as e:
                error_msg = f"{RED}❌ WAN generation failed for clip {clip_index + 1}: {e}{RESET_COLOR}"
                print(error_msg)
                raise RuntimeError(f"WAN failed to generate clip {clip_index + 1}: {e}")
        
        # Update root with generation info
        if all_frames:
            root.first_frame = Image.fromarray(all_frames[0])
        else:
            raise RuntimeError("❌ No frames were generated by WAN")
        
        # Create generation summary
        summary = create_generation_summary(prompt_schedule, wan_args, total_frames_generated)
        print(f"\n{GREEN}{summary}{RESET_COLOR}")
        
        # Update animation args for video creation
        anim_args.max_frames = total_frames_generated
        
        print(f"\n{GREEN}🎉 WAN video generation completed successfully!{RESET_COLOR}")
        print(f"Total frames generated: {total_frames_generated}")
        print(f"Output directory: {args.outdir}")
        
    except Exception as e:
        print(f"{RED}❌ FATAL ERROR in WAN video generation: {e}{RESET_COLOR}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"WAN generation failed: {e}")
        
    finally:
        # Always attempt cleanup
        try:
            wan_generator.unload_model()
            print(f"🧹 WAN model cleanup completed")
        except Exception as e:
            print(f"Warning: Error during WAN model cleanup: {e}")


def handle_frame_overlap(frames: List[np.ndarray], previous_frame: np.ndarray, overlap_count: int, is_continuation: bool) -> List[np.ndarray]:
    """
    Handle frame overlapping between clips for smooth transitions
    """
    import cv2
    
    if not frames:
        raise ValueError("Cannot process empty frames list")
    
    if not is_continuation or previous_frame is None or overlap_count <= 0:
        return frames
    
    if overlap_count >= len(frames):
        overlap_count = max(1, len(frames) // 2)
        print(f"⚠️ Reducing overlap to {overlap_count} frames (clip too short)")
    
    processed_frames = []
    
    # Add interpolated frames between previous clip and current clip
    if overlap_count > 0:
        first_frame = frames[0]
        
        # Validate frame compatibility
        if previous_frame.shape != first_frame.shape:
            print(f"⚠️ Frame shape mismatch: {previous_frame.shape} vs {first_frame.shape}")
            # Resize previous frame to match current frame
            previous_frame_pil = Image.fromarray(previous_frame)
            previous_frame_pil = previous_frame_pil.resize((first_frame.shape[1], first_frame.shape[0]))
            previous_frame = np.array(previous_frame_pil)
        
        # Create smooth transitions by blending frames
        for i in range(overlap_count):
            alpha = (i + 1) / (overlap_count + 1)
            try:
                blended_frame = cv2.addWeighted(previous_frame, 1.0 - alpha, first_frame, alpha, 0)
                processed_frames.append(blended_frame)
            except Exception as e:
                print(f"⚠️ Frame blending failed at overlap {i}, using direct transition: {e}")
                processed_frames.append(first_frame)  
    
    # Add the original frames (skip first frame if we have overlap)
    start_idx = 1 if overlap_count > 0 else 0
    processed_frames.extend(frames[start_idx:])
    
    return processed_frames


def save_clip_frames(frames: List[np.ndarray], outdir: str, timestring: str, clip_index: int, start_frame_number: int,
                    args, video_args, root) -> int:
    """
    Save frames from a clip to disk using Deforum's save system
    """
    import cv2
    
    if not frames:
        raise ValueError("Cannot save empty frames list")
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    
    saved_count = 0
    
    for i, frame in enumerate(frames):
        try:
            # Convert numpy array to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Calculate global frame number for continuous sequence
            global_frame_number = start_frame_number + i
            
            # Create filename using Deforum's convention
            filename = f"{timestring}_{global_frame_number:09d}.png"
            
            # Save using Deforum's save_image function
            save_image(pil_image, 'PIL', filename, args, video_args, root)
            saved_count += 1
            
            # Progress feedback
            if i % 10 == 0 or i == len(frames) - 1:
                print(f"💾 Saved frame {global_frame_number} (clip {clip_index+1}, frame {i+1}/{len(frames)})")
            
        except Exception as e:
            print(f"⚠️ Failed to save frame {i} of clip {clip_index}: {e}")
            continue
    
    print(f"{GREEN}💾 {saved_count} frames saved to: {outdir}{RESET_COLOR}")
    return saved_count


def create_generation_summary(prompt_schedule: List[Tuple[str, float, float]], wan_args, total_frames: int) -> str:
    """
    Create a summary of the generation process
    """
    total_duration = sum(duration for _, _, duration in prompt_schedule)
    
    summary = f"""
=== WAN Video Generation Summary (STRICT MODE) ===
Resolution: {wan_args.wan_resolution}
FPS: {wan_args.wan_fps}
Total Duration: {total_duration:.1f} seconds
Total Frames: {total_frames}
Clips Generated: {len(prompt_schedule)}

WAN Generation Settings:
- Steps: {wan_args.wan_inference_steps}
- Guidance Scale: {wan_args.wan_guidance_scale}
- Motion Strength: {wan_args.wan_motion_strength}
- Frame Overlap: {wan_args.wan_frame_overlap}

Clip Breakdown:"""
    
    for i, (prompt, start_time, duration) in enumerate(prompt_schedule):
        frames_in_clip = int(duration * wan_args.wan_fps)
        # Adjust for WAN's 4n+1 requirement
        if (frames_in_clip - 1) % 4 != 0:
            frames_in_clip = ((frames_in_clip - 1) // 4) * 4 + 1
        summary += f"\n  Clip {i+1}: {frames_in_clip} frames ({duration:.1f}s) - '{prompt[:50]}...'"
    
    return summary


def validate_wan_animation_args(args, anim_args, video_args, wan_args):
    """
    Validate arguments for WAN animation mode - STRICT
    """
    errors = []
    
    # Check that animation mode is set correctly
    if anim_args.animation_mode != 'Wan Video':
        errors.append("Animation mode must be set to 'Wan Video'")
    
    # Check WAN-specific validations
    from .wan_integration_unified import validate_wan_settings
    try:
        validate_wan_settings(wan_args)
    except ValueError as e:
        errors.extend([str(e)])
    
    # Check prompts
    if not hasattr(args, 'animation_prompts') or not args.animation_prompts:
        errors.append("Animation prompts are required for WAN video generation")
    
    # Raise validation error if any issues found
    if errors:
        raise ValueError("WAN animation validation failed: " + "; ".join(errors))


def estimate_wan_generation_time(prompt_schedule: List[Tuple[str, float, float]], wan_args) -> float:
    """
    Estimate the total generation time for WAN video
    """
    # Base time per frame (WAN is slower but higher quality)
    width, height = map(int, wan_args.wan_resolution.split('x'))
    pixels = width * height
    
    # WAN time estimation (conservative)
    base_time_per_frame = (wan_args.wan_inference_steps / 15.0) * (pixels / (512 * 512)) * 4.0
    
    total_frames = 0
    for _, _, duration in prompt_schedule:
        frame_count = int(duration * wan_args.wan_fps)
        # Adjust for WAN's 4n+1 requirement
        if (frame_count - 1) % 4 != 0:
            frame_count = ((frame_count - 1) // 4) * 4 + 1
        total_frames += frame_count
    
    estimated_time = total_frames * base_time_per_frame
    
    # Add overhead for model loading
    overhead = 120.0  # 2 minutes overhead for WAN
    
    return estimated_time + overhead


def get_wan_memory_requirements(wan_args) -> Dict[str, float]:
    """
    Estimate memory requirements for WAN generation
    """
    width, height = map(int, wan_args.wan_resolution.split('x'))
    
    # WAN memory requirements (realistic estimates)
    base_model_memory = 12.0  # GB for WAN models (larger than typical)
    
    # Video memory scales with resolution and clip length
    frame_memory = (width * height * 3 * 4) / (1024**3)  # 4 bytes per pixel, 3 channels
    max_clip_frames = int(wan_args.wan_clip_duration * wan_args.wan_fps)
    # Adjust for WAN's 4n+1 requirement
    if (max_clip_frames - 1) % 4 != 0:
        max_clip_frames = ((max_clip_frames - 1) // 4) * 4 + 1
    
    video_memory = frame_memory * max_clip_frames * 3  # 3x for WAN processing overhead
    
    return {
        'model': base_model_memory,
        'video_processing': video_memory,
        'total_minimum': base_model_memory + video_memory,
        'recommended': (base_model_memory + video_memory) * 1.8  # 80% safety margin for WAN
    }
