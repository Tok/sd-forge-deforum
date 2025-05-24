"""
Fixed Wan Video Rendering Module - Uses Stable Diffusion
Handles video generation using existing SD models instead of placeholder patterns
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

# Fixed Wan integration that uses actual SD models
from .wan_integration_fixed import WanVideoGenerator, WanPromptScheduler, validate_wan_settings
from .video_audio_utilities import ffmpeg_stitch_video, get_ffmpeg_params, get_ffmpeg_paths
from .save_images import save_image
from .rendering.util.log_utils import YELLOW, GREEN, RED, RESET_COLOR
from .parseq_adapter import ParseqAdapter
from .animation_key_frames import DeformAnimKeys


def render_wan_animation(args, anim_args, video_args, wan_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, root):
    """
    Fixed rendering function for Wan video generation using Stable Diffusion
    """
    
    print(f"{YELLOW}=== Starting Fixed Wan Video Generation (Using SD Models) ==={RESET_COLOR}")
    print(f"Resolution: {wan_args.wan_resolution}")
    print(f"FPS: {wan_args.wan_fps}")
    print(f"Clip Duration: {wan_args.wan_clip_duration}s")
    print(f"Inference Steps: {wan_args.wan_inference_steps}")
    print(f"Guidance Scale: {wan_args.wan_guidance_scale}")
    
    # Validate all arguments before starting
    try:
        validate_wan_settings(wan_args)
    except ValueError as e:
        print(f"{RED}VALIDATION ERROR: {e}{RESET_COLOR}")
        raise RuntimeError(f"Wan validation failed: {e}")
    
    # Validate that we have prompts
    if not hasattr(root, 'animation_prompts') or not root.animation_prompts:
        raise ValueError("No animation prompts provided")
    
    # Initialize fixed Wan generator (no model loading needed)
    wan_generator = WanVideoGenerator(wan_args.wan_model_path, root.device)
    
    # Set up parseq adapter for SD generation compatibility
    parseq_adapter = ParseqAdapter(parseq_args, anim_args, video_args, controlnet_args, loop_args, freeu_args, kohya_hrfix_args)
    
    # Set up animation keys for SD generation
    keys = DeformAnimKeys(anim_args, args.seed) if not parseq_adapter.use_parseq else parseq_adapter.anim_keys
    
    try:
        print(f"{YELLOW}Parsing animation prompts...{RESET_COLOR}")
        prompt_scheduler = WanPromptScheduler(root.animation_prompts, wan_args, video_args)
        prompt_schedule = prompt_scheduler.parse_prompts_and_timing()
        
        print(f"📋 Found {len(prompt_schedule)} clips to generate:")
        for i, (prompt, start_time, duration) in enumerate(prompt_schedule):
            frame_count = int(duration * wan_args.wan_fps)
            print(f"  Clip {i+1}: {frame_count} frames ({duration:.1f}s) - '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        # Generate video clips using actual Stable Diffusion
        all_frames = []
        total_frames_generated = 0
        previous_frame = None
        
        for clip_index, (prompt, start_time, duration) in enumerate(prompt_schedule):
            print(f"\n{YELLOW}🎬 Generating Clip {clip_index + 1}/{len(prompt_schedule)} with Stable Diffusion{RESET_COLOR}")
            print(f"Prompt: {prompt}")
            print(f"Duration: {duration}s")
            
            try:
                # Determine if this is a continuation (image2video) or new generation (text2video)
                is_continuation = (clip_index > 0 and previous_frame is not None and wan_args.wan_frame_overlap > 0)
                
                if is_continuation:
                    print(f"🔗 Generating image-to-video continuation from previous frame")
                    clip_frames = wan_generator.generate_img2video(
                        init_image=previous_frame,
                        prompt=prompt,
                        duration=duration,
                        fps=wan_args.wan_fps,
                        resolution=wan_args.wan_resolution,
                        steps=wan_args.wan_inference_steps,
                        guidance_scale=wan_args.wan_guidance_scale,
                        seed=args.seed if hasattr(args, 'seed') else -1,
                        motion_strength=wan_args.wan_motion_strength,
                        # Pass SD generation parameters
                        args=args,
                        keys=keys,
                        anim_args=anim_args,
                        loop_args=loop_args,
                        controlnet_args=controlnet_args,
                        freeu_args=freeu_args,
                        kohya_hrfix_args=kohya_hrfix_args,
                        root=root,
                        parseq_adapter=parseq_adapter
                    )
                else:
                    print(f"🎨 Generating text-to-video from prompt")
                    clip_frames = wan_generator.generate_txt2video(
                        prompt=prompt,
                        duration=duration,
                        fps=wan_args.wan_fps,
                        resolution=wan_args.wan_resolution,
                        steps=wan_args.wan_inference_steps,
                        guidance_scale=wan_args.wan_guidance_scale,
                        seed=args.seed if hasattr(args, 'seed') else -1,
                        motion_strength=wan_args.wan_motion_strength,
                        # Pass SD generation parameters
                        args=args,
                        keys=keys,
                        anim_args=anim_args,
                        loop_args=loop_args,
                        controlnet_args=controlnet_args,
                        freeu_args=freeu_args,
                        kohya_hrfix_args=kohya_hrfix_args,
                        root=root,
                        parseq_adapter=parseq_adapter
                    )
                
                if not clip_frames:
                    raise RuntimeError(f"No frames generated for clip {clip_index + 1}")
                
                print(f"✅ Generated {len(clip_frames)} frames for clip {clip_index + 1}")
                
                # Handle frame overlapping between clips
                if clip_index > 0 and wan_args.wan_frame_overlap > 0:
                    processed_frames = handle_frame_overlap(
                        frames=clip_frames,
                        previous_frame=previous_frame,
                        overlap_count=wan_args.wan_frame_overlap,
                        is_continuation=True
                    )
                else:
                    processed_frames = clip_frames
                
                # Save frames to disk
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
                
                # Store last frame for potential continuation
                if processed_frames:
                    previous_frame = wan_generator.extract_last_frame(processed_frames)
                
                print(f"{GREEN}✅ Clip {clip_index + 1} completed: {frames_saved} frames saved{RESET_COLOR}")
                
            except Exception as e:
                print(f"{RED}❌ Error generating clip {clip_index + 1}: {e}{RESET_COLOR}")
                raise RuntimeError(f"Failed to generate clip {clip_index + 1}: {e}")
        
        # Update root with generation info
        root.first_frame = Image.fromarray(cv2.cvtColor(all_frames[0], cv2.COLOR_BGR2RGB)) if all_frames else None
        
        # Create generation summary
        summary = create_generation_summary(prompt_schedule, wan_args, total_frames_generated)
        print(f"\n{GREEN}{summary}{RESET_COLOR}")
        
        # Update animation args for video creation
        anim_args.max_frames = total_frames_generated
        
        print(f"\n{GREEN}🎉 Fixed Wan video generation completed successfully!{RESET_COLOR}")
        print(f"Total frames generated: {total_frames_generated}")
        print(f"Output directory: {args.outdir}")
        
    except Exception as e:
        print(f"{RED}❌ FATAL ERROR in Wan video generation: {e}{RESET_COLOR}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Wan generation failed: {e}")
        
    finally:
        # Cleanup
        wan_generator.unload_model()


def handle_frame_overlap(frames: List[np.ndarray], previous_frame: np.ndarray, overlap_count: int, is_continuation: bool) -> List[np.ndarray]:
    """
    Handle frame overlapping between clips for smooth transitions
    """
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
            previous_frame_pil = Image.fromarray(cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB))
            previous_frame_pil = previous_frame_pil.resize((first_frame.shape[1], first_frame.shape[0]))
            previous_frame = cv2.cvtColor(np.array(previous_frame_pil), cv2.COLOR_RGB2BGR)
        
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
=== Fixed Wan Video Generation Summary (Using Stable Diffusion) ===
Resolution: {wan_args.wan_resolution}
FPS: {wan_args.wan_fps}
Total Duration: {total_duration:.1f} seconds
Total Frames: {total_frames}
Clips Generated: {len(prompt_schedule)}

Generation Settings:
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
    Validate arguments for fixed Wan animation mode
    """
    errors = []
    
    # Check that animation mode is set correctly
    if anim_args.animation_mode != 'Wan Video':
        errors.append("Animation mode must be set to 'Wan Video'")
    
    # Check Wan-specific validations
    try:
        validate_wan_settings(wan_args)
    except ValueError as e:
        errors.extend([str(e)])
    
    # Check prompts
    if not hasattr(args, 'animation_prompts') or not args.animation_prompts:
        errors.append("Animation prompts are required for Wan video generation")
    
    # Raise validation error if any issues found
    if errors:
        raise ValueError("Wan animation validation failed: " + "; ".join(errors))


def estimate_wan_generation_time(prompt_schedule: List[Tuple[str, float, float]], wan_args) -> float:
    """
    Estimate the total generation time for fixed Wan video
    """
    # Base time per frame using SD (more realistic estimate)
    width, height = map(int, wan_args.wan_resolution.split('x'))
    pixels = width * height
    
    # Time estimation for SD generation (more realistic)
    base_time_per_frame = (wan_args.wan_inference_steps / 20.0) * (pixels / (512 * 512)) * 1.5
    
    total_frames = sum(int(duration * wan_args.wan_fps) for _, _, duration in prompt_schedule)
    estimated_time = total_frames * base_time_per_frame
    
    # Add overhead for SD model operations
    overhead = 30.0  # 30 second overhead
    
    return estimated_time + overhead


def get_wan_memory_requirements(wan_args) -> Dict[str, float]:
    """
    Estimate memory requirements for fixed Wan generation using SD
    """
    width, height = map(int, wan_args.wan_resolution.split('x'))
    
    # Memory requirements for SD-based generation
    sd_model_memory = 4.0  # GB for SD model (already loaded)
    
    # Video memory scales with resolution and clip length
    frame_memory = (width * height * 3 * 4) / (1024**3)  # 4 bytes per pixel, 3 channels
    max_clip_frames = int(wan_args.wan_clip_duration * wan_args.wan_fps)
    video_memory = frame_memory * max_clip_frames * 2  # 2x for processing overhead
    
    return {
        'sd_model': sd_model_memory,
        'video_processing': video_memory,
        'total_minimum': sd_model_memory + video_memory,
        'recommended': (sd_model_memory + video_memory) * 1.3  # 30% safety margin
    }
