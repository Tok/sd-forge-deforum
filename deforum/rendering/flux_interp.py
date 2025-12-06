"""
Flux + Interpolation Mode: Flux Keyframes + Choice of Interpolation

Supports two interpolation methods:
- Wan FLF2V (default): AI-generated video interpolation with semantic understanding
- FILM: Google's Frame Interpolation for Large Motion (handles dramatic changes)

Note: RIFE is NOT available here (defaults to single frames on dramatic changes).
      RIFE is available for post-processing smooth videos only.

Architecture:
  Phase 1: Generate ALL keyframes with Flux/SD
  Phase 2: Batch interpolation between each consecutive keyframe pair (Wan/FILM)
  Phase 3: Stitch final video

This combines the best of both worlds:
- High-quality Flux-generated keyframes
- Smooth interpolation between keyframes using your choice of method
"""

import os
import json
from pathlib import Path
from typing import List
import cv2

try:
    from modules import shared  # type: ignore
except ImportError:
    shared = None  # type: ignore

from .data.render_data import RenderData
from .data.frame import KeyFrameDistribution, DiffusionFrame
from .data.taqaddumat import Taqaddumat
from deforum.rendering.helpers import webui as web_ui_utils
from deforum.utils.image import processing as image_utils
from deforum.rendering.helpers import filename as filename_utils
from deforum.integrations.wan.wan_simple_integration import WanSimpleIntegration
from deforum.media.video_audio_utilities import ffmpeg_stitch_video
from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()



def render_flux_interp(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, wan_args, root):
    """
    Flux + Interpolation rendering mode: Flux for keyframes + choice of interpolation

    1. Generate all keyframes with Flux/SD
    2. Interpolate between keyframes with Wan FLF2V or FILM
    3. Stitch final video

    Interpolation method is selected via wan_args.flux_interpolation_method
    """
    logger.info(f"{emoji_if_enabled('🎬')} Flux + Interpolation Mode: Flux Keyframes + ML Interpolation")

    # Pre-download soundtrack if specified (same as core.py)
    if video_args.add_soundtrack == 'File' and video_args.soundtrack_path is not None:
        if video_args.soundtrack_path.startswith(('http://', 'https://')):
            logger.info(f"Pre-downloading soundtrack at the beginning of the render process: {video_args.soundtrack_path}")
            try:
                from deforum.media.video_audio_utilities import download_audio
                video_args.soundtrack_path = download_audio(video_args.soundtrack_path)
                logger.info(f"Audio successfully pre-downloaded to: {video_args.soundtrack_path}")
            except Exception as e:
                logger.info(f"Error pre-downloading audio: {e}")

    # Create render data
    data = RenderData.create(args, parseq_args, anim_args, video_args, loop_args, controlnet_args, root)

    # Initialize progress tracking
    web_ui_utils.init_job(data)
    shared.total_tqdm = Taqaddumat()

    # Get keyframe distribution
    keyframe_distribution = KeyFrameDistribution.from_UI_tab(data)
    all_frames = DiffusionFrame.create_all_frames(data, keyframe_distribution)

    # Initialize progress bars with all frames (keyframes + tweens)
    shared.total_tqdm.reset(data, all_frames)

    # Extract only keyframes (frames with is_keyframe=True)
    keyframes = [f for f in all_frames if f.is_keyframe]

    logger.info(f"{emoji_if_enabled('📊')} Flux/Wan Workflow:")
    logger.info(f"   Total frames: {anim_args.max_frames}")
    logger.info(f"   Keyframes to generate: {len(keyframes)}")
    logger.info(f"   FLF2V segments: {len(keyframes) - 1}")

    # Resume info (compact)
    logger.debug(f"Resume: {anim_args.resume_from_timestring}, outdir: {args.outdir}, exists: {os.path.exists(data.output_directory)}")
    if os.path.exists(data.output_directory):
        img_count = len([f for f in os.listdir(data.output_directory) if f.endswith(('.png', '.jpg', '.jpeg'))])
        logger.debug(f"Existing images in output dir: {img_count}")

    # ====================
    # PHASE 1: Batch Generate All Keyframes with Flux/SD
    # ====================
    logger.separator(char="=")
    logger.info("PHASE 1: Batch Keyframe Generation with Flux/SD")
    logger.separator(char="=")

    # Check for resume mode - scan for existing keyframes
    keyframe_images = {}  # {frame_index: image_path}
    is_resuming = anim_args.resume_from_timestring
    
    if is_resuming:
        logger.info(f"{emoji_if_enabled('🔄')} Resume mode: Scanning for existing keyframes in {data.output_directory}...")
        for frame in keyframes:
            # Check simple format first (matches our save format: 000000001.png)
            simple_filename = f"{frame.i:09d}.png"
            simple_path = os.path.join(data.output_directory, simple_filename)

            # Also check for filename with timestring prefix (legacy from old runs)
            timestring_filename = filename_utils.frame_filename(data, frame.i)
            timestring_path = os.path.join(data.output_directory, timestring_filename)

            if os.path.exists(simple_path):
                keyframe_images[frame.i] = simple_path
                logger.info(f"   {emoji_if_enabled('✓')} Found existing keyframe: {simple_filename}")
            elif os.path.exists(timestring_path):
                keyframe_images[frame.i] = timestring_path
                logger.info(f"   {emoji_if_enabled('✓')} Found existing keyframe (timestring format): {timestring_filename}")
            else:
                logger.debug(f"   {emoji_if_enabled('✗')} Missing keyframe at frame {frame.i} (tried: {simple_filename}, {timestring_filename})")
        
        if len(keyframe_images) > 0:
            logger.info(f"{emoji_if_enabled('✅')} Found {len(keyframe_images)}/{len(keyframes)} existing keyframes")

    # Count how many keyframes need to be generated
    keyframes_to_generate = [f for f in keyframes if f.i not in keyframe_images]
    keyframes_existing = [f for f in keyframes if f.i in keyframe_images]
    
    if keyframes_existing:
        logger.info(f"{emoji_if_enabled('✅')} Found {len(keyframes_existing)} existing keyframes from previous run")
    if keyframes_to_generate:
        logger.debug(f"{emoji_if_enabled('📸')} Need to generate {len(keyframes_to_generate)} new keyframes")
    
    for idx, frame in enumerate(keyframes):
        # Skip if keyframe already exists (resume mode)
        if frame.i in keyframe_images:
            continue

        logger.debug(f"\n{emoji_if_enabled('📸')} Generating NEW keyframe {idx + 1}/{len(keyframes)} (frame {frame.i})...")

        # Set scheduled parameters for this frame (prompt, cfg_scale, distilled_cfg_scale, checkpoint, etc.)
        keys = data.animation_keys.deform_keys
        # Clamp frame index to valid range (prompt_series has max_frames entries indexed 0 to max_frames-1)
        frame_idx = min(frame.i, data.args.anim_args.max_frames - 1)
        data.args.args.prompt = data.prompt_series[frame_idx]  # Set prompt for current frame
        data.args.args.cfg_scale = keys.cfg_scale_schedule_series[frame_idx]
        data.args.args.distilled_cfg_scale = keys.distilled_cfg_scale_schedule_series[frame_idx]

        # Checkpoint scheduling (disabled for Flux/Wan mode - always use loaded Flux model)
        if data.args.anim_args.enable_checkpoint_scheduling:
            data.args.args.checkpoint = keys.checkpoint_schedule_series[frame_idx]
        else:
            data.args.args.checkpoint = None

        # Reset progress tracking for this frame
        shared.total_tqdm.reset_step_count(frame.actual_steps(data))

        # Generate keyframe image using Flux/SD
        web_ui_utils.update_job(data, frame.i)
        image = frame.generate(data, shared.total_tqdm)

        if image is None:
            raise RuntimeError(f"Failed to generate keyframe at frame {frame.i}")

        # Save keyframe
        keyframe_path = save_keyframe(data, frame, image)
        keyframe_images[frame.i] = keyframe_path

        logger.info(f"{emoji_if_enabled('✅')} Keyframe {idx + 1} saved: {os.path.basename(keyframe_path)}")

        # Set first_frame for UI display (use first generated keyframe)
        if idx == 0:
            data.args.root.first_frame = image

    newly_generated = len(keyframes_to_generate)
    from_resume = len(keyframes_existing)

    # Ensure first_frame is set for UI display (load from disk if not yet set)
    if data.args.root.first_frame is None and keyframes:
        first_keyframe_path = keyframe_images.get(keyframes[0].i)
        if first_keyframe_path and os.path.exists(first_keyframe_path):
            from PIL import Image
            data.args.root.first_frame = Image.open(first_keyframe_path)
            logger.info(f"{emoji_if_enabled('✅')} Loaded first frame from disk for UI display")
    
    logger.info(f"\n{emoji_if_enabled('✅')} Phase 1 Complete: {len(keyframes)} keyframes ready")
    if from_resume > 0:
        logger.info(f"   ({newly_generated} newly generated, {from_resume} from previous run)")
    else:
        logger.info(f"   (All {newly_generated} keyframes newly generated with Flux/SD)")

    # CRITICAL: Aggressive VRAM cleanup between Phase 1 and Phase 2
    # This allows switching from Flux/Z-Image to DA3-GIANT without VRAM conflicts
    logger.info(f"\n{emoji_if_enabled('🧹')} Cleaning up VRAM before Phase 2...")
    import gc
    import torch
    from modules import devices

    # Unload Flux/SD models completely
    try:
        devices.torch_gc()
    except:
        pass

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    logger.info(f"{emoji_if_enabled('✓')} VRAM cleanup complete - ready for interpolation models")

    # ====================
    # PHASE 2: Batch Frame Interpolation (Wan/FILM/DA3-3DGS)
    # ====================
    logger.separator(char="=")
    logger.info("PHASE 2: Batch Frame Interpolation")
    logger.separator(char="=")

    # Get interpolation method (check once for all segments)
    interp_method = getattr(wan_args, 'flux_flf2v_interpolation_method', 'Wan')
    logger.info(f"{emoji_if_enabled('📊')} Interpolation method: {interp_method}")

    # Check if DA3-3DGS is selected (not yet implemented)
    if interp_method == "DA3-3DGS":
        logger.warning(f"{emoji_if_enabled('⚠️')} DA3-3DGS interpolation is not yet implemented!")
        logger.warning(f"   This feature requires DA3-GIANT models with trained 3DGS heads.")
        logger.warning(f"   Falling back to Wan FLF2V interpolation...")
        interp_method = "Wan"  # Fallback to Wan

    # Unload Flux model to free GPU memory
    logger.info(f"{emoji_if_enabled('🗑')}️  Unloading Flux model to free GPU memory...")
    from backend import memory_management
    memory_management.unload_all_models()
    memory_management.soft_empty_cache()
    logger.info(f"{emoji_if_enabled('✅')} GPU memory freed")

    # Initialize Wan only if needed
    wan_integration = None
    if interp_method == "Wan":
        wan_integration = WanSimpleIntegration(device='cuda')

        # Discover and load Wan model
        logger.info(f"{emoji_if_enabled('🔍')} Discovering Wan FLF2V models...")
        discovered_models = wan_integration.discover_models()

        if not discovered_models:
            raise RuntimeError("No Wan models found. Please download a Wan model to models/Deforum/wan directory first.")

        # Use best available FLF2V model
        flf2v_models = [m for m in discovered_models if m['type'] == 'FLF2V']
        if not flf2v_models:
            ti2v_models = [m['name'] for m in discovered_models if m['type'] in ('TI2V', 'T2V', 'I2V')]
            logger.error(f"{emoji_if_enabled('❌')} No FLF2V model found!")
            if ti2v_models:
                logger.warning(f"   Found T2V/TI2V models: {', '.join(ti2v_models)}")
                logger.warning(f"   {emoji_if_enabled('⚠')}️  TI2V/T2V models CANNOT do FLF2V interpolation!")
            logger.info("   Download FLF2V model: huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers --local-dir models/Deforum/wan/Wan2.1-FLF2V-14B")
            raise RuntimeError("FLF2V model required but not found. TI2V models cannot do FLF2V interpolation.")

        model_info = flf2v_models[0]
        logger.info(f"{emoji_if_enabled('📦')} Loading Wan model: {model_info['name']}")

        # Load the Wan pipeline
        success = wan_integration.load_simple_wan_pipeline(model_info, wan_args)
        if not success:
            raise RuntimeError(f"Failed to load Wan model: {model_info['name']}")

    # Generate FLF2V segments
    all_segment_frames = []

    for idx in range(len(keyframes) - 1):
        first_kf = keyframes[idx]
        last_kf = keyframes[idx + 1]

        first_frame_idx = first_kf.i
        last_frame_idx = last_kf.i
        num_tween_frames = last_frame_idx - first_frame_idx - 1  # ONLY in-between frames (exclude both keyframes)

        logger.warning(f"\n{emoji_if_enabled('🎞')}️ Interpolation Segment {idx + 1}/{len(keyframes) - 1}:")
        logger.warning(f"   From keyframe: {first_frame_idx}")
        logger.warning(f"   To keyframe: {last_frame_idx}")
        logger.warning(f"   In-between frames to generate: {num_tween_frames} (frames {first_frame_idx+1} to {last_frame_idx-1})")

        # Check if all frames in this segment already exist (resume mode)
        if is_resuming:
            segment_complete = True
            segment_existing_frames = []
            for frame_offset in range(num_tween_frames):
                check_frame_idx = first_frame_idx + frame_offset + 1  # +1 to skip first keyframe

                # Check simple format first (matches our save format: 000000001.png)
                simple_filename = f"{check_frame_idx:09d}.png"
                simple_path = os.path.join(data.output_directory, simple_filename)

                # Also check timestring format (legacy from old runs)
                timestring_filename = filename_utils.frame_filename(data, check_frame_idx)
                timestring_path = os.path.join(data.output_directory, timestring_filename)

                if os.path.exists(simple_path):
                    segment_existing_frames.append(simple_path)
                elif os.path.exists(timestring_path):
                    segment_existing_frames.append(timestring_path)
                else:
                    segment_complete = False
                    break
            
            if segment_complete:
                logger.debug(f"{emoji_if_enabled('⏭')}️  Skipping segment {idx + 1} - all {num_tween_frames} frames already exist")
                all_segment_frames.extend(segment_existing_frames)
                continue

        # Get prompts for BOTH keyframes
        first_prompt_idx = min(first_frame_idx, len(data.prompt_series) - 1)
        last_prompt_idx = min(last_frame_idx, len(data.prompt_series) - 1)
        first_prompt_raw = data.prompt_series[first_prompt_idx]
        last_prompt_raw = data.prompt_series[last_prompt_idx]

        # Strip --neg negative prompts (Wan doesn't understand this syntax and will interpret them positively!)
        def strip_negative_prompt(prompt_text):
            """Remove --neg ... portion from Deforum prompts to avoid Wan interpreting them as positive."""
            if '--neg' in prompt_text:
                return prompt_text.split('--neg')[0].strip()
            return prompt_text.strip()

        first_prompt = strip_negative_prompt(first_prompt_raw)
        last_prompt = strip_negative_prompt(last_prompt_raw)

        # Load keyframe images - use PIL since they were saved with PIL (RGB format)
        # Using cv2.imread() on PIL-saved images causes BGR/RGB confusion
        from PIL import Image
        first_image = Image.open(keyframe_images[first_frame_idx])
        last_image = Image.open(keyframe_images[last_frame_idx])

        # Resize keyframes if resolution changed (e.g., for VRAM savings)
        target_width = data.width()
        target_height = data.height()
        if first_image.size != (target_width, target_height):
            logger.debug(f"   Resizing keyframes from {first_image.size} to {target_width}x{target_height}")
            first_image = first_image.resize((target_width, target_height), Image.LANCZOS)
            last_image = last_image.resize((target_width, target_height), Image.LANCZOS)

        # For FLF2V interpolation, use balanced guidance for semantic interpolation
        # High guidance forces prompt adherence, low guidance allows natural interpolation
        flf2v_guidance = getattr(wan_args, 'wan_flf2v_guidance_scale', 3.5)  # Default 3.5 for smooth morphing

        # Decide how to handle prompts for FLF2V
        # Options: 'none', 'first', 'last', 'blend'
        flf2v_prompt_mode = getattr(wan_args, 'wan_flf2v_prompt_mode', 'blend')  # Default to blend for semantic guidance
        
        if flf2v_prompt_mode == 'none':
            flf2v_prompt = ""
        elif flf2v_prompt_mode == 'first':
            flf2v_prompt = first_prompt
        elif flf2v_prompt_mode == 'last':
            flf2v_prompt = last_prompt
        elif flf2v_prompt_mode == 'blend':
            # Create a blended prompt describing the transition
            flf2v_prompt = f"{first_prompt} transitioning to {last_prompt}"
        else:
            flf2v_prompt = ""  # Default to no prompt
        
        logger.info(f"   {emoji_if_enabled('🎯')} Interpolation Settings:")
        logger.info(f"      Method: {interp_method}")

        # Route to appropriate interpolation function
        if interp_method == "FILM":
            logger.info(f"      Using FILM (Frame Interpolation for Large Motion)")
            segment_frames = generate_film_segment(
                first_image=first_image,
                last_image=last_image,
                num_frames=num_tween_frames,
                height=data.height(),
                width=data.width(),
                first_frame_idx=first_frame_idx,
                output_dir=data.output_directory,
                fps=video_args.fps
            )
        else:  # Default: Wan
            logger.info(f"      Guidance scale: {flf2v_guidance} {'(pure interpolation)' if flf2v_guidance == 0.0 else ''}")
            logger.info(f"      Prompt mode: {flf2v_prompt_mode}")
            logger.info(f"      First keyframe prompt: {first_prompt[:60]}...")
            logger.info(f"      Last keyframe prompt: {last_prompt[:60]}...")
            logger.info(f"      → Using: '{flf2v_prompt[:80]}...' {'(empty = pure interpolation)' if not flf2v_prompt else ''}")
            logger.info(f"      Inference steps: {wan_args.wan_inference_steps}")

            # Call Wan FLF2V
            segment_frames = generate_flf2v_segment(
                wan_integration=wan_integration,
                first_image=first_image,
                last_image=last_image,
                prompt=flf2v_prompt,
                num_frames=num_tween_frames,
                height=data.height(),
                width=data.width(),
                num_inference_steps=wan_args.wan_inference_steps,
                guidance_scale=flf2v_guidance,
                first_frame_idx=first_frame_idx,
                output_dir=data.output_directory
            )

        all_segment_frames.extend(segment_frames)

        logger.info(f"{emoji_if_enabled('✅')} Segment {idx + 1} complete: {len(segment_frames)} frames")

    logger.info(f"\n{emoji_if_enabled('✅')} Phase 2 Complete: {len(all_segment_frames)} total frames from {interp_method}")

    # ====================
    # PHASE 3: Stitch Final Video
    # ====================
    logger.separator(char="=")
    logger.info("PHASE 3: Stitching Final Video")
    logger.separator(char="=")

    # Stitch video using existing utilities
    output_video_path = stitch_wan_flux_video(
        data=data,
        frame_paths=all_segment_frames,
        video_args=video_args,
        interp_method=interp_method
    )

    logger.info(f"\n{emoji_if_enabled('🎉')} Flux + Interpolation Generation Complete!")
    logger.info(f"{emoji_if_enabled('📁')} Output: {output_video_path}")

    # Cleanup Wan if it was loaded
    if wan_integration is not None:
        wan_integration.unload_model()


def save_keyframe(data: RenderData, frame: DiffusionFrame, image):
    """Save keyframe image to disk with simple frame number naming (no timestring prefix)"""
    # Use simple format like Flux/Wan mode: 000000001.png instead of timestring_000000001.png
    filename = f"{frame.i:09d}.png"
    filepath = os.path.join(data.output_directory, filename)

    # Convert CV2 image to PIL if needed, then save
    if image_utils.is_PIL(image):
        image.save(filepath)
    else:
        # Convert numpy/cv2 image to PIL
        pil_image = image_utils.numpy_to_pil(image)
        pil_image.save(filepath)

    return filepath


def generate_flf2v_segment(wan_integration, first_image, last_image, prompt, num_frames,
                           height, width, num_inference_steps, guidance_scale,
                           first_frame_idx, output_dir):
    """Generate frames for one FLF2V segment"""

    # Adjust frame count to Wan's 4n+1 requirement (ROUND UP to avoid gaps)
    import math
    adjusted_frames = math.ceil((num_frames - 1) / 4) * 4 + 1
    if adjusted_frames != num_frames:
        logger.debug(f"   Wan requires 4n+1 frames: {num_frames} → {adjusted_frames} (will generate extra, use first {num_frames})")

    # Generate FLF2V interpolation
    result = wan_integration.pipeline.generate_flf2v(
        first_frame=first_image,
        last_frame=last_image,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=adjusted_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )

    # Extract frames from result (handle different output formats)
    frames = None
    if isinstance(result, tuple):
        frames = result[0]
    elif hasattr(result, 'frames'):
        frames = result.frames
    elif hasattr(result, 'images'):
        frames = result.images
    elif hasattr(result, 'videos'):
        frames = result.videos
    else:
        frames = result

    # Convert frames if needed
    if isinstance(frames, list) and len(frames) > 0:
        # Already a list of PIL Images
        frame_list = frames
    elif hasattr(frames, 'shape'):
        # It's a tensor or numpy array
        import torch
        import numpy as np
        from PIL import Image
        
        logger.info(f"   Processing FLF2V tensor/array with shape: {frames.shape}")
        
        if hasattr(frames, 'cpu'):
            frames_np = frames.cpu().numpy()
        else:
            frames_np = np.array(frames)
        
        # Handle different tensor formats
        if len(frames_np.shape) == 5:
            frames_np = frames_np.squeeze(0)
        if len(frames_np.shape) == 4 and frames_np.shape[0] <= 4:
            frames_np = np.transpose(frames_np, (1, 2, 3, 0))
        
        # Extract individual frames
        frame_list = []
        for i in range(frames_np.shape[0]):
            frame = frames_np[i]
            if frame.dtype in [np.float32, np.float64]:
                if frame.min() < 0:
                    frame = (frame + 1.0) / 2.0
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frame_list.append(Image.fromarray(frame))
        
        logger.info(f"   Extracted {len(frame_list)} FLF2V frames")
        
    elif hasattr(frames, '__getitem__') and hasattr(frames, '__len__'):
        # It's indexable and has length (like a tensor or array)
        logger.info(f"   Converting indexable FLF2V output (length: {len(frames)})")
        frame_list = [frames[i] for i in range(len(frames))]
    else:
        logger.error("Unable to extract frames from FLF2V output")
        raise RuntimeError(f"Unexpected FLF2V output format: {type(result)}")

    # Save frames (only first num_frames, discard extras from 4n+1 padding)
    frame_paths = []
    frames_to_save = min(num_frames, len(frame_list))
    
    if len(frame_list) > num_frames:
        logger.debug(f"   Generated {len(frame_list)} frames, using first {num_frames} (discarding {len(frame_list) - num_frames} padding frames)")
    
    for local_idx in range(frames_to_save):
        frame = frame_list[local_idx]
        # Start at first_frame_idx + 1 to avoid overwriting the first keyframe
        # This generates frames BETWEEN the keyframes, not including them
        global_frame_idx = first_frame_idx + 1 + local_idx
        filename = f"{global_frame_idx:09d}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Ensure it's a PIL Image
        if hasattr(frame, 'save'):
            frame.save(filepath)
        else:
            from PIL import Image
            Image.fromarray(frame).save(filepath)
        
        frame_paths.append(filepath)

    return frame_paths


def stitch_wan_flux_video(data, frame_paths, video_args, interp_method="Wan"):
    """Stitch all frames into final video using ffmpeg concat demuxer"""
    from deforum.media.video_audio_utilities import get_ffmpeg_params
    import glob
    import subprocess

    # Get ffmpeg parameters from settings
    ffmpeg_location, ffmpeg_crf, ffmpeg_preset = get_ffmpeg_params()

    # Output video path - use interpolation method in filename
    # Wan → flux_wan, RIFE → flux_rife, FILM → flux_film
    method_suffix = interp_method.lower()  # "wan", "rife", "film"
    output_filename = f"{data.args.root.timestring}_flux_{method_suffix}.mp4"
    output_path = os.path.join(data.output_directory, output_filename)

    # Collect ALL frame files (keyframes + tweens) sorted numerically
    all_frames = sorted(
        glob.glob(os.path.join(data.output_directory, "[0-9]" * 9 + ".png")),
        key=lambda x: int(os.path.basename(x).split('.')[0])
    )

    total_frames = len(all_frames)
    logger.info(f"{emoji_if_enabled('🎬')} Stitching {total_frames} total frames into video...")

    # Create concat file list for ffmpeg
    concat_file = os.path.join(data.output_directory, f"_{data.args.root.timestring}_concat.txt")
    with open(concat_file, 'w') as f:
        for frame_path in all_frames:
            # Escape single quotes for ffmpeg concat demuxer
            escaped_path = frame_path.replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
            # Duration for each frame (1/fps seconds)
            f.write(f"duration {1.0 / video_args.fps}\n")
        # Last frame needs to be repeated without duration for proper video ending
        if all_frames:
            escaped_path = all_frames[-1].replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")

    try:
        # Build ffmpeg command using concat demuxer
        cmd = [
            ffmpeg_location,
            '-y',  # Overwrite output
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', str(ffmpeg_crf),
            '-preset', ffmpeg_preset,
            output_path
        ]

        logger.info(f"   Running ffmpeg concat...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"FFmpeg failed: {stderr}")
            raise RuntimeError(f"FFmpeg failed with return code {process.returncode}")

        logger.info(f"{emoji_if_enabled('✅')} Video stitched successfully")

        # Add audio if specified (use pre-downloaded path from video_args)
        logger.debug(f"Soundtrack: add={video_args.add_soundtrack}, path={video_args.soundtrack_path}")
        if video_args.add_soundtrack == 'File' and video_args.soundtrack_path:
            logger.info(f"{emoji_if_enabled('🎵')} Adding audio track...")
            temp_output = output_path + '.temp.mp4'

            audio_cmd = [
                ffmpeg_location,
                '-y',
                '-i', output_path,
                '-i', video_args.soundtrack_path,  # Already downloaded by render orchestrator
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', 'copy',
                '-shortest',
                temp_output
            ]

            audio_process = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            audio_stdout, audio_stderr = audio_process.communicate()

            if audio_process.returncode != 0:
                logger.warning(f"Failed to add audio: {audio_stderr}")
            else:
                os.replace(temp_output, output_path)
                logger.info(f"{emoji_if_enabled('✅')} Audio added successfully")

    finally:
        # Cleanup concat file
        if os.path.exists(concat_file):
            os.remove(concat_file)

    return output_path


def generate_film_segment(first_image, last_image, num_frames, height, width,
                         first_frame_idx, output_dir, fps=30):
    """
    Generate frames for one FILM (Frame Interpolation for Large Motion) segment.

    Uses Google's ML-based frame interpolation to generate smooth transitions
    between two keyframes.

    Args:
        first_image: PIL Image of first keyframe
        last_image: PIL Image of last keyframe
        num_frames: Number of tween frames to generate (excluding keyframes)
        height: Frame height
        width: Frame width
        first_frame_idx: Global index of first keyframe
        output_dir: Directory to save generated frames
        fps: Target FPS for interpolation

    Returns:
        List of paths to generated tween frames
    """
    import shutil
    import math
    from deforum.integrations.external_repos.film_interpolation.film_inference import run_film_interp_infer
    from deforum.media.interpolation.frame_interpolation import check_and_download_film_model

    logger.info(f"   {emoji_if_enabled('🎬')} FILM interpolation: {num_frames} frames")

    # Create working directory for FILM in the output directory (not /tmp)
    # This avoids cleanup issues and keeps intermediate files with the project
    film_work_dir = os.path.join(output_dir, f"_film_segment_{first_frame_idx}_to_{first_frame_idx + num_frames + 1}")
    temp_input = os.path.join(film_work_dir, "input")
    temp_output = os.path.join(film_work_dir, "output")
    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)

    try:
        # Save first and last keyframes to temp input directory
        # FILM expects numbered frames: 0.png, 1.png
        first_image.save(os.path.join(temp_input, "0000000.png"))
        last_image.save(os.path.join(temp_input, "0000001.png"))

        logger.debug(f"   Generating {num_frames} intermediate frames")
        logger.debug(f"   Temp input: {temp_input}")

        # Ensure FILM model is downloaded
        film_model_folder = os.path.join(os.getcwd(), "models", "Deforum")
        film_model_path = os.path.join(film_model_folder, "film_net_fp16.pt")

        logger.debug(f"   Checking FILM model: {film_model_path}")
        check_and_download_film_model('film_net_fp16.pt', film_model_folder)

        # FILM's inter_frames parameter = number of frames to ADD between input frames
        # For 10 tween frames needed, pass inter_frames=10 (not recursion depth)
        run_film_interp_infer(
            model_path=film_model_path,
            input_folder=temp_input,
            save_folder=temp_output,
            inter_frames=num_frames  # Number of intermediate frames to generate
        )

        # Find FILM output frames
        film_frames = sorted([f for f in os.listdir(temp_output) if f.endswith('.png')])

        # Skip first and last frame (those are the keyframes)
        tween_frames = film_frames[1:-1] if len(film_frames) > 2 else film_frames

        # Move tween frames to output directory with correct naming
        frame_paths = []
        for local_idx, film_filename in enumerate(tween_frames[:num_frames]):
            # Calculate global frame index (start after first keyframe)
            global_frame_idx = first_frame_idx + 1 + local_idx
            target_filename = f"{global_frame_idx:09d}.png"
            target_path = os.path.join(output_dir, target_filename)

            # Copy frame from FILM output to final output directory
            film_frame_path = os.path.join(temp_output, film_filename)
            shutil.copy2(film_frame_path, target_path)
            frame_paths.append(target_path)

        logger.info(f"   {emoji_if_enabled('✅')} FILM generated {len(frame_paths)} tween frames")
        return frame_paths

    finally:
        # Cleanup FILM working directory
        if os.path.exists(film_work_dir):
            shutil.rmtree(film_work_dir, ignore_errors=True)
