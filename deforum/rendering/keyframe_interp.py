"""
Flux + Interpolation Mode: Flux Keyframes + Choice of Interpolation

Supports two interpolation methods:
- Wan FLF2V (default): AI-generated video interpolation with semantic understanding
- FILM: Google's Frame Interpolation for Large Motion (handles dramatic changes)

Note: RIFE is NOT available here (defaults to single frames on dramatic changes).
      RIFE is available for post-processing smooth videos only.

Architecture:
  Phase 1: Generate ALL keyframes with diffusion models (Flux/Z-Image/Lumina/SD)
  Phase 2: Batch interpolation between each consecutive keyframe pair (Wan/FILM/DA3-3DGS)
  Phase 3: Stitch final video

This combines the best of both worlds:
- High-quality diffusion-generated keyframes
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
    Keyframes + Interpolation rendering mode: Diffusion for keyframes + choice of interpolation

    1. Generate all keyframes with diffusion models (Flux/Z-Image/Lumina/SD)
    2. Interpolate between keyframes with Wan FLF2V, FILM, or DA3-3DGS
    3. Stitch final video

    Interpolation method is selected via wan_args.flux_interpolation_method
    """
    logger.info(f"{emoji_if_enabled('🎬')} Keyframes + Interpolation Mode: Diffusion Keyframes + ML Interpolation")

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

    # Initialize specialized dashboard for Flux+Interpolation mode
    from deforum.rendering import options as opt_utils
    dashboard = None
    if opt_utils.is_dashboard_enabled():
        from deforum.utils.ui.interpolation_dashboard import InterpolationDashboard
        dashboard = InterpolationDashboard()
        # Initialize progress totals
        # Phase 1: Keyframes to generate
        dashboard.phase1_total = len(keyframes)
        # Phase 2: Interpolation segments (keyframes - 1)
        dashboard.phase2_total = len(keyframes) - 1

        # Set up signal handler for clean Ctrl+C (same as core.py)
        import signal
        import threading
        import time

        if threading.current_thread() is threading.main_thread():
            original_sigint = signal.getsignal(signal.SIGINT)

            sigint_state = {
                'count': 0,
                'first_time': 0,
                'confirmation_window': 3.0
            }

            def sigint_handler(sig, frame_obj):
                current_time = time.time()
                sigint_state['count'] += 1

                if sigint_state['count'] >= 3:
                    print("\n\n⚠️  FORCE QUIT - Exiting immediately without cleanup")
                    signal.signal(signal.SIGINT, original_sigint)
                    if dashboard:
                        try:
                            dashboard._is_active = False
                        except:
                            pass
                    raise KeyboardInterrupt

                if sigint_state['count'] == 1:
                    sigint_state['first_time'] = current_time
                    print("\n\n⚠️  Interrupt detected. Press Ctrl+C again within 3 seconds to confirm exit")
                    print("   (Press Ctrl+C a 3rd time anytime to force quit)")
                    return

                if sigint_state['count'] == 2:
                    time_since_first = current_time - sigint_state['first_time']

                    if time_since_first <= sigint_state['confirmation_window']:
                        print("\n\n✓ Exit confirmed. Cleaning up...")
                        if dashboard:
                            try:
                                dashboard.stop()
                            except:
                                pass
                        signal.signal(signal.SIGINT, original_sigint)
                        if callable(original_sigint):
                            original_sigint(sig, frame_obj)
                        else:
                            raise KeyboardInterrupt
                    else:
                        sigint_state['count'] = 1
                        sigint_state['first_time'] = current_time
                        print("\n\n⚠️  Interrupt detected. Press Ctrl+C again within 3 seconds to confirm exit")
                        print("   (Press Ctrl+C a 3rd time anytime to force quit)")
                        return

            signal.signal(signal.SIGINT, sigint_handler)

        dashboard.start()

        # Store dashboard on data so Taqaddumat can access it
        data.dashboard = dashboard

    logger.info(f"{emoji_if_enabled('📊')} Flux/Wan Workflow:")
    logger.info(f"   Total frames: {anim_args.max_frames}")
    logger.info(f"   Keyframes to generate: {len(keyframes)}")
    logger.info(f"   FLF2V segments: {len(keyframes) - 1}")

    # Resume info (compact)
    logger.debug(f"Resume: {anim_args.resume_from_timestring}, outdir: {args.outdir}, exists: {os.path.exists(data.output_directory)}")
    if os.path.exists(data.output_directory):
        img_count = len([f for f in os.listdir(data.output_directory) if f.endswith(('.png', '.jpg', '.jpeg'))])
        logger.debug(f"Existing images in output dir: {img_count}")

    # Check interpolation method early (needed for Phase 1 keyframe saving logic)
    # Try new parameter name first, fallback to old for backward compatibility
    interp_method = getattr(wan_args, 'diffusion_interpolation_method',
                           getattr(wan_args, 'flux_flf2v_interpolation_method', 'Wan'))
    use_da3_3dgs = (interp_method == "DA3-3DGS")

    # ====================
    # PHASE 1: Batch Generate All Keyframes (Flux/Z-Image/Lumina/SD)
    # ====================
    logger.separator(char="=")
    logger.info("PHASE 1: Batch Keyframe Generation")
    logger.separator(char="=")

    # Check for resume mode - scan for existing keyframes
    keyframe_images = {}  # {frame_index: image_path}
    is_resuming = anim_args.resume_from_timestring
    
    if is_resuming:
        logger.info(f"{emoji_if_enabled('🔄')} Resume mode: Scanning for existing keyframes...")
        for frame in keyframes:
            # Check simple format first (matches our save format: 000000001.png)
            simple_filename = f"{frame.i:09d}.png"

            # For DA3-3DGS mode, check _diffusion/ subdirectory first
            if use_da3_3dgs:
                diffusion_dir = os.path.join(data.output_directory, "_diffusion")
                diffusion_path = os.path.join(diffusion_dir, simple_filename)
                if os.path.exists(diffusion_path):
                    keyframe_images[frame.i] = diffusion_path
                    logger.info(f"   {emoji_if_enabled('✓')} Found existing keyframe in _diffusion/: {simple_filename}")
                    continue

            # Check root directory
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
        # Update dashboard for Phase 1
        if dashboard:
            dashboard.update_phase1(idx, len(keyframes))
            dashboard.set_operation(f"Generating keyframe {idx + 1}/{len(keyframes)} (frame {frame.i})")
            dashboard.update_vram_from_torch()

        # Skip if keyframe already exists (resume mode)
        if frame.i in keyframe_images:
            if dashboard:
                dashboard.update_phase1(idx + 1, len(keyframes))
                dashboard.set_operation(f"Skipped existing keyframe {idx + 1}/{len(keyframes)}")
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

        # Generate keyframe image using diffusion model
        web_ui_utils.update_job(data, frame.i)
        image = frame.generate(data, shared.total_tqdm)

        if image is None:
            raise RuntimeError(f"Failed to generate keyframe at frame {frame.i}")

        # Save keyframe (to _diffusion/ if using DA3-3DGS for easy Phase 2 retries)
        keyframe_path = save_keyframe(data, frame, image, use_diffusion_subdir=use_da3_3dgs)
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
        logger.info(f"   (All {newly_generated} keyframes newly generated)")

    # CRITICAL: Aggressive VRAM cleanup between Phase 1 and Phase 2
    # This allows switching from Flux/Z-Image/Lumina to DA3-GIANT without VRAM conflicts
    logger.info(f"\n{emoji_if_enabled('🧹')} Cleaning up VRAM before Phase 2...")
    import gc
    import torch
    from modules import devices

    # Unload diffusion models completely
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
    # PHASE 2: Batch Frame Interpolation (Wan/FILM/DA3-Multiview/DA3-3DGS)
    # ====================
    logger.separator(char="=")
    logger.info("PHASE 2: Batch Frame Interpolation")
    logger.separator(char="=")
    logger.info(f"{emoji_if_enabled('📊')} Interpolation method: {interp_method}")

    # Unload diffusion models to free GPU memory
    logger.info(f"{emoji_if_enabled('🗑')}️  Unloading diffusion models to free GPU memory...")
    from backend import memory_management
    memory_management.unload_all_models()
    memory_management.soft_empty_cache()
    logger.info(f"{emoji_if_enabled('✅')} GPU memory freed")

    # Initialize depth model if needed for DA3-Multiview
    if interp_method == "DA3-Multiview":
        logger.info(f"{emoji_if_enabled('🔍')} Initializing DA3 depth model for multi-view geometry...")

        # Get model size from user parameter and build full model name
        model_size = getattr(wan_args, 'da3_multiview_model_size', 'Small')
        depth_algorithm = f'Depth-Anything-V3-AnyView-{model_size}'
        logger.info(f"{emoji_if_enabled('🎯')} DA3-Multiview will use {depth_algorithm}")

        # Initialize depth model
        from deforum.depth.depth import DepthModel
        from modules import devices

        models_path = os.path.join(os.getcwd(), 'models', 'Deforum')
        device = devices.get_optimal_device()

        # DepthModel uses __new__ with positional args (models_path, device) + kwargs
        data.depth_model = DepthModel(
            models_path,  # Positional arg 0
            device,       # Positional arg 1
            keep_in_vram=True,  # Keep loaded for all segments
            depth_algorithm=depth_algorithm
        )

        logger.info(f"{emoji_if_enabled('✅')} Depth model loaded: {depth_algorithm}")

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

    # Check scene strategy
    scene_strategy = getattr(wan_args, 'da3_3dgs_scene_strategy', 'per_segment')

    # For per-prompt mode with DA3-3DGS, group segments by prompt first
    if interp_method == "DA3-3DGS" and scene_strategy == "per_prompt":
        logger.info(f"{emoji_if_enabled('📦')} Using PER-PROMPT scene strategy: Grouping segments by prompt")

        # Group segments by prompt
        def group_segments_by_prompt(keyframes, prompt_series):
            """Group consecutive segments that share the same prompt."""
            prompt_groups = []
            current_group = []
            current_prompt = None

            for idx in range(len(keyframes) - 1):
                first_kf = keyframes[idx]
                last_kf = keyframes[idx + 1]

                # Get prompt for this segment (use first keyframe's prompt)
                segment_prompt = prompt_series[first_kf.i]

                # Start new group if prompt changed
                if segment_prompt != current_prompt:
                    if current_group:
                        prompt_groups.append((current_prompt, current_group))
                    current_group = [(first_kf.i, last_kf.i)]
                    current_prompt = segment_prompt
                else:
                    current_group.append((first_kf.i, last_kf.i))

            # Add final group
            if current_group:
                prompt_groups.append((current_prompt, current_group))

            return prompt_groups

        prompt_groups = group_segments_by_prompt(keyframes, data.prompt_series)

        logger.info(f"   Found {len(prompt_groups)} distinct prompt regions")
        for i, (prompt, segments) in enumerate(prompt_groups):
            keyframe_span = f"{segments[0][0]}-{segments[-1][1]}"
            logger.info(f"   Group {i+1}: {len(segments)} segments ({keyframe_span}), prompt: {prompt[:60]}...")

        # Now generate interpolations for each prompt group using shared 3DGS scene
        all_segment_frames = []

        # Import DA3-3DGS modules once
        from deforum.rendering.da3_3dgs_novel_view import generate_da3_3dgs_interpolation
        from deforum.rendering.da3_3dgs_quality import parse_densification_factor, log_vram_usage_estimate
        from PIL import Image
        import torch

        # Load all keyframe images into memory
        all_keyframes_pil = {}
        for kf_idx, kf_path in keyframe_images.items():
            all_keyframes_pil[kf_idx] = Image.open(kf_path)

        # Parse quality settings once
        densification_input = getattr(wan_args, 'da3_3dgs_densification_factor', 'Auto (Max Quality for VRAM)')
        densification_factor = parse_densification_factor(densification_input)
        near_clip_distance = getattr(wan_args, 'da3_3dgs_near_clip_distance', 0.0)
        resolution = (data.width(), data.height())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_selection = getattr(wan_args, 'da3_3dgs_model', 'DA3-GIANT')
        max_prompt_keyframes = getattr(wan_args, 'da3_3dgs_max_prompt_keyframes', 50)

        # Process each prompt group
        for group_idx, (prompt, segments) in enumerate(prompt_groups):
            logger.info(f"\n{emoji_if_enabled('🌌')} Prompt Group {group_idx + 1}/{len(prompt_groups)}")
            logger.info(f"   Prompt: {prompt[:80]}...")
            logger.info(f"   Segments: {len(segments)} ({segments[0][0]}-{segments[-1][1]})")

            if dashboard:
                dashboard.set_operation(f"Prompt group {group_idx + 1}/{len(prompt_groups)}")

            # Collect all keyframes for this prompt group
            group_keyframe_indices = set()
            for first_idx, last_idx in segments:
                group_keyframe_indices.add(first_idx)
                group_keyframe_indices.add(last_idx)
            group_keyframe_indices = sorted(group_keyframe_indices)

            logger.info(f"   Keyframes in group: {len(group_keyframe_indices)} ({min(group_keyframe_indices)}-{max(group_keyframe_indices)})")

            # Check if we need to split this group (too many keyframes for VRAM)
            if len(group_keyframe_indices) > max_prompt_keyframes:
                logger.warning(f"   ⚠️  Group has {len(group_keyframe_indices)} keyframes, exceeds max {max_prompt_keyframes}")
                logger.warning(f"   Splitting into sub-groups of {max_prompt_keyframes} keyframes each")
                # TODO: Implement sub-group splitting if needed
                # For now, just use the first max_prompt_keyframes
                group_keyframe_indices = group_keyframe_indices[:max_prompt_keyframes]
                logger.warning(f"   Using first {len(group_keyframe_indices)} keyframes (TODO: implement proper splitting)")

            # Collect keyframes for this group as PIL images
            group_keyframe_pil = [all_keyframes_pil[idx] for idx in group_keyframe_indices]

            # Collect ALL target frame indices for all segments in this group
            all_group_targets = []
            for first_idx, last_idx in segments:
                segment_targets = list(range(first_idx + 1, last_idx))
                all_group_targets.extend(segment_targets)

            logger.info(f"   Generating {len(all_group_targets)} tween frames from ONE shared 3DGS scene")

            # Build ONE 3DGS scene for this entire prompt group
            # Use existing generate_da3_3dgs_interpolation - it handles everything correctly
            group_frames = generate_da3_3dgs_interpolation(
                keyframe_images=group_keyframe_pil,
                keyframe_indices=group_keyframe_indices,
                target_frame_indices=all_group_targets,
                model_selection=model_selection,
                output_dir=data.output_directory,
                device=device,
                render_keyframes=getattr(wan_args, 'da3_3dgs_render_keyframes', True),
                segment_first_idx=segments[0][0],  # First segment's first frame
                segment_last_idx=segments[-1][1],  # Last segment's last frame
                densification_factor=densification_factor,
                near_clip_distance=near_clip_distance,
                dashboard=dashboard,
                deform_keys=data.animation_keys.deform_keys if getattr(wan_args, 'da3_3dgs_use_deforum_motion', False) else None
            )

            all_segment_frames.extend(group_frames)
            logger.info(f"{emoji_if_enabled('✅')} Prompt group {group_idx + 1} complete: {len(group_frames)} frames")

        logger.info(f"\n{emoji_if_enabled('✅')} Phase 2 Complete: {len(all_segment_frames)} total frames from PER-PROMPT DA3-3DGS")

    else:
        # Original per-segment mode (default)
        # Generate FLF2V segments
        all_segment_frames = []

        for idx in range(len(keyframes) - 1):
            # Update dashboard operation (3DGS sub-operations will update their own progress)
            if dashboard:
                dashboard.set_operation(f"Interpolating segment {idx + 1}/{len(keyframes) - 1}")

            first_kf = keyframes[idx]
            last_kf = keyframes[idx + 1]

            first_frame_idx = first_kf.i
            last_frame_idx = last_kf.i
            num_tween_frames = last_frame_idx - first_frame_idx - 1  # ONLY in-between frames (exclude both keyframes)

            logger.info(f"\n{emoji_if_enabled('🎞')}️ Interpolation Segment {idx + 1}/{len(keyframes) - 1}:")
            logger.info(f"   From keyframe: {first_frame_idx}")
            logger.info(f"   To keyframe: {last_frame_idx}")
            logger.info(f"   In-between frames to generate: {num_tween_frames} (frames {first_frame_idx+1} to {last_frame_idx-1})")

            # Check if all frames in this segment already exist (resume mode)
            # Skip only if NOT regenerating tweens
            if is_resuming and not anim_args.resume_regenerate_tweens:
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
            elif is_resuming and anim_args.resume_regenerate_tweens:
                logger.info(f"{emoji_if_enabled('🔄')} Regenerating tweens for segment {idx + 1} (resume_regenerate_tweens=True)")

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
        
            # Route to appropriate interpolation function
            if interp_method == "FILM":
                logger.info(f"   {emoji_if_enabled('🎯')} Interpolation: FILM (Frame Interpolation for Large Motion)")
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
            elif interp_method == "DA3-Multiview":
                logger.info(f"   {emoji_if_enabled('🎯')} Interpolation: DA3-Multiview (depth warping with multi-view geometry)")
                segment_frames = generate_da3_multiview_segment(
                    first_image=first_image,
                    last_image=last_image,
                    num_frames=num_tween_frames,
                    height=data.height(),
                    width=data.width(),
                    first_frame_idx=first_frame_idx,
                    output_dir=data.output_directory,
                    data=data
                )
            elif interp_method == "DA3-3DGS":
                model_selection = getattr(wan_args, 'da3_3dgs_model', 'DA3-GIANT')
                neighbor_segments = getattr(wan_args, 'da3_3dgs_neighbor_segments', 1)
                logger.info(f"   {emoji_if_enabled('🎯')} Interpolation: DA3-3DGS, model={model_selection}, neighbors={neighbor_segments}")

                # Use new proper 3DGS interpolation module
                from deforum.rendering.da3_3dgs_novel_view import (
                    collect_nearby_keyframes,
                    generate_da3_3dgs_interpolation
                )
                from PIL import Image
                import torch

                # Load all keyframe images into memory for collection
                all_keyframes_pil = {}
                for kf_idx, kf_path in keyframe_images.items():
                    all_keyframes_pil[kf_idx] = Image.open(kf_path)

                # Collect keyframes from current segment + neighbors
                collected_images, collected_indices = collect_nearby_keyframes(
                    all_keyframe_images=all_keyframes_pil,
                    segment_first_idx=first_frame_idx,
                    segment_last_idx=last_frame_idx,
                    num_neighbor_segments=neighbor_segments
                )

                # Generate target frame indices (tweens to create)
                target_indices = list(range(first_frame_idx + 1, last_frame_idx))

                # Note: Original diffusion keyframes are already in _diffusion/ subdirectory (saved during Phase 1)
                # This ensures they're available for Phase 2 retries without polluting the root directory

                # Generate interpolated frames using 3DGS
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Parse densification factor (handle "Auto" mode)
                from deforum.rendering.da3_3dgs_quality import (
                    parse_densification_factor, log_vram_usage_estimate
                )
                densification_input = getattr(wan_args, 'da3_3dgs_densification_factor', 'Auto (Max Quality for VRAM)')
                densification_factor = parse_densification_factor(densification_input)

                # Disable near-clip filtering (0.0) because DA3's camera poses can be inside the scene
                # This prevents filtering out 99%+ of splats when cameras are positioned incorrectly
                near_clip_distance = getattr(wan_args, 'da3_3dgs_near_clip_distance', 0.0)

                # Log VRAM usage estimate
                resolution = (data.width(), data.height())
                log_vram_usage_estimate(densification_factor, resolution)

                segment_frames = generate_da3_3dgs_interpolation(
                    keyframe_images=collected_images,
                    keyframe_indices=collected_indices,
                    target_frame_indices=target_indices,
                    model_selection=model_selection,
                    output_dir=data.output_directory,
                    device=device,
                    render_keyframes=getattr(wan_args, 'da3_3dgs_render_keyframes', True),
                    segment_first_idx=first_frame_idx,
                    segment_last_idx=last_frame_idx,
                    densification_factor=densification_factor,
                    near_clip_distance=near_clip_distance,
                    dashboard=dashboard,
                    deform_keys=data.animation_keys.deform_keys if getattr(wan_args, 'da3_3dgs_use_deforum_motion', False) else None
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
    output_video_path = stitch_keyframe_interpolation_video(
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

    # Stop dashboard when done
    if dashboard:
        dashboard.stop()


def save_keyframe(data: RenderData, frame: DiffusionFrame, image, use_diffusion_subdir=False):
    """Save keyframe image to disk with simple frame number naming (no timestring prefix)

    Args:
        data: RenderData object containing output directory
        frame: DiffusionFrame containing frame index
        image: Image to save (PIL or numpy array)
        use_diffusion_subdir: If True, save to _diffusion/ subdirectory (for DA3-3DGS mode)

    Returns:
        str: Full path to saved keyframe
    """
    # Use simple format: 000000001.png (no timestring prefix)
    filename = f"{frame.i:09d}.png"

    # For DA3-3DGS mode, save to _diffusion/ subdirectory
    # This keeps original keyframes available for Phase 2 retries without polluting root dir
    if use_diffusion_subdir:
        diffusion_dir = os.path.join(data.output_directory, "_diffusion")
        os.makedirs(diffusion_dir, exist_ok=True)
        filepath = os.path.join(diffusion_dir, filename)
    else:
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


def detect_model_prefix(checkpoint_name: str) -> str:
    """Detect diffusion model type from checkpoint name.

    Args:
        checkpoint_name: Name of the checkpoint file

    Returns:
        Model prefix string: 'flux', 'zit', 'lumina', or 'diffusion'
    """
    name_lower = checkpoint_name.lower()

    if "flux" in name_lower:
        return "flux"
    elif any(keyword in name_lower for keyword in ["z-image", "zimage", "zit"]):
        return "zit"
    elif "lumina" in name_lower:
        return "lumina"
    else:
        return "diffusion"


def build_output_filename(timestring: str, model_prefix: str, interp_method: str) -> str:
    """Build output video filename from components.

    Args:
        timestring: Timestamp string for the render
        model_prefix: Model type ('flux', 'zit', 'lumina', 'diffusion')
        interp_method: Interpolation method ('Wan', 'FILM', 'DA3-3DGS')

    Returns:
        Filename string (e.g., '20251207002039_zit_da3-3dgs.mp4')
    """
    method_suffix = interp_method.lower()
    return f"{timestring}_{model_prefix}_{method_suffix}.mp4"


def stitch_keyframe_interpolation_video(data, frame_paths, video_args, interp_method="Wan"):
    """Stitch all frames into final video using ffmpeg concat demuxer.

    Args:
        data: RenderData object containing output directory and settings
        frame_paths: List of frame file paths (may be unused, frames collected from disk)
        video_args: Video arguments containing fps, audio settings
        interp_method: Interpolation method name for filename

    Returns:
        str: Path to output video file
    """
    from deforum.media.video_audio_utilities import get_ffmpeg_params
    import glob
    import subprocess

    # Get ffmpeg parameters from settings
    ffmpeg_location, ffmpeg_crf, ffmpeg_preset = get_ffmpeg_params()

    # Build output path with model-aware filename
    checkpoint_name = getattr(data.args.args, 'checkpoint', '') or ""
    model_prefix = detect_model_prefix(checkpoint_name)
    output_filename = build_output_filename(
        data.args.root.timestring,
        model_prefix,
        interp_method
    )
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


def generate_da3_multiview_segment(first_image, last_image, num_frames, height, width,
                                   first_frame_idx, output_dir, data):
    """
    Generate frames for one DA3-Multiview segment.

    Uses Depth Anything V3's multi-view geometry (depth + camera pose estimation)
    to create smooth tweens between keyframes via depth warping.

    Args:
        first_image: PIL Image of first keyframe
        last_image: PIL Image of last keyframe
        num_frames: Number of tween frames to generate (excluding keyframes)
        height: Frame height
        width: Frame width
        first_frame_idx: Global index of first keyframe
        output_dir: Directory to save generated frames
        data: RenderData object (for depth model and animation keys)

    Returns:
        List of paths to generated tween frames
    """
    import numpy as np
    import torch
    from PIL import Image
    from deforum.animation.animation import transform_image_3d_switcher

    logger.info(f"   {emoji_if_enabled('🎯')} DA3-Multiview interpolation: {num_frames} frames")

    # Ensure depth model is initialized and is DA3-AnyView
    if data.depth_model is None:
        raise RuntimeError("DA3-Multiview requires depth model to be initialized")

    if not hasattr(data.depth_model, 'is_v3') or not data.depth_model.is_v3:
        raise RuntimeError("DA3-Multiview requires Depth Anything V3 (not V2)")

    # Check if model is AnyView variant
    da3_model = data.depth_model.depth_anything
    if not hasattr(da3_model, 'variant') or da3_model.variant != 'any-view':
        raise RuntimeError(
            f"DA3-Multiview requires AnyView variant, got '{getattr(da3_model, 'variant', 'unknown')}'. "
            "Select a Depth-Anything-V3-AnyView-* model in 3D Depth tab."
        )

    logger.debug(f"   Using DA3 AnyView model for multi-view geometry")

    # Convert PIL images to numpy BGR (OpenCV format)
    first_img_np = np.array(first_image)[:, :, ::-1]  # RGB -> BGR
    last_img_np = np.array(last_image)[:, :, ::-1]    # RGB -> BGR

    # Run DA3 multi-view inference to get depths + camera poses
    logger.debug(f"   Running DA3 multi-view inference...")
    result = da3_model.predict_multiview([first_img_np, last_img_np])

    # Extract multi-view data
    depth_maps = result.get('depth', None)
    camera_extrinsics = result.get('camera_extrinsics', None)

    if depth_maps is None or len(depth_maps) < 2:
        raise RuntimeError("DA3 multi-view inference failed to return depth maps")

    if camera_extrinsics is None or len(camera_extrinsics) < 2:
        logger.warning("DA3 did not return camera extrinsics, using identity transforms")
        # Fallback: identity matrices
        camera_extrinsics = [np.eye(4), np.eye(4)]

    depth_first = depth_maps[0]  # [1, 1, H, W] tensor
    depth_last = depth_maps[1]   # [1, 1, H, W] tensor
    pose_first = camera_extrinsics[0]  # [4, 4] matrix
    pose_last = camera_extrinsics[1]   # [4, 4] matrix

    logger.debug(f"   First depth: {depth_first.shape}, pose: {pose_first.shape if hasattr(pose_first, 'shape') else type(pose_first)}")
    logger.debug(f"   Last depth: {depth_last.shape}, pose: {pose_last.shape if hasattr(pose_last, 'shape') else type(pose_last)}")

    # Convert poses to numpy if they're tensors
    if hasattr(pose_first, 'cpu'):
        pose_first = pose_first.cpu().numpy()
    if hasattr(pose_last, 'cpu'):
        pose_last = pose_last.cpu().numpy()

    # Generate tween frames by interpolating camera pose and warping first keyframe
    frame_paths = []
    device = data.depth_model.device

    for tween_idx in range(num_frames):
        # Calculate interpolation factor (0 < t < 1)
        # tween_idx goes from 0 to num_frames-1
        # t should go from 1/(num_frames+1) to num_frames/(num_frames+1)
        t = (tween_idx + 1) / (num_frames + 1)

        logger.debug(f"   Generating tween {tween_idx + 1}/{num_frames} (t={t:.3f})")

        # Interpolate camera pose (simple linear for now)
        # TODO: Use proper SLERP for rotation component
        pose_interp = pose_first * (1.0 - t) + pose_last * t

        import torch.nn.functional as F
        img_h, img_w = first_img_np.shape[:2]

        # Bidirectional warping: warp from BOTH keyframes and blend
        # This reduces cumulative distortion vs always warping from first frame

        # Resize both depth maps to image size
        depth_first_resized = F.interpolate(
            depth_first,
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=False
        )
        depth_last_resized = F.interpolate(
            depth_last,
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=False
        )

        # Calculate relative transforms FROM first frame TO interpolated pose
        # We need the inverse transform: how to go from tween pose back to first frame
        pose_first_inv = np.linalg.inv(pose_first)
        pose_from_first = pose_first_inv @ pose_interp  # Transform from first to tween
        rot_from_first = torch.from_numpy(pose_from_first[:3, :3]).float().to(device)
        trans_from_first = torch.from_numpy(pose_from_first[:3, 3]).float().to(device)

        # Calculate relative transforms FROM last frame TO interpolated pose
        pose_last_inv = np.linalg.inv(pose_last)
        pose_from_last = pose_last_inv @ pose_interp  # Transform from last to tween
        rot_from_last = torch.from_numpy(pose_from_last[:3, :3]).float().to(device)
        trans_from_last = torch.from_numpy(pose_from_last[:3, 3]).float().to(device)

        # Warp first keyframe to tween position
        warped_from_first = transform_image_3d_switcher(
            device=device,
            prev_img_cv2=first_img_np,
            depth_tensor=depth_first_resized,
            rot_mat=rot_from_first,
            translate=trans_from_first,
            anim_args=data.args.anim_args,
            keys=data.animation_keys.deform_keys,
            frame_idx=first_frame_idx + tween_idx + 1
        )

        # Warp last keyframe to tween position
        warped_from_last = transform_image_3d_switcher(
            device=device,
            prev_img_cv2=last_img_np,
            depth_tensor=depth_last_resized,
            rot_mat=rot_from_last,
            translate=trans_from_last,
            anim_args=data.args.anim_args,
            keys=data.animation_keys.deform_keys,
            frame_idx=first_frame_idx + tween_idx + 1
        )

        # Blend the two warped frames based on t
        # When t=0 (near first), use mostly warped_from_first
        # When t=1 (near last), use mostly warped_from_last
        warped_img = (warped_from_first * (1.0 - t) + warped_from_last * t).astype(np.uint8)

        # Save tween frame
        global_frame_idx = first_frame_idx + tween_idx + 1
        filename = f"{global_frame_idx:09d}.png"
        filepath = os.path.join(output_dir, filename)

        # Convert BGR numpy to RGB PIL and save
        warped_img_rgb = warped_img[:, :, ::-1]  # BGR -> RGB
        Image.fromarray(warped_img_rgb).save(filepath)

        frame_paths.append(filepath)

    logger.info(f"   {emoji_if_enabled('✅')} DA3-Multiview generated {len(frame_paths)} tween frames")
    return frame_paths


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
