"""Render Integration for Zero-HITL

Bridges Zero-HITL parameters to Deforum's render pipeline.

ISOLATION: Only affects Zero-HITL tab. Normal Deforum remains unchanged.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

from deforum.utils.zero_hitl.parameter_randomizer import SlopcoreParameters
from deforum.utils.system.logging import get_logger

logger = get_logger()


def build_args_from_slopcore(
    params: SlopcoreParameters,
    prompts: List[Dict[str, Any]],
    audio_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """Build Deforum args from slopcore parameters.

    Args:
        params: Slopcore parameters
        prompts: Generated prompts with frame numbers
        audio_path: Path to generated audio
        output_dir: Output directory

    Returns:
        Dict with all required Deforum args
    """
    logger.info("=" * 80)
    logger.info("🔍 ENTRY: build_args_from_slopcore() CALLED")
    logger.info(f"🔍 output_dir parameter: {output_dir}")
    logger.info("=" * 80)

    # Convert prompts to Deforum format (JSON string and dict)
    animation_prompts_dict = {
        str(prompt['frame']): prompt['prompt']
        for prompt in prompts
    }
    animation_prompts_json = json.dumps(animation_prompts_dict)

    # Calculate total frames
    # For zero-HITL we use the exact duration * fps
    # (In normal Deforum, max_frames is user-controlled)
    duration_seconds = len(prompts) / (params.fps / 3)  # Approx based on ~3 prompts/sec
    max_frames = int(params.fps * duration_seconds)

    # Map render mode to animation_mode
    animation_mode_map = {
        "Classic 3D": "3D",
        "New 3D": "3D",
        "Keyframes Only": "3D"
    }
    animation_mode = animation_mode_map.get(params.render_mode, "3D")

    # Build args dictionaries
    # These match the structure expected by Deforum's render_animation()

    args_dict = {
        # Output directory (CRITICAL - required by save_settings_txt)
        'outdir': output_dir,

        # Prompts (required by save_settings_txt)
        'prompts': animation_prompts_dict,
        'positive_prompts': '',  # Zero-HITL uses simple prompts
        'negative_prompts': '',  # Zero-HITL uses simple prompts

        # Basic generation settings
        'W': params.resolution[0],
        'H': params.resolution[1],
        'steps': params.steps,
        'sampler': params.sampler,
        'scheduler': 'Automatic',  # Let Forge select appropriate scheduler
        'cfg_scale': params.cfg_scale,
        'seed': params.seed,
        'seed_behavior': 'schedule' if params.seed == 0 else 'fixed',

        # Model settings (use current loaded model)
        'seed_iter_N': 1,
        'use_init': False,
        'strength': 0.0,  # Not used in animation mode
        'strength_0_no_init': True,
        'init_image': None,
        'tiling': False,
        'restore_faces': False,

        # Masking (disabled for zero-HITL)
        'use_mask': False,
        'use_alpha_as_mask': False,
        'mask_file': '',
        'invert_mask': False,
        'mask_contrast_adjust': 1.0,
        'mask_brightness_adjust': 1.0,

        # Hybrid video removed
        'video_init_path': '',
        'extract_nth_frame': 1,
        'overwrite_extracted_frames': False,
        'use_mask_video': False,
        'video_mask_path': '',

        # Interpolation settings
        'interpolate_key_frames': False,
        'interpolate_x_frames': 4,

        # Resume settings
        'resume_from_timestring': False,
        'resume_timestring': '',

        # Other
        'prompts_use_csv': False,
        'prompt_csv_dir': '',
        'batch_name': f"slop_{int(time.time())}",
        'filename_format': '{timestring}_{index}_{seed}.png',
        'seed_resize_from_w': 0,
        'seed_resize_from_h': 0,
    }

    anim_args_dict = {
        # Render mode (primary workflow selector)
        'render_mode': 'New 3D',

        # Animation mode (legacy, kept for compatibility)
        'animation_mode': animation_mode,
        'max_frames': max_frames,
        'border': 'replicate',

        # Keyframe distribution (New 3D uses REDISTRIBUTED)
        'keyframe_distribution': 'REDISTRIBUTED',

        # Cadence (New 3D default)
        'diffusion_cadence': 5,  # New 3D default cadence for redistributed keyframes

        # Angle/Zoom/Translation (2D)
        'angle': '0:(0)',
        'zoom': '0:(1.0)',
        'translation_x': '0:(0)',
        'translation_y': '0:(0)',
        'transform_center_x': '0:(0.5)',
        'transform_center_y': '0:(0.5)',

        # 3D settings
        'translation_z': '0:(0)',
        'rotation_3d_x': '0:(0)',
        'rotation_3d_y': '0:(0)',
        'rotation_3d_z': '0:(0)',

        # Perspective flip
        'enable_perspective_flip': params.enable_perspective_flip,
        'perspective_flip_theta': '0:(0)',
        'perspective_flip_phi': '0:(0)',
        'perspective_flip_gamma': '0:(0)',
        'perspective_flip_fv': '0:(53)',

        # Generation strength schedules
        'strength_schedule': f'0:({params.strength})',
        'keyframe_strength_schedule': f'0:({params.strength})',
        'cfg_scale_schedule': f'0:({params.cfg_scale})',
        'distilled_cfg_scale_schedule': '0:(0)',

        # Prompts
        'animation_prompts': animation_prompts_json,

        # Noise settings
        'noise_type': params.noise_type,
        'noise_schedule': '0:(0.02)',
        'enable_noise_multiplier_scheduling': False,
        'noise_multiplier_schedule': '0:(1.0)',
        'perlin_octaves': 4,
        'perlin_persistence': 0.5,
        'perlin_w': 8,
        'perlin_h': 8,

        # Coherence
        'color_coherence': params.color_coherence,
        'color_force_grayscale': False,
        'legacy_colormatch': False,
        'color_coherence_image_path': '',
        'color_coherence_video_every_N_frames': 1,
        'diffusion_redo': 0,
        'contrast_schedule': '0:(1.0)',

        # Depth settings
        'use_depth_warping': True,
        'depth_algorithm': params.depth_model,
        'midas_weight': params.midas_weight,
        'fov': 40,
        'fov_schedule': '0:(40)',
        'near_schedule': '0:(200)',
        'far_schedule': '0:(10000)',
        'padding_mode': 'border',
        'sampling_mode': 'bicubic',
        'save_depth_maps': False,

        # Anti-blur
        'amount_schedule': '0:(0.1)',
        'kernel_schedule': '0:(5)',
        'sigma_schedule': '0:(1.0)',
        'threshold_schedule': '0:(0.0)',

        # Optical flow (RAFT) - disabled for zero-HITL
        'use_optical_flow': False,
        'optical_flow_cadence': 0,
        'optical_flow_redo_generation': 'None',
        'redo_flow_factor_schedule': '0:(1.0)',  # Minimal valid schedule
        'cadence_flow_factor_schedule': '0:(1.0)',  # Minimal valid schedule
        'raft_model_size': 'large',
        'raft_flow_iterations': 12,
        'show_flow_arrows': False,

        # Seed scheduling - DISABLED (minimal schedules to satisfy parser)
        'enable_subseed_scheduling': False,
        'subseed_schedule': '0:(1)',  # Minimal valid schedule
        'subseed_strength_schedule': '0:(0)',  # Minimal valid schedule

        # Steps scheduling - DISABLED
        'enable_steps_scheduling': False,
        'steps_schedule': f'0:({params.steps})',  # Use actual steps value

        # Sampler scheduling - DISABLED
        'enable_sampler_scheduling': False,
        'sampler_schedule': f'0:({params.sampler})',  # Use actual sampler

        # Scheduler scheduling - DISABLED
        'enable_scheduler_scheduling': False,
        'scheduler_schedule': '0:(Automatic)',  # Valid scheduler name

        # ETA scheduling - DISABLED
        'enable_ddim_eta_scheduling': False,
        'ddim_eta_schedule': '0:(0)',  # Minimal valid schedule
        'enable_ancestral_eta_scheduling': False,
        'ancestral_eta_schedule': '0:(1)',  # Minimal valid schedule

        # Additional scheduling fields
        'aspect_ratio_schedule': '0:(1)',  # Keep for compatibility
        'aspect_ratio_use_old_formula': False,
        'noise_mask_schedule': '0:(None)',  # None for no mask
        'use_noise_mask': False,
        'mask_schedule': '0:(None)',  # None for no mask
        'keyframe_type_schedule': '0:(Keyframe)',  # Valid keyframe type

        # Checkpoint scheduling - DISABLED
        'enable_checkpoint_scheduling': False,
        'checkpoint_schedule': '0:(model1.ckpt)',  # Valid checkpoint name

        # CLIP skip - DISABLED
        'enable_clipskip_scheduling': False,
        'clipskip_schedule': '0:(1)',  # Minimal valid schedule

        # Seed schedule (keep for seed behavior)
        'seed_schedule': f'0:({params.seed})',

        # Flux ControlNet (disabled for zero-HITL)
        'enable_flux_controlnet': False,
        'flux_controlnet_type': 'canny',
        'flux_controlnet_model': 'instantx',
        'flux_controlnet_strength': 0.5,
        'flux_controlnet_canny_low': 100,
        'flux_controlnet_canny_high': 200,
        'flux_base_model': 'flux1-dev-bnb-nf4-v2',
        'flux_guidance_scale': 3.5,

        # WAN FLF2V (disabled for zero-HITL, using normal 3D warping)
        'enable_wan_flf2v': False,
        'wan_flf2v_chunk_size': 13,
    }

    video_args_dict = {
        'fps': params.fps,
        'add_soundtrack': 'File',
        'soundtrack_path': audio_path,
        'skip_video_creation': False,
        'delete_imgs': False,
        'delete_input_frames': False,
        'image_path': output_dir,
        'mp4_path': output_dir,
        'store_frames_in_ram': False,
        'ffmpeg_crf': 40,  # Maximum compression artifacts (per Qwen's spec)
        'ffmpeg_preset': 'slow',
        'render_steps': False,
        'path_name_modifier': 'x0_pred',
        'make_gif': False,
        'gif_fps': params.fps,
        'make_mp4_transparent': False,
        'add_frame_number_to_video': False,
        'soundtrack_volume': 0.8,
    }

    parseq_args_dict = {
        'parseq_manifest': '',
        'parseq_use_deltas': True,
        'parseq_non_schedule_overrides': True,
    }

    loop_args_dict = {
        'use_looper': False,
        'init_images': '',
        'image_strength_schedule': '0:(0.75)',
        'image_keyframe_strength_schedule': '0:(0.75)',
        'blendFactorMax': '0:(0.35)',
        'blendFactorSlope': '0:(0.25)',
        'tweening_frames_schedule': '0:(20)',
        'color_correction_factor': '0:(0.075)',
    }

    controlnet_args_dict = {
        # ControlNet disabled for zero-HITL
    }

    root_dict = {
        'timestring': f"slop_{int(time.time())}",
        'raw_batch_name': f"slop_{int(time.time())}",
    }

    logger.info(f"🔍 DEBUG: root_dict created with keys: {list(root_dict.keys())}")
    logger.info(f"🔍 DEBUG: args_dict['outdir'] = {args_dict.get('outdir', 'MISSING!')}")

    return {
        'args': args_dict,
        'anim_args': anim_args_dict,
        'video_args': video_args_dict,
        'parseq_args': parseq_args_dict,
        'loop_args': loop_args_dict,
        'controlnet_args': controlnet_args_dict,
        'root': root_dict,
    }


def execute_render(
    params: SlopcoreParameters,
    prompts: List[Dict[str, Any]],
    audio_path: str,
    output_dir: str
) -> str:
    """Execute Deforum render with zero-HITL parameters.

    Args:
        params: Slopcore parameters
        prompts: Generated prompts
        audio_path: Path to audio file
        output_dir: Output directory

    Returns:
        Path to generated video

    Raises:
        Exception: If render fails
    """
    logger.info("=" * 80)
    logger.info("🔍 ENTRY: execute_render() CALLED")
    logger.info(f"🔍 output_dir parameter: {output_dir}")
    logger.info("=" * 80)
    logger.info("🎬 Executing Deforum render with zero-HITL parameters...")

    try:
        # Build args from slopcore parameters
        all_args = build_args_from_slopcore(params, prompts, audio_path, output_dir)

        # Convert dicts to SimpleNamespace (Deforum expects this)
        from types import SimpleNamespace
        args = SimpleNamespace(**all_args['args'])
        anim_args = SimpleNamespace(**all_args['anim_args'])
        video_args = SimpleNamespace(**all_args['video_args'])
        parseq_args = SimpleNamespace(**all_args['parseq_args'])
        loop_args = SimpleNamespace(**all_args['loop_args'])
        controlnet_args = all_args['controlnet_args']
        root = SimpleNamespace(**all_args['root'])  # Convert root to SimpleNamespace too

        logger.info(f"🔍 DEBUG: args namespace attributes: {dir(args)}")
        logger.info(f"🔍 DEBUG: args has outdir: {hasattr(args, 'outdir')}")
        if hasattr(args, 'outdir'):
            logger.info(f"🔍 DEBUG: args.outdir value: {args.outdir}")

        # Call Deforum render
        logger.info("🎬 About to call render_animation()...")
        from deforum.orchestration.render import render_animation

        try:
            render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root)
            logger.info("✓ render_animation() completed successfully")
        except Exception as render_error:
            logger.error(f"❌ render_animation() failed with exception: {render_error}")
            logger.error(f"   Exception type: {type(render_error).__name__}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            raise

        # Find generated video (Deforum creates it based on timestring)
        video_pattern = f"{root.timestring}*.mp4"
        output_path = Path(output_dir)
        videos = list(output_path.glob(video_pattern))

        if videos:
            video_path = str(videos[0])
            logger.info(f"✓ Video generated: {video_path}")
            return video_path
        else:
            # Fallback: find most recent mp4
            videos = list(output_path.glob("*.mp4"))
            if videos:
                videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                video_path = str(videos[0])
                logger.warning(f"⚠️ Video found (fallback): {video_path}")
                return video_path
            else:
                raise FileNotFoundError("No video file found after render")

    except Exception as e:
        logger.error(f"❌ Render execution failed: {e}")
        raise
