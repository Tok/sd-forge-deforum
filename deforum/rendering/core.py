import os

from pathlib import Path
from typing import List

import numpy as np

# noinspection PyUnresolvedReferences
try:
    from modules import shared  # type: ignore  # provided by Forge
except ImportError:
    # Allow imports in test environments without Forge
    shared = None  # type: ignore

from . import img_2_img_tubes
from .data.frame import KeyFrameDistribution, DiffusionFrame
from .data.render_data import RenderData
from .data.taqaddumat import Taqaddumat
from deforum.rendering.helpers import filename as filename_utils
from deforum.utils.image import processing as image_utils
from deforum.media.load_images import load_img
from deforum.utils.system.logging import log as log_utils
from deforum.rendering.helpers import memory as memory_utils
from deforum.rendering.helpers import subtitle as subtitle_utils
from deforum.rendering.helpers import webui as web_ui_utils
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


IS_USE_PROFILER = False


class NoImageGenerated(Exception):
    pass


def _strip_negative_prompt(prompt_text):
    """Remove --neg ... portion from Deforum prompts to avoid Wan interpreting them as positive.

    Wan doesn't understand Deforum's --neg syntax and will interpret negative prompts as positive,
    potentially generating unwanted content (e.g., "nsfw" in negative becomes positive).
    """
    if '--neg' in prompt_text:
        return prompt_text.split('--neg')[0].strip()
    return prompt_text.strip()


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    # Pre-download soundtrack if specified
    if video_args.add_soundtrack == 'File' and video_args.soundtrack_path is not None:
        if video_args.soundtrack_path.startswith(('http://', 'https://')):
            logger.info(f"Pre-downloading soundtrack at the beginning of the render process: {video_args.soundtrack_path}")
            try:
                from deforum.media.video_audio_utilities import download_audio
                video_args.soundtrack_path = download_audio(video_args.soundtrack_path)
                logger.info(f"Audio successfully pre-downloaded to: {video_args.soundtrack_path}")
            except Exception as e:
                logger.error(f"Pre-downloading audio failed: {e}")
    
    data = RenderData.create(args, parseq_args, anim_args, video_args, loop_args, controlnet_args, root)
    check_render_conditions(data)
    web_ui_utils.init_job(data)
    diffusion_frames = DiffusionFrame.create_all_frames(data, KeyFrameDistribution.from_UI_tab(data))
    subtitle_utils.create_all_subtitles_if_active(data, diffusion_frames)
    shared.total_tqdm = Taqaddumat()
    shared.total_tqdm.reset(data, diffusion_frames)
    run_render_animation(data, diffusion_frames)
    data.animation_mode.unload_raft_and_depth_model()


def run_render_animation(data: RenderData, frames: List[DiffusionFrame]):
    # Reverse generation: process frames in reverse order (last→first), then reassemble correctly
    # This allows stable forward-motion by generating zoom-out (model fills naturally), then reversing
    if data.args.anim_args.reverse_generation:
        logger.info("Reverse Generation enabled: Processing frames in reverse order (last→first)")
        logger.info("Frames will be reassembled in correct order for final video")

        # Reverse the frame list
        generation_order_frames = list(reversed(frames))

        # Reassign tweens: save all tweens first, then reassign to next frame in generation order
        # (which is the PREVIOUS frame in timeline order)
        # This way tweens are emitted AFTER their source keyframe is generated
        saved_tweens = [frame.tweens for frame in generation_order_frames]

        # Clear all tweens first
        for frame in generation_order_frames:
            frame.tweens = []

        # Reassign: each frame's original tweens go to the next frame in generation order
        for i in range(len(generation_order_frames) - 1):
            generation_order_frames[i + 1].tweens = saved_tweens[i]

        # First frame in generation order (last in timeline) has no tweens
        generation_order_frames[0].tweens = []
    else:
        generation_order_frames = frames

    for frame in generation_order_frames:
        is_resume, full_path = is_resume_with_image(data, frame)
        if is_resume:
            shared.total_tqdm.total_animation_cycles.update()
            shared.total_tqdm.total_frames.update(len(frame.tweens))
            shared.total_tqdm.total_steps.update(frame.actual_steps(data))
            # Load existing image from disk (for resume functionality)
            existing_image, _ = load_img(full_path, None, shape=None, use_alpha_as_mask=False)
            data.images.before_previous = data.images.previous
            data.images.previous = existing_image
            continue

        profiler = maybe_start_profiler()
        try:
            process_frame(data, frame)
        except NoImageGenerated:
            log_utils.print_warning_generate_returned_no_image()
            break  # Exit the loop if no image was generated
        finally:
            maybe_end_profiler_and_print_results(profiler)


def process_frame(data, frame):
    prepare_generation(data, frame)

    # Skip traditional tweens if using Wan FLF2V
    use_wan_flf2v = should_use_wan_flf2v(data, frame)
    if not use_wan_flf2v:
        emit_tweens(data, frame)

    pre_process(data, frame)
    image = frame.generate(data, shared.total_tqdm)
    if image is None:
        raise NoImageGenerated()

    # Emit Wan FLF2V tweens AFTER keyframe generation
    if use_wan_flf2v:
        emit_wan_flf2v_tweens(data, frame, image)

    post_process(data, frame, image)


def prepare_generation(data: RenderData, frame: DiffusionFrame):
    memory_utils.handle_med_or_low_vram_before_step(data)
    web_ui_utils.update_job(data, frame.i)
    shared.total_tqdm.reset_tween_count(len(frame.tweens))
    log_utils.print_animation_frame_info(frame.i, data.args.anim_args.max_frames, frame.is_keyframe)


def emit_tweens(data: RenderData, frame: DiffusionFrame):
    if frame.has_tween_frames():
        update_pseudo_cadence(data, len(frame.tweens) - 1)
        log_utils.print_tween_frame_from_to_info(frame)
        [tween.emit_frame(data, shared.total_tqdm, frame) for tween in frame.tweens]


def should_use_wan_flf2v(data: RenderData, frame: DiffusionFrame) -> bool:
    """Check if this frame should use Wan FLF2V for tweens instead of traditional depth warping.

    Respects keyframe_type from keyframe_type_schedule:
    - "tween": Always use depth tweening
    - "flf2v": Always use FLF2V (if conditions met)
    - "auto": Auto-decide based on tween count and chunk_size
    """
    # Basic conditions required for FLF2V
    has_tweens = frame.has_tween_frames()
    has_previous = data.images.has_previous()
    is_not_first_frame = frame.i != 0

    if not (has_tweens and has_previous and is_not_first_frame):
        return False

    # Check keyframe_type from schedule
    keyframe_type = frame.keyframe_type.lower() if hasattr(frame, 'keyframe_type') else "tween"

    # Explicit tween: never use FLF2V
    if keyframe_type == "tween":
        return False

    # Explicit flf2v: use FLF2V if global enable is on
    if keyframe_type == "flf2v":
        return hasattr(data.args.anim_args, 'enable_wan_flf2v') and data.args.anim_args.enable_wan_flf2v

    # Auto mode: decide based on tween count and chunk size
    if keyframe_type == "auto":
        if not (hasattr(data.args.anim_args, 'enable_wan_flf2v') and data.args.anim_args.enable_wan_flf2v):
            return False

        # Auto-decide: use FLF2V for short sections (< 80% of chunk_size)
        num_tweens = len(frame.tweens)
        chunk_size = getattr(data.args.anim_args, 'wan_flf2v_chunk_size', 81)
        threshold = int(chunk_size * 0.8)  # 80% of chunk_size

        return num_tweens <= threshold

    # Default to tween for unknown types
    return False


def emit_wan_flf2v_tweens(data: RenderData, frame: DiffusionFrame, current_keyframe_image):
    """Generate tween frames using Wan FLF2V interpolation between previous and current keyframes.

    For short sections (< chunk_size): Direct FLF2V interpolation
    For long sections (>= chunk_size): Chaining mode with intermediate depth-tween keyframes
    """
    if not frame.has_tween_frames():
        return

    num_tweens = len(frame.tweens)
    total_frames = num_tweens + 1  # tweens + current keyframe

    # Get chunk size from settings
    chunk_size = 81  # Default
    if hasattr(data.args.anim_args, 'wan_flf2v_chunk_size'):
        chunk_size = data.args.anim_args.wan_flf2v_chunk_size

    # Determine if we need chaining
    use_chaining = total_frames > chunk_size

    if use_chaining:
        log_utils.info(f"🎬 Using Wan FLF2V CHAINING mode for {num_tweens} tween frames", log_utils.BLUE)
        log_utils.info(f"   Chunk size: {chunk_size} frames, will chain multiple FLF2V calls", log_utils.BLUE)
    else:
        log_utils.info(f"🎬 Using Wan FLF2V DIRECT mode for {num_tweens} tween frames", log_utils.BLUE)

    try:
        # Import Wan integration
        from ...wan.wan_simple_integration import WanSimpleIntegration  # type: ignore

        # Get previous and current keyframe images
        prev_keyframe = data.images.previous
        curr_keyframe = current_keyframe_image

        # Convert OpenCV images (BGR numpy array) to PIL Images (RGB)
        import cv2  # type: ignore
        from PIL import Image
        if isinstance(prev_keyframe, np.ndarray):
            prev_keyframe_pil = Image.fromarray(cv2.cvtColor(prev_keyframe, cv2.COLOR_BGR2RGB))
        else:
            prev_keyframe_pil = prev_keyframe

        if isinstance(curr_keyframe, np.ndarray):
            curr_keyframe_pil = Image.fromarray(cv2.cvtColor(curr_keyframe, cv2.COLOR_BGR2RGB))
        else:
            curr_keyframe_pil = curr_keyframe

        # Initialize Wan
        wan = WanSimpleIntegration()

        # Auto-discover and load Wan FLF2V model
        models = wan.discover_models()
        flf2v_model = None
        ti2v_models = []
        for model in models:
            if model['type'] == 'FLF2V':
                flf2v_model = model
                break
            elif model['type'] in ('TI2V', 'T2V', 'I2V'):
                ti2v_models.append(model['name'])

        if not flf2v_model:
            log_utils.error("❌ No Wan FLF2V model found!", log_utils.RED)
            if ti2v_models:
                log_utils.warning(f"   Found T2V/I2V/TI2V models: {', '.join(ti2v_models)}", log_utils.YELLOW)
                log_utils.warning("   ⚠️  TI2V/T2V/I2V models CANNOT do FLF2V interpolation!", log_utils.YELLOW)
                log_utils.info("   They will extend the first frame instead of interpolating to the last frame.", log_utils.YELLOW)
            log_utils.info("   Download FLF2V model: huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers --local-dir models/Deforum/wan/Wan2.1-FLF2V-14B", log_utils.YELLOW)
            log_utils.info("   Falling back to traditional depth-based tweens", log_utils.YELLOW)
            # Fallback to traditional tweens
            update_pseudo_cadence(data, len(frame.tweens) - 1)
            [tween.emit_frame(data, shared.total_tqdm, frame) for tween in frame.tweens]
            return

        # Load Wan FLF2V pipeline
        wan_args = None  # Get from data if available
        if hasattr(data.args, 'wan_args'):
            wan_args = data.args.wan_args

        if not wan.load_simple_wan_pipeline(flf2v_model, wan_args):
            log_utils.error("❌ Failed to load Wan FLF2V pipeline", log_utils.RED)
            # Fallback
            update_pseudo_cadence(data, len(frame.tweens) - 1)
            [tween.emit_frame(data, shared.total_tqdm, frame) for tween in frame.tweens]
            return

        # Get dimensions from data
        width = data.args.args.W
        height = data.args.args.H

        # Branch: Direct vs Chaining mode
        if use_chaining:
            # CHAINING MODE: Generate intermediate depth-tween keyframes, then FLF2V between them
            generated_frames = _emit_wan_flf2v_chaining(
                data, frame, wan, prev_keyframe_pil, curr_keyframe_pil,
                width, height, chunk_size, num_tweens
            )
        else:
            # DIRECT MODE: Single FLF2V call
            generated_frames = _emit_wan_flf2v_direct(
                data, frame, wan, prev_keyframe_pil, curr_keyframe_pil,
                width, height, total_frames
            )

        if generated_frames is None:
            # Fallback to traditional tweens
            update_pseudo_cadence(data, len(frame.tweens) - 1)
            [tween.emit_frame(data, shared.total_tqdm, frame) for tween in frame.tweens]
            return

        # Save tween frames
        for idx, tween in enumerate(frame.tweens):
            # Get corresponding frame from generated_frames
            # (skip first frame if it's the prev keyframe)
            flf2v_frame = generated_frames[idx]

            # Convert PIL to OpenCV format (BGR numpy array)
            if isinstance(flf2v_frame, Image.Image):
                tween_image = cv2.cvtColor(np.array(flf2v_frame), cv2.COLOR_RGB2BGR)
            else:
                tween_image = flf2v_frame

            # Save the tween frame
            saved_image = image_utils.save_and_return_frame(data, tween, tween_image)
            shared.total_tqdm.increment_tween_count()

            # Update reference images
            data.images.before_previous = data.images.previous
            data.images.previous = saved_image
            data.args.root.init_sample = saved_image

        mode_str = "CHAINING" if use_chaining else "DIRECT"
        log_utils.success(f"✅ Wan FLF2V ({mode_str}) generated {len(frame.tweens)} tween frames", log_utils.GREEN)

    except Exception as e:
        log_utils.error(f"❌ Wan FLF2V failed: {e}", log_utils.RED)
        import traceback
        traceback.print_exc()
        log_utils.info("   Falling back to traditional depth-based tweens", log_utils.YELLOW)
        # Fallback to traditional tweens
        update_pseudo_cadence(data, len(frame.tweens) - 1)
        [tween.emit_frame(data, shared.total_tqdm, frame) for tween in frame.tweens]


def _emit_wan_flf2v_direct(data, frame, wan, first_frame_pil, last_frame_pil, width, height, num_frames):
    """Direct FLF2V: Single call to interpolate between two keyframes."""
    prompt_raw = data.args.root.animation_prompts.get(str(frame.i), "")
    prompt = _strip_negative_prompt(prompt_raw)

    log_utils.info(f"   Generating {num_frames} frames with single FLF2V call...", log_utils.BLUE)

    try:
        result = wan.pipeline.generate_flf2v(
            first_frame=first_frame_pil,
            last_frame=last_frame_pil,
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=20,
            guidance_scale=7.5
        )

        # Extract frames from result
        if hasattr(result, 'frames'):
            all_frames = result.frames[0]  # First (and only) video
        elif isinstance(result, list):
            all_frames = result
        else:
            log_utils.error("❌ Unexpected FLF2V result format", log_utils.RED)
            return None

        # Return frames excluding first and last (they're the keyframes)
        return all_frames[1:-1]

    except Exception as e:
        log_utils.error(f"❌ Direct FLF2V failed: {e}", log_utils.RED)
        import traceback
        traceback.print_exc()
        return None


def _emit_wan_flf2v_chaining(data, frame, wan, first_frame_pil, last_frame_pil, width, height, chunk_size, num_tweens):
    """Chaining FLF2V: Generate depth-tween intermediate keyframes, then FLF2V between each pair.

    Example: 200 tweens with chunk_size=81
    - Create intermediate keyframes at: 0, 80, 160, 200
    - FLF2V: 0→80 (81 frames), 80→160 (81 frames), 160→200 (41 frames)
    - Concatenate: frames 1-79, frames 81-159, frames 161-199 = 200 tweens
    """
    log_utils.info(f"   Generating intermediate depth-tween keyframes for chaining...", log_utils.BLUE)

    # Calculate chunk positions
    chunk_positions = []  # Frame indices where we need intermediate keyframes
    remaining = num_tweens
    pos = 0

    while remaining > 0:
        chunk_positions.append(pos)
        if remaining > chunk_size - 1:
            pos += chunk_size - 1  # -1 because chunk includes both endpoints
            remaining -= (chunk_size - 1)
        else:
            pos += remaining
            remaining = 0

    chunk_positions.append(num_tweens)  # Final position (current keyframe)

    log_utils.info(f"   Chunk positions: {chunk_positions} (total: {len(chunk_positions)-1} chunks)", log_utils.BLUE)

    # Generate intermediate keyframes using depth warping
    intermediate_keyframes = [first_frame_pil]  # Start with first keyframe

    for i in range(1, len(chunk_positions) - 1):
        chunk_pos = chunk_positions[i]
        tween_at_pos = frame.tweens[chunk_pos - 1]  # -1 because tweens are 0-indexed

        # Generate this tween frame using traditional depth warping
        log_utils.info(f"   Generating intermediate keyframe at position {chunk_pos}...", log_utils.BLUE)

        # Use traditional tween generation for this intermediate keyframe
        tween_image = tween_at_pos._generate(data, frame, data.images.previous)

        # Convert to PIL
        import cv2  # type: ignore
        from PIL import Image
        if isinstance(tween_image, np.ndarray):
            intermediate_pil = Image.fromarray(cv2.cvtColor(tween_image, cv2.COLOR_BGR2RGB))
        else:
            intermediate_pil = tween_image

        intermediate_keyframes.append(intermediate_pil)

    intermediate_keyframes.append(last_frame_pil)  # End with last keyframe

    log_utils.info(f"   Generated {len(intermediate_keyframes)} intermediate keyframes", log_utils.BLUE)

    # Now FLF2V between each pair of intermediate keyframes
    all_generated_frames = []

    for i in range(len(intermediate_keyframes) - 1):
        chunk_start = chunk_positions[i]
        chunk_end = chunk_positions[i + 1]
        chunk_num_frames = chunk_end - chunk_start + 1  # +1 to include both endpoints

        prompt_raw = data.args.root.animation_prompts.get(str(frame.i), "")
        prompt = _strip_negative_prompt(prompt_raw)

        log_utils.info(f"   FLF2V chunk {i+1}/{len(intermediate_keyframes)-1}: frames {chunk_start}→{chunk_end} ({chunk_num_frames} frames)", log_utils.BLUE)

        try:
            result = wan.pipeline.generate_flf2v(
                first_frame=intermediate_keyframes[i],
                last_frame=intermediate_keyframes[i + 1],
                prompt=prompt,
                height=height,
                width=width,
                num_frames=chunk_num_frames,
                num_inference_steps=20,
                guidance_scale=7.5
            )

            # Extract frames
            if hasattr(result, 'frames'):
                chunk_frames_result = result.frames[0]
            elif isinstance(result, list):
                chunk_frames_result = result
            else:
                log_utils.error(f"❌ Unexpected FLF2V result format for chunk {i+1}", log_utils.RED)
                return None

            # Append frames (excluding first for subsequent chunks to avoid duplicates)
            if i == 0:
                # First chunk: skip first and last frame (keyframes)
                all_generated_frames.extend(chunk_frames_result[1:-1])
            else:
                # Subsequent chunks: skip first and last frame
                all_generated_frames.extend(chunk_frames_result[1:-1])

        except Exception as e:
            log_utils.error(f"❌ FLF2V chunk {i+1} failed: {e}", log_utils.RED)
            import traceback
            traceback.print_exc()
            return None

    log_utils.info(f"   Chaining complete: {len(all_generated_frames)} frames generated", log_utils.BLUE)
    return all_generated_frames


def pre_process(data: RenderData, frame: DiffusionFrame):
    shared.total_tqdm.reset_step_count(frame.actual_steps(data))
    frame_tube = img_2_img_tubes.frame_transformation_tube
    contrasted_noise_tube = img_2_img_tubes.contrasted_noise_transformation_tube
    frame.prepare_generation(data, frame_tube, contrasted_noise_tube)


def post_process(data: RenderData, frame: DiffusionFrame, image):
    df = frame
    if not image_utils.is_PIL(image):  # check is required when resuming from timestring
        image = img_2_img_tubes.conditional_frame_transformation_tube(data, df)(image)
    shared.state.assign_current_image(image)
    df.after_diffusion(data, image)
    web_ui_utils.update_status_tracker(data, frame.i)


def check_render_conditions(data: RenderData):
    log_utils.info(f"Sampler: '{data.args.args.sampler}' Scheduler: '{data.args.args.scheduler}'")
    if data.has_keyframe_distribution():
        msg = "Experimental conditions: Using 'keyframe distribution' together with '{method}'. {results}. \
               In case of problems, consider deactivating either one."
        dark_or_dist = "Resulting images may quickly end up looking dark or distorted."
        if data.has_optical_flow_cadence():
            log_utils.warning(msg.format(method="optical flow cadence", results=dark_or_dist))
        if data.has_optical_flow_redo():
            log_utils.warning(msg.format(method="optical flow generation", results=dark_or_dist))


def update_pseudo_cadence(data: RenderData, value: int):
    data.parseq_adapter.cadence = value
    data.parseq_adapter.a1111_cadence = value
    data.args.anim_args.diffusion_cadence = value
    data.args.anim_args.cadence_flow_factor_schedule = f"0: ({value})"


def is_resume_with_image(data: RenderData, frame: DiffusionFrame):
    last_index = frame.i  # same as diffusion_frame.tweens[-1].i
    filename = filename_utils.frame_filename(data, last_index)
    full_path = Path(data.output_directory) / filename
    is_file_existing = os.path.exists(full_path)
    if is_file_existing:
        log_utils.info(f"Frame {filename} exists, skipping to next keyframe.", log_utils.ORANGE)
    return is_file_existing, full_path


def maybe_start_profiler():
    if not IS_USE_PROFILER:
        return None
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler


def maybe_end_profiler_and_print_results(profiler, limit=20):
    if not IS_USE_PROFILER:
        return
    import pstats
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('time').print_stats(limit)
