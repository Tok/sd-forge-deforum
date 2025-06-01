import os

from pathlib import Path
from typing import List

# noinspection PyUnresolvedReferences
from modules import shared  # provided by Forge

from . import img_2_img_tubes
from .data.frame import KeyFrameDistribution, DiffusionFrame
from .data.render_data import RenderData
from .data.taqaddumat import Taqaddumat
from .util import filename_utils, image_utils, log_utils, memory_utils, subtitle_utils, web_ui_utils

IS_USE_PROFILER = False


class NoImageGenerated(Exception):
    pass


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args,
                     freeu_args, kohya_hrfix_args, root):
    log_utils.info("Using experimental render core.", log_utils.RED)
    
    # Pre-download soundtrack if specified
    if video_args.add_soundtrack == 'File' and video_args.soundtrack_path is not None:
        if video_args.soundtrack_path.startswith(('http://', 'https://')):
            print(f"Pre-downloading soundtrack at the beginning of the render process: {video_args.soundtrack_path}")
            try:
                from ..video_audio_utilities import download_audio
                video_args.soundtrack_path = download_audio(video_args.soundtrack_path)
                print(f"Audio successfully pre-downloaded to: {video_args.soundtrack_path}")
            except Exception as e:
                print(f"Error pre-downloading audio: {e}")
    
    data = RenderData.create(args, parseq_args, anim_args, video_args, loop_args, controlnet_args, freeu_args,
                             kohya_hrfix_args, root)
    check_render_conditions(data)
    web_ui_utils.init_job(data)
    diffusion_frames = DiffusionFrame.create_all_frames(data, KeyFrameDistribution.from_UI_tab(data))
    subtitle_utils.create_all_subtitles_if_active(data, diffusion_frames)
    shared.total_tqdm = Taqaddumat()
    shared.total_tqdm.reset(data, diffusion_frames)
    run_render_animation(data, diffusion_frames)
    data.animation_mode.unload_raft_and_depth_model()


def run_render_animation(data: RenderData, frames: List[DiffusionFrame]):
    for frame in frames:
        is_resume, full_path = is_resume_with_image(data, frame)
        if is_resume:
            shared.total_tqdm.total_animation_cycles.update()
            shared.total_tqdm.total_frames.update(len(frame.tweens))
            shared.total_tqdm.total_steps.update(frame.actual_steps(data))
            existing_image = image_utils.load_image(full_path)
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
    emit_tweens(data, frame)
    pre_process(data, frame)
    image = frame.generate(data, shared.total_tqdm)
    if image is None:
        raise NoImageGenerated()
    post_process(data, frame, image)


def prepare_generation(data: RenderData, frame: DiffusionFrame):
    memory_utils.handle_med_or_low_vram_before_step(data)
    web_ui_utils.update_job(data, frame.i)
    shared.total_tqdm.reset_tween_count(len(frame.tweens))
    log_utils.print_animation_frame_info(frame.i, data.args.anim_args.max_frames)


def emit_tweens(data: RenderData, frame: DiffusionFrame):
    if frame.has_tween_frames():
        update_pseudo_cadence(data, len(frame.tweens) - 1)
        log_utils.print_tween_frame_from_to_info(frame)
        [tween.emit_frame(data, shared.total_tqdm, frame) for tween in frame.tweens]


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
            log_utils.warn(msg.format(method="optical flow cadence", results=dark_or_dist))
        if data.has_optical_flow_redo():
            log_utils.warn(msg.format(method="optical flow generation", results=dark_or_dist))
        if data.is_hybrid_available():
            log_utils.warn(msg.format(method="hybrid video", results="Render process may not run stable."))


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
