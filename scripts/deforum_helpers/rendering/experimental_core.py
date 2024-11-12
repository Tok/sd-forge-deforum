import os
from pathlib import Path
from typing import List

# noinspection PyUnresolvedReferences
import modules.shared as shared

from . import img_2_img_tubes
from .data.combined_tqdm import CombinedTqdm
from .data.frame import KeyFrameDistribution, DiffusionFrame
from .data.render_data import RenderData
from .util import filename_utils, image_utils, log_utils, memory_utils, subtitle_utils, web_ui_utils


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args,
                     freeu_args, kohya_hrfix_args, root):
    log_utils.info("Using experimental render core.", log_utils.RED)
    data = RenderData.create(args, parseq_args, anim_args, video_args, loop_args, controlnet_args, freeu_args,
                             kohya_hrfix_args, root)
    check_render_conditions(data)
    web_ui_utils.init_job(data)
    diffusion_frames = DiffusionFrame.create_all_frames(data, KeyFrameDistribution.from_UI_tab(data))
    subtitle_utils.create_all_subtitles_if_active(data, diffusion_frames)
    shared.total_tqdm = CombinedTqdm()
    shared.total_tqdm.reset(diffusion_frames)
    run_render_animation(data, diffusion_frames)
    data.animation_mode.unload_raft_and_depth_model()


def run_render_animation(data: RenderData, frames: List[DiffusionFrame]):
    for frame in frames:
        is_resume, full_path = is_resume_with_image(data, frame)
        if is_resume:
            shared.total_tqdm.total_diffusions_tqdm.update()
            shared.total_tqdm.total_frames_tqdm.update(len(frame.tweens))
            shared.total_tqdm.total_steps_tqdm.update(frame.actual_steps())
            existing_image = image_utils.load_image(full_path)
            data.images.before_previous = data.images.previous
            data.images.previous = existing_image
            continue

        prepare_generation(data, frame)
        emit_tweens(data, frame)
        pre_process(data, frame)
        image = frame.generate(data)
        if image is None:
            log_utils.print_warning_generate_returned_no_image()
            break
        shared.total_tqdm.increment_diffusion_frame_count()
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
    shared.total_tqdm.reset_step_count(frame.actual_steps())
    frame_tube = img_2_img_tubes.frame_transformation_tube
    contrasted_noise_tube = img_2_img_tubes.contrasted_noise_transformation_tube
    frame.prepare_generation(data, frame_tube, contrasted_noise_tube)
    shared.total_tqdm.reset_step_count(frame.actual_steps())


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
