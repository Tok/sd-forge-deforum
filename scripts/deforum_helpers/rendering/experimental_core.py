import os
from pathlib import Path
from typing import List

# noinspection PyUnresolvedReferences
import modules.shared as shared
# noinspection PyUnresolvedReferences
from modules.shared import cmd_opts, progress_print_out, state
from tqdm import tqdm

from . import img_2_img_tubes
from .data.frame import KeyFrameDistribution, DiffusionFrame
from .data.render_data import RenderData
from .util import filename_utils, image_utils, log_utils, memory_utils, opt_utils, subtitle_utils, web_ui_utils


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args,
                     freeu_args, kohya_hrfix_args, root):
    log_utils.info("Using experimental render core.", log_utils.RED)
    data = RenderData.create(args, parseq_args, anim_args, video_args, loop_args, controlnet_args, freeu_args,
                             kohya_hrfix_args, root)
    check_render_conditions(data)
    web_ui_utils.init_job(data)
    diffusion_frames = DiffusionFrame.create_all_frames(data, KeyFrameDistribution.from_UI_tab(data))
    subtitle_utils.create_all_subtitles_if_active(data, diffusion_frames)
    shared.total_tqdm.updateTotal(sum(len(frame.tweens) for frame in diffusion_frames))
    run_render_animation(data, diffusion_frames)
    data.animation_mode.unload_raft_and_depth_model()


def run_render_animation(data: RenderData, diffusion_frames: List[DiffusionFrame]):
    for diffusion_frame in diffusion_frames:
        shared.total_tqdm.update()

        is_resume, full_path = is_resume_with_image(data, diffusion_frame)
        if is_resume:
            existing_image = image_utils.load_image(full_path)
            data.images.before_previous = data.images.previous
            data.images.previous = existing_image
            continue

        _pre_process_diffusion_frame_and_emit_tweens(data, diffusion_frame)
        image = diffusion_frame.generate(data)
        if image is None:
            log_utils.print_warning_generate_returned_no_image()
            break
        _post_process_diffusion_frame(data, diffusion_frame, image)


def _pre_process_diffusion_frame_and_emit_tweens(data, diffusion_frame):
    memory_utils.handle_med_or_low_vram_before_step(data)
    web_ui_utils.update_job(data, diffusion_frame.i)
    if diffusion_frame.has_tween_frames():
        emit_tweens(data, diffusion_frame)
    log_utils.print_animation_frame_info(diffusion_frame.i, data.args.anim_args.max_frames)
    frame_tube = img_2_img_tubes.frame_transformation_tube
    contrasted_noise_tube = img_2_img_tubes.contrasted_noise_transformation_tube
    diffusion_frame.prepare_generation(data, frame_tube, contrasted_noise_tube)


def _post_process_diffusion_frame(data: RenderData, diffusion_frame, image):
    df = diffusion_frame
    if not image_utils.is_PIL(image):  # check is required when resuming from timestring
        image = img_2_img_tubes.conditional_frame_transformation_tube(data, df)(image)
    state.assign_current_image(image)
    df.after_diffusion(data, image)
    web_ui_utils.update_status_tracker(data, diffusion_frame.i)


def emit_tweens(data, frame):
    _update_pseudo_cadence(data, len(frame.tweens) - 1)
    log_utils.print_tween_frame_from_to_info(frame)
    grayscale_tube = img_2_img_tubes.conditional_force_tween_to_grayscale_tube
    overlay_mask_tube = img_2_img_tubes.conditional_add_overlay_mask_tube
    tweens = _maybe_wrap_tweens_with_progress_bar(frame)
    [tween.emit_frame(data, frame, grayscale_tube, overlay_mask_tube) for tween in tweens]


def check_render_conditions(data):
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


def _update_pseudo_cadence(data, value):
    data.parseq_adapter.cadence = value
    data.parseq_adapter.a1111_cadence = value
    data.args.anim_args.diffusion_cadence = value
    data.args.anim_args.cadence_flow_factor_schedule = f"0: ({value})"


def _maybe_wrap_tweens_with_progress_bar(frame):
    # only use tween progress bar when extra console output (aka "dev mode") is disabled.
    if not opt_utils.is_verbose():
        log_utils.clear_previous_line()
        # TODO add a working total progress bar..
        wrapped = tqdm(frame.tweens, desc="Tweens progress", file=progress_print_out, position=0,
                       disable=cmd_opts.disable_console_progressbars, colour='#FFA468')
        shared.total_tqdm = wrapped
        return wrapped
    return frame.tweens


def is_resume_with_image(data, diffusion_frame):
    last_index = diffusion_frame.i  # same as diffusion_frame.tweens[-1].i
    filename = filename_utils.frame_filename(data, last_index)
    full_path = Path(data.output_directory) / filename
    is_file_existing = os.path.exists(full_path)
    if is_file_existing:
        log_utils.info(f"Frame {filename} exists, skipping to next keyframe.", log_utils.ORANGE)
    return is_file_existing, full_path
