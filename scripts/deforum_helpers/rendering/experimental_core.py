import os
from pathlib import Path
from typing import List

# noinspection PyUnresolvedReferences
from modules.shared import cmd_opts, progress_print_out, state
from tqdm import tqdm

from . import img_2_img_tubes
from .data.frame import KeyFrameDistribution, KeyFrame
from .data.render_data import RenderData
from .util import filename_utils, image_utils, log_utils, opt_utils, memory_utils, web_ui_utils


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args,
                     freeu_args, kohya_hrfix_args, root):
    log_utils.info("Using experimental render core.")
    data = RenderData.create(args, parseq_args, anim_args, video_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, root)
    _check_experimental_render_conditions(data)
    web_ui_utils.init_job(data)
    key_frames = KeyFrame.create_all_frames(data, KeyFrameDistribution.from_UI_tab(data))
    run_render_animation(data, key_frames)
    data.animation_mode.unload_raft_and_depth_model()


def run_render_animation(data: RenderData, key_frames: List[KeyFrame]):
    for key_frame in key_frames:
        if is_resume(data, key_frame):
            continue
        pre_process_key_frame_and_emit_tweens(data, key_frame)
        image = key_frame.generate()
        if image is None:
            log_utils.print_warning_generate_returned_no_image()
            break
        post_process_key_frame(key_frame, image)


def pre_process_key_frame_and_emit_tweens(data, key_frame):
    memory_utils.handle_med_or_low_vram_before_step(data)
    web_ui_utils.update_job(data)
    if key_frame.has_tween_frames():
        emit_tweens(data, key_frame)
    log_utils.print_animation_frame_info(key_frame.i, data.args.anim_args.max_frames)
    key_frame.maybe_write_frame_subtitle()
    frame_tube = img_2_img_tubes.frame_transformation_tube
    contrasted_noise_tube = img_2_img_tubes.contrasted_noise_transformation_tube
    key_frame.prepare_generation(frame_tube, contrasted_noise_tube)


def post_process_key_frame(key_frame, image):
    if not image_utils.is_PIL(image):  # check is required when resuming from timestring
        image = img_2_img_tubes.conditional_frame_transformation_tube(key_frame)(image)
    state.assign_current_image(image)
    key_frame.after_diffusion(image)
    web_ui_utils.update_status_tracker(key_frame.render_data)


def is_resume(data, key_step):
    filename = filename_utils.frame_filename(data, key_step.i)
    full_path = Path(data.output_directory) / filename
    is_file_existing = os.path.exists(full_path)
    if is_file_existing:
        log_utils.warn(f"Frame {filename} exists, skipping to next key frame.")
        key_step.render_data.args.args.seed = key_step.next_seed()
    return is_file_existing


def emit_tweens(data, key_step):
    _update_pseudo_cadence(data, len(key_step.tweens) - 1)
    log_utils.print_tween_frame_from_to_info(key_step)
    grayscale_tube = img_2_img_tubes.conditional_force_tween_to_grayscale_tube
    overlay_mask_tube = img_2_img_tubes.conditional_add_overlay_mask_tube
    tweens = _tweens_with_progress(key_step)
    [tween.emit_frame(key_step, grayscale_tube, overlay_mask_tube) for tween in tweens]


def _check_experimental_render_conditions(data):
    if data.has_parseq_keyframe_redistribution():
        msg = "Experimental conditions: Using 'Parseq keyframe redistribution' together with '{method}'. {results}. \
               In case of problems, consider deactivating either one."
        dark_or_dist = "Resulting images may quickly end up looking dark or distorted."
        if data.has_optical_flow_cadence():
            log_utils.warn(msg.format(method="optical flow cadence", results=dark_or_dist))
        if data.has_optical_flow_redo():
            log_utils.warn(msg.format(method="optical flow generation", results=dark_or_dist))
        if data.is_hybrid_available():
            log_utils.warn(msg.format(method="hybrid video", results="Render process may not run stable."))


def _update_pseudo_cadence(data, value):
    data.turbo.cadence = value
    data.parseq_adapter.cadence = value
    data.parseq_adapter.a1111_cadence = value
    data.args.anim_args.diffusion_cadence = value
    data.args.anim_args.cadence_flow_factor_schedule = f"0: ({value})"


def _tweens_with_progress(key_step):
    # only use tween progress bar when extra console output (aka "dev mode") is disabled.
    if not opt_utils.is_verbose():
        log_utils.clear_previous_line()
        return tqdm(key_step.tweens, desc="Tweens progress", file=progress_print_out,
                    disable=cmd_opts.disable_console_progressbars, colour='#FFA468')
    return key_step.tweens
