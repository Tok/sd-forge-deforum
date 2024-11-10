from cv2.typing import MatLike

from .call.anim import call_anim_frame_warp
from .call.hybrid import (call_get_flow_for_hybrid_motion_prev,
                          call_get_flow_for_hybrid_motion,
                          call_get_matrix_for_hybrid_motion,
                          call_get_matrix_for_hybrid_motion_prev)
from ...hybrid_video import (get_flow_from_images, image_transform_ransac,
                             image_transform_optical_flow, rel_flow_to_abs_flow)


def advance_optical_flow_cadence_before_animation_warping(data, last_frame, tween_frame, prev_image, image) -> MatLike:
    is_with_flow = data.is_3d_or_2d_with_optical_flow()
    if is_with_flow and _is_do_flow(data, tween_frame, last_frame.i, prev_image, image):
        method = data.args.anim_args.optical_flow_cadence  # string containing the flow method (e.g. "RAFT").
        flow = get_flow_from_images(prev_image, image, method, data.animation_mode.raft_model)
        tween_frame.cadence_flow = flow / len(last_frame.tweens)
        advanced_image = _advance_optical_flow(tween_frame, image)
        flow_factor = 1.0
        return image_transform_optical_flow(advanced_image, -tween_frame.cadence_flow, flow_factor)
    return image


def advance(data, i, image, depth):
    if depth is not None:
        warped_image, _ = call_anim_frame_warp(data, i, image, depth)
        return warped_image
    else:
        return image


def do_hybrid_video_motion(data, last_frame, tween_i, reference_images, image):
    """Warps the previous and/or the next to match the motion of the provided reference images."""
    motion = data.args.anim_args.hybrid_motion

    def _is_do_motion(motions):
        return tween_i > 0 and motion in motions

    ri = reference_images
    transformed = (_advance_hybrid_motion_ransac_transform(data, tween_i, ri, image)
                   if _is_do_motion(['Affine', 'Perspective']) else image)
    flown = (_advance_hybrid_motion_optical_tween_flow(data, tween_i, ri, last_frame, transformed)
             if _is_do_motion(['Optical Flow']) else transformed)
    return flown


def do_optical_flow_cadence_after_animation_warping(data, tween_frame, prev_image, image):
    if not data.animation_mode.is_raft_active():
        return image
    if tween_frame.cadence_flow is not None:
        new_flow, _ = call_anim_frame_warp(data, tween_frame.i, image, tween_frame.depth)
        tween_frame.cadence_flow = new_flow
        abs_flow = rel_flow_to_abs_flow(tween_frame.cadence_flow, data.width(), data.height())
        tween_frame.cadence_flow_inc = abs_flow * tween_frame.value
        image = _advance_cadence_flow(data, tween_frame, image)

    if prev_image is not None and tween_frame.value < 1.0:
        return prev_image * (1.0 - tween_frame.value) + image * tween_frame.value
    return image


def _advance_optical_flow(tween_step, image, flow_factor: int = 1):
    flow = tween_step.cadence_flow * -1
    return image_transform_optical_flow(image, flow, flow_factor)


def _advance_optical_tween_flow(last_frame, flow, image):
    flow_factor = last_frame.frame_data.flow_factor()
    return image_transform_optical_flow(image, flow, flow_factor)


def _advance_hybrid_motion_optical_tween_flow(data, tween_i, reference_images, last_frame, image):
    last_i = tween_i - 1
    flow = (call_get_flow_for_hybrid_motion(data, last_i)
            if not data.args.anim_args.hybrid_motion_use_prev_img
            else call_get_flow_for_hybrid_motion_prev(data, last_i, reference_images.previous))
    data.animation_mode.prev_flow = flow
    return _advance_optical_tween_flow(last_frame, flow, image)


def _advance_cadence_flow(data, tween_frame, image):
    ff_string = data.args.anim_args.cadence_flow_factor_schedule
    flow_factor = float(ff_string.split(": ")[1][1:-1])
    i = tween_frame.i
    flow = tween_frame.cadence_flow_inc
    return image_transform_optical_flow(image, flow, flow_factor)


def _advance_ransac_transform(data, matrix, image):
    motion = data.args.anim_args.hybrid_motion
    return image_transform_ransac(image, matrix, motion)  # TODO provide depth prediction


def _advance_hybrid_motion_ransac_transform(data, tween_i, reference_images, image):
    last_i = tween_i - 1
    matrix = (call_get_matrix_for_hybrid_motion(data, last_i)
              if not data.args.anim_args.hybrid_motion_use_prev_img
              else call_get_matrix_for_hybrid_motion_prev(data, last_i, reference_images.previous))
    return _advance_ransac_transform(data, matrix, image)


def _is_do_flow(data, tween_frame, start_i, prev_image, image):
    has_tween_schedule = data.animation_keys.deform_keys.strength_schedule_series[start_i] > 0
    has_images = prev_image is not None and image is not None
    has_step_and_images = tween_frame.cadence_flow is None and has_images
    return has_tween_schedule and has_step_and_images and data.animation_mode.is_raft_active()
