from cv2.typing import MatLike

from .call.anim import call_anim_frame_warp
from ..core.frame_processing import image_transform_optical_flow, rel_flow_to_abs_flow, get_flow_from_images
# Note: hybrid video imports removed - functionality not available


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
    # Note: hybrid video motion functionality removed
    return image


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
    # Note: hybrid motion optical flow functionality removed
    return image


def _advance_cadence_flow(data, tween_frame, image):
    ff_string = data.args.anim_args.cadence_flow_factor_schedule
    flow_factor = float(ff_string.split(": ")[1][1:-1])
    i = tween_frame.i
    flow = tween_frame.cadence_flow_inc
    return image_transform_optical_flow(image, flow, flow_factor)


def _advance_ransac_transform(data, matrix, image):
    # Note: hybrid motion functionality removed
    return image


def _advance_hybrid_motion_ransac_transform(data, tween_i, reference_images, image):
    # Note: hybrid motion ransac functionality removed
    return image


def _is_do_flow(data, tween_frame, start_i, prev_image, image):
    has_tween_schedule = data.animation_keys.deform_keys.strength_schedule_series[start_i] > 0
    has_images = prev_image is not None and image is not None
    has_step_and_images = tween_frame.cadence_flow is None and has_images
    return has_tween_schedule and has_step_and_images and data.animation_mode.is_raft_active()
