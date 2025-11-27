from cv2.typing import MatLike

from deforum.rendering.calls.anim import call_anim_frame_warp
from deforum.animation.optical_flow_utils import (get_flow_from_images, image_transform_optical_flow,
                                   abs_flow_to_rel_flow, rel_flow_to_abs_flow)


def advance_optical_flow_cadence_before_animation_warping(data, last_frame, tween_frame, prev_image, image) -> MatLike:
    """Calculate optical flow between keyframes once before any warping.

    Original algorithm: Flow is calculated ONCE between prev/next keyframes and divided by 2.
    This flow represents the motion field across the keyframe gap.
    """
    is_with_flow = data.is_3d_or_2d_with_optical_flow()
    if is_with_flow and _is_do_flow(data, tween_frame, last_frame.i, prev_image, image):
        method = data.args.anim_args.optical_flow_cadence  # string containing the flow method (e.g. "RAFT")
        flow = get_flow_from_images(prev_image, image, method, data.animation_mode.raft_model)
        # Divide by 2 (original algorithm) - flow represents motion across keyframe gap
        tween_frame.cadence_flow = flow / 2.0
    return image  # Return unmodified - flow applied AFTER animation warping


def advance(data, i, image, depth):
    """Apply 3D animation warping to image using depth."""
    if depth is not None:
        warped_image, _ = call_anim_frame_warp(data, i, image, depth)
        return warped_image
    else:
        return image


def do_optical_flow_cadence_after_animation_warping(data, tween_frame, prev_image, image):
    """Apply optical flow AFTER 3D animation warping, then blend with previous frame.

    Original algorithm order:
    1. Image already warped by 3D transforms (in advance() function)
    2. Warp the FLOW FIELD itself using same 3D transforms
    3. Apply warped flow to warped image
    4. Blend result with previous frame based on tween position
    """
    if not data.animation_mode.is_raft_active():
        # When depth warping is active, schedules handle interpolation - no blending needed
        # Blending would cause keyframe "leaking" and defeat 3D transforms
        if data.args.anim_args.use_depth_warping:
            return image  # Return warped image directly

        # No depth warping, no optical flow - blend for smooth transition
        if prev_image is not None and tween_frame.value < 1.0:
            return prev_image * (1.0 - tween_frame.value) + image * tween_frame.value
        return image

    if tween_frame.cadence_flow is not None:
        # Convert flow to relative coordinates for warping
        cadence_flow = abs_flow_to_rel_flow(tween_frame.cadence_flow, data.width(), data.height())

        # Warp the FLOW FIELD itself using 3D animation transforms
        # This is critical - the flow needs to follow the same 3D motion as the image
        # Only warp flow if we have valid depth (can't calculate depth from a 2-channel flow field)
        if tween_frame.depth is not None:
            cadence_flow, _ = call_anim_frame_warp(data, tween_frame.i, cadence_flow, tween_frame.depth)
        # If no depth, use flow as-is (unwarped)

        # Convert back to absolute coordinates and scale by tween position
        cadence_flow_inc = rel_flow_to_abs_flow(cadence_flow, data.width(), data.height()) * tween_frame.value

        # Get flow factor from schedule
        ff_string = data.args.anim_args.cadence_flow_factor_schedule
        flow_factor = float(ff_string.split(": ")[1][1:-1])

        # Apply the warped+scaled flow to the already-warped image
        image = image_transform_optical_flow(image, cadence_flow_inc, flow_factor)

    # Final blend with previous frame based on tween interpolation value
    if prev_image is not None and tween_frame.value < 1.0:
        return prev_image * (1.0 - tween_frame.value) + image * tween_frame.value
    return image


def _is_do_flow(data, tween_frame, start_i, prev_image, image):
    """Check if optical flow should be calculated for this tween."""
    # Bounds check - don't try to access schedule beyond max_frames
    max_frames = data.args.anim_args.max_frames
    if start_i >= max_frames:
        return False

    has_tween_schedule = data.animation_keys.deform_keys.strength_schedule_series[start_i] > 0
    has_images = prev_image is not None and image is not None
    has_step_and_images = tween_frame.cadence_flow is None and has_images
    return has_tween_schedule and has_step_and_images and data.animation_mode.is_raft_active()
