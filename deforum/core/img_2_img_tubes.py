from typing import Callable

import cv2
import numpy as np
from PIL import ImageOps, Image
from cv2.typing import MatLike

from .data.frame.diffusion_frame import DiffusionFrame
from .data.render_data import RenderData
from .util import image_utils, turbo_utils
from .util.call.hybrid import call_hybrid_composite
from .util.fun_utils import tube
from ..colors import maintain_colors
from ..hybrid_video import get_flow_from_images, image_transform_optical_flow
from ..masks import do_overlay_mask

"""
This module provides functions for conditionally processing images through various transformations.
The `tube` function allows chaining these transformations together to create flexible image processing pipelines.
Easily experiment by changing, or changing the order of function calls in the tube without having to worry
about the larger context and without having to invent unnecessary names for intermediary processing results.

All functions within the tube take and return an image (`img` argument). They may (and must) pass through
the original image unchanged if a specific transformation is disabled or not required.

Example:
transformed_image = my_tube(arguments)(original_image)
"""

# ImageTubes are functions that take a MatLike image and return a newly processed (or the same unchanged) MatLike image.
ImageTube = Callable[[MatLike], MatLike]
PilImageTube = Callable[[Image.Image], Image.Image]


def frame_transformation_tube(data: RenderData, frame: DiffusionFrame) -> ImageTube:
    return tube(lambda img: frame.apply_frame_warp_transform(data, img),
                lambda img: frame.do_hybrid_compositing_before_motion(data, img),
                lambda img: DiffusionFrame.apply_hybrid_motion_ransac_transform(data, img, frame.i),
                lambda img: DiffusionFrame.apply_hybrid_motion_optical_flow(data, img, frame),
                lambda img: frame.do_normal_hybrid_compositing_after_motion(data, img),
                lambda img: DiffusionFrame.apply_color_matching(data, img),
                lambda img: DiffusionFrame.transform_to_grayscale_if_active(data, img))


def contrast_transformation_tube(data: RenderData, frame: DiffusionFrame) -> ImageTube:
    return tube(lambda img: frame.apply_scaling(img),
                lambda img: frame.apply_anti_blur(data, img))


def noise_transformation_tube(data: RenderData, frame: DiffusionFrame) -> ImageTube:
    return tube(lambda img: frame.apply_frame_noising(data, frame, img))


def optical_flow_redo_tube(data: RenderData, frame: DiffusionFrame, optical_flow) -> ImageTube:
    return tube(lambda img: image_utils.pil_to_numpy(img),
                lambda img: image_utils.bgr_to_rgb(img),
                lambda img: image_transform_optical_flow(
                    img, get_flow_from_images(data.images.previous, img, optical_flow, data.animation_mode.raft_model),
                    frame.frame_data.redo_flow_factor))


def process_tween_tube(data: RenderData, last_frame, i, depth) -> ImageTube:
    return tube(lambda img: turbo_utils.advance(data, i, img, depth),
                lambda img: turbo_utils.do_hybrid_video_motion(data, last_frame, i, data.images, img))


# Conditional Tubes (can be switched on or off by providing a Callable[Boolean] `is_do_process` predicate).
def conditional_hybrid_video_after_generation_tube(data: RenderData, frame: DiffusionFrame) -> PilImageTube:
    fd = frame.frame_data
    return tube(lambda img: call_hybrid_composite(data, frame.i, img, fd.hybrid_comp_schedules),
                lambda img: image_utils.numpy_to_pil(img),
                is_do_process=lambda: frame.i > 0 and data.is_hybrid_composite_after_generation())


def conditional_extra_color_match_tube(data: RenderData, i) -> PilImageTube:
    # color matching on first frame is after generation, color match was collected earlier,
    # so we do an extra generation to avoid the corruption introduced by the color match of first output
    return tube(lambda img: maintain_colors(img, data.images.color_match, data.args.anim_args.color_coherence),
                lambda img: maintain_colors(img, data.images.color_match, data.args.anim_args.color_coherence),
                lambda img: image_utils.numpy_to_pil(img),
                is_do_process=lambda: i == 0 and data.is_initialize_color_match(
                    data.images.color_match))


def conditional_color_match_tube(data: RenderData, frame: DiffusionFrame) -> ImageTube:
    # on strength 0, set color match to generation
    return tube(lambda img: image_utils.bgr_to_rgb(np.asarray(img)),
                is_do_process=lambda: data.is_do_color_match_conversion(frame))


def conditional_force_to_grayscale_tube(data: RenderData) -> PilImageTube:
    return tube(lambda img: ImageOps.grayscale(img),
                lambda img: ImageOps.colorize(img, black="black", white="white"),
                is_do_process=lambda: data.args.anim_args.color_force_grayscale)


def conditional_add_overlay_mask_tube(data: RenderData, i) -> PilImageTube:
    is_use_overlay = data.args.args.overlay_mask
    is_use_mask = data.args.anim_args.use_mask_video or data.args.args.use_mask
    # do_overlay_mask can handle both, BGR array and PIL Image, but does a conversion if an array is passed.
    is_bgr_array = False  # No need for conversion as grayscale returns a PIL Image.
    return tube(lambda img: ImageOps.grayscale(img),
                lambda img: do_overlay_mask(data.args.args, data.args.anim_args, img, i, is_bgr_array),
                is_do_process=lambda: is_use_overlay and is_use_mask)


def conditional_force_tween_to_grayscale_tube(data: RenderData) -> ImageTube:
    return tube(lambda img: cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY),
                lambda img: cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                is_do_process=lambda: data.args.anim_args.color_force_grayscale)


# Composite Tubes, made from other Tubes.
def contrasted_noise_transformation_tube(data: RenderData, frame: DiffusionFrame) -> ImageTube:
    """Combines contrast and noise transformation tubes."""
    contrast_tube: ImageTube = contrast_transformation_tube(data, frame)
    noise_tube: ImageTube = noise_transformation_tube(data, frame)
    return tube(lambda img: noise_tube(contrast_tube(img)))


def conditional_frame_transformation_tube(data: RenderData, frame: DiffusionFrame) -> PilImageTube:
    hybrid_tube: PilImageTube = conditional_hybrid_video_after_generation_tube(data, frame)
    extra_tube: PilImageTube = conditional_extra_color_match_tube(data, frame.i)
    gray_tube: PilImageTube = conditional_force_to_grayscale_tube(data)
    mask_tube: PilImageTube = conditional_add_overlay_mask_tube(data, frame.i)
    return tube(lambda img: mask_tube(gray_tube(extra_tube(hybrid_tube(img)))))
