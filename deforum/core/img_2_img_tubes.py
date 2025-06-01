from typing import Callable, Any

import cv2
import numpy as np
from PIL import ImageOps, Image
from cv2.typing import MatLike

from .data.render_data import RenderData
from ..utils import image_utils, turbo_utils
from ..utils.fun_utils import tube
from ..utils.colors import maintain_colors
from ..utils.mask_utilities import do_overlay_mask

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


def frame_transformation_tube(data: RenderData, frame) -> ImageTube:
    from .data.frame.diffusion_frame import DiffusionFrame
    return tube(lambda img: frame.apply_frame_warp_transform(data, img),
                # Note: hybrid functionality removed
                lambda img: DiffusionFrame.apply_color_matching(data, img),
                lambda img: DiffusionFrame.transform_to_grayscale_if_active(data, img))


def contrast_transformation_tube(data: RenderData, frame) -> ImageTube:
    return tube(lambda img: frame.apply_scaling(img),
                lambda img: frame.apply_anti_blur(data, img))


def noise_transformation_tube(data: RenderData, frame) -> ImageTube:
    return tube(lambda img: frame.apply_frame_noising(data, frame, img))


def optical_flow_redo_tube(data: RenderData, frame, optical_flow) -> ImageTube:
    # Note: optical flow functionality requires hybrid imports which have been removed
    return tube(lambda img: img)


def process_tween_tube(data: RenderData, last_frame, i, depth) -> ImageTube:
    return tube(lambda img: turbo_utils.advance(data, i, img, depth),
                # Note: hybrid video motion functionality removed
                lambda img: img)


# Conditional Tubes (can be switched on or off by providing a Callable[Boolean] `is_do_process` predicate).
def conditional_hybrid_video_after_generation_tube(data: RenderData, frame) -> PilImageTube:
    # Note: hybrid video functionality removed
    return tube(lambda img: img)


def conditional_extra_color_match_tube(data: RenderData, i) -> PilImageTube:
    # color matching on first frame is after generation, color match was collected earlier,
    # so we do an extra generation to avoid the corruption introduced by the color match of first output
    return tube(lambda img: maintain_colors(img, data.images.color_match, data.args.anim_args.color_coherence),
                lambda img: maintain_colors(img, data.images.color_match, data.args.anim_args.color_coherence),
                lambda img: image_utils.numpy_to_pil(img),
                is_do_process=lambda: i == 0 and data.is_initialize_color_match(
                    data.images.color_match))


def conditional_color_match_tube(data: RenderData, frame) -> PilImageTube:
    return tube(lambda img: turbo_utils.do_color_match(data, frame.i, img))


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
def contrasted_noise_transformation_tube(data: RenderData, frame) -> ImageTube:
    """Combines contrast and noise transformation tubes."""
    contrast_tube: ImageTube = contrast_transformation_tube(data, frame)
    noise_tube: ImageTube = noise_transformation_tube(data, frame)
    return tube(lambda img: noise_tube(contrast_tube(img)))


def conditional_frame_transformation_tube(data: RenderData, frame) -> PilImageTube:
    return tube(lambda img: img,
                lambda img: turbo_utils.do_frame_warp(data, frame.i, img),
                lambda img: turbo_utils.do_color_coherence(data, img),
                lambda img: turbo_utils.do_contrast_and_noise(data, frame.i, img))


def conditional_color_coherence_tube(data: RenderData, frame) -> PilImageTube:
    return tube(lambda img: turbo_utils.do_color_coherence(data, img))


def conditional_mask_overlay_tube(data: RenderData, frame) -> PilImageTube:
    return tube(lambda img: turbo_utils.do_mask_overlay(data, frame.i, img))


def conditional_grayscale_tube(data: RenderData, frame) -> PilImageTube:
    return tube(lambda img: turbo_utils.do_grayscale(data, img))


def conditional_frame_transformation_tube_after_generation(data: RenderData, frame) -> PilImageTube:
    hybrid_tube: PilImageTube = conditional_hybrid_video_after_generation_tube(data, frame)
    color_coherence_tube: PilImageTube = conditional_color_coherence_tube(data, frame)
    mask_overlay_tube: PilImageTube = conditional_mask_overlay_tube(data, frame)
    grayscale_tube: PilImageTube = conditional_grayscale_tube(data, frame)
    color_match_tube: PilImageTube = conditional_color_match_tube(data, frame)

    return tube(hybrid_tube, color_coherence_tube, mask_overlay_tube, grayscale_tube, color_match_tube)
