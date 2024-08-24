import os

import PIL
import cv2
import numpy as np
from PIL import Image
from cv2.typing import MatLike

from . import filename_utils
from ..data.render_data import RenderData


def bgr_to_rgb(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def numpy_to_pil(np_image: MatLike) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(np_image))


def pil_to_numpy(pil_image: Image.Image) -> MatLike:
    return np.array(pil_image)


def save_cadence_frame(data: RenderData, i: int, image: MatLike, is_overwrite: bool = True):
    filename = filename_utils.frame_filename(data, i)
    save_path: str = os.path.join(data.args.args.outdir, filename)
    if is_overwrite or not os.path.exists(save_path):
        cv2.imwrite(save_path, image)


def save_cadence_frame_and_depth_map_if_active(data: RenderData, frame, i, image):
    save_cadence_frame(data, i, image)
    if data.args.anim_args.save_depth_maps:
        dm_save_path = os.path.join(data.output_directory, filename_utils.frame_filename(data, i, True))
        data.depth_model.save(dm_save_path, frame.depth)


def save_and_return_frame(data: RenderData, frame, i, image):
    save_cadence_frame_and_depth_map_if_active(data, frame, i, image)
    return image


def is_PIL(image):
    return type(image) is PIL.Image.Image
