from dataclasses import dataclass

import PIL
import cv2
import numpy as np
from cv2.typing import MatLike

from ...load_images import load_image


@dataclass(init=True, frozen=False, repr=False, eq=True)
class Images:
    color_match: MatLike = None
    previous: MatLike | None = None

    def has_previous(self):
        return self.previous is not None

    @staticmethod
    def _load_color_match_sample(init) -> MatLike:
        """get color match for 'Image' color coherence only once, before loop"""
        if init.args.anim_args.color_coherence == 'Image':
            image_box: PIL.Image.Image = None
            # noinspection PyTypeChecker
            raw_image = load_image(init.args.anim_args.color_coherence_image_path, image_box)
            resized = raw_image.resize(init.dimensions(), PIL.Image.LANCZOS)
            return cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)

    @staticmethod
    def create(data):
        return Images(Images._load_color_match_sample(data))
