from dataclasses import dataclass
from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image

from ...data.render_data import RenderData
from ...util import image_utils, log_utils, turbo_utils, web_ui_utils
from ...util.call.subtitle import call_write_frame_subtitle


@dataclass(init=True, frozen=False, repr=False, eq=False)
class Tween:
    """cadence vars"""
    value: float
    cadence_flow: Any
    cadence_flow_inc: Any
    depth: Any
    i: int

    def emit_frame(self, last_frame, grayscale_tube, overlay_mask_tube):
        """Emits this tween frame."""
        max_frames = last_frame.render_data.args.anim_args.max_frames
        if self.i >= max_frames:
            return  # skipping tween emission on the last frame

        data = last_frame.render_data
        self.handle_synchronous_status_concerns(data)

        # TODO use tube...
        next_image = self.process(last_frame, data, data.images.before_previous, data.images.previous)
        new_image = self.generate_tween_image(data, grayscale_tube, overlay_mask_tube, data.images.previous, next_image)

        saved_image = image_utils.save_and_return_frame(data, self, new_image)

        # updating reference images to calculate hybrid motions in next iteration
        data.images.before_previous = data.images.previous
        data.images.previous = saved_image

    def process(self, last_frame, data, prev_image, image):
        # TODO use tube
        advanced_image = turbo_utils.advance_optical_flow_cadence_before_animation_warping(
            data, last_frame, self, prev_image, image)

        if advanced_image is not None:
            self.depth = Tween.calculate_depth_prediction(data, advanced_image)

        next_img = turbo_utils.advance(data, self.i, advanced_image, self.depth)
        return turbo_utils.do_hybrid_video_motion(data, last_frame, self.i, data.images, next_img)

    def generate_tween_image(self, data, grayscale_tube, overlay_mask_tube, prev_image, next_image):
        warped = turbo_utils.do_optical_flow_cadence_after_animation_warping(data, self, prev_image, next_image)
        recolored = grayscale_tube(data)(warped)
        is_tween = True
        masked = overlay_mask_tube(data, is_tween)(recolored)
        return masked

    def handle_synchronous_status_concerns(self, data):
        log_utils.print_tween_frame_info(data, self.i, self.cadence_flow, self.value)
        web_ui_utils.update_progress_during_cadence(data, self.i)

    def write_tween_frame_subtitle(self, data: RenderData, previous_diffusion_frame):
        # Cadence can be asserted because subtitle generation
        # skips the last tween in favor of its parent diffusion frame.
        is_cadence = True
        decremented_index = self.i - 1
        # Since tween frames are not diffused, they don't have their own seed.
        # We provide the seed of the previous frame that was diffused (parent diffusion frame has the next seed).
        # Since the 1st frame is always diffused it never has tweens, meaning there's always a previous_diffusion_frame.
        seed = previous_diffusion_frame.seed
        subseed = previous_diffusion_frame.subseed
        call_write_frame_subtitle(data, decremented_index, is_cadence, seed, subseed)

    def has_cadence(self):
        return self.cadence_flow is not None

    @staticmethod
    def create_in_between_steps(key_frames, i, data, from_i, to_i):
        tween_count = to_i - from_i
        last_step = key_frames[i]
        return Tween.create_steps(last_step, tween_count, from_i)

    @staticmethod
    def _calculate_expected_tween_frames(num_entries):
        if num_entries <= 0:
            raise ValueError("Number of entries must be positive")
        offset = 1.0 / num_entries
        positions = [offset + (i / num_entries) for i in range(num_entries)]
        return positions

    @staticmethod
    def create_steps_from_values(last_frame, values, from_i):
        tween_count = len(values)
        r = range(tween_count)
        return list((Tween(values[i], None, None, last_frame.depth, i + from_i + 1) for i in r))

    @staticmethod
    def create_steps(last_frame, tween_count, from_i) -> Tuple[list['Tween'], list[float]]:
        if tween_count > 0:
            expected_tween_frames = Tween._calculate_expected_tween_frames(tween_count)
            return Tween.create_steps_from_values(last_frame, expected_tween_frames, from_i), expected_tween_frames
        return list(), list()

    @staticmethod
    def calculate_depth_prediction(data, image):
        has_image = image is not None
        has_depth = data.depth_model is not None
        if has_image and has_depth:
            # Ensure image is a NumPy array
            if isinstance(image, list):
                image = np.array(image)
            elif isinstance(image, Image.Image):
                image = np.array(image)

            # If it's still not a NumPy array, raise an error
            if not isinstance(image, np.ndarray):
                raise ValueError("Image must be a NumPy array.")

            weight = data.args.anim_args.midas_weight
            precision = data.args.root.half_precision
            return data.depth_model.predict(image, weight, precision)
        else:
            return None
