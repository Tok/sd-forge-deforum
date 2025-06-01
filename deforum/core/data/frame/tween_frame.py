from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ..render_data import RenderData
from ... import img_2_img_tubes
from ....utils import image_utils, log_utils, turbo_utils, web_ui_utils
from ....utils.call.subtitle import call_write_subtitle_from_to


@dataclass(init=True, frozen=False, repr=False, eq=False)
class Tween:
    """cadence vars"""
    value: float
    cadence_flow: Any
    cadence_flow_inc: Any
    depth: Any
    i: int

    def emit_frame(self, data, total_tqdm, last_frame):
        """Emits this tween frame."""
        max_frames = data.args.anim_args.max_frames
        if self.i >= max_frames:
            return  # skipping tween emission on the last frame

        self.handle_synchronous_status_concerns(data)

        new_image = self._generate(data, last_frame, data.images.previous)
        saved_image = image_utils.save_and_return_frame(data, self, new_image)
        total_tqdm.increment_tween_count()

        # updating reference images to calculate hybrid motions in next iteration
        data.images.before_previous = data.images.previous
        data.images.previous = saved_image
        data.args.root.init_sample = saved_image

    def _generate(self, data, last_frame, prev_image):
        advanced_image = turbo_utils.advance_optical_flow_cadence_before_animation_warping(
            data, last_frame, self, data.images.before_previous, data.images.previous)
        self.depth = Tween.calculate_depth_prediction(data, advanced_image)
        processed_image = img_2_img_tubes.process_tween_tube(data, last_frame, self.i, self.depth)(advanced_image)
        warped = turbo_utils.do_optical_flow_cadence_after_animation_warping(data, self, prev_image, processed_image)

        grayscale_tube = img_2_img_tubes.conditional_force_tween_to_grayscale_tube
        recolored = grayscale_tube(data)(warped)

        is_tween = True
        overlay_mask_tube = img_2_img_tubes.conditional_add_overlay_mask_tube
        masked = overlay_mask_tube(data, is_tween)(recolored)
        return masked

    def handle_synchronous_status_concerns(self, data):
        log_utils.print_tween_frame_info(data, self.i, self.cadence_flow, self.value)
        web_ui_utils.update_progress_during_cadence(data, self.i)

    def write_tween_subtitle_from_to(self, data: RenderData, sub_i, previous_diffusion_frame, from_time, to_time):
        # Cadence can be asserted because subtitle generation
        # skips the last tween in favor of its parent diffusion frame.
        is_cadence = True
        decremented_index = self.i - 1
        # Since tween frames are not diffused, they don't have their own seed.
        # We provide the seed of the previous frame that was diffused (parent diffusion frame has the next seed).
        # Since the 1st frame is always diffused it never has tweens, meaning there's always a previous_diffusion_frame.
        seed = previous_diffusion_frame.seed
        subseed = previous_diffusion_frame.subseed
        call_write_subtitle_from_to(data, sub_i, decremented_index, is_cadence, seed, subseed, from_time, to_time)

    def has_cadence(self):
        return self.cadence_flow is not None

    def is_last(self, last_keyframe):
        return self.i == last_keyframe.i

    @staticmethod
    def create_in_between_steps(key_frames, i, from_i, to_i):
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
    def create_steps(last_frame, tween_count, from_i) -> list['Tween']:
        if tween_count > 0:
            expected_tween_frames = Tween._calculate_expected_tween_frames(tween_count)
            return Tween.create_steps_from_values(last_frame, expected_tween_frames, from_i)
        return list()

    @staticmethod
    def calculate_depth_prediction(data, image):
        has_image = image is not None
        has_depth = data.depth_model is not None
        if has_image and has_depth:
            image = Tween.ensure_image_is_a_numpy_array(image)
            weight = data.args.anim_args.midas_weight
            precision = data.args.root.half_precision
            return data.depth_model.predict(image, weight, precision)
        else:
            return None

    @staticmethod
    def ensure_image_is_a_numpy_array(image):
        def convert(img):
            return np.array(img) if isinstance(img, list) or isinstance(img, Image.Image) else img
        numpy_array = convert(image)
        if not isinstance(numpy_array, np.ndarray):
            raise ValueError("Image must be a NumPy array.")
        return numpy_array
