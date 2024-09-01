from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable, Tuple, List

from ..turbo import Turbo
from ...data.indexes import Indexes, IndexWithStart
from ...data.render_data import RenderData
from ...util import image_utils, log_utils, web_ui_utils
from ...util.call.subtitle import call_write_frame_subtitle


@dataclass(init=True, frozen=False, repr=False, eq=False)
class Tween:
    """cadence vars"""
    indexes: Indexes
    value: float
    # Since tweens are not diffused, the seed or 'shadow-seed' is currently only relevant for subtitles.
    # We generate and assign a seed any ways, so the behaviour is consistent and comparable with parseq setups.
    shadow_seed: int
    cadence_flow: Any
    cadence_flow_inc: Any
    depth: Any

    def i(self):
        return self.indexes.tween.i

    def emit_frame(self, last_frame, grayscale_tube, overlay_mask_tube):
        """Emits this tween frame."""
        max_frames = last_frame.render_data.args.anim_args.max_frames
        if self.i() >= max_frames:
            return  # skipping tween emission on the last frame

        data = last_frame.render_data
        # data.turbo.steps = len(last_step.tweens)
        self.handle_synchronous_status_concerns(data)
        self.process(last_frame, data)

        new_image = self.generate_tween_image(data, grayscale_tube, overlay_mask_tube)
        new_image = image_utils.save_and_return_frame(data, self, self.i(), new_image)

        # updating reference images to calculate hybrid motions in next iteration
        data.images.previous = new_image

    def generate_tween_image(self, data, grayscale_tube, overlay_mask_tube):
        warped = data.turbo.do_optical_flow_cadence_after_animation_warping(data, self.indexes, self)
        recolored = grayscale_tube(data)(warped)
        is_tween = True
        masked = overlay_mask_tube(data, is_tween)(recolored)
        return masked

    def process(self, last_frame, data):
        data.turbo.advance_optical_flow_cadence_before_animation_warping(data, last_frame, self)
        self.depth = Tween.calculate_depth_prediction(data, data.turbo)
        data.turbo.advance(data, self.i(), self.depth)
        data.turbo.do_hybrid_video_motion(data, last_frame, self.indexes, data.images)

    def handle_synchronous_status_concerns(self, data):
        log_utils.print_tween_frame_info(data, self.indexes, self.cadence_flow, self.value)
        web_ui_utils.update_progress_during_cadence(data, self.indexes)

    def write_tween_frame_subtitle(self, data: RenderData):
        # Cadence can be asserted because subtitle generation
        # skips the last tween in favor of its parent diffusion frame.
        is_cadence = True
        decremented_index = self.indexes.tween.i - 1
        call_write_frame_subtitle(data, decremented_index, is_cadence, self.shadow_seed)

    def has_cadence(self):
        return self.cadence_flow is not None

    @staticmethod
    def create_in_between_steps(key_frames, i, data, from_i, to_i):
        tween_range = range(from_i, to_i)
        tween_indexes_list: List[Indexes] = Tween.create_indexes(data.indexes, tween_range)
        last_step = key_frames[i]
        tween_steps_and_values = Tween.create_steps(last_step, tween_indexes_list)
        for tween in tween_steps_and_values[0]:
            tween.indexes.update_tween_index(tween.i() + key_frames[i].i)
        return tween_steps_and_values

    @staticmethod
    def _calculate_expected_tween_frames(num_entries):
        if num_entries <= 0:
            raise ValueError("Number of entries must be positive")
        offset = 1.0 / num_entries
        positions = [offset + (i / num_entries) for i in range(num_entries)]
        return positions

    @staticmethod
    def _increment(original_indexes, tween_count, from_start):
        inc = original_indexes.frame.i - tween_count - original_indexes.tween.start + from_start
        original_indexes.tween = IndexWithStart(original_indexes.tween.start, original_indexes.tween.start + inc)
        return original_indexes

    @staticmethod
    def create_steps_from_values(last_frame, values):
        count = len(values)
        r = range(count)
        indexes_list = [Tween._increment(last_frame.render_data.indexes.copy(), count, i + 1) for i in r]
        return list((Tween(indexes_list[i], values[i], -1, None, None, last_frame.depth) for i in r))

    @staticmethod
    def create_indexes(base_indexes: Indexes, frame_range: Iterable[int]) -> list[Indexes]:
        return list(chain.from_iterable([Indexes.create_from_last(base_indexes, i)] for i in frame_range))

    @staticmethod
    def create_steps(last_frame, tween_indexes_list: list[Indexes]) -> Tuple[list['Tween'], list[float]]:
        if len(tween_indexes_list) > 0:
            expected_tween_frames = Tween._calculate_expected_tween_frames(len(tween_indexes_list))
            return Tween.create_steps_from_values(last_frame, expected_tween_frames), expected_tween_frames
        return list(), list()

    @staticmethod
    def calculate_depth_prediction(data, turbo: Turbo):
        has_depth = data.depth_model is not None
        has_next = turbo.next.image is not None
        if has_depth and has_next:
            image = turbo.next.image
            weight = data.args.anim_args.midas_weight
            precision = data.args.root.half_precision
            return data.depth_model.predict(image, weight, precision)
