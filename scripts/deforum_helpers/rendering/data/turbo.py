from dataclasses import dataclass

from cv2.typing import MatLike

from ..util import opt_utils
from ..util.call.anim import call_anim_frame_warp
from ..util.call.hybrid import (call_get_flow_for_hybrid_motion_prev, call_get_flow_for_hybrid_motion,
                                call_get_matrix_for_hybrid_motion, call_get_matrix_for_hybrid_motion_prev)
from ..util.call.resume import call_get_resume_vars
from ...hybrid_video import (get_flow_from_images, image_transform_ransac,
                             image_transform_optical_flow, rel_flow_to_abs_flow)


@dataclass(init=True, frozen=False, repr=False, eq=True)
class ImageFrame:
    image: MatLike | None
    index: int


# Disabling transformations of previous frames may not be suited for all scenarios,
# but depending on setup can speed up generations significantly and without changing
# the visual output in a noticeable way. Leaving it off should be fine for current use cases.
IS_TRANSFORM_PREV = False  # TODO? benchmark and visually compare results. make configurable from UI or remove?


@dataclass(frozen=False)
class Turbo:
    cadence: int
    prev: ImageFrame
    next: ImageFrame

    @staticmethod
    def create(data):
        steps = 1 if data.has_video_input() else data.cadence()
        return Turbo(steps, ImageFrame(None, 0), ImageFrame(None, 0))

    def advance(self, data, i: int, depth):
        if self._has_prev_image() and IS_TRANSFORM_PREV:
            self.prev.image, _ = call_anim_frame_warp(data, i, self.prev.image, depth)
        if self._has_next_image():
            self.next.image, _ = call_anim_frame_warp(data, i, self.next.image, depth)

    def do_hybrid_video_motion(self, data, last_frame, indexes, reference_images):
        """Warps the previous and/or the next to match the motion of the provided reference images."""
        motion = data.args.anim_args.hybrid_motion

        def _is_do_motion(motions):
            return indexes.tween.i > 0 and motion in motions

        if _is_do_motion(['Affine', 'Perspective']):
            self.advance_hybrid_motion_ransac_transform(data, indexes, reference_images)
        if _is_do_motion(['Optical Flow']):
            self.advance_hybrid_motion_optical_tween_flow(data, indexes, reference_images, last_frame)

    def advance_optical_flow(self, tween_step, flow_factor: int = 1):
        flow = tween_step.cadence_flow * -1
        self.next.image = image_transform_optical_flow(self.next.image, flow, flow_factor)

    def advance_optical_tween_flow(self, indexes, last_frame, flow):
        flow_factor = last_frame.step_data.flow_factor()
        i = indexes.tween.i
        if self.is_advance_prev(i):
            self.prev.image = image_transform_optical_flow(self.prev.image, flow, flow_factor)
        if self.is_advance_next(i):
            self.next.image = image_transform_optical_flow(self.next.image, flow, flow_factor)

    def advance_hybrid_motion_optical_tween_flow(self, data, indexes, reference_images, last_frame):
        last_i = indexes.tween.i - 1
        flow = (call_get_flow_for_hybrid_motion(data, last_i)
                if not data.args.anim_args.hybrid_motion_use_prev_img
                else call_get_flow_for_hybrid_motion_prev(data, last_i, reference_images.previous))
        self.advance_optical_tween_flow(indexes, last_frame, flow)
        data.animation_mode.prev_flow = flow

    def advance_cadence_flow(self, data, tween_frame):
        ff_string = data.args.anim_args.cadence_flow_factor_schedule
        flow_factor = float(ff_string.split(": ")[1][1:-1])
        i = tween_frame.i()
        flow = tween_frame.cadence_flow_inc
        if self.is_advance_prev(i):
            self.prev.image = image_transform_optical_flow(self.prev.image, flow, flow_factor)
        if self.is_advance_next(i):
            self.next.image = image_transform_optical_flow(self.next.image, flow, flow_factor)

    def advance_ransac_transform(self, data, matrix):
        i = data.indexes.tween.i
        motion = data.args.anim_args.hybrid_motion
        if self.is_advance_prev(i):
            self.prev.image = image_transform_ransac(self.prev.image, matrix, motion)
        if self.is_advance_next(i):
            self.next.image = image_transform_ransac(self.next.image, matrix, motion)

    def advance_hybrid_motion_ransac_transform(self, data, indexes, reference_images):
        last_i = indexes.tween.i - 1
        matrix = (call_get_matrix_for_hybrid_motion(data, last_i)
                  if not data.args.anim_args.hybrid_motion_use_prev_img
                  else call_get_matrix_for_hybrid_motion_prev(data, last_i, reference_images.previous))
        self.advance_ransac_transform(data, matrix)

    def advance_optical_flow_cadence_before_animation_warping(self, data, last_frame, tween_frame):
        if data.is_3d_or_2d_with_optical_flow():
            if self._is_do_flow(data, tween_frame):
                method = data.args.anim_args.optical_flow_cadence  # string containing the flow method (e.g. "RAFT").
                flow = get_flow_from_images(self.prev.image, self.next.image, method, data.animation_mode.raft_model)
                tween_frame.cadence_flow = flow / len(last_frame.tweens)
            if tween_frame.has_cadence():
                self.advance_optical_flow(tween_frame)
                flow_factor = 1.0
                self.next.image = image_transform_optical_flow(self.next.image, -tween_frame.cadence_flow, flow_factor)

    def _is_do_flow(self, data, tween_frame):
        i = data.indexes.tween.start
        has_tween_schedule = data.animation_keys.deform_keys.strength_schedule_series[i] > 0
        has_images = self.prev.image is not None and self.next.image is not None
        has_step_and_images = tween_frame.cadence_flow is None and has_images
        return has_tween_schedule and has_step_and_images and data.animation_mode.is_raft_active()

    def do_optical_flow_cadence_after_animation_warping(self, data, indexes, tween_frame):
        if not data.animation_mode.is_raft_active():
            return self.next.image
        if tween_frame.cadence_flow is not None:
            # TODO Calculate all increments before running the generation (and try to avoid abs->rel->abs conversions).
            # temp_flow = abs_flow_to_rel_flow(tween_step.cadence_flow, data.width(), data.height())
            # new_flow, _ = call_anim_frame_warp(data, indexes.tween.i, temp_flow, None)
            new_flow, _ = call_anim_frame_warp(data, indexes.tween.i, self.prev.image, None)
            tween_frame.cadence_flow = new_flow
            abs_flow = rel_flow_to_abs_flow(tween_frame.cadence_flow, data.width(), data.height())
            tween_frame.cadence_flow_inc = abs_flow * tween_frame.value
            self.advance_cadence_flow(data, tween_frame)
        self.prev.index = self.next.frame_idx = indexes.tween.i if indexes is not None else 0
        if self.prev.image is not None and tween_frame.value < 1.0:
            return self.prev.image * (1.0 - tween_frame.value) + self.next.image * tween_frame.value
        return self.next.image

    def progress_step(self, indexes, opencv_image):
        self.prev.image, self.prev.index = self.next.image, self.next.index
        self.next.image, self.next.index = opencv_image, indexes.frame.i
        return self.cadence

    def _set_up_step_vars(self, data):
        # determine last frame and frame to start on
        prev_frame, next_frame, prev_img, next_img = call_get_resume_vars(data, self)
        if self.cadence > 1:
            self.prev.image, self.prev.index = prev_img, prev_frame if prev_frame >= 0 else 0
            self.next.image, self.next.index = next_img, next_frame if next_frame >= 0 else 0

    def find_start(self, data) -> int:
        """Maybe resume animation (requires at least two frames - see function)."""
        if data.is_resuming_from_timestring():
            # set start_frame to next frame
            self._set_up_step_vars(data)
        # instead of "self.next.index + 1" we always return 0, to print a message
        # for every frame that is skipped because it already exists.
        return 0

    def has_steps(self):
        return self.cadence > 1

    def _has_prev_image(self):
        return self.prev.image is not None

    def is_advance_prev(self, i: int) -> bool:
        return IS_TRANSFORM_PREV and self._has_prev_image() and i > self.prev.index

    def _has_next_image(self):
        return self.next.image is not None

    def is_advance_next(self, i: int) -> bool:
        return i > self.next.index

    def is_first_step(self) -> bool:
        return self.cadence == 1

    def is_first_step_with_subtitles(self) -> bool:
        return self.is_first_step() and opt_utils.is_subtitle_generation_active()

    def is_emit_in_between_frames(self) -> bool:
        return self.cadence > 1
