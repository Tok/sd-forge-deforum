import os
from dataclasses import dataclass
from typing import Any, List

import PIL
import cv2
import numpy as np
from PIL import Image

from . import KeyFrameData, KeyFrameDistribution
from .tween_frame import Tween
from .. import RenderData, Schedule
from ... import img_2_img_tubes
from ...util import depth_utils, filename_utils, log_utils, opt_utils, utils
from ...util.call.anim import call_anim_frame_warp
from ...util.call.gen import call_generate
from ...util.call.hybrid import (
    call_get_flow_for_hybrid_motion, call_get_flow_for_hybrid_motion_prev, call_get_matrix_for_hybrid_motion,
    call_get_matrix_for_hybrid_motion_prev, call_hybrid_composite)
from ...util.call.images import call_add_noise
from ...util.call.mask import call_compose_mask_with_check, call_unsharp_mask
from ...util.call.subtitle import call_format_animation_params, call_write_frame_subtitle
from ...util.call.video_and_audio import call_render_preview
from ....colors import maintain_colors
from ....hybrid_video import image_transform_ransac, image_transform_optical_flow
from ....save_images import save_image
from ....seed import next_seed


@dataclass(init=True, frozen=False, repr=False, eq=False)
class KeyFrame:
    """Key steps are the steps for frames that actually get diffused (as opposed to tween frame steps)."""
    i: int
    step_data: KeyFrameData
    render_data: RenderData
    schedule: Schedule
    depth: Any  # assigned during generation
    subtitle_params_to_print: Any
    subtitle_params_string: str
    last_preview_frame: int
    tweens: List[Tween]
    tween_values: List[float]

    def has_tween_frames(self):
        return len(self.tweens) > 0

    def is_optical_flow_redo_before_generation(self, optical_flow_redo_generation, images):
        has_flow_redo = optical_flow_redo_generation != 'None'
        return has_flow_redo and images.has_previous() and self.step_data.has_strength()

    def maybe_write_frame_subtitle(self):
        data = self.render_data
        if data.turbo.is_first_step_with_subtitles():
            params_string = opt_utils.generation_info_for_subtitles()
            self.subtitle_params_to_print = params_string
            self.subtitle_params_string = call_format_animation_params(data, data.indexes.frame.i, params_string)
            call_write_frame_subtitle(data, data.indexes.frame.i, params_string)

    def apply_frame_warp_transform(self, data: RenderData, image):
        is_not_last_frame = self.i < data.args.anim_args.max_frames
        if is_not_last_frame:
            previous, self.depth = call_anim_frame_warp(data, self.i, image, None)
            return previous

    def _do_hybrid_compositing_on_cond(self, data: RenderData, image, condition):
        i = data.indexes.frame.i
        schedules = self.step_data.hybrid_comp_schedules
        if condition:
            _, composed = call_hybrid_composite(data, i, image, schedules)
            return composed
        return image

    def do_hybrid_compositing_before_motion(self, data: RenderData, image):
        condition = data.is_hybrid_composite_before_motion()
        return self._do_hybrid_compositing_on_cond(data, image, condition)

    def do_normal_hybrid_compositing_after_motion(self, data: RenderData, image):
        condition = data.is_normal_hybrid_composite()
        return self._do_hybrid_compositing_on_cond(data, image, condition)

    def apply_scaling(self, image):
        return (image * self.step_data.contrast).round().astype(np.uint8)

    def apply_anti_blur(self, data: RenderData, image):
        if self.step_data.amount > 0:
            return call_unsharp_mask(data, self, image, data.mask)
        return image

    def apply_frame_noising(self, data: RenderData, mask, image):
        is_use_any_mask = data.args.args.use_mask or data.args.anim_args.use_noise_mask
        if is_use_any_mask:
            seq = self.schedule.noise_mask
            vals = mask.noise_vals
            contrast_image = image
            data.args.root.noise_mask = call_compose_mask_with_check(data, seq, vals, contrast_image)
        return call_add_noise(data, self, image)

    def create_color_match_for_video(self):
        data = self.render_data
        if data.args.anim_args.color_coherence == 'Video Input' and data.is_hybrid_available():
            if int(data.indexes.frame.i) % int(data.args.anim_args.color_coherence_video_every_N_frames) == 0:
                prev_vid_img = Image.open(filename_utils.preview_video_image_path(data, data.indexes))
                prev_vid_img = prev_vid_img.resize(data.dimensions(), PIL.Image.LANCZOS)
                data.images.color_match = np.asarray(prev_vid_img)
                return cv2.cvtColor(data.images.color_match, cv2.COLOR_RGB2BGR)
        return None

    def _generate_and_update_noise(self, data, image, contrasted_noise_tube):
        noised_image = contrasted_noise_tube(data, self)(image)
        data.update_sample_and_args_for_current_progression_step(self, noised_image)
        return image  # return original as passed.

    def transform_and_update_noised_sample(self, frame_tube, contrasted_noise_tube):
        data = self.render_data
        if data.images.has_previous():  # skipping 1st iteration
            transformed_image = frame_tube(data, self)(data.images.previous)
            if transformed_image is None:
                log_utils.warn("Image transformation failed, using fallback.")
                transformed_image = data.images.previous
            return self._generate_and_update_noise(data, transformed_image, contrasted_noise_tube)
        return None

    def prepare_generation(self, frame_tube, contrasted_noise_tube):
        self.render_data.images.color_match = self.create_color_match_for_video()
        self.render_data.images.previous = self.transform_and_update_noised_sample(frame_tube, contrasted_noise_tube)
        self.render_data.prepare_generation(self.render_data, self, self.i)
        self.maybe_redo_optical_flow()
        self.maybe_redo_diffusion()

    # Conditional Redoes
    def maybe_redo_optical_flow(self):
        data = self.render_data
        optical_flow_redo_generation = data.optical_flow_redo_generation_if_not_in_preview_mode()
        is_redo_optical_flow = self.is_optical_flow_redo_before_generation(optical_flow_redo_generation, data.images)
        if is_redo_optical_flow:
            data.args.root.init_sample = self.do_optical_flow_redo_before_generation()

    def maybe_redo_diffusion(self):
        data = self.render_data
        is_pos_redo = data.has_positive_diffusion_redo
        is_diffusion_redo = is_pos_redo and data.images.has_previous() and self.step_data.has_strength()
        is_not_preview = data.is_not_in_motion_preview_mode()
        if is_diffusion_redo and is_not_preview:
            self.do_diffusion_redo()

    def generate(self):
        return call_generate(self.render_data, self)

    def after_diffusion(self, image):
        self.render_data.images.color_match = img_2_img_tubes.conditional_color_match_tube(self)(image)
        self.progress_and_save(image)
        self.render_data.args.args.seed = self.next_seed()
        self.update_render_preview()

    def progress_and_save(self, image):
        next_index = self._progress_save_and_get_next_index(image)
        self.render_data.indexes.update_frame(next_index)

    def _progress_save_and_get_next_index(self, image):
        data = self.render_data
        """Will progress frame or turbo-frame step, save the image, update `self.depth` and return next index."""
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if not data.animation_mode.has_video_input:
            data.images.previous = opencv_image
        filename = filename_utils.frame_filename(data, self.i)

        is_overwrite = True  # replace the processed tween frame with the original one? (probably not)
        if is_overwrite or not os.path.exists(os.path.join(data.args.args.outdir, filename)):
            # In many cases, the original images may look more detailed or 'better' than the processed ones,
            # but we only save the frames that were processed tough the flows to keep the output consistent.
            # However, it may be preferable to use them for the 1st and for the last frame, or as thumbnails.
            # TODO? add option to save original frames in a different sub dir.
            save_image(image, 'PIL', filename, data.args.args, data.args.video_args, data.args.root)

        self.depth = depth_utils.generate_and_save_depth_map_if_active(data, opencv_image)
        if data.turbo.has_steps():
            return data.indexes.frame.i + data.turbo.progress_step(data.indexes, opencv_image)
        return data.indexes.frame.i + 1  # normal (i.e. 'non-turbo') step always increments by 1.

    def next_seed(self):
        return next_seed(self.render_data.args.args, self.render_data.args.root)

    def update_render_preview(self):
        self.last_preview_frame = call_render_preview(self.render_data, self.last_preview_frame)

    def do_optical_flow_redo_before_generation(self):
        data = self.render_data
        redo = data.args.anim_args.optical_flow_redo_generation
        stored_seed = data.args.args.seed  # keep original to reset it after executing the optical flow
        data.args.args.seed = utils.generate_random_seed()  # create and set a new random seed
        log_utils.print_optical_flow_info(data, redo)

        sample_image = call_generate(data, self)
        optical_tube = img_2_img_tubes.optical_flow_redo_tube(data, self, redo)
        transformed_sample_image = optical_tube(sample_image)

        data.args.args.seed = stored_seed  # restore stored seed
        return Image.fromarray(transformed_sample_image)

    def do_diffusion_redo(self):
        data = self.render_data
        stored_seed = data.args.args.seed
        last_diffusion_redo_index = int(data.args.anim_args.diffusion_redo)
        for n in range(0, last_diffusion_redo_index):
            log_utils.print_redo_generation_info(data, n)
            data.args.args.seed = utils.generate_random_seed()
            diffusion_redo_image = call_generate(data, self)
            diffusion_redo_image = cv2.cvtColor(np.array(diffusion_redo_image), cv2.COLOR_RGB2BGR)
            # color match on last one only
            is_last_iteration = n == last_diffusion_redo_index
            if is_last_iteration:
                mode = data.args.anim_args.color_coherence
                diffusion_redo_image = maintain_colors(data.images.previous, data.images.color_match, mode)
            data.args.args.seed = stored_seed
            data.args.root.init_sample = Image.fromarray(cv2.cvtColor(diffusion_redo_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def create(data: RenderData):
        step_data = KeyFrameData.create(data)
        schedule = Schedule.create(data)
        return KeyFrame(0, step_data, data, schedule, None, None, "", 0, list(), list())

    @staticmethod
    def apply_color_matching(data: RenderData, image):
        return KeyFrame.apply_color_coherence(image, data) if data.has_color_coherence() else image

    @staticmethod
    def apply_color_coherence(image, data: RenderData):
        if data.images.color_match is None:
            # Initialize color_match for next iteration with current image, but don't do anything yet.
            if image is not None:
                data.images.color_match = image.copy()
            return image
        return maintain_colors(image, data.images.color_match, data.args.anim_args.color_coherence)

    @staticmethod
    def transform_to_grayscale_if_active(data: RenderData, image):
        if data.args.anim_args.color_force_grayscale:
            grayscale = cv2.cvtColor(data.images.previous, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        return image

    @staticmethod
    def apply_hybrid_motion_ransac_transform(data: RenderData, image):
        """hybrid video motion - warps `images.previous` to match motion, usually to prepare for compositing"""
        motion = data.args.anim_args.hybrid_motion
        if motion in ['Affine', 'Perspective']:
            last_i = data.indexes.frame.i - 1
            reference_images = data.images
            matrix = call_get_matrix_for_hybrid_motion_prev(data, last_i, reference_images.previous) \
                if data.args.anim_args.hybrid_motion_use_prev_img \
                else call_get_matrix_for_hybrid_motion(data, last_i)
            return image_transform_ransac(image, matrix, data.args.anim_args.hybrid_motion)
        return image

    @staticmethod
    def apply_hybrid_motion_optical_flow(data: RenderData, key_frame, image):
        motion = data.args.anim_args.hybrid_motion
        if motion in ['Optical Flow']:
            last_i = data.indexes.frame.i - 1
            reference_images = data.images
            flow = call_get_flow_for_hybrid_motion_prev(data, last_i, reference_images.previous) \
                if data.args.anim_args.hybrid_motion_use_prev_img \
                else call_get_flow_for_hybrid_motion(data, last_i)
            transformed = image_transform_optical_flow(
                reference_images.previous, flow, key_frame.step_data.flow_factor())
            data.animation_mode.prev_flow = flow
            return transformed
        return image

    @staticmethod
    def create_all_frames(data, index_dist: KeyFrameDistribution = KeyFrameDistribution.default()):
        """Creates a list of key steps for the entire animation."""
        start_index = data.turbo.find_start(data)
        num_key_steps = 1 + int((data.args.anim_args.max_frames - start_index) / data.cadence())
        if data.parseq_adapter.use_parseq and index_dist is KeyFrameDistribution.PARSEQ_ONLY:
            num_key_steps = len(data.parseq_adapter.parseq_json["keyframes"])

        key_steps = [KeyFrame.create(data) for _ in range(0, num_key_steps)]
        actual_num_key_steps = len(key_steps)

        recalculated_key_steps = KeyFrame._recalculate_and_check_tweens(
            data, key_steps, start_index, actual_num_key_steps, data.parseq_adapter, index_dist)
        log_utils.print_tween_step_creation_info(key_steps, index_dist)

        return recalculated_key_steps

    @staticmethod
    def _recalculate_and_check_tweens(data, key_frames, start_index, num_key_steps,
                                      parseq_adapter, index_distribution):
        max_frames = data.args.anim_args.max_frames
        key_frames = index_distribution.calculate(key_frames, start_index, max_frames, num_key_steps, parseq_adapter)
        key_frames = KeyFrame._add_tweens_to_key_steps(key_frames)
        log_utils.print_key_step_debug_info_if_verbose(key_frames)

        # The number of generated tweens depends on index since last key-frame. The last tween has the same
        # index as the key_step it belongs to and is meant to replace the unprocessed original key frame.
        assert len(key_frames) == num_key_steps
        assert key_frames[0].i == 1  # 1st key frame is at index 1
        assert key_frames[0].tweens == []  # 1st key frame has no tweens
        if index_distribution != KeyFrameDistribution.PARSEQ_ONLY:  # just using however many key frames Parseq defines.
            assert key_frames[-1].i == max_frames  # last index is same as max frames

        return key_frames

    @staticmethod
    def _add_tweens_to_key_steps(key_steps):
        log_utils.info(f"Adding tweens to {len(key_steps)} keyframes...")
        for i in range(1, len(key_steps)):  # skipping 1st key frame
            data = key_steps[i].render_data
            from_i = key_steps[i - 1].i
            to_i = key_steps[i].i
            tweens, values = Tween.create_in_between_steps(key_steps, i, data, from_i, to_i)
            log_utils.debug(f"Creating {len(tweens)} tweens ({from_i}->{to_i}) for key frame at {key_steps[i].i}")
            key_steps[i].tweens = tweens
            key_steps[i].tween_values = values
            key_steps[i].render_data.indexes.update_tween_start(data.turbo)
        return key_steps
