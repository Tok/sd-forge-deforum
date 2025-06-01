import os
from dataclasses import dataclass
from math import ceil
from typing import Any, List, Optional

import PIL
import cv2
import numpy as np
from PIL import Image
# noinspection PyUnresolvedReferences
from modules import shared
from tqdm import tqdm

from . import DiffusionFrameData, KeyFrameDistribution
from .tween_frame import Tween
from .. import RenderData, Schedule
from ..taqaddumat import Taqaddumat
from ... import img_2_img_tubes
from ...util import depth_utils, filename_utils, log_utils, utils
from ...util.call.anim import call_anim_frame_warp
from ...util.call.gen import call_generate
from ...util.call.images import call_add_noise
from ...util.call.mask import call_compose_mask_with_check, call_unsharp_mask
from ...util.call.subtitle import call_write_subtitle_from_to
from ...util.call.video_and_audio import call_render_preview
from ....colors import maintain_colors
from ...util import image_utils


@dataclass(init=True, frozen=False, repr=False, eq=False)
class DiffusionFrame:
    """DiffusionFrames are the frames that actually get diffused (as opposed to Tween frames)."""
    i: int
    is_keyframe: bool
    seed: int  # TODO avoid reassignment after creation:
    subseed: int
    subseed_strength: float
    strength: float
    frame_data: DiffusionFrameData  # immutable collection of less essential frame data.
    schedule: Schedule  # assigned after the seed is available
    depth: Any  # assigned during generation
    last_preview_frame: int
    tweens: List[Tween]

    def actual_steps(self, data):
        if self.i == 1 and not data.args.args.use_init:
            return self.schedule.steps
        return int(ceil(self.schedule.steps * self.strength)) + 1

    def has_tween_frames(self):
        return len(self.tweens) > 0

    def is_optical_flow_redo_before_generation(self, optical_flow_redo_generation, images):
        has_flow_redo = optical_flow_redo_generation != 'None'
        return has_flow_redo and images.has_previous() and self.has_strength()

    def write_subtitle_from_to(self, data: RenderData, subtitle_index, frame_i, from_time, to_time):
        # Non-cadence can be asserted because subtitle creation gives priority to diffusion frames over tween ones.
        is_cadence = False
        call_write_subtitle_from_to(data, subtitle_index, frame_i, is_cadence,
                                    self.seed, self.subseed, from_time, to_time)

    def apply_frame_warp_transform(self, data: RenderData, image):
        is_not_last_frame = self.i < data.args.anim_args.max_frames
        if is_not_last_frame:
            previous, self.depth = call_anim_frame_warp(data, self.i, image, None)
            return previous

    def apply_scaling(self, image):
        return (image * self.frame_data.contrast).round().astype(np.uint8)

    def apply_anti_blur(self, data: RenderData, image):
        if self.frame_data.amount > 0:
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

    def create_color_match_for_video(self, data: RenderData):
        if data.args.anim_args.color_coherence == 'Video Input':
            # Note: hybrid video functionality removed
            pass
        return None

    def _generate_and_update_noise(self, data: RenderData, image, contrasted_noise_tube):
        noised_image = contrasted_noise_tube(data, self)(image)
        data.update_sample_and_args_for_current_progression_step(self, noised_image)
        return image  # return original as passed.

    def transform_and_update_noised_sample(self, data: RenderData, frame_tube, contrasted_noise_tube):
        if data.images.has_previous():  # skipping 1st iteration
            transformed_image = frame_tube(data, self)(data.images.previous)
            if transformed_image is None:
                log_utils.warn("Image transformation failed, using fallback.")
                transformed_image = data.images.previous
            return self._generate_and_update_noise(data, transformed_image, contrasted_noise_tube)
        return None

    def prepare_generation(self, data: RenderData, frame_tube, contrasted_noise_tube):
        data.images.color_match = self.create_color_match_for_video(data)
        data.images.previous = self.transform_and_update_noised_sample(data, frame_tube, contrasted_noise_tube)
        data.prepare_generation(data, self, self.i)
        self.maybe_redo_optical_flow(data)
        self.maybe_redo_diffusion(data)

    # Conditional Redoes
    def maybe_redo_optical_flow(self, data: RenderData):
        optical_flow_redo_generation = data.optical_flow_redo_generation_if_not_in_preview_mode()
        is_redo_optical_flow = self.is_optical_flow_redo_before_generation(optical_flow_redo_generation, data.images)
        if is_redo_optical_flow:
            data.args.root.init_sample = self.do_optical_flow_redo_before_generation(data)

    def maybe_redo_diffusion(self, data: RenderData):
        is_pos_redo = data.has_positive_diffusion_redo
        is_diffusion_redo = is_pos_redo and data.images.has_previous() and self.has_strength()
        is_not_preview = data.is_not_in_motion_preview_mode()
        if is_diffusion_redo and is_not_preview:
            self.do_diffusion_redo(data)

    def has_strength(self):
        return self.strength > 0

    def generate(self, data: RenderData, total_tqdm):
        image = call_generate(data, self)
        total_tqdm.increment_animation_cycle_count()
        return image

    def after_diffusion(self, data: RenderData, image):
        data.images.color_match = img_2_img_tubes.conditional_color_match_tube(data, self)(image)
        self.progress_and_save(data, image)
        self.update_render_preview(data)

    def progress_and_save(self, data: RenderData, image):
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
        self.depth = depth_utils.generate_and_save_depth_map_if_active(data, opencv_image, self.i)

    def update_render_preview(self, data: RenderData):
        self.last_preview_frame = call_render_preview(data, self.last_preview_frame, self.i)

    def do_optical_flow_redo_before_generation(self, data):
        redo = data.args.anim_args.optical_flow_redo_generation
        stored_seed = data.args.args.seed  # keep original to reset it after executing the optical flow
        random_seed = utils.generate_random_seed()  # create and set a new random seed
        log_utils.print_optical_flow_info(data, redo, random_seed)

        sample_image = call_generate(data, self, random_seed)
        optical_tube = img_2_img_tubes.optical_flow_redo_tube(data, self, redo)
        transformed_sample_image = optical_tube(sample_image)

        data.args.args.seed = stored_seed  # restore stored seed
        return Image.fromarray(transformed_sample_image)

    def do_diffusion_redo(self, data: RenderData):
        last_diffusion_redo_index = int(data.args.anim_args.diffusion_redo)
        for n in range(0, last_diffusion_redo_index):
            log_utils.print_redo_generation_info(data, n)
            diffusion_redo_image = call_generate(data, self, utils.generate_random_seed())
            diffusion_redo_image = cv2.cvtColor(np.array(diffusion_redo_image), cv2.COLOR_RGB2BGR)
            # color match on last one only
            is_last_iteration = n == last_diffusion_redo_index
            if is_last_iteration:
                mode = data.args.anim_args.color_coherence
                diffusion_redo_image = maintain_colors(data.images.previous, data.images.color_match, mode)
            data.args.root.init_sample = Image.fromarray(cv2.cvtColor(diffusion_redo_image, cv2.COLOR_BGR2RGB))

    def tween_values(self):
        return list(map(lambda _: _.value, self.tweens))

    @staticmethod
    def create(data: RenderData, i):
        initial_index = i  # replaced once keyframes are arranged.
        frame_data = DiffusionFrameData.create(data, initial_index)
        return DiffusionFrame(initial_index, False, -1, -1, 1.0, 0.0, frame_data, None, "", 0, list())

    @staticmethod
    def apply_color_matching(data: RenderData, image):
        return DiffusionFrame.apply_color_coherence(image, data) if data.has_color_coherence() else image

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
    def precalculate_diffusion_frame_count(data: RenderData, keyframe_distribution, start_index, max_frames):
        # TODO change implementation so KeyFrames can be instantiated without any pre-calculations.
        if keyframe_distribution is KeyFrameDistribution.OFF:
            return 0  # not relevant
        elif keyframe_distribution is KeyFrameDistribution.KEYFRAMES_ONLY:
            return (len(data.parseq_adapter.parseq_json['keyframes'])
                    if data.parseq_adapter.use_parseq
                    else len(data.args.root.prompt_keyframes) + 1)  # +1 because last frame is not defined in prompts
        elif keyframe_distribution is KeyFrameDistribution.REDISTRIBUTED:
            return 1 + int((data.args.anim_args.max_frames - start_index) / data.cadence())
        elif keyframe_distribution is KeyFrameDistribution.ADDITIVE:
            # TODO refactor to make pre-calc obsolete.
            temp_diffusion_frame_count = 1 + int((data.args.anim_args.max_frames - start_index) / data.cadence())
            uniform_indices = KeyFrameDistribution.uniform_indexes(start_index, max_frames, temp_diffusion_frame_count)
            keyframes = KeyFrameDistribution.select_keyframes(data)

            precalculated_diffusion_frames = list(set(set(uniform_indices) | set(keyframes)))
            return len(precalculated_diffusion_frames)
        else:
            raise ValueError(f"Invalid keyframe_distribution: {keyframe_distribution}")

    @staticmethod
    def create_all_frames(data: RenderData, keyframe_dist: KeyFrameDistribution = KeyFrameDistribution.default()):
        """Creates a list of key steps for the entire animation."""
        start_index = 0
        max_frames = (data.args.anim_args.max_frames
                      if not data.parseq_adapter.use_parseq
                      else data.args.anim_args.max_frames - 1)
        diffusion_frame_count = DiffusionFrame.precalculate_diffusion_frame_count(
            data, keyframe_dist, start_index, max_frames)

        diffusion_frames = [DiffusionFrame.create(data, i) for i in range(0, diffusion_frame_count)]
        actual_diffusion_frame_count = len(diffusion_frames)
        recalculated_frames = DiffusionFrame._recalculate_and_check_tweens(
            data, diffusion_frames, start_index, actual_diffusion_frame_count, keyframe_dist)
        log_utils.print_tween_frame_creation_info(diffusion_frames, keyframe_dist)
        return recalculated_frames

    @staticmethod
    def _recalculate_and_check_tweens(data: RenderData, diffusion_frames, start_index, diffusion_frame_count,
                                      keyframe_distribution):
        # TODO? move everything here into KeyFrame.create
        assert diffusion_frame_count == len(diffusion_frames)  # FIXME? calculate instead of pass diffusion_frame_count

        key_indices = keyframe_distribution.calculate(data, start_index, diffusion_frame_count)
        assert len(diffusion_frames) == len(key_indices)

        for i, key_i in enumerate(key_indices):
            # TODO separate handling from calculation. this should be done on init.
            is_kf = KeyFrameDistribution.is_deforum_keyframe(data, key_i)
            diffusion_frames[i].i = key_i
            diffusion_frames[i].is_keyframe = is_kf
            diffusion_frames[i].strength = DiffusionFrame._select_keyframe_or_cadence_strength(data, key_i, is_kf)

        diffusion_frames = DiffusionFrame.add_tweens_to_diffusion_frames(diffusion_frames)
        log_utils.print_key_frame_debug_info_if_verbose(diffusion_frames)

        pseudo_cadence = data.args.anim_args.max_frames / len(diffusion_frames)
        log_utils.info(f"Calculated pseudo cadence: {log_utils.ORANGE}{pseudo_cadence:.2f}{log_utils.RESET_COLOR}")

        # The number of generated tweens depends on index since last diffusion_frame. The last tween has the same
        # index as the diffusion_frame it belongs to and is meant to replace the unprocessed original key frame.
        assert len(diffusion_frames) == diffusion_frame_count
        assert diffusion_frames[0].i == 1  # 1st diffusion frame is at index 1
        assert diffusion_frames[0].tweens == []  # 1st diffusion frame has no tweens
        if keyframe_distribution != KeyFrameDistribution.KEYFRAMES_ONLY:
            assert diffusion_frames[-1].i == data.args.anim_args.max_frames  # last index is same as max frames

        DiffusionFrame._assign_initial_seeds_and_schedules(data, diffusion_frames)

        return diffusion_frames

    @staticmethod
    def _select_keyframe_or_cadence_strength(data: RenderData, index, is_keyframe):
        # Applies `strength_schedule` if Parseq is active or if there is no entry with index i in the Deforum prompts.
        # otherwise `keyframe_strength_schedule` is applied, which should be set lower (=more denoise on keyframes).
        # schedule series indices shifted to start at 0.
        keys = data.animation_keys.deform_keys
        idx = index - 1
        
        # Check if index is within bounds
        if idx >= len(keys.strength_schedule_series) or idx >= len(keys.keyframe_strength_schedule_series):
            log_utils.warn(f"Frame index {index} (0-indexed: {idx}) exceeds strength schedule series length, using default value.")
            return 0.85  # Default strength value
        
        return (keys.strength_schedule_series[idx]
                if data.parseq_adapter.use_parseq or is_keyframe
                else keys.keyframe_strength_schedule_series[idx])

    @staticmethod
    def _assign_initial_seeds_and_schedules(data: RenderData, diffusion_frames):
        log_utils.info(f"Precalculating {len(diffusion_frames)} seeds with behaviour '{data.args.args.seed_behavior}'.")
        is_seed_managed_by_parseq = data.parseq_adapter.manages_seed()
        if is_seed_managed_by_parseq:
            DiffusionFrame.process_parseq_diffusion_frames(diffusion_frames, data)
        else:
            DiffusionFrame.process_deforum_diffusion_frames(diffusion_frames, data)

    @staticmethod
    def process_parseq_diffusion_frames(diffusion_frames, data):
        data.args.anim_args.enable_subseed_scheduling = True
        keys = data.animation_keys.deform_keys  # Parseq keys are decorated in 'ParseqAnimKeysDecorator'
        for diffusion_frame in tqdm(diffusion_frames, desc='Precalculations', unit='diffusion-frames',
                                    dynamic_ncols=True, file=shared.progress_print_out,
                                    bar_format=Taqaddumat.NO_ETA_BAR_FORMAT,
                                    disable=shared.cmd_opts.disable_console_progressbars,
                                    colour=log_utils.HEX_YELLOW):
            i = diffusion_frame.i - 1
            # Ensure i is within bounds of the series
            if i >= len(keys.seed_schedule_series):
                log_utils.warn(f"Frame index {diffusion_frame.i} (0-indexed: {i}) exceeds max_frames, using fallback values.")
                diffusion_frame.seed = data.args.args.seed
                diffusion_frame.subseed = -1
                diffusion_frame.subseed_strength = 0.0
            else:
                diffusion_frame.seed = int(keys.seed_schedule_series[i])
                diffusion_frame.subseed = int(keys.subseed_schedule_series[i])
                diffusion_frame.subseed_strength = float(keys.subseed_strength_schedule_series[i])
            
            # Always use a safe index for schedule creation
            safe_i = min(i, len(keys.steps_schedule_series)-1)
            diffusion_frame.schedule = Schedule.create(data, diffusion_frame.seed, safe_i)

    @staticmethod
    def process_deforum_diffusion_frames(diffusion_frames, data):
        behavior = data.args.args.seed_behavior

        def _start_seed():
            return data.args.args.seed + (-1 if behavior == 'iter' else (1 if behavior == 'ladder' else 0))

        def _start_control():
            return 0 if behavior != 'ladder' and behavior != 'alternate' else 1

        last_seed = _start_seed()
        last_seed_control = _start_control()  # same as 'data.args.root.seed_internal', but it's passed directly.
        for diffusion_frame in tqdm(diffusion_frames, desc='Precalculations', unit='diffusion-frames',
                                    dynamic_ncols=True, file=shared.progress_print_out,
                                    bar_format=Taqaddumat.NO_ETA_BAR_FORMAT,
                                    disable=shared.cmd_opts.disable_console_progressbars,
                                    colour=log_utils.HEX_YELLOW):
            DiffusionFrame._assign_subseed_properties(data, diffusion_frame, False)
            the_next_seed, the_next_seed_control = generate_next_seed(data.args.args, last_seed, last_seed_control)
            log_utils.debug(f"Seed {the_next_seed:010}. " +
                            (f"Subseed {diffusion_frame.subseed:010} at {diffusion_frame.subseed_strength}."
                             if diffusion_frame.subseed != -1 else ""))
            diffusion_frame.seed = the_next_seed
            
            # Ensure the frame index is within bounds for schedule creation
            i = diffusion_frame.i
            keys = data.animation_keys.deform_keys
            if i >= len(keys.steps_schedule_series):
                log_utils.warn(f"Frame index {i} exceeds max_frames length, using safe index for schedule creation.")
                safe_i = min(i, len(keys.steps_schedule_series)-1)
                diffusion_frame.schedule = Schedule.create(data, diffusion_frame.seed, safe_i)
            else:
                diffusion_frame.schedule = Schedule.create(data, diffusion_frame.seed, diffusion_frame.i)
            last_seed = the_next_seed
            last_seed_control = the_next_seed_control

    @staticmethod
    def _assign_subseed_properties(data: RenderData, diffusion_frame, is_seed_managed_by_parseq):
        keys = data.animation_keys.deform_keys
        is_subseed_scheduling_enabled = data.args.anim_args.enable_subseed_scheduling
        if is_subseed_scheduling_enabled:
            i = diffusion_frame.i - 1
            # Check if index is within bounds
            if i >= len(keys.subseed_schedule_series):
                log_utils.warn(f"Frame index {diffusion_frame.i} (0-indexed: {i}) exceeds subseed schedule series length, using default values.")
                diffusion_frame.subseed = -1
                diffusion_frame.subseed_strength = 0.0
            else:
                diffusion_frame.subseed = int(keys.subseed_schedule_series[i])
                diffusion_frame.subseed_strength = float(keys.subseed_strength_schedule_series[i])

    @staticmethod
    def add_tweens_to_diffusion_frames(diffusion_frames):
        log_utils.debug(f"Adding tweens to {len(diffusion_frames)} keyframes...")
        for i in range(1, len(diffusion_frames)):  # skipping 1st key frame
            from_i = diffusion_frames[i - 1].i
            to_i = diffusion_frames[i].i
            tweens = Tween.create_in_between_steps(diffusion_frames, i, from_i, to_i)
            from_to = f"({from_i:05}->{to_i:05})"
            log_utils.debug(f"Creating {len(tweens):03} tweens {from_to} for frame #{diffusion_frames[i].i:09}")
            diffusion_frames[i].tweens = tweens
        return diffusion_frames
