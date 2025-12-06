from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from deforum.rendering.data.render_data import RenderData
from deforum.rendering import img_2_img_tubes
from deforum.rendering.helpers import image as image_utils
from deforum.utils.system.logging import log as log_utils
from deforum.rendering.helpers import turbo as turbo_utils
from deforum.rendering.helpers import webui as web_ui_utils
from deforum.rendering.calls.subtitle import call_write_subtitle_from_to


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

        # Store tween for DA3 Gaussian splatting scene building (if 'all' mode)
        if hasattr(data, 'generated_keyframes'):
            collection_mode = getattr(data.args.anim_args, 'da3_3dgs_frame_collection', 'keyframes')
            max_frames_limit = getattr(data.args.anim_args, 'da3_3dgs_max_frames', 30)
            current_count = len(data.generated_keyframes)

            # Only store tweens in 'all' mode
            if collection_mode == 'all' and current_count < max_frames_limit:
                import numpy as np
                # Convert saved_image (OpenCV BGR) to numpy if needed
                if isinstance(saved_image, np.ndarray):
                    image_np = saved_image
                else:
                    image_np = np.array(saved_image)

                frame_data = {
                    'frame_idx': self.i,
                    'image': image_np,
                    'depth': self.depth,
                    'seed': last_frame.seed,  # Tweens inherit seed from parent keyframe
                    'is_keyframe': False,
                    'is_tween': True
                }
                data.generated_keyframes.append(frame_data)

        # updating reference images for next iteration
        data.images.before_previous = data.images.previous
        data.images.previous = saved_image
        data.args.root.init_sample = saved_image

    def _generate(self, data, last_frame, prev_image):
        # Check tween generation mode
        tween_mode = getattr(data.args.anim_args, 'tween_generation_mode', 'depth_warp')

        if tween_mode == 'da3_multiview':
            # Phase 2: Multi-view tween generation
            return self._generate_multiview(data, last_frame, prev_image)
        elif tween_mode == 'da3_gaussian':
            # Phase 3: 3D Gaussian Splatting scene rendering
            return self._generate_gaussian(data, last_frame, prev_image)

        # Standard depth warp pipeline (default)
        advanced_image = turbo_utils.advance_optical_flow_cadence_before_animation_warping(
            data, last_frame, self, data.images.before_previous, data.images.previous)
        # Temporarily calculate depth from input for warping
        temp_depth = Tween.calculate_depth_prediction(data, advanced_image)
        processed_image = img_2_img_tubes.process_tween_tube(data, last_frame, self.i, temp_depth)(advanced_image)
        warped = turbo_utils.do_optical_flow_cadence_after_animation_warping(data, self, prev_image, processed_image)

        grayscale_tube = img_2_img_tubes.conditional_force_tween_to_grayscale_tube
        recolored = grayscale_tube(data)(warped)

        is_tween = True
        overlay_mask_tube = img_2_img_tubes.conditional_add_overlay_mask_tube
        masked = overlay_mask_tube(data, is_tween)(recolored)

        # CRITICAL FIX: Calculate depth from the FINAL OUTPUT, not the input
        # Next frame needs the depth of THIS frame's actual output to warp correctly
        self.depth = Tween.calculate_depth_prediction(data, masked)

        return masked

    def _generate_multiview(self, data, last_frame, prev_image):
        """Generate tween using DA3 multi-view geometry (Phase 2)."""
        try:
            from deforum.rendering.tween_generators.da3_multiview import DA3MultiViewTweenGenerator

            # Create generator with current depth model
            generator = DA3MultiViewTweenGenerator(data.depth_model)

            # Get keyframe images
            prev_keyframe = data.images.before_previous if data.images.before_previous is not None else prev_image
            next_keyframe = data.images.previous

            # Generate tween using multi-view geometry
            tween_image = generator.generate_tween(
                data,
                self,  # Tween frame with value (0-1)
                prev_keyframe,
                next_keyframe,
                depth=None  # Multi-view doesn't use pre-computed depth
            )

            # Apply post-processing (grayscale, masks)
            grayscale_tube = img_2_img_tubes.conditional_force_tween_to_grayscale_tube
            recolored = grayscale_tube(data)(tween_image)

            is_tween = True
            overlay_mask_tube = img_2_img_tubes.conditional_add_overlay_mask_tube
            masked = overlay_mask_tube(data, is_tween)(recolored)

            # Calculate depth from final output
            self.depth = Tween.calculate_depth_prediction(data, masked)

            return masked

        except ImportError as e:
            log_utils.error(f"DA3 multi-view generator not available: {e}")
            log_utils.warning("Falling back to standard depth warp")
            # Fall back to standard generation
            return self._generate_standard_depth_warp(data, last_frame, prev_image)
        except Exception as e:
            log_utils.error(f"DA3 multi-view tween generation failed: {e}")
            log_utils.warning("Falling back to standard depth warp")
            return self._generate_standard_depth_warp(data, last_frame, prev_image)

    def _generate_gaussian(self, data, last_frame, prev_image):
        """Generate tween using 3D Gaussian Splatting (Phase 3)."""
        try:
            from deforum.rendering.tween_generators.da3_gaussian import DA3GaussianTweenGenerator

            # Check if we've already attempted scene building
            if hasattr(data, 'gaussian_scene_build_attempted') and data.gaussian_scene_build_attempted:
                # Scene building was already attempted and failed - fall back
                if data.gaussian_scene is None:
                    return self._generate_standard_depth_warp(data, last_frame, prev_image)

            # Check if 3DGS scene is already built
            if not hasattr(data, 'gaussian_scene') or data.gaussian_scene is None:
                # Mark that we're attempting scene building (prevents retries on failure)
                data.gaussian_scene_build_attempted = True

                log_utils.info("3D Gaussian scene not built yet, building now...")
                # Collect all keyframes from diffusion_frames
                # NOTE: This should ideally be done once after all keyframes are generated
                # For now, we build on first tween (inefficient but functional)
                keyframes = self._collect_keyframes(data)
                if len(keyframes) < 2:
                    log_utils.warning("Not enough keyframes for 3DGS, need at least 2. Falling back to depth warp.")
                    return self._generate_standard_depth_warp(data, last_frame, prev_image)

                # Build 3DGS scene
                generator = DA3GaussianTweenGenerator(data.depth_model, keyframes)
                data.gaussian_scene = generator.build_scene(keyframes, data.animation_keys)
                data.gaussian_generator = generator  # Store generator for later use

                if data.gaussian_scene is None:
                    log_utils.error("3DGS scene building failed, falling back to depth warp for all remaining tweens")
                    return self._generate_standard_depth_warp(data, last_frame, prev_image)

            # Generate tween using existing 3DGS scene
            generator = data.gaussian_generator
            tween_image = generator.generate_tween(
                data,
                self,  # Tween frame with value (0-1)
                prev_image,
                data.images.previous,
                depth=None  # 3DGS doesn't use pre-computed depth
            )

            # Apply post-processing (grayscale, masks)
            grayscale_tube = img_2_img_tubes.conditional_force_tween_to_grayscale_tube
            recolored = grayscale_tube(data)(tween_image)

            is_tween = True
            overlay_mask_tube = img_2_img_tubes.conditional_add_overlay_mask_tube
            masked = overlay_mask_tube(data, is_tween)(recolored)

            # Calculate depth from final output
            self.depth = Tween.calculate_depth_prediction(data, masked)

            return masked

        except ImportError as e:
            log_utils.error(f"DA3 Gaussian generator not available: {e}")
            log_utils.warning("Falling back to standard depth warp")
            return self._generate_standard_depth_warp(data, last_frame, prev_image)
        except Exception as e:
            log_utils.error(f"DA3 Gaussian tween generation failed: {e}")
            import traceback
            log_utils.debug(traceback.format_exc())
            log_utils.warning("Falling back to standard depth warp")
            return self._generate_standard_depth_warp(data, last_frame, prev_image)

    def _collect_keyframes(self, data):
        """Collect generated frames for 3DGS scene building.

        Collection strategy based on da3_3dgs_frame_collection setting:
        - 'keyframes': Keyframes only (minimal, default)
        - 'diffusion': Keyframes + non-key diffusion frames (New 3D mode)
        - 'all': Keyframes + diffusion + tweens (maximum quality)

        Returns:
            List of numpy array images (BGR format) for DA3 3DGS processing
        """
        if hasattr(data, 'generated_keyframes'):
            frame_dicts = data.generated_keyframes
            # Extract just the images from the frame dicts
            # DA3 expects List[np.ndarray], not List[dict]
            images = [kf['image'] for kf in frame_dicts]

            # Count frame types for logging
            keyframe_count = sum(1 for kf in frame_dicts if kf.get('is_keyframe', False))
            tween_count = sum(1 for kf in frame_dicts if kf.get('is_tween', False))
            diffusion_count = len(images) - tween_count  # All non-tweens are diffusion frames

            # Get collection mode
            collection_mode = getattr(data.args.anim_args, 'da3_3dgs_frame_collection', 'keyframes')

            # Log what we collected
            if collection_mode == 'keyframes':
                log_utils.info(f"Collected {len(images)} keyframes for 3DGS scene building")
            elif collection_mode == 'diffusion':
                log_utils.info(
                    f"Collected {len(images)} diffusion frames for 3DGS scene building "
                    f"({keyframe_count} keyframes + {diffusion_count - keyframe_count} cadence)"
                )
            elif collection_mode == 'all':
                log_utils.info(
                    f"Collected {len(images)} total frames for 3DGS scene building "
                    f"({keyframe_count} keyframes + {diffusion_count - keyframe_count} diffusion + {tween_count} tweens)"
                )

            return images
        else:
            log_utils.warning("No frames collected yet - generated_keyframes not initialized")
            return []

    def _generate_standard_depth_warp(self, data, last_frame, prev_image):
        """Standard depth warp pipeline (extracted for fallback)."""
        advanced_image = turbo_utils.advance_optical_flow_cadence_before_animation_warping(
            data, last_frame, self, data.images.before_previous, data.images.previous)
        temp_depth = Tween.calculate_depth_prediction(data, advanced_image)
        processed_image = img_2_img_tubes.process_tween_tube(data, last_frame, self.i, temp_depth)(advanced_image)
        warped = turbo_utils.do_optical_flow_cadence_after_animation_warping(data, self, prev_image, processed_image)

        grayscale_tube = img_2_img_tubes.conditional_force_tween_to_grayscale_tube
        recolored = grayscale_tube(data)(warped)

        is_tween = True
        overlay_mask_tube = img_2_img_tubes.conditional_add_overlay_mask_tube
        masked = overlay_mask_tube(data, is_tween)(recolored)

        self.depth = Tween.calculate_depth_prediction(data, masked)
        return masked

    def handle_synchronous_status_concerns(self, data):
        log_utils.print_tween_frame_info(data, self.i, self.cadence_flow, self.value)
        web_ui_utils.update_progress_during_cadence(data, self.i)

    def write_tween_subtitle_from_to(self, data: RenderData, sub_i, previous_diffusion_frame, from_time, to_time):
        # Cadence can be asserted because subtitle generation
        # skips the last tween in favor of its parent diffusion frame.
        is_cadence = True
        # With 0-based indexing, use frame index directly
        frame_index = self.i
        # Since tween frames are not diffused, they don't have their own seed.
        # We provide the seed of the previous frame that was diffused (parent diffusion frame has the next seed).
        # Since the 1st frame is always diffused it never has tweens, meaning there's always a previous_diffusion_frame.
        seed = previous_diffusion_frame.seed
        subseed = previous_diffusion_frame.subseed
        call_write_subtitle_from_to(data, sub_i, frame_index, is_cadence, seed, subseed, from_time, to_time)

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
            return data.depth_model.predict(
                image,
                use_ray_pose=data.args.anim_args.da3_use_ray_pose,
                conf_thresh_percentile=data.args.anim_args.da3_conf_thresh_percentile
            )
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
