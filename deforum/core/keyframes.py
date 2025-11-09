"""Core keyframe scheduling and interpolation classes.

This module contains the fundamental keyframe system that powers Deforum's
animation scheduling. It handles parsing of schedule strings (e.g., "0:(1.0), 30:(1.5)")
and interpolation between keyframes to create smooth animations.

The keyframe system supports:
- Numeric schedules with interpolation (Linear, Quadratic, Cubic)
- String schedules for model/sampler switching
- Expression evaluation using numexpr (e.g., "0:(sin(t/10))")
- Dynamic schedule generation from UI arguments

Classes:
    FrameInterpolater: Core parsing and interpolation engine
    DeformAnimKeys: Animation parameter schedules (movement, effects, etc.)
    ControlNetKeys: ControlNet weight and guidance schedules
    LooperAnimKeys: Guided image/looper schedules
"""

import re
from typing import Any, Dict
import numpy as np
import numexpr
import pandas as pd

# Import pure utilities
from deforum.utils.parsing.strings import sanitize_keyframe_value

# Optional imports with fallback for testing
try:
    from deforum.utils.generation.prompts import check_is_number
except ImportError:
    # Fallback for unit tests
    def check_is_number(x: str) -> bool:
        """Check if string represents a number."""
        return x.replace('.', '').replace('-', '').isdigit()

try:
    from modules import shared
except ImportError:
    # Fallback for unit tests
    shared = None


class FrameInterpolater:
    """Parse keyframe schedules and interpolate values between keyframes.

    This class handles the core logic of Deforum's keyframe system:
    1. Parse schedule strings like "0:(1.0), 30:(2.0), 60:(1.5)"
    2. Evaluate mathematical expressions using numexpr (e.g., "0:(sin(t/10))")
    3. Interpolate between keyframes using Linear/Quadratic/Cubic methods
    4. Handle both numeric and string schedules

    Attributes:
        max_frames: Total number of frames in the animation
        seed: Random seed for expression evaluation (available as 's' variable)

    Examples:
        >>> fi = FrameInterpolater(max_frames=100, seed=42)
        >>> series = fi.parse_inbetweens("0:(1.0), 50:(2.0), 100:(1.0)")
        >>> series[25]  # Value at frame 25
        1.5

        >>> # Expression evaluation
        >>> series = fi.parse_inbetweens("0:(sin(t/10)*2 + 1)")
        >>> # Uses numexpr with variables: t=frame, max_f=max_frames-1, s=seed
    """

    def __init__(self, max_frames: int = 0, seed: int = -1) -> None:
        """Initialize frame interpolater.

        Args:
            max_frames: Total number of frames to interpolate
            seed: Random seed for expression evaluation
        """
        self.max_frames = max_frames
        self.seed = seed

    def parse_inbetweens(
        self,
        value: str,
        filename: str = 'unknown',
        is_single_string: bool = False
    ) -> pd.Series:
        """Parse schedule string and return interpolated series.

        This is the main entry point for schedule parsing. It combines
        parse_key_frames() and get_inbetweens() into a single call.

        Args:
            value: Schedule string (e.g., "0:(1.0), 30:(2.0)")
            filename: Name for error reporting
            is_single_string: If True, treat values as strings (no interpolation)

        Returns:
            Pandas Series with one value per frame

        Raises:
            RuntimeError: If schedule string is malformed
            SyntaxError: If expression evaluation fails

        Examples:
            >>> fi = FrameInterpolater(max_frames=100)
            >>> fi.parse_inbetweens("0:(1.0), 100:(2.0)")
            0      1.000000
            1      1.010101
            ...
            100    2.000000
            Length: 100, dtype: float64
        """
        key_frames = self.parse_key_frames(value, filename=filename)
        return self.get_inbetweens(
            key_frames,
            filename=filename,
            is_single_string=is_single_string
        )

    def sanitize_value(self, value: str) -> str:
        """Sanitize keyframe value by removing quotes and whitespace.

        Wrapper for backward compatibility - delegates to pure function.

        Args:
            value: Raw keyframe value

        Returns:
            Sanitized value
        """
        return sanitize_keyframe_value(value)

    def get_inbetweens(
        self,
        key_frames: Dict[int, str],
        integer: bool = False,
        interp_method: str = 'Linear',
        is_single_string: bool = False,
        filename: str = 'unknown'
    ) -> pd.Series:
        """Interpolate values between keyframes.

        Takes a dictionary of keyframes and creates a full series with
        interpolated values for every frame. Supports expression evaluation
        and multiple interpolation methods.

        Args:
            key_frames: Dict mapping frame number to value expression
            integer: If True, return integer series
            interp_method: 'Linear', 'Quadratic', or 'Cubic'
            is_single_string: If True, replicate strings without interpolation
            filename: Name for error reporting

        Returns:
            Pandas Series with interpolated values

        Raises:
            SyntaxError: If expression evaluation fails

        Notes:
            - Expressions have access to: t (frame), max_f (last frame), s (seed)
            - Cubic requires 4+ keyframes, Quadratic requires 3+
            - Falls back to simpler interpolation if not enough keyframes
        """
        key_frame_series = pd.Series([np.nan for a in range(self.max_frames)])

        # Set up variables for numexpr evaluation
        max_f = self.max_frames - 1
        s = self.seed

        # Evaluate expressions at keyframes
        for i in range(0, self.max_frames):
            value_is_number = False
            if i in key_frames:
                value = key_frames[i]
                sanitized_value = self.sanitize_value(value)
                value_is_number = check_is_number(sanitized_value)
                if value_is_number:
                    key_frame_series[i] = float(sanitized_value)

            if not value_is_number:
                t = i
                # Evaluate expression or use string value
                try:
                    key_frame_series[i] = (
                        numexpr.evaluate(value) if not is_single_string
                        else float(sanitized_value)
                    )
                except SyntaxError as e:
                    e.filename = f"{filename}@frame#{i}"
                    raise e
            elif is_single_string:
                # Replicate previous string value
                key_frame_series[i] = key_frame_series[i-1]

        # Convert to appropriate type
        key_frame_series = (
            key_frame_series.astype(float) if not is_single_string
            else key_frame_series
        )

        # Downgrade interpolation method if not enough keyframes
        if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
            interp_method = 'Quadratic'
        if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
            interp_method = 'Linear'

        # Fill endpoints and interpolate
        key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
        key_frame_series[self.max_frames-1] = key_frame_series[
            key_frame_series.last_valid_index()
        ]
        key_frame_series = key_frame_series.interpolate(
            method=interp_method.lower(),
            limit_direction='both'
        )

        if integer:
            return key_frame_series.astype(int)
        return key_frame_series

    def parse_key_frames(self, string: str, filename: str = 'unknown') -> Dict[int, str]:
        """Parse keyframe schedule string into frame→value dictionary.

        Parses schedule strings in format: "frame:(value), frame:(value), ..."
        Supports mathematical expressions for both frame numbers and values.

        Args:
            string: Schedule string to parse
            filename: Name for error reporting

        Returns:
            Dictionary mapping frame numbers to value expressions

        Raises:
            RuntimeError: If string is malformed
            SyntaxError: If frame number expression evaluation fails

        Examples:
            >>> fi = FrameInterpolater(max_frames=100, seed=42)
            >>> fi.parse_key_frames("0:(1.0), 50:(2.0), 100:(1.5)")
            {0: '1.0', 50: '2.0', 100: '1.5'}

            >>> # Expression in frame number
            >>> fi.parse_key_frames("0:(1.0), max_f/2:(2.0)")
            {0: '1.0', 49: '2.0'}
        """
        frames = dict()
        max_f = self.max_frames - 1
        s = self.seed

        for match_object in string.split(","):
            frameParam = match_object.split(":")
            try:
                # Evaluate frame number (may be expression like "max_f/2")
                frame_str = self.sanitize_value(frameParam[0].strip())
                if check_is_number(frame_str):
                    frame = int(frame_str)
                else:
                    # Remove quotes and evaluate expression
                    cleaned = frame_str.replace("'", "", 1).replace('"', "", 1)
                    cleaned = cleaned[::-1].replace("'", "", 1).replace('"', "", 1)[::-1]
                    frame = int(numexpr.evaluate(cleaned))

                frames[frame] = frameParam[1].strip()
            except SyntaxError as e:
                e.filename = filename
                raise e

        if frames == {} and len(string) != 0:
            raise RuntimeError('Key Frame string not correctly formatted')

        return frames


class DeformAnimKeys:
    """Animation parameter keyframe schedules.

    Parses all animation schedules from UI arguments into interpolated series.
    This includes motion parameters (translation, rotation, zoom), rendering
    parameters (steps, cfg_scale, seed), and effect parameters (noise, contrast).

    Each schedule is parsed from a string like "0:(1.0), 30:(2.0)" and stored
    as a pandas Series with interpolated values for every frame.

    Attributes:
        fi: FrameInterpolater instance used for all parsing
        angle_series: 2D rotation angle schedule
        zoom_series: Zoom factor schedule
        translation_x/y/z_series: Translation schedules
        rotation_3d_x/y/z_series: 3D rotation schedules
        perspective_flip_*_series: Perspective flip parameters
        strength_schedule_series: Denoising strength schedule
        cfg_scale_schedule_series: CFG scale schedule
        seed_schedule_series: Seed schedule
        ... and many more (see source for complete list)

    Examples:
        >>> keys = DeformAnimKeys(anim_args, seed=42)
        >>> keys.zoom_series[25]  # Get zoom at frame 25
        1.5
    """

    def __init__(self, anim_args: Any, seed: int = -1):
        """Initialize animation keyframe schedules.

        Args:
            anim_args: Animation arguments object with schedule strings
            seed: Random seed for expression evaluation
        """
        self.fi = FrameInterpolater(anim_args.max_frames, seed)

        # Motion parameters
        self.angle_series = self.fi.parse_inbetweens(anim_args.angle, 'angle')
        self.transform_center_x_series = self.fi.parse_inbetweens(
            anim_args.transform_center_x, 'transform_center_x'
        )
        self.transform_center_y_series = self.fi.parse_inbetweens(
            anim_args.transform_center_y, 'transform_center_y'
        )
        self.zoom_series = self.fi.parse_inbetweens(anim_args.zoom, 'zoom')
        self.translation_x_series = self.fi.parse_inbetweens(
            anim_args.translation_x, 'translation_x'
        )
        self.translation_y_series = self.fi.parse_inbetweens(
            anim_args.translation_y, 'translation_y'
        )
        self.translation_z_series = self.fi.parse_inbetweens(
            anim_args.translation_z, 'translation_z'
        )
        self.rotation_3d_x_series = self.fi.parse_inbetweens(
            anim_args.rotation_3d_x, 'rotation_3d_x'
        )
        self.rotation_3d_y_series = self.fi.parse_inbetweens(
            anim_args.rotation_3d_y, 'rotation_3d_y'
        )
        self.rotation_3d_z_series = self.fi.parse_inbetweens(
            anim_args.rotation_3d_z, 'rotation_3d_z'
        )

        # Perspective flip
        self.perspective_flip_theta_series = self.fi.parse_inbetweens(
            anim_args.perspective_flip_theta, 'perspective_flip_theta'
        )
        self.perspective_flip_phi_series = self.fi.parse_inbetweens(
            anim_args.perspective_flip_phi, 'perspective_flip_phi'
        )
        self.perspective_flip_gamma_series = self.fi.parse_inbetweens(
            anim_args.perspective_flip_gamma, 'perspective_flip_gamma'
        )
        self.perspective_flip_fv_series = self.fi.parse_inbetweens(
            anim_args.perspective_flip_fv, 'perspective_flip_fv'
        )

        # Rendering parameters
        self.noise_schedule_series = self.fi.parse_inbetweens(
            anim_args.noise_schedule, 'noise_schedule'
        )
        self.strength_schedule_series = self.fi.parse_inbetweens(
            anim_args.strength_schedule, 'strength_schedule'
        )
        self.keyframe_strength_schedule_series = self.fi.parse_inbetweens(
            anim_args.keyframe_strength_schedule, 'keyframe_strength_schedule'
        )
        self.contrast_schedule_series = self.fi.parse_inbetweens(
            anim_args.contrast_schedule, 'contrast_schedule'
        )
        self.cfg_scale_schedule_series = self.fi.parse_inbetweens(
            anim_args.cfg_scale_schedule, 'cfg_scale_schedule'
        )
        self.distilled_cfg_scale_schedule_series = self.fi.parse_inbetweens(
            anim_args.distilled_cfg_scale_schedule, 'distilled_cfg_scale_schedule'
        )
        self.ddim_eta_schedule_series = self.fi.parse_inbetweens(
            anim_args.ddim_eta_schedule, 'ddim_eta_schedule'
        )
        self.ancestral_eta_schedule_series = self.fi.parse_inbetweens(
            anim_args.ancestral_eta_schedule, 'ancestral_eta_schedule'
        )

        # Seed and sampler schedules
        self.subseed_schedule_series = self.fi.parse_inbetweens(
            anim_args.subseed_schedule, 'subseed_schedule'
        )
        self.subseed_strength_schedule_series = self.fi.parse_inbetweens(
            anim_args.subseed_strength_schedule, 'subseed_strength_schedule'
        )
        self.checkpoint_schedule_series = self.fi.parse_inbetweens(
            anim_args.checkpoint_schedule, 'checkpoint_schedule', is_single_string=True
        )
        self.steps_schedule_series = self.fi.parse_inbetweens(
            anim_args.steps_schedule, 'steps_schedule'
        )
        self.seed_schedule_series = self.fi.parse_inbetweens(
            anim_args.seed_schedule, 'seed_schedule'
        )
        self.sampler_schedule_series = self.fi.parse_inbetweens(
            anim_args.sampler_schedule, 'sampler_schedule', is_single_string=True
        )
        self.scheduler_schedule_series = self.fi.parse_inbetweens(
            anim_args.scheduler_schedule, 'scheduler_schedule', is_single_string=True
        )
        self.clipskip_schedule_series = self.fi.parse_inbetweens(
            anim_args.clipskip_schedule, 'clipskip_schedule'
        )
        self.noise_multiplier_schedule_series = self.fi.parse_inbetweens(
            anim_args.noise_multiplier_schedule, 'noise_multiplier_schedule'
        )

        # Mask and effect schedules
        self.mask_schedule_series = self.fi.parse_inbetweens(
            anim_args.mask_schedule, 'mask_schedule', is_single_string=True
        )
        self.noise_mask_schedule_series = self.fi.parse_inbetweens(
            anim_args.noise_mask_schedule, 'noise_mask_schedule', is_single_string=True
        )
        self.kernel_schedule_series = self.fi.parse_inbetweens(
            anim_args.kernel_schedule, 'kernel_schedule'
        )
        self.sigma_schedule_series = self.fi.parse_inbetweens(
            anim_args.sigma_schedule, 'sigma_schedule'
        )
        self.amount_schedule_series = self.fi.parse_inbetweens(
            anim_args.amount_schedule, 'amount_schedule'
        )
        self.threshold_schedule_series = self.fi.parse_inbetweens(
            anim_args.threshold_schedule, 'threshold_schedule'
        )

        # Camera parameters
        self.aspect_ratio_series = self.fi.parse_inbetweens(
            anim_args.aspect_ratio_schedule, 'aspect_ratio_schedule'
        )
        self.fov_series = self.fi.parse_inbetweens(anim_args.fov_schedule, 'fov_schedule')
        self.near_series = self.fi.parse_inbetweens(anim_args.near_schedule, 'near_schedule')
        self.far_series = self.fi.parse_inbetweens(anim_args.far_schedule, 'far_schedule')

        # Flow and cadence
        self.cadence_flow_factor_schedule_series = self.fi.parse_inbetweens(
            anim_args.cadence_flow_factor_schedule, 'cadence_flow_factor_schedule'
        )
        self.redo_flow_factor_schedule_series = self.fi.parse_inbetweens(
            anim_args.redo_flow_factor_schedule, 'redo_flow_factor_schedule'
        )

        # Keyframe type
        self.keyframe_type_schedule_series = self.fi.parse_inbetweens(
            anim_args.keyframe_type_schedule, 'keyframe_type_schedule', is_single_string=True
        )


class ControlNetKeys:
    """ControlNet weight and guidance keyframe schedules.

    Dynamically parses ControlNet schedules for multiple models based on the
    configured number of ControlNet units. Each unit has weight, guidance_start,
    and guidance_end schedules.

    Attributes:
        fi: FrameInterpolater instance used for all parsing
        schedules: Dict mapping schedule names to pandas Series
        cn_N_weight_schedule_series: Weight schedule for unit N
        cn_N_guidance_start_schedule_series: Guidance start for unit N
        cn_N_guidance_end_schedule_series: Guidance end for unit N

    Examples:
        >>> keys = ControlNetKeys(anim_args, controlnet_args)
        >>> keys.cn_1_weight_schedule_series[25]  # Weight for unit 1 at frame 25
        0.8
    """

    def __init__(self, anim_args: Any, controlnet_args: Any):
        """Initialize ControlNet keyframe schedules.

        Args:
            anim_args: Animation arguments with max_frames
            controlnet_args: ControlNet arguments with schedule strings
        """
        self.fi = FrameInterpolater(max_frames=anim_args.max_frames)
        self.schedules = {}

        # Get max ControlNet units from settings
        if shared is not None:
            max_models = shared.opts.data.get(
                "control_net_unit_count",
                shared.opts.data.get("control_net_max_models_num", 5)
            )
        else:
            max_models = 5

        num_of_models = 5 if max_models <= 5 else max_models

        # Parse schedules for each unit
        for i in range(1, num_of_models + 1):
            for suffix in ['weight', 'guidance_start', 'guidance_end']:
                prefix = f"cn_{i}"
                input_key = f"{prefix}_{suffix}"
                output_key = f"{input_key}_schedule_series"

                schedule_value = getattr(controlnet_args, input_key)
                self.schedules[output_key] = self.fi.parse_inbetweens(
                    schedule_value, input_key
                )
                setattr(self, output_key, self.schedules[output_key])


class LooperAnimKeys:
    """Guided image/looper keyframe schedules.

    Parses schedules for the guided images / looper feature, which allows
    blending reference images into the animation at specified strengths.

    Attributes:
        fi: FrameInterpolater instance used for all parsing
        use_looper: Whether looper is enabled
        imagesToKeyframe: Dictionary of images to use
        image_strength_schedule_series: Image influence schedule
        blendFactorMax_series: Max blend factor schedule
        blendFactorSlope_series: Blend slope schedule
        tweening_frames_schedule_series: Tweening frames schedule
        color_correction_factor_series: Color correction schedule

    Examples:
        >>> keys = LooperAnimKeys(loop_args, anim_args, seed=42)
        >>> keys.image_strength_schedule_series[25]
        0.5
    """

    def __init__(self, loop_args: Any, anim_args: Any, seed: int):
        """Initialize looper keyframe schedules.

        Args:
            loop_args: Looper arguments with schedule strings
            anim_args: Animation arguments with max_frames
            seed: Random seed for expression evaluation
        """
        self.fi = FrameInterpolater(anim_args.max_frames, seed)

        self.use_looper = loop_args.use_looper
        self.imagesToKeyframe = loop_args.init_images

        self.image_strength_schedule_series = self.fi.parse_inbetweens(
            loop_args.image_strength_schedule, 'image_strength_schedule'
        )
        self.image_keyframe_strength_schedule_series = self.fi.parse_inbetweens(
            loop_args.image_keyframe_strength_schedule, 'image_keyframe_strength_schedule'
        )
        self.blendFactorMax_series = self.fi.parse_inbetweens(
            loop_args.blendFactorMax, 'blendFactorMax'
        )
        self.blendFactorSlope_series = self.fi.parse_inbetweens(
            loop_args.blendFactorSlope, 'blendFactorSlope'
        )
        self.tweening_frames_schedule_series = self.fi.parse_inbetweens(
            loop_args.tweening_frames_schedule, 'tweening_frames_schedule'
        )
        self.color_correction_factor_series = self.fi.parse_inbetweens(
            loop_args.color_correction_factor, 'color_correction_factor'
        )
