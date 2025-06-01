"""
Immutable Animation Schedule Models

This module provides immutable dataclasses for animation schedules to replace
the mutable objects with setattr patterns in animation_key_frames.py and 
parseq_adapter.py.

Replaces patterns like:
    setattr(self, output_key, self.schedules[output_key])
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

# Backward compatibility alias for tests
try:
    from .animation_key_frames import FrameInterpolater
except ImportError:
    # Fallback if module not available
    FrameInterpolater = None


@dataclass(frozen=True)
class AnimationSchedules:
    """
    Immutable animation schedules to replace DeformAnimKeys class.
    
    Replaces mutable pattern in animation_key_frames.py where schedules are
    created as attributes using setattr.
    """
    # Movement schedules
    angle_series: Tuple[float, ...] = field(default_factory=tuple)
    zoom_series: Tuple[float, ...] = field(default_factory=tuple)
    translation_x_series: Tuple[float, ...] = field(default_factory=tuple)
    translation_y_series: Tuple[float, ...] = field(default_factory=tuple)
    translation_z_series: Tuple[float, ...] = field(default_factory=tuple)
    rotation_3d_x_series: Tuple[float, ...] = field(default_factory=tuple)
    rotation_3d_y_series: Tuple[float, ...] = field(default_factory=tuple)
    rotation_3d_z_series: Tuple[float, ...] = field(default_factory=tuple)
    
    # Transform center
    transform_center_x_series: Tuple[float, ...] = field(default_factory=tuple)
    transform_center_y_series: Tuple[float, ...] = field(default_factory=tuple)
    
    # Perspective flip
    perspective_flip_theta_series: Tuple[float, ...] = field(default_factory=tuple)
    perspective_flip_phi_series: Tuple[float, ...] = field(default_factory=tuple)
    perspective_flip_gamma_series: Tuple[float, ...] = field(default_factory=tuple)
    perspective_flip_fv_series: Tuple[float, ...] = field(default_factory=tuple)
    
    # Quality schedules
    noise_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    strength_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    keyframe_strength_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    contrast_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    cfg_scale_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    distilled_cfg_scale_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    
    # Sampling schedules
    steps_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    seed_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    sampler_schedule_series: Tuple[str, ...] = field(default_factory=tuple)
    scheduler_schedule_series: Tuple[str, ...] = field(default_factory=tuple)
    
    # Advanced schedules
    ddim_eta_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    ancestral_eta_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    subseed_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    subseed_strength_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    checkpoint_schedule_series: Tuple[str, ...] = field(default_factory=tuple)
    clipskip_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    noise_multiplier_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    
    # Mask schedules
    mask_schedule_series: Tuple[str, ...] = field(default_factory=tuple)
    noise_mask_schedule_series: Tuple[str, ...] = field(default_factory=tuple)
    
    # Processing schedules
    kernel_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    sigma_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    amount_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    threshold_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    
    # View schedules
    aspect_ratio_series: Tuple[float, ...] = field(default_factory=tuple)
    fov_series: Tuple[float, ...] = field(default_factory=tuple)
    near_series: Tuple[float, ...] = field(default_factory=tuple)
    far_series: Tuple[float, ...] = field(default_factory=tuple)
    
    # Optical flow schedules
    cadence_flow_factor_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    redo_flow_factor_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    
    @classmethod
    def from_anim_args(cls, anim_args: Any, max_frames: int = 100, seed: int = -1) -> 'AnimationSchedules':
        """
        Pure factory method to create schedules from animation args.
        
        Replaces the mutable DeformAnimKeys.__init__ pattern.
        """
        fi = FrameInterpolater(max_frames, seed)
        
        return cls(
            angle_series=tuple(fi.parse_inbetweens(anim_args.angle, 'angle')),
            zoom_series=tuple(fi.parse_inbetweens(anim_args.zoom, 'zoom')),
            translation_x_series=tuple(fi.parse_inbetweens(anim_args.translation_x, 'translation_x')),
            translation_y_series=tuple(fi.parse_inbetweens(anim_args.translation_y, 'translation_y')),
            translation_z_series=tuple(fi.parse_inbetweens(anim_args.translation_z, 'translation_z')),
            rotation_3d_x_series=tuple(fi.parse_inbetweens(anim_args.rotation_3d_x, 'rotation_3d_x')),
            rotation_3d_y_series=tuple(fi.parse_inbetweens(anim_args.rotation_3d_y, 'rotation_3d_y')),
            rotation_3d_z_series=tuple(fi.parse_inbetweens(anim_args.rotation_3d_z, 'rotation_3d_z')),
            transform_center_x_series=tuple(fi.parse_inbetweens(anim_args.transform_center_x, 'transform_center_x')),
            transform_center_y_series=tuple(fi.parse_inbetweens(anim_args.transform_center_y, 'transform_center_y')),
            perspective_flip_theta_series=tuple(fi.parse_inbetweens(anim_args.perspective_flip_theta, 'perspective_flip_theta')),
            perspective_flip_phi_series=tuple(fi.parse_inbetweens(anim_args.perspective_flip_phi, 'perspective_flip_phi')),
            perspective_flip_gamma_series=tuple(fi.parse_inbetweens(anim_args.perspective_flip_gamma, 'perspective_flip_gamma')),
            perspective_flip_fv_series=tuple(fi.parse_inbetweens(anim_args.perspective_flip_fv, 'perspective_flip_fv')),
            noise_schedule_series=tuple(fi.parse_inbetweens(anim_args.noise_schedule, 'noise_schedule')),
            strength_schedule_series=tuple(fi.parse_inbetweens(anim_args.strength_schedule, 'strength_schedule')),
            keyframe_strength_schedule_series=tuple(fi.parse_inbetweens(anim_args.keyframe_strength_schedule, 'keyframe_strength_schedule')),
            contrast_schedule_series=tuple(fi.parse_inbetweens(anim_args.contrast_schedule, 'contrast_schedule')),
            cfg_scale_schedule_series=tuple(fi.parse_inbetweens(anim_args.cfg_scale_schedule, 'cfg_scale_schedule')),
            distilled_cfg_scale_schedule_series=tuple(fi.parse_inbetweens(anim_args.distilled_cfg_scale_schedule, 'distilled_cfg_scale_schedule')),
            steps_schedule_series=tuple(fi.parse_inbetweens(anim_args.steps_schedule, 'steps_schedule')),
            seed_schedule_series=tuple(fi.parse_inbetweens(anim_args.seed_schedule, 'seed_schedule')),
            sampler_schedule_series=tuple(fi.parse_inbetweens(anim_args.sampler_schedule, 'sampler_schedule', is_single_string=True)),
            scheduler_schedule_series=tuple(fi.parse_inbetweens(anim_args.scheduler_schedule, 'scheduler_schedule', is_single_string=True)),
            ddim_eta_schedule_series=tuple(fi.parse_inbetweens(anim_args.ddim_eta_schedule, 'ddim_eta_schedule')),
            ancestral_eta_schedule_series=tuple(fi.parse_inbetweens(anim_args.ancestral_eta_schedule, 'ancestral_eta_schedule')),
            subseed_schedule_series=tuple(fi.parse_inbetweens(anim_args.subseed_schedule, 'subseed_schedule')),
            subseed_strength_schedule_series=tuple(fi.parse_inbetweens(anim_args.subseed_strength_schedule, 'subseed_strength_schedule')),
            checkpoint_schedule_series=tuple(fi.parse_inbetweens(anim_args.checkpoint_schedule, 'checkpoint_schedule', is_single_string=True)),
            clipskip_schedule_series=tuple(fi.parse_inbetweens(anim_args.clipskip_schedule, 'clipskip_schedule')),
            noise_multiplier_schedule_series=tuple(fi.parse_inbetweens(anim_args.noise_multiplier_schedule, 'noise_multiplier_schedule')),
            mask_schedule_series=tuple(fi.parse_inbetweens(anim_args.mask_schedule, 'mask_schedule', is_single_string=True)),
            noise_mask_schedule_series=tuple(fi.parse_inbetweens(anim_args.noise_mask_schedule, 'noise_mask_schedule', is_single_string=True)),
            kernel_schedule_series=tuple(fi.parse_inbetweens(anim_args.kernel_schedule, 'kernel_schedule')),
            sigma_schedule_series=tuple(fi.parse_inbetweens(anim_args.sigma_schedule, 'sigma_schedule')),
            amount_schedule_series=tuple(fi.parse_inbetweens(anim_args.amount_schedule, 'amount_schedule')),
            threshold_schedule_series=tuple(fi.parse_inbetweens(anim_args.threshold_schedule, 'threshold_schedule')),
            aspect_ratio_series=tuple(fi.parse_inbetweens(anim_args.aspect_ratio_schedule, 'aspect_ratio_schedule')),
            fov_series=tuple(fi.parse_inbetweens(anim_args.fov_schedule, 'fov_schedule')),
            near_series=tuple(fi.parse_inbetweens(anim_args.near_schedule, 'near_schedule')),
            far_series=tuple(fi.parse_inbetweens(anim_args.far_schedule, 'far_schedule')),
            cadence_flow_factor_schedule_series=tuple(fi.parse_inbetweens(anim_args.cadence_flow_factor_schedule, 'cadence_flow_factor_schedule')),
            redo_flow_factor_schedule_series=tuple(fi.parse_inbetweens(anim_args.redo_flow_factor_schedule, 'redo_flow_factor_schedule'))
        )


@dataclass(frozen=True)
class ControlNetSchedules:
    """
    Immutable ControlNet schedules to replace the setattr pattern in ControlNetKeys.
    """
    schedules: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    
    @classmethod
    def from_args(cls, anim_args: Any, controlnet_args: Any, max_models: int = 5) -> 'ControlNetSchedules':
        """
        Pure factory method to create ControlNet schedules.
        
        Replaces the mutable ControlNetKeys.__init__ pattern with setattr.
        """
        fi = FrameInterpolater(max_frames=anim_args.max_frames)
        schedules = {}
        
        for i in range(1, max_models + 1):
            for suffix in ['weight', 'guidance_start', 'guidance_end']:
                prefix = f"cn_{i}"
                input_key = f"{prefix}_{suffix}"
                output_key = f"{input_key}_schedule_series"
                
                try:
                    input_value = getattr(controlnet_args, input_key, "0: (1.0)")
                    schedule_data = tuple(fi.parse_inbetweens(input_value, input_key))
                    schedules[output_key] = schedule_data
                except AttributeError:
                    # Default schedule if attribute doesn't exist
                    schedules[output_key] = tuple([1.0] * anim_args.max_frames)
        
        return cls(schedules=schedules)
    
    def get_schedule(self, schedule_name: str) -> Optional[Tuple[float, ...]]:
        """Get a specific schedule by name"""
        return self.schedules.get(schedule_name)


@dataclass(frozen=True)
class FreeUSchedules:
    """
    Immutable FreeU schedules to replace FreeUAnimKeys.
    """
    freeu_enabled: bool = False
    freeu_b1_series: Tuple[float, ...] = field(default_factory=tuple)
    freeu_b2_series: Tuple[float, ...] = field(default_factory=tuple)
    freeu_s1_series: Tuple[float, ...] = field(default_factory=tuple)
    freeu_s2_series: Tuple[float, ...] = field(default_factory=tuple)
    
    @classmethod
    def from_args(cls, anim_args: Any, freeu_args: Any) -> 'FreeUSchedules':
        """Pure factory method to create FreeU schedules"""
        fi = FrameInterpolater(max_frames=anim_args.max_frames)
        
        # Get defaults
        try:
            from .args import FreeUArgs
            defaults = FreeUArgs()
        except ImportError:
            defaults = {}
        
        return cls(
            freeu_enabled=getattr(freeu_args, 'freeu_enabled', False),
            freeu_b1_series=tuple(fi.parse_inbetweens(
                getattr(freeu_args, 'freeu_b1', defaults.get('freeu_b1', {}).get('value', '0: (1.0)')),
                'freeu_args.b1'
            )),
            freeu_b2_series=tuple(fi.parse_inbetweens(
                getattr(freeu_args, 'freeu_b2', defaults.get('freeu_b2', {}).get('value', '0: (1.0)')),
                'freeu_args.b2'
            )),
            freeu_s1_series=tuple(fi.parse_inbetweens(
                getattr(freeu_args, 'freeu_s1', defaults.get('freeu_s1', {}).get('value', '0: (1.0)')),
                'freeu_args.s1'
            )),
            freeu_s2_series=tuple(fi.parse_inbetweens(
                getattr(freeu_args, 'freeu_s2', defaults.get('freeu_s2', {}).get('value', '0: (1.0)')),
                'freeu_args.s2'
            ))
        )


@dataclass(frozen=True)
class KohyaSchedules:
    """
    Immutable Kohya HR Fix schedules to replace KohyaHRFixAnimKeys.
    """
    kohya_hrfix_enabled: bool = False
    block_number_series: Tuple[float, ...] = field(default_factory=tuple)
    downscale_factor_series: Tuple[float, ...] = field(default_factory=tuple)
    start_percent_series: Tuple[float, ...] = field(default_factory=tuple)
    end_percent_series: Tuple[float, ...] = field(default_factory=tuple)
    
    @classmethod
    def from_args(cls, anim_args: Any, kohya_hrfix_args: Any) -> 'KohyaSchedules':
        """Pure factory method to create Kohya schedules"""
        fi = FrameInterpolater(max_frames=anim_args.max_frames)
        
        # Get defaults
        try:
            from .args import KohyaHRFixArgs
            defaults = KohyaHRFixArgs()
        except ImportError:
            defaults = {}
        
        return cls(
            kohya_hrfix_enabled=getattr(kohya_hrfix_args, 'kohya_hrfix_enabled', False),
            block_number_series=tuple(fi.parse_inbetweens(
                getattr(kohya_hrfix_args, 'kohya_hrfix_block_number', 
                       defaults.get('kohya_hrfix_block_number', {}).get('value', '0: (2)')),
                'kohya_hrfix.block_number'
            )),
            downscale_factor_series=tuple(fi.parse_inbetweens(
                getattr(kohya_hrfix_args, 'kohya_hrfix_downscale_factor',
                       defaults.get('kohya_hrfix_downscale_factor', {}).get('value', '0: (2.0)')),
                'kohya_hrfix.downscale_factor'
            )),
            start_percent_series=tuple(fi.parse_inbetweens(
                getattr(kohya_hrfix_args, 'kohya_hrfix_start_percent',
                       defaults.get('kohya_hrfix_start_percent', {}).get('value', '0: (0.0)')),
                'kohya_hrfix.start_percent'
            )),
            end_percent_series=tuple(fi.parse_inbetweens(
                getattr(kohya_hrfix_args, 'kohya_hrfix_end_percent',
                       defaults.get('kohya_hrfix_end_percent', {}).get('value', '0: (1.0)')),
                'kohya_hrfix.end_percent'
            ))
        )


@dataclass(frozen=True)
class LooperSchedules:
    """
    Immutable looper schedules to replace LooperAnimKeys.
    """
    use_looper: bool = False
    images_to_keyframe: Tuple[str, ...] = field(default_factory=tuple)
    image_strength_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    image_keyframe_strength_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    blend_factor_max_series: Tuple[float, ...] = field(default_factory=tuple)
    blend_factor_slope_series: Tuple[float, ...] = field(default_factory=tuple)
    tweening_frames_schedule_series: Tuple[float, ...] = field(default_factory=tuple)
    color_correction_factor_series: Tuple[float, ...] = field(default_factory=tuple)
    
    @classmethod
    def from_args(cls, loop_args: Any, anim_args: Any, seed: int = -1) -> 'LooperSchedules':
        """Pure factory method to create looper schedules"""
        fi = FrameInterpolater(anim_args.max_frames, seed)
        
        return cls(
            use_looper=getattr(loop_args, 'use_looper', False),
            images_to_keyframe=tuple(getattr(loop_args, 'init_images', [])),
            image_strength_schedule_series=tuple(fi.parse_inbetweens(
                getattr(loop_args, 'image_strength_schedule', '0: (0.75)'), 
                'image_strength_schedule'
            )),
            image_keyframe_strength_schedule_series=tuple(fi.parse_inbetweens(
                getattr(loop_args, 'image_keyframe_strength_schedule', '0: (0.50)'),
                'image_keyframe_strength_schedule'
            )),
            blend_factor_max_series=tuple(fi.parse_inbetweens(
                getattr(loop_args, 'blendFactorMax', '0: (0.35)'),
                'blendFactorMax'
            )),
            blend_factor_slope_series=tuple(fi.parse_inbetweens(
                getattr(loop_args, 'blendFactorSlope', '0: (0.25)'),
                'blendFactorSlope'
            )),
            tweening_frames_schedule_series=tuple(fi.parse_inbetweens(
                getattr(loop_args, 'tweening_frames_schedule', '0: (20)'),
                'tweening_frames_schedule'
            )),
            color_correction_factor_series=tuple(fi.parse_inbetweens(
                getattr(loop_args, 'color_correction_factor', '0: (0.075)'),
                'color_correction_factor'
            ))
        )


@dataclass(frozen=True)
class ParseqScheduleData:
    """
    Immutable Parseq schedule data to replace setattr pattern in parseq_adapter.py.
    
    Replaces patterns like:
        setattr(inst, name, definedField)
    """
    frame_data: Dict[str, Any] = field(default_factory=dict)
    use_deltas: bool = True
    max_frames: int = 100
    
    def get_schedule_series(self, name: str) -> Optional[Tuple[float, ...]]:
        """
        Pure function to extract schedule series from frame data.
        
        Replaces the dynamic parseq_to_series method.
        """
        # Check if value is present in first frame
        if not self.frame_data:
            return None
            
        first_frame = self.frame_data[0] if self.frame_data else {}
        if name not in first_frame:
            return None
        
        # Build series
        key_frame_series = [np.nan] * self.max_frames
        
        for frame in self.frame_data:
            frame_idx = frame.get('frame', 0)
            if frame_idx < self.max_frames and name in frame:
                key_frame_series[frame_idx] = frame[name]
        
        # Convert to pandas series and interpolate
        series = pd.Series(key_frame_series)
        series = series.interpolate(method='linear', limit_direction='both')
        
        return tuple(series.values)
    
    @classmethod
    def from_parseq_frames(cls, frames: List[Dict[str, Any]], use_deltas: bool = True, max_frames: int = 100) -> 'ParseqScheduleData':
        """Create ParseqScheduleData from raw frame data"""
        return cls(
            frame_data=frames,
            use_deltas=use_deltas,
            max_frames=max_frames
        ) 