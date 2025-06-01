import copy
import json
import logging
import operator
from operator import itemgetter
import numpy as np
import pandas as pd
import requests
import os
from .animation_key_frames import DeformAnimKeys, ControlNetKeys, LooperAnimKeys
from .rendering.util import log_utils
from .rich import console
from .general_utils import tickOrCross


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

IGNORED_FIELDS = ['fi', 'use_looper', 'imagesToKeyframe', 'schedules']

class ParseqAdapter():
    def __init__(self, parseq_args, anim_args, video_args, controlnet_args, loop_args, mute=False):
        self.parseq_args = parseq_args
        self.anim_args = anim_args
        self.video_args = video_args
        self.controlnet_args = controlnet_args
        self.loop_args = loop_args
        self.mute = mute
        
        self.use_parseq = parseq_args.parseq_manifest != None and parseq_args.parseq_manifest.strip()
        
        if self.use_parseq:
            # Initialize parseq
            self.parseq_json = self.load_parseq_json()
            self.rendered_frames = self.parseq_json.get('rendered_frames', [])
            self.frame_count = len(self.rendered_frames)
            
            # Create key objects with parseq decorators
            self.anim_keys = ParseqAnimKeysDecorator(self, DeformAnimKeys(anim_args, anim_args.seed))
            self.controlnet_keys = ParseqControlNetKeysDecorator(self, ControlNetKeys(anim_args, controlnet_args)) if controlnet_args else None
            self.looper_keys = ParseqLooperKeysDecorator(self, LooperAnimKeys(loop_args, anim_args, anim_args.seed)) if loop_args else None
            
            if not self.mute:
                log_utils.info(f"Using Parseq: {tickOrCross(self.use_parseq)}")
        else:
            self.anim_keys = None
            self.controlnet_keys = None
            self.looper_keys = None
            
    def load_parseq_json(self):
        """Load and validate parseq JSON manifest"""
        try:
            if os.path.exists(self.parseq_args.parseq_manifest):
                with open(self.parseq_args.parseq_manifest, 'r') as f:
                    return json.load(f)
            else:
                # Try to parse as JSON string
                return json.loads(self.parseq_args.parseq_manifest)
        except Exception as e:
            log_utils.error(f"Failed to load parseq manifest: {e}")
            return {}
    
    def manages_prompts(self):
        """Check if parseq manages prompts"""
        return self.use_parseq and any('prompt' in frame for frame in self.rendered_frames)
    
    def manages_seed(self):
        """Check if parseq manages seed"""
        return self.use_parseq and any('seed' in frame for frame in self.rendered_frames)


class ParseqAbstractDecorator():  

    def __init__(self, adapter: ParseqAdapter, fallback_keys):
        self.adapter = adapter
        self.fallback_keys = fallback_keys

    def parseq_to_series(self, seriesName):
        
        # Check if value is present in first frame of JSON data. If not, assume it's undefined.
        # The Parseq contract is that the first frame (at least) must define values for all fields.
        try:
            if self.adapter.rendered_frames[0][seriesName] is not None:
                logging.debug(f"Found {seriesName} in first frame of Parseq data. Assuming it's defined.")
        except KeyError:
            return None
        
        required_frames = self.adapter.anim_args.max_frames

        key_frame_series = pd.Series([np.nan for a in range(required_frames)])
        
        for frame in self.adapter.rendered_frames:
            frame_idx = frame['frame']
            if frame_idx < required_frames:                
                if not np.isnan(key_frame_series[frame_idx]):
                    logging.warning(f"Duplicate frame definition {frame_idx} detected for data {seriesName}. Latest wins.")        
                key_frame_series[frame_idx] = frame[seriesName]

        # If the animation will have more frames than Parseq defines,
        # duplicate final value to match the required frame count.
        while (frame_idx < required_frames):
            key_frame_series[frame_idx] = operator.itemgetter(-1)(self.adapter.rendered_frames)[seriesName]
            frame_idx += 1

        return key_frame_series

    # fallback to anim_args if the series is not defined in the Parseq data
    def __getattribute__(inst, name):
        try:
            definedField = super(ParseqAbstractDecorator, inst).__getattribute__(name)
        except AttributeError:
            # No field with this name has been explicitly extracted from the JSON data.
            # It must be a new parameter. Let's see if it's in the raw JSON.

            parseqName = inst.strip_suffixes(name)
            
            # returns None if not defined in Parseq JSON data
            definedField = inst.parseq_to_series(parseqName)
            if (definedField is not None):
                # add the field to the instance so we don't compute it again.
                setattr(inst, name, definedField)

        if (definedField is not None):
            return definedField
        else:
            logging.debug(f"Data for {name} not defined in Parseq data. Falling back to standard Deforum values.")
            return getattr(inst.fallback_keys, name)

    
    # parseq doesn't use _series, _schedule or _schedule_series suffixes in the
    # JSON data - remove them.        
    def strip_suffixes(self, name):
        strippableSuffixes = ['_series', '_schedule']
        parseqName = name
        while any(parseqName.endswith(suffix) for suffix in strippableSuffixes):
            for suffix in strippableSuffixes:
                if parseqName.endswith(suffix):
                    parseqName = parseqName[:-len(suffix)]
        return parseqName
    
    # parseq prefixes some field names for clarity. These prefixes are not present in the original Deforum names.
    def strip_parseq_prefixes(self, name):
        strippablePrefixes = ['guided_']
        parseqName = name
        while any(parseqName.startswith(prefix) for prefix in strippablePrefixes):
            for prefix in strippablePrefixes:
                if parseqName.startswith(prefix):
                    parseqName = parseqName[len(prefix):]
        return parseqName    
    
    def all_parseq_fields(self):
        return [self.strip_parseq_prefixes(field) for field in self.adapter.rendered_frames[0].keys() if (not field.endswith('_delta') and not field.endswith('_pc'))]

    def managed_fields(self):
        all_parseq_fields = self.all_parseq_fields()
        deforum_fields = [self.strip_suffixes(property) for property, _ in vars(self.fallback_keys).items() if property not in IGNORED_FIELDS and not property.startswith('_')]
        return [field for field in deforum_fields if field in all_parseq_fields]

    def unmanaged_fields(self):
        all_parseq_fields = self.all_parseq_fields()
        deforum_fields = [self.strip_suffixes(property) for property, _ in vars(self.fallback_keys).items() if property not in IGNORED_FIELDS and not property.startswith('_')]
        return [field for field in deforum_fields if field not in all_parseq_fields]


class ParseqControlNetKeysDecorator(ParseqAbstractDecorator):
    def __init__(self, adapter: ParseqAdapter, cn_keys):
        super().__init__(adapter, cn_keys)


class ParseqLooperKeysDecorator(ParseqAbstractDecorator):
    def __init__(self, adapter: ParseqAdapter, looper_keys):
        super().__init__(adapter, looper_keys)


class ParseqAnimKeysDecorator(ParseqAbstractDecorator):
    def __init__(self, adapter: ParseqAdapter, anim_keys):
        super().__init__(adapter, anim_keys)

        # Parseq treats input values as absolute values. So if you want to 
        # progressively rotate 180 degrees over 4 frames, you specify: 45, 90, 135, 180.
        # However, many animation parameters are relative to the previous frame if there is enough
        # loopback strength. So if you want to rotate 180 degrees over 5 frames, the animation engine expects:
        # 45, 45, 45, 45. Therefore, for such parameter, we use the fact that Parseq supplies delta values.
        optional_delta = '_delta' if self.adapter.use_parseq else ''
        self.angle_series = super().parseq_to_series('angle' + optional_delta)
        self.zoom_series = super().parseq_to_series('zoom' + optional_delta)        
        self.translation_x_series = super().parseq_to_series('translation_x' + optional_delta)
        self.translation_y_series = super().parseq_to_series('translation_y' + optional_delta)
        self.translation_z_series = super().parseq_to_series('translation_z' + optional_delta)
        self.rotation_3d_x_series = super().parseq_to_series('rotation_3d_x' + optional_delta)
        self.rotation_3d_y_series = super().parseq_to_series('rotation_3d_y' + optional_delta)
        self.rotation_3d_z_series = super().parseq_to_series('rotation_3d_z' + optional_delta)
        self.perspective_flip_theta_series = super().parseq_to_series('perspective_flip_theta' + optional_delta)
        self.perspective_flip_phi_series = super().parseq_to_series('perspective_flip_phi' + optional_delta)
        self.perspective_flip_gamma_series = super().parseq_to_series('perspective_flip_gamma' + optional_delta)
 
        # Non-motion animation args - never use deltas for these.
        self.perspective_flip_fv_series = super().parseq_to_series('perspective_flip_fv')
        self.noise_schedule_series = super().parseq_to_series('noise')
        self.strength_schedule_series = super().parseq_to_series('strength')
        # TODO not implemented
        # self.keyframe_strength_schedule_series = super().parseq_to_series('keyframe_strength')
        self.sampler_schedule_series = super().parseq_to_series('sampler_schedule')
        # TODO implement scheduler_schedule in Parseq
        # self.scheduler_schedule_series = super().parseq_to_series('scheduler_schedule')  # Not implemented
        self.contrast_schedule_series = super().parseq_to_series('contrast')
        self.cfg_scale_schedule_series = super().parseq_to_series('scale')
        # TODO implement distilled_scale schedule in Parseq
        # self.distilled_cfg_scale_schedule_series = super().parseq_to_series('distilled_scale')  # Not implemented
        self.steps_schedule_series = super().parseq_to_series("steps_schedule")
        self.seed_schedule_series = super().parseq_to_series('seed')
        self.fov_series = super().parseq_to_series('fov')
        self.near_series = super().parseq_to_series('near')
        self.far_series = super().parseq_to_series('far')
        self.subseed_schedule_series = super().parseq_to_series('subseed')
        self.subseed_strength_schedule_series = super().parseq_to_series('subseed_strength')
        self.kernel_schedule_series = super().parseq_to_series('antiblur_kernel')
        self.sigma_schedule_series = super().parseq_to_series('antiblur_sigma')
        self.amount_schedule_series = super().parseq_to_series('antiblur_amount')
        self.threshold_schedule_series = super().parseq_to_series('antiblur_threshold')

        # TODO - move to a different decorator?
        self.prompts = super().parseq_to_series('deforum_prompt') # formatted as "{positive} --neg {negative}"

