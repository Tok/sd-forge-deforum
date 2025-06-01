from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.interpolate import CubicSpline

from .enum.shakify_key import ShakifyKey as Key
from .enum.xyz import Xyz
from .shake_data import SHAKE_LIST
from ....utils import log_utils
from ....config.defaults import get_camera_shake_list

FRAME = 'frame'
VALUE = 'value'
COLUMNS = [FRAME, VALUE]


@dataclass(init=True, frozen=True, repr=False, eq=False)
class Shaker:
    shake_name: str
    shake_intensity: float
    shake_speed: float
    transform: dict[str, dict[str, DataFrame]]
    is_enabled: bool = True

    def get_data(self, transform_type: str, axis: str, i: int):
        """
        Parameters:
        - transform_type (str): The type of transformation to retrieve data for.
          Expected values are 'translation' or 'rotation_3d'.
        - axis (str): The axis for which to retrieve the data.
          Expected values are 'x', 'y', or 'z'.
        """
        return Shaker._wrap_around(self.transform[transform_type][axis], i)

    @staticmethod
    def is_enabled(data: 'RenderData'):
        aa = data.args.anim_args
        return aa.shake_name and aa.shake_name != 'None'

    @staticmethod
    def create(data: 'RenderData') -> 'Shaker':
        is_enabled = Shaker.is_enabled(data)
        if not is_enabled:
            return Shaker('None', 1.0, 1.0, {}, False)

        aa = data.args.anim_args
        shake_name = aa.shake_name
        intensity = aa.shake_intensity
        speed = aa.shake_speed

        log_utils.info(f"Calculating shake '{shake_name}' at intensity {intensity} with speed {speed}.")
        shake_key = Shaker.shake_name_to_key(shake_name)
        if shake_key is None:
            log_utils.warn(f"Camera shake name '{shake_name}' not found! Defaulting to 'None'. Valid options are: {list(get_camera_shake_list().values())}")
            return Shaker('None', 1.0, 1.0, {}, False)
        
        name, source_fps, shake_data = SHAKE_LIST[shake_key]

        def _proc(key, xyz):
            return Shaker._process(data, shake_data, key, xyz, source_fps, data.fps())

        log_utils.info(f"Interpolating camera shake loop from {int(source_fps)} to {data.fps()} FPS.")
        return Shaker(name, intensity, speed, {
            Key.LOC.deforum(): {'x': _proc(Key.LOC, Xyz.X), 'y': _proc(Key.LOC, Xyz.Y), 'z': _proc(Key.LOC, Xyz.Z)},
            Key.ROT.deforum(): {'x': _proc(Key.ROT, Xyz.X), 'y': _proc(Key.ROT, Xyz.Y), 'z': _proc(Key.ROT, Xyz.Z)}
        })

    @staticmethod
    def _wrap_around(df: DataFrame, i: int) -> float:
        # Camera shake data is meant to be looped, so we can just modulo the index.
        return df.loc[i % len(df), VALUE]

    @staticmethod
    def shake_name_to_key(shake_name):
        # First check if shake_name is already a valid key
        camera_shake_list = get_camera_shake_list()
        if shake_name in camera_shake_list.keys():
            return shake_name
        # If not, try to find a key with matching display name
        return next((key for key, value in camera_shake_list.items() if value == shake_name), None)

    @staticmethod
    def _process(data, shake_data, shakify_key: Key, xyz: Xyz, source_fps: int, target_fps: int):
        i = xyz.to_i()
        axis = xyz.value
        aa = data.args.anim_args
        intensity = aa.shake_intensity
        speed = aa.shake_speed

        data_frame = Shaker._parse_frame(shake_data, i, shakify_key.shakify())
        spline = Shaker._create_spline(data_frame, intensity, speed)

        source_frame_count = Shaker._source_frame_count(data_frame)
        target_frame_count = Shaker._target_frame_count(data_frame, source_fps, target_fps)
        log_utils.debug(f"Interpolating {source_frame_count} '{shakify_key.shakify()}' frames from Camera Shakify "
                        f"into {target_frame_count} points for '{shakify_key.deforum()}' on axis {axis}.")
        return Shaker._stretch_spline_along_x_axis(spline, source_frame_count, target_frame_count)

    @staticmethod
    def _parse_frame(shake_data, i, shakify_key):
        composite_key = (shakify_key, i)
        if composite_key not in shake_data:
            if shakify_key == Key.LOC.shakify():
                # 2D shakes don't provide location data, so we provide translation as 0.
                # Since the series is meant to be looped, just one zero-frame would be enough,
                # but we add two points so a spline can be calculated between them.
                return pd.DataFrame([(0, 0.0), (1, 0.0)], columns=COLUMNS)
            raise ValueError(f"Missing shakify data for key {composite_key}.")
        return pd.DataFrame(shake_data[composite_key], columns=COLUMNS)

    @staticmethod
    def _create_spline(df: DataFrame, intensity, speed) -> CubicSpline:
        return CubicSpline(df[FRAME] * speed, df[VALUE] * intensity)

    @staticmethod
    def _source_frame_count(data_frame):
        return len(data_frame) + 1

    @staticmethod
    def _target_frame_count(data_frame, source_fps, target_fps):
        return int(Shaker._source_frame_count(data_frame) * target_fps / source_fps)

    @staticmethod
    def _stretch_spline_along_x_axis(spline, num_frames_source, num_frames_target):
        frames_target = np.linspace(0, num_frames_source, num_frames_target)
        return pd.DataFrame({FRAME: frames_target, VALUE: spline(frames_target)})

    @staticmethod
    def shake_keys():
        return list(SHAKE_LIST.keys())

    @staticmethod
    def shake_names():
        return [value[0] for value in SHAKE_LIST.values()]

    @staticmethod
    def shakes():
        return {key: value[0] for key, value in SHAKE_LIST.items()}
