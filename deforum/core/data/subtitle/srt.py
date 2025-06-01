import os

from dataclasses import dataclass
from decimal import Decimal

from ....utils import opt_utils
from ....utils.subtitle_handler import init_srt_file


@dataclass(init=True, frozen=True, repr=False, eq=False)
class Srt:
    filename: str
    frame_duration: Decimal

    @staticmethod
    def create_if_active(out_dir: str, timestring: str, fps: float) -> 'Srt | None':
        if not opt_utils.is_subtitle_generation_active():
            return None
        else:
            # create .srt file and set timeframe mechanism using FPS
            filename = os.path.join(out_dir, f"{timestring}.srt")
            frame_duration = init_srt_file(filename, fps)
            return Srt(filename, frame_duration)
