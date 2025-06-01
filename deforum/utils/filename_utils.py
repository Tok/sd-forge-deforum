import os
from enum import Enum
from pathlib import Path

from ..media.video_audio_pipeline import get_frame_name


class FileFormat(Enum):
    JPG = "jpg"
    PNG = "png"

    @staticmethod
    def frame_format():
        return FileFormat.PNG

    @staticmethod
    def video_frame_format():
        return FileFormat.JPG


def _frame_filename_index(i: int, file_format: FileFormat) -> str:
    return f"{i:09}.{file_format.value}"


def frame_filename(data, i: int, is_depth=False, file_format=FileFormat.frame_format()) -> str:
    infix = "_depth_" if is_depth else "_"
    return f"{data.args.root.timestring}{infix}{_frame_filename_index(i, file_format)}"


def depth_frame(data, i) -> str:
    return frame_filename(data, i, True)


def preview_video_image_path(data, i) -> Path:
    frame_name = get_frame_name(data.args.anim_args.video_init_path)
    index = _frame_filename_index(i, FileFormat.video_frame_format())
    return Path(data.output_directory) / "inputframes" / (frame_name + index)
