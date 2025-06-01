"""
RIFE frame interpolation integration.
"""

from .inference_video import *

__all__ = [
    'run_rife_new_video_infer',
    'clear_write_buffer',
    'build_read_buffer',
    'make_inference',
    'pad_image',
    'stitch_video',
] 