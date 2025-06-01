from ...utils import filename_utils
from ...media.video_audio_pipeline import save_frame as media_save_frame
import os


def save_frame(data, image, frame_idx):
    """
    Save a frame using the standard Deforum frame saving logic.
    This wraps the media save_frame function with proper filename generation.
    """
    filename = filename_utils.frame_filename(data, frame_idx)
    file_path = os.path.join(data.args.args.outdir, filename)
    
    # Convert PIL image to opencv format if needed
    if hasattr(image, 'save'):  # PIL Image
        import cv2
        import numpy as np
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        media_save_frame(opencv_image, file_path)
    else:
        # Assume it's already in opencv format
        media_save_frame(image, file_path) 