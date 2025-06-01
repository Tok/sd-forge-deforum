import cv2
from PIL import Image

from ..composable_mask_system import compose_mask_with_check
from ...media.image_enhancement import unsharp_mask


def call_compose_mask_with_check(init, mask_seq, val_masks, image):
    return compose_mask_with_check(init.args.root, init.args.args, mask_seq, val_masks,
                                   Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))


def call_unsharp_mask(init, frame, image, mask):
    kernel_size = (frame.frame_data.kernel, frame.frame_data.kernel)
    mask_image = mask.image if init.args.args.use_mask else None
    return unsharp_mask(image, kernel_size, frame.frame_data.sigma, frame.frame_data.amount,
                        frame.frame_data.threshold, mask_image)
