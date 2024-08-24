from ....composable_masks import compose_mask_with_check
from ....image_sharpening import unsharp_mask


def call_compose_mask_with_check(init, mask_seq, val_masks, image):
    return compose_mask_with_check(init.args.root, init.args.args, mask_seq, val_masks,
                                   Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))


def call_unsharp_mask(init, step, image, mask):
    kernel_size = (step.step_data.kernel, step.step_data.kernel)
    mask_image = mask.image if init.args.args.use_mask else None
    return unsharp_mask(image, kernel_size, step.step_data.sigma, step.step_data.amount,
                        step.step_data.threshold, mask_image)
