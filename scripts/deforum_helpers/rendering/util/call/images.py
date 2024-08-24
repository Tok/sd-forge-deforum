from ....load_images import get_mask_from_file
from ....noise import add_noise


def call_add_noise(init, step, image):
    aa = init.args.anim_args
    amount: float = step.step_data.noise
    seed: int = init.args.args.seed
    n_type: str = aa.noise_type
    perlin_arguments = (aa.perlin_w, aa.perlin_h, aa.perlin_octaves, aa.perlin_persistence)
    mask = init.args.root.noise_mask
    is_do_maks_invert = init.args.args.invert_mask
    return add_noise(image, amount, seed, n_type, perlin_arguments, mask, is_do_maks_invert)


def call_get_mask_from_file(init, i, is_mask: bool = False):
    next_frame = get_next_frame(init.output_directory, init.args.anim_args.video_mask_path, i, is_mask)
    return get_mask_from_file(next_frame, init.args.args)


def call_get_mask_from_file_with_frame(init, frame):
    return get_mask_from_file(frame, init.args.args)
