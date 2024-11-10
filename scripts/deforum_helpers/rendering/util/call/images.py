from ....load_images import get_mask_from_file
from ....noise import add_noise
from ....video_audio_utilities import get_next_frame


def call_add_noise(data, frame, image):
    aa = data.args.anim_args
    amount: float = frame.frame_data.noise
    seed: int = data.args.args.seed
    n_type: str = aa.noise_type
    perlin_arguments = (aa.perlin_w, aa.perlin_h, aa.perlin_octaves, aa.perlin_persistence)
    mask = data.args.root.noise_mask
    is_do_maks_invert = data.args.args.invert_mask
    return add_noise(image, amount, seed, n_type, perlin_arguments, mask, is_do_maks_invert)


def call_get_mask_from_file(data, i, is_mask: bool = False):
    next_frame = get_next_frame(data.output_directory, data.args.anim_args.video_mask_path, i, is_mask)
    return get_mask_from_file(next_frame, data.args.args)


def call_get_mask_from_file_with_frame(data, frame):
    return get_mask_from_file(frame, data.args.args)
