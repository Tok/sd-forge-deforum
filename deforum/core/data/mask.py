from dataclasses import dataclass
from typing import Any

from PIL import Image

from ..util import put_all
from ..util.utils import create_img, call_or_use_on_cond
from ...load_images import get_mask, load_img
from ...rendering.util.call.images import call_get_mask_from_file


@dataclass(init=True, frozen=True, repr=False, eq=False)
class Mask:
    image: Image
    vals: Any
    noise_vals: Any

    def has_mask_image(self):
        return self.image is not None

    @staticmethod
    def _create_vals(count, dimensions):
        return list(map(lambda _: {'everywhere': create_img(dimensions)}, range(count)))

    @staticmethod
    def _assign(init, i, is_mask_image, dicts):
        # Grab the first frame masks since they wont be provided until next frame
        # Video mask overrides the init image mask, also, won't be searching for init_mask if use_mask_video is set
        # Made to solve https://github.com/deforum-art/deforum-for-automatic1111-webui/issues/386
        key = 'video_mask'
        if init.args.anim_args.use_mask_video:
            mask = call_get_mask_from_file(init, i, True)
            init.args.args.mask_file = mask
            init.args.root.noise_mask = mask
            put_all(dicts, key, mask)
        elif is_mask_image is None and init.is_use_mask:
            put_all(dicts, key, get_mask(init.args.args))

    @staticmethod
    def _load_mask(init, args):
        return load_img(args.init_image, args.init_image_box, shape=init.dimensions(),
                        use_alpha_as_mask=args.use_alpha_as_mask)[1]

    @staticmethod
    def _create_mask_image(init):
        args = init.args.args
        return call_or_use_on_cond(init.is_using_init_image_or_box(), lambda: Mask._load_mask(init, args))

    @staticmethod
    def _create(init, i, mask_image):
        mask_and_noise_mask = Mask._create_vals(2, init.dimensions())
        put_all(mask_and_noise_mask, 'video_mask', mask_image)
        Mask._assign(init, i, mask_image, mask_and_noise_mask)
        return Mask(mask_image, mask_and_noise_mask[0], mask_and_noise_mask[1])

    @staticmethod
    def create(init, i):
        mask_image = Mask._create_mask_image(init)
        return call_or_use_on_cond(mask_image is not None, Mask._create(init, i, mask_image))
