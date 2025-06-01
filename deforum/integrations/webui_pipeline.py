# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

from modules.processing import StableDiffusionProcessingImg2Img
from modules.shared import opts, sd_model
import os

def get_webui_sd_pipeline(args, root):
    # Set up the pipeline
    p = StableDiffusionProcessingImg2Img(
        sd_model=sd_model,
        outpath_samples = opts.outdir_samples or opts.outdir_img2img_samples,
    )    # we'll set up the rest later
    
    os.makedirs(args.outdir, exist_ok=True)
    p.width, p.height = map(lambda x: x - x % 8, (args.W, args.H))
    p.steps = args.steps
    p.seed = args.seed
    p.sampler_name = args.sampler
    p.scheduler_name = args.scheduler
    p.tiling = args.tiling
    p.restore_faces = args.restore_faces
    p.subseed = root.subseed
    p.subseed_strength = root.subseed_strength
    p.seed_resize_from_w = args.seed_resize_from_w
    p.seed_resize_from_h = args.seed_resize_from_h
    p.fill = args.fill
    p.batch_size = 1  # b.size 1 as this is DEFORUM :)
    p.seed = args.seed
    p.do_not_save_samples = True  # Setting this to False will trigger webui's saving mechanism - and we will end up with duplicated files, and another folder within our destination folder - big no no.
    p.sampler_name = args.sampler
    p.scheduler = args.scheduler
    p.mask_blur = args.mask_overlay_blur
    p.extra_generation_params["Mask blur"] = args.mask_overlay_blur
    p.n_iter = 1
    p.steps = args.steps
    p.denoising_strength = 1 - args.strength
    p.outpath_samples = args.outdir

    # Guidance scales:
    # Separate CFG scale schedules for Img2Img and for Txt2Img pipes have been removed in favor of unified ones.
    p.cfg_scale = args.cfg_scale  # see "StableDiffusionProcessing" in <webUI-dir>/modules/processing.py
    p.distilled_cfg_scale = args.distilled_cfg_scale
    # Additionally passed to the Img2Img pipe only (probably for override?), so we can just pass the same value again:
    p.image_cfg_scale = args.cfg_scale  # image specific, only used in "StableDiffusionProcessingImg2Img" (img2img.py)
    # p.image_distilled_cfg_scale  # <-- does not exist (which is fine).

    return p
