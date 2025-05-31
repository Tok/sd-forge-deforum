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

import gc
import cv2
import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat
from modules import devices
from modules.shared import cmd_opts
from .depth_anything_v2 import DepthAnything
from .depth_midas import MidasDepth
from .general_utils import debug_print

class DepthModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        keep_in_vram = kwargs.get('keep_in_vram', False)
        depth_algorithm = kwargs.get('depth_algorithm', 'Depth-Anything-V2-Small')
        Width, Height = kwargs.get('Width', 512), kwargs.get('Height', 512)
        midas_weight = kwargs.get('midas_weight', 0.2)
        model_switched = cls._instance and cls._instance.depth_algorithm != depth_algorithm
        model_deleted = cls._instance and cls._instance.should_delete

        should_reload = (cls._instance is None or model_deleted or model_switched)

        if should_reload:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(models_path=args[0], device=args[1], half_precision=not cmd_opts.no_half, keep_in_vram=keep_in_vram, depth_algorithm=depth_algorithm, Width=Width, Height=Height, midas_weight=midas_weight)
        elif cls._instance.should_delete and keep_in_vram:
            cls._instance._initialize(models_path=args[0], device=args[1], half_precision=not cmd_opts.no_half, keep_in_vram=keep_in_vram, depth_algorithm=depth_algorithm, Width=Width, Height=Height, midas_weight=midas_weight)
        cls._instance.should_delete = not keep_in_vram
        return cls._instance

    def _initialize(self, models_path, device, half_precision=not cmd_opts.no_half, keep_in_vram=False, depth_algorithm='Midas-3-Hybrid', Width=512, Height=512, midas_weight=1.0):
        self.models_path = models_path
        self.device = device
        self.half_precision = half_precision
        self.keep_in_vram = keep_in_vram
        self.depth_algorithm = depth_algorithm
        self.Width, self.Height = Width, Height
        self.midas_weight = midas_weight
        self.depth_min, self.depth_max = 1000, -1000
        self.should_delete = False
        self._initialize_model()

    def _initialize_model(self):
        depth_algo = self.depth_algorithm.lower()
        if depth_algo.startswith('depth-anything'):
            self.depth_anything = DepthAnything(self.device)
        elif depth_algo.startswith('midas'):
            self.midas_depth = MidasDepth(self.models_path, self.device, half_precision=self.half_precision, midas_model_type=self.depth_algorithm)
        else:
            raise Exception(f"Unknown depth_algorithm: {self.depth_algorithm}. Only 'Depth-Anything-V2-Small' and 'Midas-3-Hybrid' are supported.")

    def predict(self, prev_img_cv2, midas_weight, half_precision) -> torch.Tensor:
        img_pil = Image.fromarray(cv2.cvtColor(prev_img_cv2.astype(np.uint8), cv2.COLOR_RGB2BGR))

        if self.depth_algorithm.lower().startswith('depth-anything'):
            depth_tensor = self.depth_anything.predict(img_pil)
        elif self.depth_algorithm.lower().startswith('midas'):
            depth_tensor = self.midas_depth.predict(prev_img_cv2, half_precision)
        else:
            raise Exception(f"Unknown depth_algorithm passed to depth.predict function: {self.depth_algorithm}")

        return depth_tensor
        
    def to(self, device):
        self.device = device
        if self.depth_algorithm.lower().startswith('midas'):
            self.midas_depth.to(device)
        # Depth-Anything-V2 handles device management internally
        gc.collect()
        torch.cuda.empty_cache()
        
    def to_image(self, depth: torch.Tensor):
        depth = depth.cpu().numpy()
        depth = np.expand_dims(depth, axis=0) if len(depth.shape) == 2 else depth
        self.depth_min, self.depth_max = min(self.depth_min, depth.min()), max(self.depth_max, depth.max())
        denom = max(1e-8, self.depth_max - self.depth_min)
        temp = rearrange((depth - self.depth_min) / denom * 255, 'c h w -> h w c')
        return Image.fromarray(repeat(temp, 'h w 1 -> h w c', c=3).astype(np.uint8))

    def save(self, filename: str, depth: torch.Tensor):
        self.to_image(depth).save(filename)

    def delete_model(self):
        if hasattr(self, 'midas_depth'):
            del self.midas_depth

        if hasattr(self, 'depth_anything'):
            del self.depth_anything

        gc.collect()
        torch.cuda.empty_cache()
        devices.torch_gc()
