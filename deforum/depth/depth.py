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
from .depth_anything_v2 import DepthAnything

class DepthModel:
    """
    Simplified depth model using only Depth Anything V2

    Supports three model sizes:
    - Depth-Anything-V2-Small (fastest)
    - Depth-Anything-V2-Base (balanced)
    - Depth-Anything-V2-Large (best quality)
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        keep_in_vram = kwargs.get('keep_in_vram', False)
        depth_algorithm = kwargs.get('depth_algorithm', 'Depth-Anything-V2-Small')
        model_deleted = cls._instance and cls._instance.should_delete
        model_switched = cls._instance and cls._instance.depth_algorithm != depth_algorithm

        should_reload = (cls._instance is None or model_deleted or model_switched)

        if should_reload:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(
                models_path=args[0],
                device=args[1],
                keep_in_vram=keep_in_vram,
                depth_algorithm=depth_algorithm
            )
        elif cls._instance.should_delete and keep_in_vram:
            cls._instance._initialize(
                models_path=args[0],
                device=args[1],
                keep_in_vram=keep_in_vram,
                depth_algorithm=depth_algorithm
            )
        cls._instance.should_delete = not keep_in_vram
        return cls._instance

    def _initialize(self, models_path, device, keep_in_vram=False, depth_algorithm='Depth-Anything-V2-Small'):
        self.models_path = models_path
        self.device = device
        self.keep_in_vram = keep_in_vram
        self.depth_algorithm = depth_algorithm
        self.depth_min, self.depth_max = 1000, -1000
        self.should_delete = False
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Depth Anything V2 model with the selected size"""
        # Extract model size from depth_algorithm string
        # 'Depth-Anything-V2-Small' -> 'small'
        # 'Depth-Anything-V2-Base' -> 'base'
        # 'Depth-Anything-V2-Large' -> 'large'
        model_size = self.depth_algorithm.lower().split('-')[-1]  # Get last part after last dash
        if model_size not in ['small', 'base', 'large']:
            model_size = 'small'  # Fallback to small if unknown

        self.depth_anything = DepthAnything(self.device, model_size=model_size)

    def predict(self, prev_img_cv2, midas_weight=None, half_precision=None) -> torch.Tensor:
        """
        Predict depth map from image

        Args:
            prev_img_cv2: Input image as numpy array (BGR, uint8) - OpenCV format
            midas_weight: Legacy parameter, ignored (kept for backward compatibility)
            half_precision: Legacy parameter, ignored (kept for backward compatibility)

        Returns:
            torch.Tensor: Depth map tensor
        """
        img_pil = Image.fromarray(cv2.cvtColor(prev_img_cv2.astype(np.uint8), cv2.COLOR_RGB2BGR))
        depth_tensor = self.depth_anything.predict(img_pil)
        return depth_tensor

    def to(self, device):
        """Move model to specified device"""
        self.device = device
        if hasattr(self, 'depth_anything'):
            self.depth_anything.pipe.model.to(device)
        gc.collect()
        torch.cuda.empty_cache()

    def to_image(self, depth: torch.Tensor):
        """Convert depth tensor to PIL Image"""
        depth = depth.cpu().numpy()
        # Remove extra dimensions (batch, etc.) - squeeze to at most 3D
        while len(depth.shape) > 3:
            depth = depth.squeeze(0)
        depth = np.expand_dims(depth, axis=0) if len(depth.shape) == 2 else depth
        self.depth_min, self.depth_max = min(self.depth_min, depth.min()), max(self.depth_max, depth.max())
        denom = max(1e-8, self.depth_max - self.depth_min)
        temp = rearrange((depth - self.depth_min) / denom * 255, 'c h w -> h w c')
        return Image.fromarray(repeat(temp, 'h w 1 -> h w c', c=3).astype(np.uint8))

    def save(self, filename: str, depth: torch.Tensor):
        """Save depth map to file"""
        self.to_image(depth).save(filename)

    def delete_model(self):
        """Clean up model from memory"""
        if hasattr(self, 'depth_anything'):
            del self.depth_anything

        gc.collect()
        torch.cuda.empty_cache()
        devices.torch_gc()
