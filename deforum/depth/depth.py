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
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()

class DepthModel:
    """
    Depth model supporting Depth Anything V2 and V3

    Supported models:
    - Depth-Anything-V2-Small/Base/Large (stable, default)
    - Depth-Anything-V3-Mono-Small/Base/Large (better quality)
    - Depth-Anything-V3-AnyView-Small/Base/Large (multi-view support)
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
        """Initialize Depth Anything V2 or V3 model based on depth_algorithm"""
        # Parse depth_algorithm string
        # DA2: 'Depth-Anything-V2-Small' -> version='v2', size='small', variant=None
        # DA3: 'Depth-Anything-V3-Mono-Large' -> version='v3', size='large', variant='mono'
        parts = self.depth_algorithm.lower().split('-')

        # Determine version
        is_v3 = 'v3' in parts

        # Extract model size (always last part)
        model_size = parts[-1]
        if model_size not in ['small', 'base', 'large']:
            logger.warning(f"Unknown model size '{model_size}', defaulting to 'small'")
            model_size = 'small'

        if is_v3:
            # DA3 model - extract variant (mono or anyview)
            # 'depth-anything-v3-mono-small' -> variant='mono'
            # 'depth-anything-v3-anyview-base' -> variant='any-view'
            variant = 'mono'  # Default
            if 'anyview' in parts:
                variant = 'any-view'
            elif 'mono' in parts:
                variant = 'mono'

            logger.info(f"Loading Depth Anything V3 ({variant}, {model_size})")

            try:
                from .depth_anything_v3 import DepthAnythingV3
                self.depth_anything = DepthAnythingV3(
                    self.device,
                    model_size=model_size,
                    variant=variant
                )
                self.is_v3 = True
            except ImportError as e:
                logger.error(
                    "Depth Anything V3 not available. Install with: pip install depth-anything-3 xformers"
                )
                logger.warning("Falling back to Depth Anything V2 Small")
                self.depth_anything = DepthAnything(self.device, model_size='small')
                self.is_v3 = False
        else:
            # DA2 model
            logger.info(f"Loading Depth Anything V2 ({model_size})")
            self.depth_anything = DepthAnything(self.device, model_size=model_size)
            self.is_v3 = False

    def predict(self, prev_img_cv2, use_ray_pose=False, conf_thresh_percentile=40.0) -> torch.Tensor:
        """
        Predict depth map from image

        Args:
            prev_img_cv2: Input image as numpy array (BGR, uint8) - OpenCV format
            use_ray_pose: DA3 only - use ray-based pose estimation (more accurate, slower)
            conf_thresh_percentile: DA3 only - confidence threshold percentile (0-100)

        Returns:
            torch.Tensor: Depth map tensor
        """
        img_pil = Image.fromarray(cv2.cvtColor(prev_img_cv2.astype(np.uint8), cv2.COLOR_RGB2BGR))

        # Pass DA3 parameters if this is a V3 model
        if self.is_v3:
            depth_tensor = self.depth_anything.predict(
                img_pil,
                use_ray_pose=use_ray_pose,
                conf_thresh_percentile=conf_thresh_percentile
            )
        else:
            # DA2 doesn't use these parameters
            depth_tensor = self.depth_anything.predict(img_pil)

        return depth_tensor

    def to(self, device):
        """Move model to specified device"""
        self.device = device
        if hasattr(self, 'depth_anything'):
            if hasattr(self, 'is_v3') and self.is_v3:
                # DA3 model has different structure
                self.depth_anything.model.to(device)
            else:
                # DA2 model uses pipeline
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
