"""Depth Anything V3 integration for enhanced monocular and multi-view depth estimation.

DA3 provides several improvements over DA2:
- Better monocular depth quality
- Multi-view geometry support
- Depth-ray representation for pose estimation
- Temporal consistency for video sequences

This module provides drop-in replacement for DA2 in Phase 1, with hooks for
Phase 2 (multi-view) and Phase 3 (3DGS) capabilities.
"""

from torchvision import transforms
import torch
import numpy as np
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


class DepthAnythingV3:
    """Depth Anything V3 model wrapper.

    Provides drop-in replacement for DepthAnythingV2 with enhanced capabilities.

    Args:
        device: torch device (cpu/cuda)
        model_size: 'small', 'base', or 'large' (default: 'small')
        variant: 'mono' or 'any-view' (default: 'mono')
            - 'mono': Optimized for single-image depth (Phase 1)
            - 'any-view': Multi-view geometry support (Phase 2)
    """

    def __init__(self, device, model_size='small', variant='mono'):
        """Initialize Depth Anything V3 model.

        Models auto-download from HuggingFace on first use to cache directory.
        """
        self.device = device
        self.model_size = model_size
        self.variant = variant

        # Model name mapping
        model_map = {
            ('mono', 'small'): 'depth-anything/Depth-Anything-V3-Small',
            ('mono', 'base'): 'depth-anything/Depth-Anything-V3-Base',
            ('mono', 'large'): 'depth-anything/Depth-Anything-V3-Large',
            ('any-view', 'small'): 'depth-anything/DA3-Small',
            ('any-view', 'base'): 'depth-anything/DA3-Base',
            ('any-view', 'large'): 'depth-anything/DA3-Large',
        }

        key = (variant.lower(), model_size.lower())
        if key not in model_map:
            logger.warning(
                f"Invalid DA3 config: variant={variant}, size={model_size}. "
                f"Defaulting to mono/small."
            )
            key = ('mono', 'small')

        model_name = model_map[key]

        logger.info(f"Loading Depth Anything V3 ({variant} {model_size}) from {model_name}...")
        logger.info("Model will auto-download to HuggingFace cache if not present")

        try:
            # Try to import DA3
            from depth_anything_3.api import DepthAnything3
            self.model = DepthAnything3.from_pretrained(model_name)
            self.model.to(device)
            logger.info(f"✓ Depth Anything V3 loaded successfully on {device}")

        except ImportError as e:
            logger.error(
                "Depth Anything V3 package not found. "
                "Install with: pip install depth-anything-3"
            )
            raise ImportError(
                "depth-anything-3 package required for DA3 support. "
                "Run: pip install depth-anything-3 xformers"
            ) from e

        except Exception as e:
            logger.error(f"Failed to load DA3 model: {str(e)}")
            raise

    def predict(self, image, weight=0.5, half_precision=False):
        """Predict depth map from single image (drop-in replacement for DA2).

        Args:
            image: Input image (numpy array or PIL Image)
            weight: Depth map weight (not used by DA3, kept for compatibility)
            half_precision: Use FP16 precision (not used by DA3, kept for compatibility)

        Returns:
            Depth map tensor [1, 1, H, W] compatible with DA2 output format
        """
        import torch.nn.functional as F

        # Store original image dimensions
        if isinstance(image, np.ndarray):
            original_h, original_w = image.shape[:2]
            from PIL import Image
            # Assume BGR format from cv2
            image_rgb = image[:, :, ::-1]
            image = Image.fromarray(image_rgb)
        else:
            original_w, original_h = image.size

        # Run DA3 inference (may downsample internally for processing)
        result = self.model.inference([image])

        # Extract depth map from Prediction object (dataclass with .depth attribute)
        # result.depth is np.ndarray with shape [N, H, W] where N is number of images
        depth_np = result.depth[0]  # First (and only) image -> [H, W]

        # Convert to tensor format matching DA2 output: [1, 1, H, W]
        if isinstance(depth_np, torch.Tensor):
            depth = depth_np
            if depth.ndim == 2:
                depth = depth.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
            elif depth.ndim == 3:
                depth = depth.unsqueeze(0)  # [1,H,W] -> [1,1,H,W]
        else:
            # Convert numpy to tensor [H,W] -> [1,1,H,W]
            depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()

        # CRITICAL: Normalize depth to 0-1 range for consistency with DA2
        # DA3 returns depth in arbitrary range (e.g., 0.94-1.04), but depth warping
        # expects normalized depth where 0=nearest, 1=farthest
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:  # Avoid division by zero
            depth = (depth - depth_min) / (depth_max - depth_min)

        # CRITICAL: Resize depth map to match original image dimensions
        # DA3 downsamples during processing (e.g., 1920x480 -> 504x280)
        # but depth warping expects depth to match image size exactly
        current_h, current_w = depth.shape[2], depth.shape[3]
        if current_h != original_h or current_w != original_w:
            depth = F.interpolate(
                depth,
                size=(original_h, original_w),
                mode='bilinear',
                align_corners=True  # Changed from False - prevents spatial misalignment
            )

        return depth

    def predict_multiview(self, images):
        """Predict depth with multi-view consistency (Phase 2 capability).

        Args:
            images: List of input images (numpy arrays or PIL Images)

        Returns:
            Dictionary with:
                - 'depth': List of depth maps [N, H, W]
                - 'confidence': Confidence maps [N, H, W]
                - 'camera_extrinsics': Camera poses [N, 4, 4]
                - 'camera_intrinsics': Camera intrinsics [N, 3, 3]
        """
        if self.variant != 'any-view':
            logger.warning(
                f"predict_multiview() requires variant='any-view', "
                f"got '{self.variant}'. Single-view fallback."
            )
            # Fallback to single-view for each image
            depths = [self.predict(img) for img in images]
            return {
                'depth': depths,
                'confidence': [torch.ones_like(d) for d in depths],
                'camera_extrinsics': None,
                'camera_intrinsics': None,
            }

        # Convert numpy arrays to PIL Images if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                from PIL import Image
                img_rgb = img[:, :, ::-1]  # BGR -> RGB
                pil_images.append(Image.fromarray(img_rgb))
            else:
                pil_images.append(img)

        # Run DA3 multi-view inference
        result = self.model.inference(pil_images)

        return result

    def estimate_3d_gaussians(self, images):
        """Estimate 3D Gaussian Splatting parameters (Phase 3 capability).

        Args:
            images: List of keyframe images

        Returns:
            Dictionary with 3DGS parameters:
                - 'means': 3D positions [N, 3]
                - 'rotations': Quaternions [N, 4]
                - 'scales': Scale parameters [N, 3]
                - 'opacities': Opacity values [N, 1]
                - 'colors': RGB colors [N, 3]
        """
        logger.info("3D Gaussian Splatting estimation (Phase 3 - not yet implemented)")

        # Convert numpy arrays to PIL Images if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                from PIL import Image
                img_rgb = img[:, :, ::-1]  # BGR -> RGB
                pil_images.append(Image.fromarray(img_rgb))
            else:
                pil_images.append(img)

        # Run DA3 inference with 3DGS enabled
        try:
            result = self.model.inference(pil_images, infer_gs=True)
            # Return Prediction object with .gaussians attribute
            return result
        except Exception as e:
            logger.error(f"3DGS estimation failed: {str(e)}")
            logger.info("This feature requires DA3NESTED-GIANT-LARGE model")
            return None