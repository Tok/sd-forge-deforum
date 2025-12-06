"""Depth Anything V3 integration for enhanced monocular and multi-view depth estimation.

DA3 provides several improvements over DA2:
- Better monocular depth quality
- Multi-view geometry support
- Depth-ray representation for pose estimation
- Temporal consistency for video sequences

This module provides drop-in replacement for DA2 in Phase 1, with hooks for
Phase 2 (multi-view) and Phase 3 (3DGS) capabilities.
"""

from typing import Dict, List, Tuple, Union, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()

# Constants
DEPTH_OUTPUT_FORMAT = (1, 1)  # Target depth tensor format: [1, 1, H, W]
BGR_TO_RGB_SLICE = slice(None, None, -1)  # Reverse color channel order


def _get_model_name(variant: str, size: str) -> str:
    """Get HuggingFace model name for DA3 variant and size.

    Args:
        variant: 'mono' or 'any-view'
        size: 'small', 'base', or 'large'

    Returns:
        HuggingFace model identifier string
    """
    model_map: Dict[Tuple[str, str], str] = {
        ('mono', 'small'): 'depth-anything/Depth-Anything-V3-Small',
        ('mono', 'base'): 'depth-anything/Depth-Anything-V3-Base',
        ('mono', 'large'): 'depth-anything/Depth-Anything-V3-Large',
        ('any-view', 'small'): 'depth-anything/DA3-Small',
        ('any-view', 'base'): 'depth-anything/DA3-Base',
        ('any-view', 'large'): 'depth-anything/DA3-Large',
    }

    key = (variant.lower(), size.lower())
    if key not in model_map:
        logger.warning(
            f"Invalid DA3 config: variant={variant}, size={size}. "
            f"Defaulting to mono/small."
        )
        key = ('mono', 'small')

    return model_map[key]


def _convert_bgr_to_rgb_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to RGB PIL Image.

    Args:
        image_bgr: Numpy array in BGR format (from OpenCV)

    Returns:
        PIL Image in RGB format
    """
    image_rgb = image_bgr[:, :, BGR_TO_RGB_SLICE]
    return Image.fromarray(image_rgb)


def _extract_image_dimensions(image: Union[np.ndarray, Image.Image]) -> Tuple[int, int]:
    """Extract height and width from image.

    Args:
        image: Numpy array or PIL Image

    Returns:
        Tuple of (height, width)
    """
    if isinstance(image, np.ndarray):
        return image.shape[:2]  # (H, W) for numpy
    else:
        w, h = image.size
        return (h, w)  # PIL gives (W, H), return (H, W)


def _prepare_image_for_inference(
    image: Union[np.ndarray, Image.Image]
) -> Tuple[Image.Image, int, int]:
    """Prepare image for DA3 inference by converting to PIL RGB and extracting dimensions.

    Args:
        image: Input image (numpy array in BGR or PIL Image in RGB)

    Returns:
        Tuple of (PIL Image in RGB, original_height, original_width)
    """
    original_h, original_w = _extract_image_dimensions(image)

    if isinstance(image, np.ndarray):
        pil_image = _convert_bgr_to_rgb_pil(image)
    else:
        pil_image = image

    return pil_image, original_h, original_w


def _convert_depth_to_tensor(depth_np: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert depth array/tensor to standardized format [1, 1, H, W].

    Args:
        depth_np: Depth data (numpy array or torch tensor)

    Returns:
        Depth tensor in format [1, 1, H, W]
    """
    if isinstance(depth_np, torch.Tensor):
        depth = depth_np
        if depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
        elif depth.ndim == 3:
            depth = depth.unsqueeze(0)  # [1,H,W] -> [1,1,H,W]
    else:
        # Convert numpy to tensor [H,W] -> [1,1,H,W]
        depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()

    return depth


def _normalize_depth_range(depth: torch.Tensor) -> torch.Tensor:
    """Normalize depth values to [0, 1] range.

    DA3 returns depth in arbitrary ranges (e.g., 0.94-1.04).
    Depth warping expects normalized depth where 0=nearest, 1=farthest.

    Args:
        depth: Depth tensor with arbitrary value range

    Returns:
        Depth tensor normalized to [0, 1]
    """
    depth_min = depth.min()
    depth_max = depth.max()

    if depth_max > depth_min:  # Avoid division by zero
        return (depth - depth_min) / (depth_max - depth_min)

    return depth


def _resize_depth_to_match_image(
    depth: torch.Tensor,
    target_h: int,
    target_w: int
) -> torch.Tensor:
    """Resize depth map to match original image dimensions.

    DA3 downsamples during processing (e.g., 1920x480 -> 504x280).
    Depth warping expects depth to match image size exactly.

    Args:
        depth: Depth tensor [1, 1, H, W]
        target_h: Target height
        target_w: Target width

    Returns:
        Resized depth tensor [1, 1, target_h, target_w]
    """
    current_h, current_w = depth.shape[2], depth.shape[3]

    if current_h != target_h or current_w != target_w:
        # align_corners=True prevents spatial misalignment (cross/quadrant artifacts)
        depth = F.interpolate(
            depth,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=True
        )

    return depth


def _convert_images_to_pil(images: List[Union[np.ndarray, Image.Image]]) -> List[Image.Image]:
    """Convert list of images to PIL format.

    Args:
        images: List of numpy arrays (BGR) or PIL Images

    Returns:
        List of PIL Images in RGB format
    """
    pil_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            pil_images.append(_convert_bgr_to_rgb_pil(img))
        else:
            pil_images.append(img)
    return pil_images


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

    def __init__(
        self,
        device: torch.device,
        model_size: str = 'small',
        variant: str = 'mono'
    ) -> None:
        """Initialize Depth Anything V3 model.

        Models auto-download from HuggingFace on first use to cache directory.
        """
        self.device = device
        self.model_size = model_size
        self.variant = variant

        model_name = _get_model_name(variant, model_size)

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

    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
        weight: float = 0.5,
        half_precision: bool = False,
        use_ray_pose: bool = False,
        conf_thresh_percentile: float = 40.0
    ) -> torch.Tensor:
        """Predict depth map from single image (drop-in replacement for DA2).

        Args:
            image: Input image (numpy array or PIL Image)
            weight: Depth map weight (not used by DA3, kept for compatibility)
            half_precision: Use FP16 precision (not used by DA3, kept for compatibility)
            use_ray_pose: Use ray-based pose estimation (more accurate but slower)
            conf_thresh_percentile: Adaptive confidence threshold percentile (0-100)

        Returns:
            Depth map tensor [1, 1, H, W] compatible with DA2 output format
        """
        # Prepare image and extract dimensions
        pil_image, original_h, original_w = _prepare_image_for_inference(image)

        # Run DA3 inference (may downsample internally for processing)
        # Pass tuning parameters to DA3 model
        result = self.model.inference(
            [pil_image],
            use_ray_pose=use_ray_pose,
            conf_thresh_percentile=conf_thresh_percentile
        )

        # Extract depth map from Prediction object (dataclass with .depth attribute)
        # result.depth is np.ndarray with shape [N, H, W] where N is number of images
        depth_np = result.depth[0]  # First (and only) image -> [H, W]

        # Convert to tensor format matching DA2 output: [1, 1, H, W]
        depth = _convert_depth_to_tensor(depth_np)

        # Normalize depth to 0-1 range for consistency with DA2
        depth = _normalize_depth_range(depth)

        # Resize depth map to match original image dimensions
        depth = _resize_depth_to_match_image(depth, original_h, original_w)

        return depth

    def predict_multiview(
        self,
        images: List[Union[np.ndarray, Image.Image]]
    ) -> Union[Dict[str, Any], Any]:
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
        pil_images = _convert_images_to_pil(images)

        # Run DA3 multi-view inference
        result = self.model.inference(pil_images)

        return result

    def estimate_3d_gaussians(
        self,
        images: List[Union[np.ndarray, Image.Image]]
    ) -> Optional[Any]:
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
        # Convert numpy arrays to PIL Images if needed
        pil_images = _convert_images_to_pil(images)

        # Run DA3 inference with 3DGS enabled
        try:
            logger.debug(f"Attempting 3DGS estimation with {len(pil_images)} images...")
            result = self.model.inference(pil_images, infer_gs=True)
            # Return Prediction object with .gaussians attribute
            return result
        except (AttributeError, TypeError) as e:
            # Model doesn't have gs_head/gs_adapter - this is expected for current DA3 models
            logger.debug(f"3DGS estimation not supported by current model: {str(e)}")
            logger.warning(
                "⚠️  3D Gaussian Splatting requires DA3 models with trained 3DGS heads "
                "(e.g., DA3-GIANT-LARGE). Current DA3 models don't support this feature yet."
            )
            return None
        except Exception as e:
            logger.error(f"3DGS estimation failed: {str(e)}")
            return None
