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
import subprocess
import sys
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()

# Suppress verbose DA3 internal logging at module level
# DA3 logs timing info during inference - suppress before any imports
import logging
for logger_name in ['dinov2', 'depth_anything_v2', '__main__']:
    try:
        da_logger = logging.getLogger(logger_name)
        da_logger.setLevel(logging.WARNING)
        da_logger.propagate = False
    except:
        pass


def _ensure_da3_package_installed() -> bool:
    """Ensure depth-anything-3 package is installed, auto-install if needed.

    Returns:
        True if package is available (already installed or just installed)
        False if installation failed
    """
    try:
        import depth_anything_3
        return True
    except ImportError:
        logger.warning("Depth Anything V3 package not found, attempting auto-install...")
        logger.info(f"Installing to: {sys.executable}")
        logger.info("Running: pip install --upgrade depth-anything-3 numpy<2.0 trimesh")
        logger.info("This may take a few minutes (downloading ~50MB + dependencies)...")
        logger.info("Note: Using --upgrade to resolve dependency conflicts (numpy, pillow, trimesh)")

        try:
            # Step 1: Install DA3 package WITHOUT xformers (xformers often fails on Python 3.12+)
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                '--upgrade',
                'git+https://github.com/ByteDance-Seed/Depth-Anything-3.git',
                'numpy>=1.23.0,<2.0.0',  # Compatible numpy version
                'trimesh',  # Ensure trimesh uses compatible numpy
            ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            logger.info("✓ Depth Anything V3 package installed successfully to venv")

            # Step 2: Try to install xformers separately (OPTIONAL - don't fail if this fails)
            logger.info("Attempting to install xformers (optional optimization)...")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install',
                    'xformers>=0.0.20',
                ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                logger.info("✓ xformers installed successfully")
            except subprocess.CalledProcessError:
                logger.warning("⚠️  xformers installation failed (common on Python 3.12+)")
                logger.warning("   DA3 will work without it, but may be slower")
                logger.warning("   To install manually (requires CUDA toolkit):")
                logger.warning(f"     {sys.executable} -m pip install xformers")

            # Verify DA3 installation
            import depth_anything_3
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to auto-install DA3 package: {e}")
            logger.error("You may need to install manually:")
            logger.error(f"  {sys.executable} -m pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git")
            return False
        except ImportError as e:
            logger.error("DA3 package installed but import still failed")
            logger.error(f"Import error: {str(e)}")

            # Check if it's a dependency conflict
            if 'numpy' in str(e).lower() or 'trimesh' in str(e).lower():
                logger.error("")
                logger.error("⚠️  Dependency conflict detected (numpy/trimesh).")
                logger.error("Attempting to fix by upgrading conflicting packages...")

                try:
                    # Try to fix by upgrading numpy and trimesh
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install',
                        '--upgrade', '--force-reinstall',
                        'numpy>=1.23.0,<2.0.0',
                        'trimesh'
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                    logger.info("✓ Dependencies upgraded. RESTART THE APPLICATION for changes to take effect.")
                    logger.info("   Then try your generation again.")
                    return False  # Still return False since restart needed

                except subprocess.CalledProcessError:
                    logger.error("Failed to auto-fix dependency conflicts.")
                    logger.error("Manual fix required - restart application after running:")
                    logger.error(f"  {sys.executable} -m pip install --upgrade numpy trimesh")
                    return False
            else:
                logger.error("Try restarting the application")
                return False

# Constants
DEPTH_OUTPUT_FORMAT = (1, 1)  # Target depth tensor format: [1, 1, H, W]
BGR_TO_RGB_SLICE = slice(None, None, -1)  # Reverse color channel order


def _get_model_name(variant: str, size: str) -> str:
    """Get HuggingFace model name for DA3 variant and size.

    Args:
        variant: 'mono', 'any-view', or 'giant' (3DGS capable)
        size: 'small', 'base', 'large', 'giant', or 'nested-giant-large'

    Returns:
        HuggingFace model identifier string
    """
    model_map: Dict[Tuple[str, str], str] = {
        # Mono models - single-view depth only (ONLY LARGE available on HF)
        ('mono', 'small'): 'depth-anything/DA3MONO-LARGE',  # No small variant, use large
        ('mono', 'base'): 'depth-anything/DA3MONO-LARGE',   # No base variant, use large
        ('mono', 'large'): 'depth-anything/DA3MONO-LARGE',
        # Any-view models - multi-view geometry (NOT 3DGS capable)
        ('any-view', 'small'): 'depth-anything/DA3-SMALL',
        ('any-view', 'base'): 'depth-anything/DA3-BASE',
        ('any-view', 'large'): 'depth-anything/DA3-LARGE',
        # GIANT models - 3DGS capable
        ('giant', 'giant'): 'depth-anything/DA3-GIANT',
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


# Module-level cache for DA3 models (key: (variant, model_size, device))
_DA3_MODEL_CACHE = {}


def clear_model_cache():
    """Clear the DA3 model cache. Useful for tests and memory management."""
    global _DA3_MODEL_CACHE
    _DA3_MODEL_CACHE.clear()


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

        # Check cache first
        cache_key = (variant, model_size, str(device))
        if cache_key in _DA3_MODEL_CACHE:
            logger.debug(f"Using cached DA3 model ({variant} {model_size})")
            self.model = _DA3_MODEL_CACHE[cache_key]
            return

        model_name = _get_model_name(variant, model_size)

        logger.info(f"Loading DA3 ({variant} {model_size})...")

        # Ensure DA3 package is installed (auto-install if needed)
        if not _ensure_da3_package_installed():
            raise ImportError(
                "Failed to install depth-anything-3 package. "
                "Try manually: pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git xformers"
            )

        try:
            # Import DA3 (package is now guaranteed to be installed)
            from depth_anything_3.api import DepthAnything3

            # DA3's from_pretrained() uses HF Hub internally (whatever version is in venv)
            # Model files (~1.4GB for Large) auto-download to HF cache on first use
            self.model = DepthAnything3.from_pretrained(model_name)
            self.model.to(device)

            # Cache the model for future use
            _DA3_MODEL_CACHE[cache_key] = self.model

            logger.info(f"✓ DA3 loaded on {device}")

            # Suppress verbose DA3 internal logging after model is ready
            # DA3 logs timing info during inference - suppress to INFO level
            import logging
            for logger_name in ['dinov2', 'depth_anything_v2', '__main__']:
                try:
                    da_logger = logging.getLogger(logger_name)
                    da_logger.setLevel(logging.WARNING)
                    da_logger.propagate = False
                except:
                    pass

        except ImportError as e:
            logger.error(
                "❌ DA3 package installed but import failed. Try restarting the application."
            )
            raise ImportError(
                "depth-anything-3 import failed after installation. "
                "Restart the application and try again."
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
        conf_thresh_percentile: float = 40.0,
        return_full_result: bool = False
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Predict depth map from single image (drop-in replacement for DA2).

        Args:
            image: Input image (numpy array or PIL Image)
            weight: Depth map weight (not used by DA3, kept for compatibility)
            half_precision: Use FP16 precision (not used by DA3, kept for compatibility)
            use_ray_pose: Use ray-based pose estimation (more accurate but slower)
            conf_thresh_percentile: Adaptive confidence threshold percentile (0-100)
            return_full_result: If True, return dict with depth, confidence, rays, etc.

        Returns:
            If return_full_result=False: Depth map tensor [1, 1, H, W] (default)
            If return_full_result=True: Dict with all DA3 outputs
        """
        # Prepare image and extract dimensions
        pil_image, original_h, original_w = _prepare_image_for_inference(image)

        # Run DA3 inference (may downsample internally for processing)
        # Suppress DA3's verbose INFO logging by temporarily redirecting stdout
        import logging
        import sys
        import io

        # Save original loggers and set all known DA3 loggers to WARNING
        saved_levels = {}
        for logger_name in ['dinov2', 'depth_anything_v2', 'depth_anything_3', '__main__', '']:
            try:
                da_logger = logging.getLogger(logger_name)
                saved_levels[logger_name] = da_logger.level
                da_logger.setLevel(logging.CRITICAL)  # Suppress everything except CRITICAL
            except:
                pass

        # Also suppress stdout (DA3 may be using print())
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()  # Redirect to dummy buffer

        try:
            # Pass tuning parameters to DA3 model
            result = self.model.inference(
                [pil_image],
                use_ray_pose=use_ray_pose,
                conf_thresh_percentile=conf_thresh_percentile
            )
        finally:
            # Restore stdout
            sys.stdout = original_stdout

            # Restore logger levels
            for logger_name, level in saved_levels.items():
                try:
                    da_logger = logging.getLogger(logger_name)
                    da_logger.setLevel(level)
                except:
                    pass

        # Extract depth map from Prediction object (dataclass with .depth attribute)
        # result.depth is np.ndarray with shape [N, H, W] where N is number of images
        depth_np = result.depth[0]  # First (and only) image -> [H, W]

        # Convert to tensor format matching DA2 output: [1, 1, H, W]
        depth = _convert_depth_to_tensor(depth_np)

        # Normalize depth to 0-1 range for consistency with DA2
        depth = _normalize_depth_range(depth)

        # Resize depth map to match original image dimensions
        depth = _resize_depth_to_match_image(depth, original_h, original_w)

        # Return full result if requested (for visualization/debugging)
        if return_full_result:
            full_result = {
                'depth': depth,
                'raw_result': result,  # Store original result object
                'original_h': original_h,
                'original_w': original_w,
            }

            # Extract additional data if available (AnyView models only)
            if hasattr(result, 'confidence') and result.confidence is not None:
                full_result['confidence'] = result.confidence[0]  # [H, W]

            if hasattr(result, 'ray_direction') and result.ray_direction is not None:
                full_result['ray_direction'] = result.ray_direction[0]  # [H, W, 3]

            if hasattr(result, 'ray_origin') and result.ray_origin is not None:
                full_result['ray_origin'] = result.ray_origin[0]  # [H, W, 3]

            if hasattr(result, 'extrinsics') and result.extrinsics is not None:
                full_result['extrinsics'] = result.extrinsics[0]  # [4, 4]

            if hasattr(result, 'intrinsics') and result.intrinsics is not None:
                full_result['intrinsics'] = result.intrinsics[0]  # [3, 3]

            return full_result

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

        # Convert Prediction object to dictionary
        # Extract depth maps (convert to tensors for consistency)
        depths = []
        for i in range(len(result.depth)):
            depth_np = result.depth[i]  # [H, W]
            depth = _convert_depth_to_tensor(depth_np)  # [1, 1, H, W]
            depth = _normalize_depth_range(depth)
            depths.append(depth)

        # Build result dictionary
        multiview_result = {
            'depth': depths,
            'confidence': None,
            'camera_extrinsics': None,
            'camera_intrinsics': None,
        }

        # Extract optional multi-view data
        if hasattr(result, 'confidence') and result.confidence is not None:
            multiview_result['confidence'] = result.confidence  # [N, H, W]

        if hasattr(result, 'extrinsics') and result.extrinsics is not None:
            multiview_result['camera_extrinsics'] = result.extrinsics  # [N, 4, 4]

        if hasattr(result, 'intrinsics') and result.intrinsics is not None:
            multiview_result['camera_intrinsics'] = result.intrinsics  # [N, 3, 3]

        return multiview_result

    def estimate_3d_gaussians(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        use_ray_pose: bool = False,
        confidence_threshold: float = 0.0
    ) -> Optional[Any]:
        """Estimate 3D Gaussian Splatting parameters (Phase 3 capability).

        Args:
            images: List of keyframe images
            use_ray_pose: Use DA3 ray head for more accurate camera poses (slower but better geometry)
            confidence_threshold: Filter splats by confidence percentile (0=disabled, 50=top 50%, 90=very confident only)

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

        # Log quality settings if enabled
        if use_ray_pose:
            logger.info("🎯 Using ray pose estimation for more accurate camera poses")
        if confidence_threshold > 0:
            logger.info(f"💎 Filtering splats by confidence threshold: {confidence_threshold}%")

        # Run DA3 inference with 3DGS enabled
        try:
            logger.debug(f"Attempting 3DGS estimation with {len(pil_images)} images...")

            # Build inference kwargs
            inference_kwargs = {"infer_gs": True}

            # Add ray pose estimation if requested (DA3 may support this via inference params)
            if use_ray_pose:
                inference_kwargs["use_ray_pose"] = True

            result = self.model.inference(pil_images, **inference_kwargs)

            # Apply confidence filtering if threshold is set
            if confidence_threshold > 0 and result is not None and hasattr(result, 'gaussians'):
                result = self._filter_gaussians_by_confidence(result, confidence_threshold)

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

    def _filter_gaussians_by_confidence(self, result: Any, confidence_threshold: float) -> Any:
        """Filter 3D gaussians by confidence percentile.

        Args:
            result: Prediction result with .gaussians attribute
            confidence_threshold: Percentile threshold (0-100, e.g., 90 = keep top 10%)

        Returns:
            Filtered result with updated gaussians
        """
        import torch
        import numpy as np

        if not hasattr(result, 'gaussians') or result.gaussians is None:
            logger.warning("No gaussians to filter")
            return result

        gaussians = result.gaussians

        # Try to get confidence/opacity values from gaussians
        # DA3 3DGS may store confidence in various attributes
        confidence_values = None

        if hasattr(gaussians, 'confidence'):
            confidence_values = gaussians.confidence
        elif hasattr(gaussians, 'opacities'):
            # Use opacity as proxy for confidence
            confidence_values = gaussians.opacities
        elif hasattr(gaussians, 'features') and hasattr(gaussians.features, 'confidence'):
            confidence_values = gaussians.features.confidence

        if confidence_values is None:
            logger.warning("Could not find confidence values in gaussians, skipping filtering")
            return result

        # Convert to numpy if needed
        if torch.is_tensor(confidence_values):
            confidence_values = confidence_values.cpu().numpy()

        # Flatten if multi-dimensional
        if len(confidence_values.shape) > 1:
            confidence_values = confidence_values.flatten()

        # Calculate threshold value from percentile
        # If threshold is 90%, we want to keep splats with confidence >= 90th percentile
        threshold_value = np.percentile(confidence_values, confidence_threshold)

        # Create mask for high-confidence splats
        keep_mask = confidence_values >= threshold_value

        num_original = len(confidence_values)
        num_kept = np.sum(keep_mask)
        logger.info(f"💎 Confidence filtering: keeping {num_kept}/{num_original} splats ({num_kept/num_original*100:.1f}%)")
        logger.info(f"   Threshold: {threshold_value:.4f} (top {100-confidence_threshold:.0f}% of splats)")

        # Apply mask to all gaussian attributes
        try:
            # Filter each attribute of gaussians if it exists and has matching shape
            if hasattr(gaussians, 'means') and gaussians.means is not None:
                if torch.is_tensor(gaussians.means):
                    gaussians.means = gaussians.means[keep_mask]
                else:
                    gaussians.means = gaussians.means[keep_mask]

            if hasattr(gaussians, 'rotations') and gaussians.rotations is not None:
                if torch.is_tensor(gaussians.rotations):
                    gaussians.rotations = gaussians.rotations[keep_mask]
                else:
                    gaussians.rotations = gaussians.rotations[keep_mask]

            if hasattr(gaussians, 'scales') and gaussians.scales is not None:
                if torch.is_tensor(gaussians.scales):
                    gaussians.scales = gaussians.scales[keep_mask]
                else:
                    gaussians.scales = gaussians.scales[keep_mask]

            if hasattr(gaussians, 'opacities') and gaussians.opacities is not None:
                if torch.is_tensor(gaussians.opacities):
                    gaussians.opacities = gaussians.opacities[keep_mask]
                else:
                    gaussians.opacities = gaussians.opacities[keep_mask]

            if hasattr(gaussians, 'colors') and gaussians.colors is not None:
                if torch.is_tensor(gaussians.colors):
                    gaussians.colors = gaussians.colors[keep_mask]
                else:
                    gaussians.colors = gaussians.colors[keep_mask]

            if hasattr(gaussians, 'confidence') and gaussians.confidence is not None:
                if torch.is_tensor(gaussians.confidence):
                    gaussians.confidence = gaussians.confidence[keep_mask]
                else:
                    gaussians.confidence = gaussians.confidence[keep_mask]

        except Exception as e:
            logger.warning(f"Error applying confidence filter to gaussians: {e}")
            logger.warning("Returning unfiltered gaussians")
            return result

        return result
