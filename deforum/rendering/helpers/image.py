"""Image utilities - Mixed pure and impure functions.

This module contains:
- Pure conversion functions imported from deforum.utils.image.processing
- Impure I/O functions for saving/loading frames (kept here due to side effects)
"""

import os

import cv2
from cv2.typing import MatLike

from deforum.rendering.helpers import filename as filename_utils
from deforum.rendering.data.render_data import RenderData

# Import pure conversion functions from refactored utils module
from deforum.utils.image.processing import (
    bgr_to_rgb,
    numpy_to_pil,
    pil_to_numpy,
    is_PIL,
)
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


def _can_visualize_rays(data: RenderData) -> bool:
    """Check if current depth model supports ray visualization.

    Args:
        data: RenderData with depth model

    Returns:
        True if model is DA3 AnyView/Giant (supports ray maps)
    """
    if data.depth_model is None:
        return False

    # Check if it's DA3 with ray support (not Mono)
    if hasattr(data.depth_model, 'is_v3') and data.depth_model.is_v3:
        # Check for AnyView or Giant variant
        if hasattr(data.depth_model, 'depth_anything'):
            da3_model = data.depth_model.depth_anything
            if hasattr(da3_model, 'variant'):
                variant = da3_model.variant.lower()
                return variant in ['any-view', 'giant']

    return False


def _apply_ray_visualization(data: RenderData, source_image, depth_image, frame):
    """Apply ray direction visualization to depth preview.

    Args:
        data: RenderData with depth model and args
        source_image: Original frame image (for re-running depth prediction)
        depth_image: Depth preview image to overlay rays on
        frame: Current frame (unused, for future extensions)

    Returns:
        Depth image with ray visualization overlay
    """
    try:
        from deforum.utils.visualization import create_combined_visualization
        import numpy as np

        # Re-run depth prediction with full result to get ray data
        # We need the original frame image, not the depth image
        full_result = data.depth_model.depth_anything.predict(
            source_image,
            use_ray_pose=getattr(data.args.anim_args, 'da3_use_ray_pose', False),
            conf_thresh_percentile=getattr(data.args.anim_args, 'da3_conf_thresh_percentile', 40.0),
            return_full_result=True
        )

        # Extract ray data
        rays = full_result.get('ray_direction', None)
        confidence = full_result.get('confidence', None)

        if rays is not None:
            # Convert rays to numpy if needed
            if not isinstance(rays, np.ndarray):
                rays = np.array(rays)

            # Apply visualization
            depth_image = create_combined_visualization(
                depth_image,
                rays,
                confidence=confidence,
                show_rays=True,
                show_confidence=False  # Keep it simple, just rays
            )

            logger.debug(f"Applied ray visualization to depth preview (frame {frame.i})")
        else:
            logger.warning("Ray data not available, skipping visualization")

    except Exception as e:
        logger.warning(f"Failed to apply ray visualization: {e}")
        # Return original depth image on failure
        pass

    return depth_image


def save_cadence_frame(data: RenderData, i: int, image: MatLike, is_overwrite: bool = True):
    filename = filename_utils.frame_filename(data, i)
    save_path: str = os.path.join(data.args.args.outdir, filename)
    if is_overwrite or not os.path.exists(save_path):
        cv2.imwrite(save_path, image)

        # Also copy to frame-preview.png for UI live preview
        preview_path = os.path.join(data.args.args.outdir, "frame-preview.png")
        cv2.imwrite(preview_path, image)


def save_cadence_frame_and_depth_map_if_active(data: RenderData, frame, image):
    import cv2

    save_cadence_frame(data, frame.i, image)

    # Create depth preview whenever depth is available (always in 3D mode)
    # regardless of save_depth_maps setting
    if frame.depth is not None and data.depth_model is not None:
        depth_preview_path = os.path.join(data.args.args.outdir, "depth-preview.png")

        # Convert depth to image format (depth_model has the conversion logic)
        # We need to save to a temp location first to get the converted image
        temp_depth_path = os.path.join(data.args.args.outdir, "_temp_depth.png")
        data.depth_model.save(temp_depth_path, frame.depth)
        # Read as color image (depth_model.save() already converts to uint8)
        depth_image = cv2.imread(temp_depth_path, cv2.IMREAD_COLOR)

        # Check if ray visualization is enabled
        visualize_rays = getattr(data.args.anim_args, 'da3_visualize_rays', False)

        # Add ray visualization if enabled and model supports it
        if visualize_rays and _can_visualize_rays(data):
            depth_image = _apply_ray_visualization(data, image, depth_image, frame)

        # Create preview with optional flow arrows
        show_flow_arrows = getattr(data.args.anim_args, 'show_flow_arrows', False)
        if show_flow_arrows and hasattr(frame, 'cadence_flow') and frame.cadence_flow is not None:
            from deforum.animation.optical_flow_utils import draw_flow_arrows
            depth_with_flow = draw_flow_arrows(depth_image, frame.cadence_flow)
            cv2.imwrite(depth_preview_path, depth_with_flow)
        else:
            cv2.imwrite(depth_preview_path, depth_image)

        # Clean up temp file
        if os.path.exists(temp_depth_path):
            os.remove(temp_depth_path)

    # Save depth maps to depth-maps directory if user wants to keep them
    # NEVER draw arrows on saved depth maps - arrows are ONLY for preview
    if data.args.anim_args.save_depth_maps and frame.depth is not None:
        dm_save_path = os.path.join(data.output_directory, filename_utils.frame_filename(data, frame.i, True))
        data.depth_model.save(dm_save_path, frame.depth)


def load_image(image_path):
    if not os.path.isfile(image_path):
        logger.info(f"File does not exist: {image_path}")
        return None
    return cv2.imread(str(image_path))


def save_and_return_frame(data: RenderData, frame, image):
    save_cadence_frame_and_depth_map_if_active(data, frame, image)
    return image
