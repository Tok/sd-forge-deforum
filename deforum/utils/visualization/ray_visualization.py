"""Ray direction visualization utilities for DA3 depth-ray representation.

Provides functions to visualize DA3's ray maps as:
- Color-encoded direction maps (RGB = normalized xyz)
- Arrow overlays showing direction vectors
- Confidence heatmaps
"""

import numpy as np
import cv2
from typing import Optional, Tuple


def normalize_rays(rays: np.ndarray) -> np.ndarray:
    """Normalize ray direction vectors to unit length.

    Args:
        rays: Ray directions [H, W, 3]

    Returns:
        Normalized rays [H, W, 3]
    """
    magnitude = np.linalg.norm(rays, axis=2, keepdims=True)
    magnitude = np.maximum(magnitude, 1e-8)  # Avoid division by zero
    return rays / magnitude


def rays_to_rgb(rays: np.ndarray) -> np.ndarray:
    """Convert ray directions to RGB color encoding.

    Maps 3D direction vectors to RGB colors:
    - X direction → Red channel
    - Y direction → Green channel
    - Z direction → Blue channel

    Args:
        rays: Ray directions [H, W, 3]

    Returns:
        RGB image [H, W, 3] in uint8 format (0-255)
    """
    # Normalize to unit vectors
    rays_norm = normalize_rays(rays)

    # Map from [-1, 1] to [0, 255]
    rgb = ((rays_norm + 1.0) * 127.5).astype(np.uint8)

    return rgb


def draw_ray_arrows(
    depth_image: np.ndarray,
    rays: np.ndarray,
    grid_size: int = 32,
    arrow_scale: float = 20.0,
    arrow_color: Tuple[int, int, int] = (0, 255, 0),
    arrow_thickness: int = 1,
    confidence: Optional[np.ndarray] = None,
    conf_threshold: float = 0.5
) -> np.ndarray:
    """Draw arrow overlay on depth image showing ray directions.

    Args:
        depth_image: Base depth image [H, W, 3] in BGR format
        rays: Ray directions [H, W, 3]
        grid_size: Spacing between arrows in pixels
        arrow_scale: Arrow length multiplier
        arrow_color: Arrow color in BGR format (default: green)
        arrow_thickness: Arrow line thickness
        confidence: Optional confidence map [H, W] (0-1)
        conf_threshold: Minimum confidence to draw arrow

    Returns:
        Depth image with arrow overlay [H, W, 3] in BGR format
    """
    output = depth_image.copy()
    h, w = rays.shape[:2]

    # Normalize rays for consistent arrow lengths
    rays_norm = normalize_rays(rays)

    # Sample grid of points
    for y in range(grid_size // 2, h, grid_size):
        for x in range(grid_size // 2, w, grid_size):
            # Skip if confidence too low
            if confidence is not None and confidence[y, x] < conf_threshold:
                continue

            # Get ray direction at this point
            ray = rays_norm[y, x]

            # Project to 2D (use X and Y components, ignore Z)
            dx = ray[0] * arrow_scale
            dy = ray[1] * arrow_scale

            # Draw arrow
            start_point = (x, y)
            end_point = (int(x + dx), int(y + dy))

            cv2.arrowedLine(
                output,
                start_point,
                end_point,
                arrow_color,
                arrow_thickness,
                tipLength=0.3
            )

    return output


def create_ray_visualization(
    depth_image: np.ndarray,
    rays: np.ndarray,
    mode: str = 'arrows',
    confidence: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """Create ray direction visualization.

    Args:
        depth_image: Base depth image [H, W, 3] in BGR format
        rays: Ray directions [H, W, 3]
        mode: Visualization mode ('arrows', 'color', 'both')
        confidence: Optional confidence map [H, W]
        **kwargs: Additional arguments for specific modes:
            - grid_size: Arrow spacing (default: 32)
            - arrow_scale: Arrow length (default: 20.0)
            - arrow_color: Arrow BGR color (default: green)
            - conf_threshold: Min confidence (default: 0.5)

    Returns:
        Visualization image [H, W, 3] in BGR format

    Raises:
        ValueError: If mode is invalid
    """
    if mode == 'color':
        # Pure color-encoded ray directions
        ray_rgb = rays_to_rgb(rays)
        return cv2.cvtColor(ray_rgb, cv2.COLOR_RGB2BGR)

    elif mode == 'arrows':
        # Arrows on depth image
        return draw_ray_arrows(depth_image, rays, confidence=confidence, **kwargs)

    elif mode == 'both':
        # Color-encoded background with arrow overlay
        ray_rgb = rays_to_rgb(rays)
        ray_bgr = cv2.cvtColor(ray_rgb, cv2.COLOR_RGB2BGR)

        # Blend with depth image (50/50)
        blended = cv2.addWeighted(depth_image, 0.5, ray_bgr, 0.5, 0)

        # Add arrows on top
        return draw_ray_arrows(blended, rays, confidence=confidence, **kwargs)

    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'arrows', 'color', or 'both'")


def create_confidence_heatmap(
    confidence: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Create confidence heatmap visualization.

    Args:
        confidence: Confidence map [H, W] with values 0-1
        colormap: OpenCV colormap (default: COLORMAP_JET)

    Returns:
        Heatmap image [H, W, 3] in BGR format
    """
    # Convert to uint8
    conf_uint8 = (confidence * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(conf_uint8, colormap)

    return heatmap


def create_combined_visualization(
    depth_image: np.ndarray,
    rays: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    show_rays: bool = True,
    show_confidence: bool = False
) -> np.ndarray:
    """Create combined depth + ray + confidence visualization.

    Args:
        depth_image: Base depth image [H, W, 3] in BGR format
        rays: Ray directions [H, W, 3]
        confidence: Optional confidence map [H, W]
        show_rays: Whether to show ray arrows
        show_confidence: Whether to blend confidence heatmap

    Returns:
        Combined visualization [H, W, 3] in BGR format
    """
    output = depth_image.copy()

    # Add confidence heatmap blend
    if show_confidence and confidence is not None:
        heatmap = create_confidence_heatmap(confidence)
        output = cv2.addWeighted(output, 0.7, heatmap, 0.3, 0)

    # Add ray arrows
    if show_rays and rays is not None:
        output = draw_ray_arrows(
            output,
            rays,
            confidence=confidence,
            conf_threshold=0.5,
            arrow_color=(0, 255, 0),
            grid_size=32,
            arrow_scale=20.0
        )

    return output
