"""Frame Overlap Simulator - Calculate preservation/novelty metrics for camera movements.

This module simulates how camera movements (zoom, rotation, translation) affect frame overlap
between consecutive frames. It calculates:
- Preservation %: How much of the previous frame is still visible
- Novelty %: How much new space needs to be generated

Used for:
- Visualizing frame trails ("worm effect") showing previous frame positions
- Auto-tuning camera paths for optimal depth warping performance
- Detecting excessive camera movement that may cause artifacts
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray


# Constants
DEFAULT_TRAIL_LENGTH = 15  # Number of previous frames to show in worm trail
MIN_PRESERVATION_THRESHOLD = 0.3  # Warn if preservation drops below 30%
MAX_NOVELTY_THRESHOLD = 0.7  # Warn if novelty exceeds 70%


@dataclass(frozen=True)
class Rectangle:
    """Immutable rectangle representation.

    Attributes:
        center_x: X coordinate of rectangle center
        center_y: Y coordinate of rectangle center
        width: Rectangle width in pixels
        height: Rectangle height in pixels
        rotation: Rotation angle in degrees (0 = no rotation)
    """
    center_x: float
    center_y: float
    width: float
    height: float
    rotation: float = 0.0

    def get_corners(self) -> NDArray[np.float64]:
        """Get the four corner points of this rectangle.

        Returns:
            4x2 numpy array of corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            Corners are ordered clockwise starting from top-left
        """
        half_w = self.width / 2.0
        half_h = self.height / 2.0

        # Unrotated corners relative to center
        corners = np.array([
            [-half_w, -half_h],  # Top-left
            [half_w, -half_h],   # Top-right
            [half_w, half_h],    # Bottom-right
            [-half_w, half_h],   # Bottom-left
        ], dtype=np.float64)

        # Apply rotation if needed
        if abs(self.rotation) > 0.01:
            angle_rad = np.radians(self.rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ], dtype=np.float64)
            corners = corners @ rotation_matrix.T

        # Translate to actual center position
        corners[:, 0] += self.center_x
        corners[:, 1] += self.center_y

        return corners


@dataclass(frozen=True)
class FrameMetrics:
    """Frame overlap metrics for a single frame transition.

    Attributes:
        frame_index: Frame number
        preservation: Percentage of previous frame still visible (0.0-1.0)
        novelty: Percentage of new space to generate (0.0-1.0)
        overlap_area: Actual overlap area in pixels²
        viewport_area: Total viewport area in pixels²
        prev_frame_rect: Rectangle representing previous frame position
        curr_viewport_rect: Rectangle representing current viewport
    """
    frame_index: int
    preservation: float
    novelty: float
    overlap_area: float
    viewport_area: float
    prev_frame_rect: Rectangle
    curr_viewport_rect: Rectangle


def apply_translation(rect: Rectangle, translation_x: float, translation_y: float) -> Rectangle:
    """Apply translation to a rectangle.

    Args:
        rect: Input rectangle
        translation_x: X-axis translation in pixels
        translation_y: Y-axis translation in pixels

    Returns:
        New rectangle with translation applied
    """
    return Rectangle(
        center_x=rect.center_x + translation_x,
        center_y=rect.center_y + translation_y,
        width=rect.width,
        height=rect.height,
        rotation=rect.rotation
    )


def apply_rotation(rect: Rectangle, rotation_degrees: float) -> Rectangle:
    """Apply rotation to a rectangle.

    Args:
        rect: Input rectangle
        rotation_degrees: Rotation angle in degrees (positive = clockwise)

    Returns:
        New rectangle with rotation applied
    """
    return Rectangle(
        center_x=rect.center_x,
        center_y=rect.center_y,
        width=rect.width,
        height=rect.height,
        rotation=rect.rotation + rotation_degrees
    )


def apply_zoom(rect: Rectangle, zoom_factor: float) -> Rectangle:
    """Apply zoom to a rectangle.

    Zoom semantics:
    - zoom > 1.0 = zoom in (previous frame appears larger, gets cropped)
    - zoom < 1.0 = zoom out (previous frame appears smaller, black borders appear)
    - zoom = 1.0 = no zoom

    Args:
        rect: Input rectangle
        zoom_factor: Zoom multiplier (>1.0 = zoom in, <1.0 = zoom out)

    Returns:
        New rectangle with zoom applied
    """
    return Rectangle(
        center_x=rect.center_x,
        center_y=rect.center_y,
        width=rect.width * zoom_factor,  # Zoom in = larger, zoom out = smaller
        height=rect.height * zoom_factor,
        rotation=rect.rotation
    )


def calculate_polygon_area(corners: NDArray[np.float64]) -> float:
    """Calculate area of a polygon using the shoelace formula.

    Args:
        corners: Nx2 array of polygon vertices [[x1,y1], [x2,y2], ...]

    Returns:
        Area of the polygon in pixels²
    """
    n = len(corners)
    if n < 3:
        return 0.0

    # Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i, 0] * corners[j, 1]
        area -= corners[j, 0] * corners[i, 1]

    return abs(area) / 2.0


def calculate_rectangle_intersection_area(rect1: Rectangle, rect2: Rectangle) -> float:
    """Calculate intersection area between two rectangles.

    This uses the Sutherland-Hodgman polygon clipping algorithm to handle
    rotated rectangles correctly.

    Args:
        rect1: First rectangle
        rect2: Second rectangle

    Returns:
        Intersection area in pixels²
    """
    # For axis-aligned rectangles (no rotation), use simple AABB intersection
    if abs(rect1.rotation) < 0.01 and abs(rect2.rotation) < 0.01:
        return _calculate_aabb_intersection(rect1, rect2)

    # For rotated rectangles, use polygon clipping
    return _calculate_polygon_intersection(rect1, rect2)


def _calculate_aabb_intersection(rect1: Rectangle, rect2: Rectangle) -> float:
    """Fast axis-aligned bounding box intersection (no rotation).

    Args:
        rect1: First rectangle (unrotated)
        rect2: Second rectangle (unrotated)

    Returns:
        Intersection area in pixels²
    """
    # Calculate AABB bounds
    r1_left = rect1.center_x - rect1.width / 2.0
    r1_right = rect1.center_x + rect1.width / 2.0
    r1_top = rect1.center_y - rect1.height / 2.0
    r1_bottom = rect1.center_y + rect1.height / 2.0

    r2_left = rect2.center_x - rect2.width / 2.0
    r2_right = rect2.center_x + rect2.width / 2.0
    r2_top = rect2.center_y - rect2.height / 2.0
    r2_bottom = rect2.center_y + rect2.height / 2.0

    # Calculate intersection bounds
    intersect_left = max(r1_left, r2_left)
    intersect_right = min(r1_right, r2_right)
    intersect_top = max(r1_top, r2_top)
    intersect_bottom = min(r1_bottom, r2_bottom)

    # Check if intersection exists
    if intersect_right <= intersect_left or intersect_bottom <= intersect_top:
        return 0.0

    width = intersect_right - intersect_left
    height = intersect_bottom - intersect_top

    return width * height


def _calculate_polygon_intersection(rect1: Rectangle, rect2: Rectangle) -> float:
    """Calculate intersection area between rotated rectangles using polygon clipping.

    Args:
        rect1: First rectangle
        rect2: Second rectangle

    Returns:
        Intersection area in pixels²
    """
    # Get corner points of both rectangles
    poly1 = rect1.get_corners()
    poly2 = rect2.get_corners()

    # Use Sutherland-Hodgman algorithm to clip poly1 against poly2
    clipped = _sutherland_hodgman_clip(poly1, poly2)

    if len(clipped) < 3:
        return 0.0

    return calculate_polygon_area(clipped)


def _sutherland_hodgman_clip(subject: NDArray[np.float64], clip: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sutherland-Hodgman polygon clipping algorithm.

    Args:
        subject: Subject polygon vertices (Nx2 array)
        clip: Clipping polygon vertices (Mx2 array)

    Returns:
        Clipped polygon vertices (Kx2 array, K may be 0 if no intersection)
    """
    def inside_edge(point: NDArray[np.float64], edge_start: NDArray[np.float64], edge_end: NDArray[np.float64]) -> bool:
        """Check if point is on the inside (left) of an edge."""
        return np.cross(edge_end - edge_start, point - edge_start) >= 0

    def line_intersection(p1: NDArray[np.float64], p2: NDArray[np.float64],
                         p3: NDArray[np.float64], p4: NDArray[np.float64]) -> NDArray[np.float64]:
        """Find intersection point between line segments p1-p2 and p3-p4."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return p1  # Lines are parallel

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float64)

    output = subject.copy()

    # Clip against each edge of the clipping polygon
    for i in range(len(clip)):
        if len(output) == 0:
            break

        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]

        input_list = output
        output = []

        for j in range(len(input_list)):
            current = input_list[j]
            previous = input_list[j - 1]

            current_inside = inside_edge(current, edge_start, edge_end)
            previous_inside = inside_edge(previous, edge_start, edge_end)

            if current_inside:
                if not previous_inside:
                    # Entering: add intersection then current
                    intersection = line_intersection(previous, current, edge_start, edge_end)
                    output.append(intersection)
                output.append(current)
            elif previous_inside:
                # Exiting: add intersection only
                intersection = line_intersection(previous, current, edge_start, edge_end)
                output.append(intersection)

        output = np.array(output, dtype=np.float64) if output else np.array([], dtype=np.float64).reshape(0, 2)

    return output


def calculate_frame_metrics(
    prev_frame_rect: Rectangle,
    translation_x: float,
    translation_y: float,
    rotation_3d_y: float,
    zoom: float,
    viewport_width: float,
    viewport_height: float,
    frame_index: int
) -> FrameMetrics:
    """Calculate preservation/novelty metrics for a single frame transition.

    Args:
        prev_frame_rect: Rectangle representing previous frame position
        translation_x: X-axis translation delta (pixels)
        translation_y: Y-axis translation delta (pixels)
        rotation_3d_y: Y-axis rotation delta (degrees) - horizontal pan
        zoom: Zoom delta (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out)
        viewport_width: Current viewport width (pixels)
        viewport_height: Current viewport height (pixels)
        frame_index: Current frame number

    Returns:
        FrameMetrics object with preservation/novelty calculations
    """
    # Current viewport is always centered at origin
    curr_viewport = Rectangle(
        center_x=0.0,
        center_y=0.0,
        width=viewport_width,
        height=viewport_height,
        rotation=0.0
    )

    # Apply transforms to previous frame in order: translate → rotate → zoom
    transformed = apply_translation(prev_frame_rect, translation_x, translation_y)
    transformed = apply_rotation(transformed, rotation_3d_y)
    transformed = apply_zoom(transformed, zoom)

    # Calculate intersection between transformed previous frame and current viewport
    overlap_area = calculate_rectangle_intersection_area(transformed, curr_viewport)
    viewport_area = viewport_width * viewport_height

    # Calculate metrics
    preservation = overlap_area / viewport_area if viewport_area > 0 else 0.0
    novelty = 1.0 - preservation

    return FrameMetrics(
        frame_index=frame_index,
        preservation=preservation,
        novelty=novelty,
        overlap_area=overlap_area,
        viewport_area=viewport_area,
        prev_frame_rect=transformed,
        curr_viewport_rect=curr_viewport
    )


def simulate_camera_path(
    translation_x_schedule: List[float],
    translation_y_schedule: List[float],
    rotation_3d_y_schedule: List[float],
    zoom_schedule: List[float],
    viewport_width: float,
    viewport_height: float
) -> List[FrameMetrics]:
    """Simulate entire camera path and calculate metrics for all frames.

    Args:
        translation_x_schedule: Per-frame X translation deltas
        translation_y_schedule: Per-frame Y translation deltas
        rotation_3d_y_schedule: Per-frame Y rotation deltas (degrees)
        zoom_schedule: Per-frame zoom factors (1.0 = no zoom)
        viewport_width: Viewport width in pixels
        viewport_height: Viewport height in pixels

    Returns:
        List of FrameMetrics for each frame transition
    """
    num_frames = len(translation_x_schedule)
    metrics: List[FrameMetrics] = []

    # Start with viewport centered at origin
    current_rect = Rectangle(
        center_x=0.0,
        center_y=0.0,
        width=viewport_width,
        height=viewport_height,
        rotation=0.0
    )

    for i in range(num_frames):
        # Calculate metrics for this frame
        frame_metrics = calculate_frame_metrics(
            prev_frame_rect=current_rect,
            translation_x=translation_x_schedule[i],
            translation_y=translation_y_schedule[i],
            rotation_3d_y=rotation_3d_y_schedule[i],
            zoom=zoom_schedule[i],
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            frame_index=i
        )

        metrics.append(frame_metrics)

        # Update current rectangle for next iteration
        current_rect = frame_metrics.prev_frame_rect

    return metrics


def analyze_metrics(metrics: List[FrameMetrics]) -> Dict[str, float]:
    """Analyze frame metrics and calculate summary statistics.

    Args:
        metrics: List of FrameMetrics from simulate_camera_path()

    Returns:
        Dictionary containing:
        - avg_preservation: Average preservation across all frames
        - min_preservation: Minimum preservation (worst case)
        - avg_novelty: Average novelty across all frames
        - max_novelty: Maximum novelty (worst case)
        - preservation_below_threshold: % of frames below MIN_PRESERVATION_THRESHOLD
        - novelty_above_threshold: % of frames above MAX_NOVELTY_THRESHOLD
    """
    if not metrics:
        return {
            'avg_preservation': 0.0,
            'min_preservation': 0.0,
            'avg_novelty': 0.0,
            'max_novelty': 0.0,
            'preservation_below_threshold': 0.0,
            'novelty_above_threshold': 0.0
        }

    preservations = [m.preservation for m in metrics]
    novelties = [m.novelty for m in metrics]

    below_threshold = sum(1 for p in preservations if p < MIN_PRESERVATION_THRESHOLD)
    above_threshold = sum(1 for n in novelties if n > MAX_NOVELTY_THRESHOLD)

    return {
        'avg_preservation': np.mean(preservations),
        'min_preservation': np.min(preservations),
        'avg_novelty': np.mean(novelties),
        'max_novelty': np.max(novelties),
        'preservation_below_threshold': below_threshold / len(metrics),
        'novelty_above_threshold': above_threshold / len(metrics)
    }
