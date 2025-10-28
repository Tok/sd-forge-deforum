"""3D Camera Path Spline Generation

Generates smooth camera paths using splines with support for:
- Bezier curves
- Catmull-Rom splines
- Rotate-around presets
- Look-at logic for curve following
- Schedule population for Deforum animation

Key concepts:
- Control points define the path
- Spline interpolation creates smooth curves
- Tangent vectors determine camera direction (look-at)
- Rotate-around uses translation_x with rotation_3d_y at -5x factor
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from scipy import interpolate


@dataclass(frozen=True)
class CameraPoint:
    """Single point on camera path with position and orientation."""
    x: float  # Translation X (left/right)
    y: float  # Translation Y (up/down)
    z: float  # Translation Z (forward/back, zoom)
    rot_x: float  # Rotation 3D X (tilt up/down)
    rot_y: float  # Rotation 3D Y (pan left/right)
    rot_z: float  # Rotation 3D Z (roll)
    frame: int  # Frame number


@dataclass(frozen=True)
class SplineConfig:
    """Configuration for spline generation."""
    num_frames: int
    num_control_points: int  # Number of waypoints along path
    spline_type: str  # "bezier", "catmull_rom", "linear"
    closed_loop: bool  # Whether path loops back to start
    smoothness: float  # 0-1, affects control point spacing


def generate_control_points_circle(
    num_points: int,
    radius: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
    height_variation: float = 0.0
) -> List[Tuple[float, float, float]]:
    """Generate control points in a circular pattern.

    Args:
        num_points: Number of waypoints around circle
        radius: Radius of circle
        center_x, center_y, center_z: Center position
        height_variation: Amount of up/down variation (0 = flat circle)

    Returns:
        List of (x, y, z) tuples
    """
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = center_x + radius * np.cos(angle)
        z = center_z + radius * np.sin(angle)
        # Add height variation using sine wave
        y = center_y + height_variation * np.sin(4 * angle)
        points.append((x, y, z))
    return points


def generate_control_points_figure_eight(
    num_points: int,
    scale: float = 100.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0
) -> List[Tuple[float, float, float]]:
    """Generate control points in a figure-8 pattern.

    Args:
        num_points: Number of waypoints
        scale: Size of figure-8
        center_x, center_y, center_z: Center position

    Returns:
        List of (x, y, z) tuples
    """
    points = []
    for i in range(num_points):
        t = 2 * np.pi * i / num_points
        x = center_x + scale * np.sin(t)
        z = center_z + scale * np.sin(t) * np.cos(t)
        y = center_y + scale * 0.2 * np.sin(2 * t)  # Some height variation
        points.append((x, y, z))
    return points


def catmull_rom_spline(
    control_points: List[Tuple[float, float, float]],
    num_samples: int,
    closed: bool = False,
    tension: float = 0.5
) -> np.ndarray:
    """Generate Catmull-Rom spline through control points.

    Args:
        control_points: List of (x, y, z) waypoints
        num_samples: Number of points to sample along spline
        closed: Whether to close the loop
        tension: 0 (loose) to 1 (tight), affects curve sharpness

    Returns:
        Array of shape (num_samples, 3) with interpolated points
    """
    points = np.array(control_points)

    if closed:
        # Add wraparound points for smooth loop
        points = np.vstack([points[-1:], points, points[:1]])

    # Create parameter values for each control point
    t_control = np.linspace(0, 1, len(points))
    t_sample = np.linspace(0, 1, num_samples)

    # Interpolate each dimension separately
    x_interp = interpolate.CubicSpline(t_control, points[:, 0], bc_type='periodic' if closed else 'not-a-knot')
    y_interp = interpolate.CubicSpline(t_control, points[:, 1], bc_type='periodic' if closed else 'not-a-knot')
    z_interp = interpolate.CubicSpline(t_control, points[:, 2], bc_type='periodic' if closed else 'not-a-knot')

    # Sample the spline
    spline_points = np.column_stack([
        x_interp(t_sample),
        y_interp(t_sample),
        z_interp(t_sample)
    ])

    return spline_points


def calculate_tangent_vectors(spline_points: np.ndarray) -> np.ndarray:
    """Calculate tangent vectors (forward direction) at each point.

    Args:
        spline_points: Array of shape (N, 3) with path points

    Returns:
        Array of shape (N, 3) with normalized tangent vectors
    """
    # Calculate differences between adjacent points
    tangents = np.zeros_like(spline_points)
    tangents[1:-1] = spline_points[2:] - spline_points[:-2]  # Central difference
    tangents[0] = spline_points[1] - spline_points[0]  # Forward difference at start
    tangents[-1] = spline_points[-1] - spline_points[-2]  # Backward difference at end

    # Normalize
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / (norms + 1e-8)  # Avoid division by zero

    return tangents


def tangent_to_rotation(tangent: np.ndarray) -> Tuple[float, float, float]:
    """Convert tangent vector to rotation angles (Euler angles).

    Args:
        tangent: Normalized tangent vector (forward direction)

    Returns:
        (rot_x, rot_y, rot_z) in degrees
    """
    # Extract components
    dx, dy, dz = tangent

    # Pan (rotation_y): horizontal angle
    rot_y = np.degrees(np.arctan2(dx, dz))

    # Tilt (rotation_x): vertical angle
    horizontal_dist = np.sqrt(dx**2 + dz**2)
    rot_x = -np.degrees(np.arctan2(dy, horizontal_dist + 1e-8))

    # Roll (rotation_z): typically 0 for camera following path
    rot_z = 0.0

    return (rot_x, rot_y, rot_z)


def generate_camera_path(
    config: SplineConfig,
    control_points: List[Tuple[float, float, float]],
    look_at_curve: bool = True
) -> List[CameraPoint]:
    """Generate complete camera path with positions and orientations.

    Args:
        config: Spline configuration
        control_points: List of (x, y, z) waypoints
        look_at_curve: If True, camera looks tangent to curve (forward along path)

    Returns:
        List of CameraPoint objects, one per frame
    """
    # Generate spline path
    if config.spline_type == "catmull_rom":
        spline_points = catmull_rom_spline(
            control_points,
            config.num_frames,
            closed=config.closed_loop,
            tension=1.0 - config.smoothness
        )
    else:
        # Default to linear interpolation
        points = np.array(control_points)
        if config.closed_loop:
            points = np.vstack([points, points[:1]])
        t_control = np.linspace(0, 1, len(points))
        t_sample = np.linspace(0, 1, config.num_frames)
        spline_points = np.column_stack([
            np.interp(t_sample, t_control, points[:, 0]),
            np.interp(t_sample, t_control, points[:, 1]),
            np.interp(t_sample, t_control, points[:, 2])
        ])

    # Calculate tangents for look-at
    if look_at_curve:
        tangents = calculate_tangent_vectors(spline_points)

    # Generate camera points
    camera_path = []
    for frame_idx in range(config.num_frames):
        x, y, z = spline_points[frame_idx]

        if look_at_curve:
            rot_x, rot_y, rot_z = tangent_to_rotation(tangents[frame_idx])
        else:
            rot_x = rot_y = rot_z = 0.0

        camera_path.append(CameraPoint(
            x=x,
            y=y,
            z=z,
            rot_x=rot_x,
            rot_y=rot_y,
            rot_z=rot_z,
            frame=frame_idx
        ))

    return camera_path


def generate_rotate_around_path(
    num_frames: int,
    radius: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    height: float = 0.0,
    rotation_factor: float = -5.0,
    center_z: float = 0.0,
    use_sphere: bool = True
) -> List[CameraPoint]:
    """Generate rotate-around camera path on sphere surface.

    The camera moves randomly around a sphere while rotating to look at center.
    Uses spherical coordinates with random variation for interesting paths.

    Args:
        num_frames: Number of frames
        radius: Radius of sphere
        center_x, center_y, center_z: Center position
        height: Additional height offset
        rotation_factor: Rotation multiplier (typically -5 for smooth rotate-around)
        use_sphere: If True, randomize around sphere; if False, flat circle

    Returns:
        List of CameraPoint objects
    """
    camera_path = []

    for frame_idx in range(num_frames):
        if use_sphere:
            # Spherical rotation with random wobble
            # Theta (azimuth) - horizontal rotation
            theta = 2 * np.pi * frame_idx / num_frames

            # Phi (elevation) - varies between -pi/3 and pi/3 (avoid poles)
            # Add sinusoidal variation for interesting paths
            phi_base = np.sin(3 * theta) * (np.pi / 4)  # Oscillate elevation
            phi_noise = np.sin(7 * theta) * (np.pi / 8)  # Add higher frequency wobble
            phi = phi_base + phi_noise

            # Spherical to Cartesian coordinates
            x = center_x + radius * np.cos(phi) * np.cos(theta)
            y = center_y + height + radius * np.sin(phi)
            z = center_z + radius * np.cos(phi) * np.sin(theta)
        else:
            # Flat circle (classic mode)
            angle = 2 * np.pi * frame_idx / num_frames
            x = center_x + radius * np.cos(angle)
            z = center_z + radius * np.sin(angle)
            y = center_y + height

        # Rotation to look at center
        # Calculate angle to look at center point
        dx = center_x - x
        dy = (center_y + height) - y
        dz = center_z - z

        # Pan angle (rotation_y)
        rot_y = np.degrees(np.arctan2(dx, dz))

        # Tilt angle (rotation_x)
        horizontal_dist = np.sqrt(dx**2 + dz**2)
        rot_x = np.degrees(np.arctan2(dy, horizontal_dist))

        rot_z = 0.0

        camera_path.append(CameraPoint(
            x=x,
            y=y,
            z=z,
            rot_x=rot_x,
            rot_y=rot_y,
            rot_z=rot_z,
            frame=frame_idx
        ))

    return camera_path


def generate_street_path(
    num_frames: int,
    street_length: float = 500.0,
    lane_weave: float = 20.0,
    center_x: float = 0.0,
    center_y: float = 10.0,  # Eye level ~10 units
    center_z: float = 0.0
) -> List[CameraPoint]:
    """Generate street/dashcam style forward-moving path.

    Camera moves forward along a street with gentle lane weaving.
    Always faces forward (dashcam/bodycam perspective).

    Args:
        num_frames: Number of frames
        street_length: Total distance traveled
        lane_weave: Amount of left/right weaving (lane changes)
        center_x, center_y, center_z: Starting position

    Returns:
        List of CameraPoint objects
    """
    camera_path = []

    for frame_idx in range(num_frames):
        t = frame_idx / num_frames

        # Forward motion (Z increases)
        z = center_z + t * street_length

        # Gentle left/right weaving (sine wave with some variation)
        x = center_x + lane_weave * np.sin(2 * np.pi * t * 2)  # 2 weaves per path
        x += lane_weave * 0.3 * np.sin(2 * np.pi * t * 5)  # Add high-freq wobble

        # Slight up/down (road bumps)
        y = center_y + 2 * np.sin(2 * np.pi * t * 8)  # Small bumps

        # Camera always faces forward (down +Z axis)
        rot_x = 0.0  # Level horizon
        rot_y = 0.0  # Straight ahead
        rot_z = 0.0  # No roll

        camera_path.append(CameraPoint(
            x=x,
            y=y,
            z=z,
            rot_x=rot_x,
            rot_y=rot_y,
            rot_z=rot_z,
            frame=frame_idx
        ))

    return camera_path


def camera_path_to_schedules(camera_path: List[CameraPoint]) -> Dict[str, str]:
    """Convert camera path to Deforum schedule strings.

    Args:
        camera_path: List of CameraPoint objects

    Returns:
        Dict with schedule strings for each parameter:
        - translation_x, translation_y, translation_z
        - rotation_3d_x, rotation_3d_y, rotation_3d_z
    """
    schedules = {
        'translation_x': [],
        'translation_y': [],
        'translation_z': [],
        'rotation_3d_x': [],
        'rotation_3d_y': [],
        'rotation_3d_z': []
    }

    for point in camera_path:
        schedules['translation_x'].append(f"{point.frame}: ({point.x:.2f})")
        schedules['translation_y'].append(f"{point.frame}: ({point.y:.2f})")
        schedules['translation_z'].append(f"{point.frame}: ({point.z:.2f})")
        schedules['rotation_3d_x'].append(f"{point.frame}: ({point.rot_x:.2f})")
        schedules['rotation_3d_y'].append(f"{point.frame}: ({point.rot_y:.2f})")
        schedules['rotation_3d_z'].append(f"{point.frame}: ({point.rot_z:.2f})")

    # Join with commas
    return {
        key: ', '.join(values)
        for key, values in schedules.items()
    }