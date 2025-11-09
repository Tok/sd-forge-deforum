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
- Rotate-around uses quaternion-based look-at targeting center point
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from scipy import interpolate
from deforum.utils.math.quaternion import look_at_target
from deforum.utils.system.logging import log as log_utils


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
    center_z: float = 0.0,
    use_sphere: bool = True,
    frames_per_loop: float = None
) -> List[CameraPoint]:
    """Generate rotate-around camera path on sphere surface.

    The camera moves around a sphere while always looking at center (quaternion-based).

    Args:
        num_frames: Number of frames to generate
        radius: Radius of sphere
        center_x, center_y, center_z: Center position
        height: Additional height offset
        use_sphere: If True, randomize around sphere; if False, flat circle
        frames_per_loop: Frames for one complete rotation
                        If None (default), uses 120 frames per orbit for visible rotation angles

    Returns:
        List of CameraPoint objects with rotations calculated via quaternion look-at
    """
    # Default: ~120 frames per orbit for sharper rotation angles visible to depth warping
    # This creates ~3° per frame rotation instead of ~1°, making depth warping work better
    if frames_per_loop is None:
        frames_per_loop = 120.0
    camera_path = []

    for frame_idx in range(num_frames):
        if use_sphere:
            # Spherical rotation with sinusoidal wobble
            # Theta (azimuth) - horizontal rotation (slowed to 1 loop/min)
            theta = 2 * np.pi * frame_idx / frames_per_loop

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
            # Flat circle (classic mode) - also slowed to 1 loop/min
            angle = 2 * np.pi * frame_idx / frames_per_loop
            x = center_x + radius * np.cos(angle)
            z = center_z + radius * np.sin(angle)
            y = center_y + height

        # Rotation to look at center (using quaternion-based look-at)
        target = (center_x, center_y + height, center_z)
        camera = (x, y, z)
        rot_x, rot_y, rot_z = look_at_target(camera, target)

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
    Camera always faces forward (zero rotation) - fixed to vehicle/body like dashcam/bodycam.

    Args:
        num_frames: Number of frames
        street_length: Total distance traveled
        lane_weave: Amount of left/right weaving (lane changes)
        center_x, center_y, center_z: Starting position

    Returns:
        List of CameraPoint objects
    """
    # Generate camera points with zero rotation (dashcam/bodycam behavior)
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

        camera_path.append(CameraPoint(
            x=x,
            y=y,
            z=z,
            rot_x=0.0,  # Always level (dashcam/bodycam fixed to vehicle/body)
            rot_y=0.0,  # Always facing forward
            rot_z=0.0,  # No roll
            frame=frame_idx
        ))

    return camera_path


def _normalize_angle_delta(delta: float) -> float:
    """Normalize angle delta to shortest rotation (-180 to +180 degrees)."""
    while delta > 180:
        delta -= 360
    while delta < -180:
        delta += 360
    return delta


def _print_camera_path_analysis(delta_analysis: Dict[str, List[float]], camera_path: List[CameraPoint]):
    """Print translation/rotation ratio analysis to console.

    For circular orbit paths, the expected ratio is: rot_y / trans_x ≈ 57.3 / radius
    - radius=100: expect ~0.57
    - radius=50: expect ~1.15
    - radius=25: expect ~2.29

    Args:
        delta_analysis: Dict of delta lists for each axis
        camera_path: Original camera path for frame count
    """
    # Calculate average absolute deltas (ignore direction for ratio calc)
    avg_trans_x = np.mean(np.abs(delta_analysis['translation_x']))
    avg_trans_y = np.mean(np.abs(delta_analysis['translation_y']))
    avg_trans_z = np.mean(np.abs(delta_analysis['translation_z']))
    avg_rot_x = np.mean(np.abs(delta_analysis['rotation_x']))
    avg_rot_y = np.mean(np.abs(delta_analysis['rotation_y']))
    avg_rot_z = np.mean(np.abs(delta_analysis['rotation_z']))

    # Calculate ratios (avoid division by zero)
    def safe_ratio(a, b):
        return a / b if abs(b) > 0.01 else 0.0

    ratio_y_to_x = safe_ratio(avg_rot_y, avg_trans_x)  # Primary orbit ratio
    ratio_x_to_y = safe_ratio(avg_rot_x, avg_trans_y)  # Vertical tilt ratio
    ratio_z_to_z = safe_ratio(avg_rot_z, avg_trans_z)  # Roll ratio

    # Estimate orbit radius from translation magnitude (rough approximation)
    # For circular orbit: trans ≈ 2πR / frames_per_orbit
    # Expected ratio: rot_y / trans_x ≈ 180 / (π * R) ≈ 57.3 / R
    estimated_radius = 57.3 / ratio_y_to_x if ratio_y_to_x > 0.1 else 0.0

    log_utils.info("📊 Camera Path Analysis:", log_utils.BLUE)
    log_utils.info(f"   Frames: {len(camera_path)}", log_utils.BLUE)
    log_utils.info(f"   Avg Translation Deltas: X={avg_trans_x:.3f}, Y={avg_trans_y:.3f}, Z={avg_trans_z:.3f}", log_utils.BLUE)
    log_utils.info(f"   Avg Rotation Deltas: X={avg_rot_x:.3f}°, Y={avg_rot_y:.3f}°, Z={avg_rot_z:.3f}°", log_utils.BLUE)
    log_utils.info("   Translation/Rotation Ratios:", log_utils.BLUE)
    log_utils.info(f"      rot_y / trans_x = {ratio_y_to_x:.2f} (estimated orbit radius: ~{estimated_radius:.0f})", log_utils.BLUE)
    log_utils.info(f"      rot_x / trans_y = {ratio_x_to_y:.2f} (vertical tilt)", log_utils.BLUE)
    log_utils.info(f"      rot_z / trans_z = {ratio_z_to_z:.2f} (roll)", log_utils.BLUE)


def camera_path_to_schedules(
    camera_path: List[CameraPoint],
    speed_multiplier: float = 1.0,
    speed_randomization: float = 0.0,
    random_seed: int = 0
) -> Dict[str, str]:
    """Convert camera path to Deforum schedule strings (ALL DELTAS).

    Both translation and rotation use frame-to-frame deltas.
    Animation engine accumulates these deltas each frame.

    Path is normalized so first frame starts at origin (0,0,0) with zero rotation.

    Args:
        camera_path: List of CameraPoint objects (absolute positions/rotations along curve)
        speed_multiplier: Global speed control (default 1.0)
            - 0.5 = half speed (smoother, slower)
            - 1.0 = normal speed
            - 2.0 = double speed (faster motion)
        speed_randomization: Speed variation amount (default 0.0)
            - 0.0 = uniform frame spacing (no variation)
            - 0.5 = moderate speed oscillation (+/- 50%)
            - 1.0 = maximum speed variation (+/- 100%)
        random_seed: Seed for randomization (default 0 for reproducibility)

    Returns:
        Dict with DELTA schedule strings for each parameter:
        - translation_x, translation_y, translation_z (linear deltas)
        - rotation_3d_x, rotation_3d_y, rotation_3d_z (angle-wrapped deltas)
    """
    if not camera_path:
        return {
            'translation_x': '0: (0)',
            'translation_y': '0: (0)',
            'translation_z': '0: (0)',
            'rotation_3d_x': '0: (0)',
            'rotation_3d_y': '0: (0)',
            'rotation_3d_z': '0: (0)',
        }

    # Normalize path so first frame (in timeline) starts at origin
    # For positions: subtract first frame position
    # For rotations: DON'T normalize - they need to point at center in world space
    #
    # IMPORTANT: For look-at paths (rotate-around, etc.), rotations must be
    # recalculated after position offset to maintain correct look-at target.
    first_point = camera_path[0]
    offset_x = first_point.x
    offset_y = first_point.y
    offset_z = first_point.z

    # Calculate the offset center position
    # Assume center was at (0,0,0) before offset, now at:
    center_offset_x = -offset_x
    center_offset_y = -offset_y
    center_offset_z = -offset_z

    # Detect if this is a look-at path (has non-zero rotations)
    # If all rotations are near-zero, it's a fixed-rotation path (dashcam/bodycam)
    has_rotations = any(
        abs(p.rot_x) > 0.1 or abs(p.rot_y) > 0.1 or abs(p.rot_z) > 0.1
        for p in camera_path
    )

    schedules = {
        'translation_x': [],
        'translation_y': [],
        'translation_z': [],
        'rotation_3d_x': [],
        'rotation_3d_y': [],
        'rotation_3d_z': []
    }

    # Track previous values to calculate frame-to-frame deltas
    prev_x = 0.0
    prev_y = 0.0
    prev_z = 0.0
    prev_rot_x = 0.0
    prev_rot_y = 0.0
    prev_rot_z = 0.0

    # Setup speed randomization if enabled
    if speed_randomization > 0.0:
        np.random.seed(random_seed)
        # Generate smooth speed variation using combination of sine waves
        # This creates natural acceleration/deceleration patterns
        num_frames = len(camera_path)

        # Multi-frequency noise for natural variation
        t = np.linspace(0, 1, num_frames)
        speed_variation = (
            np.sin(2 * np.pi * t * 2) * 0.5 +  # Slow wave
            np.sin(2 * np.pi * t * 5) * 0.3 +  # Medium wave
            np.sin(2 * np.pi * t * 11) * 0.2   # Fast wave
        )
        # Normalize to [-1, 1] range
        speed_variation = speed_variation / np.max(np.abs(speed_variation))
        # Scale by randomization amount: 1.0 +/- randomization
        speed_per_frame = 1.0 + speed_variation * speed_randomization
    else:
        speed_per_frame = None

    # Track deltas for ratio analysis
    delta_analysis = {
        'translation_x': [],
        'translation_y': [],
        'translation_z': [],
        'rotation_x': [],
        'rotation_y': [],
        'rotation_z': []
    }

    for idx, point in enumerate(camera_path):
        # Normalize position (subtract offset so first frame is at origin)
        norm_x = point.x - offset_x
        norm_y = point.y - offset_y
        norm_z = point.z - offset_z

        # Recalculate rotation to point at offset center (for look-at paths only)
        # This maintains look-at relationship after position offset
        if has_rotations:
            # Look-at path: recalculate rotation to point at offset center
            camera_pos = (norm_x, norm_y, norm_z)
            center_pos = (center_offset_x, center_offset_y, center_offset_z)
            norm_rot_x, norm_rot_y, norm_rot_z = look_at_target(camera_pos, center_pos)
        else:
            # Fixed-rotation path (dashcam/bodycam): keep rotation as-is
            norm_rot_x = point.rot_x
            norm_rot_y = point.rot_y
            norm_rot_z = point.rot_z

        # Calculate base deltas
        delta_x = norm_x - prev_x
        delta_y = norm_y - prev_y
        delta_z = norm_z - prev_z
        delta_rot_x = _normalize_angle_delta(norm_rot_x - prev_rot_x)
        delta_rot_y = _normalize_angle_delta(norm_rot_y - prev_rot_y)
        delta_rot_z = _normalize_angle_delta(norm_rot_z - prev_rot_z)

        # Apply speed multiplier (with per-frame variation if enabled)
        if speed_per_frame is not None:
            frame_speed = speed_multiplier * speed_per_frame[idx]
        else:
            frame_speed = speed_multiplier

        # Scale ONLY translation deltas, NOT rotation
        # Rotation deltas must remain unchanged to maintain look-at relationship
        # Otherwise, slow speeds cause camera to drift instead of rotate around center
        delta_x *= frame_speed
        delta_y *= frame_speed
        delta_z *= frame_speed
        # Rotation deltas NOT scaled - they must track center regardless of speed
        # delta_rot_x, delta_rot_y, delta_rot_z unchanged

        # Output all as deltas (scaled by speed multiplier)
        schedules['translation_x'].append(f"{point.frame}: ({delta_x:.2f})")
        schedules['translation_y'].append(f"{point.frame}: ({delta_y:.2f})")
        schedules['translation_z'].append(f"{point.frame}: ({delta_z:.2f})")
        schedules['rotation_3d_x'].append(f"{point.frame}: ({delta_rot_x:.2f})")
        schedules['rotation_3d_y'].append(f"{point.frame}: ({delta_rot_y:.2f})")
        schedules['rotation_3d_z'].append(f"{point.frame}: ({delta_rot_z:.2f})")

        # Track deltas for analysis (skip first frame which is all zeros)
        if idx > 0:
            delta_analysis['translation_x'].append(delta_x)
            delta_analysis['translation_y'].append(delta_y)
            delta_analysis['translation_z'].append(delta_z)
            delta_analysis['rotation_x'].append(delta_rot_x)
            delta_analysis['rotation_y'].append(delta_rot_y)
            delta_analysis['rotation_z'].append(delta_rot_z)

        # Store current as previous for next iteration
        prev_x = norm_x
        prev_y = norm_y
        prev_z = norm_z
        prev_rot_x = norm_rot_x
        prev_rot_y = norm_rot_y
        prev_rot_z = norm_rot_z

    # Analyze translation/rotation ratios for look-at paths
    if has_rotations and len(delta_analysis['translation_x']) > 0:
        _print_camera_path_analysis(delta_analysis, camera_path)

    # Join with commas
    return {
        key: ', '.join(values)
        for key, values in schedules.items()
    }