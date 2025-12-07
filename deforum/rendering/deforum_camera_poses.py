"""Generate camera extrinsic matrices from Deforum movement schedules.

This module converts Deforum's translate_x/y/z and rotate_x/y/z schedules into
camera extrinsic matrices suitable for 3DGS rendering.

Key concept: We use DA3's scene centroid as the reference point, then apply
Deforum movements relative to that center. This gives us:
- DA3's excellent 3D scene reconstruction
- Deforum's precise movement control
- Camera Shakify integration
- Consistent camera positioning across varying scene scales
"""

import numpy as np
from typing import List, Tuple, Optional
from deforum.utils.system.logging import get_logger

logger = get_logger()


def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert Euler angles (in degrees) to 3x3 rotation matrix.

    Deforum uses:
    - rx: Rotation around X axis (pitch, up/down)
    - ry: Rotation around Y axis (yaw, left/right)
    - rz: Rotation around Z axis (roll, tilt)

    Args:
        rx: Rotation around X axis in degrees
        ry: Rotation around Y axis in degrees
        rz: Rotation around Z axis in degrees

    Returns:
        3x3 rotation matrix (world-to-camera convention)
    """
    # Convert degrees to radians
    rx_rad = np.radians(rx)
    ry_rad = np.radians(ry)
    rz_rad = np.radians(rz)

    # Rotation matrices for each axis
    # X-axis rotation (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx_rad), -np.sin(rx_rad)],
        [0, np.sin(rx_rad), np.cos(rx_rad)]
    ])

    # Y-axis rotation (yaw)
    Ry = np.array([
        [np.cos(ry_rad), 0, np.sin(ry_rad)],
        [0, 1, 0],
        [-np.sin(ry_rad), 0, np.cos(ry_rad)]
    ])

    # Z-axis rotation (roll)
    Rz = np.array([
        [np.cos(rz_rad), -np.sin(rz_rad), 0],
        [np.sin(rz_rad), np.cos(rz_rad), 0],
        [0, 0, 1]
    ])

    # Combine rotations: R = Rz * Ry * Rx (order matters!)
    R = Rz @ Ry @ Rx

    return R


def look_at_matrix(eye: np.ndarray, center: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """Create a camera look-at rotation matrix.

    Args:
        eye: Camera position [3]
        center: Point to look at [3]
        up: Up vector [3], default [0, 1, 0]

    Returns:
        3x3 rotation matrix pointing camera at center
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])

    # Calculate camera coordinate system
    forward = center - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    up_corrected = np.cross(right, forward)

    # Build rotation matrix (camera-to-world)
    # Camera looks down -Z axis, Y is up, X is right
    R_c2w = np.column_stack([right, up_corrected, -forward])

    # Convert to world-to-camera
    R = R_c2w.T

    return R


def build_extrinsic_matrix(
    position: np.ndarray,
    rotation: np.ndarray
) -> np.ndarray:
    """Build 4x4 extrinsic matrix from position and rotation.

    Args:
        position: Camera position in world coords [3]
        rotation: 3x3 rotation matrix (world-to-camera)

    Returns:
        4x4 extrinsic matrix [R|t]
    """
    # Extrinsic = [R | t] where t = -R @ camera_position
    t = -rotation @ position

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = t

    return extrinsic


def generate_camera_poses_from_deforum_schedules(
    keyframe_indices: List[int],
    segment_first_idx: int,
    segment_last_idx: int,
    target_frame_indices: List[int],
    deform_keys: dict,
    scene_centroid: np.ndarray,
    scene_bounds: Tuple[np.ndarray, np.ndarray],
    base_camera_distance: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Generate camera extrinsic matrices from Deforum movement schedules.

    Strategy:
    1. Start at scene centroid (DA3's center point)
    2. Position camera at appropriate distance from scene
    3. Apply Deforum translate_x/y/z movements
    4. Apply Deforum rotate_x/y/z rotations
    5. Apply Camera Shakify on top (if present)
    6. Generate interpolated poses for tween frames

    Args:
        keyframe_indices: All collected keyframe indices
        segment_first_idx: First keyframe index of segment
        segment_last_idx: Last keyframe index of segment
        target_frame_indices: Tween frame indices to generate
        deform_keys: Deforum animation keys with schedules
        scene_centroid: Center point of DA3's 3D scene [3]
        scene_bounds: (bbox_min, bbox_max) of scene
        base_camera_distance: Optional distance from centroid (auto-calculated if None)

    Returns:
        Tuple of (first_pose, last_pose, tween_poses)
        - first_pose: 4x4 extrinsic for segment first keyframe
        - last_pose: 4x4 extrinsic for segment last keyframe
        - tween_poses: List of 4x4 extrinsics for tween frames
    """
    bbox_min, bbox_max = scene_bounds
    scene_extent = bbox_max - bbox_min
    max_extent = np.max(scene_extent)

    # Auto-calculate camera distance if not provided
    if base_camera_distance is None:
        base_camera_distance = max_extent * 0.75  # 75% of max extent

    # Get schedules for segment boundaries
    first_tx = deform_keys.translation_x_series[segment_first_idx]
    first_ty = deform_keys.translation_y_series[segment_first_idx]
    first_tz = deform_keys.translation_z_series[segment_first_idx]
    first_rx = deform_keys.rotation_3d_x_series[segment_first_idx]
    first_ry = deform_keys.rotation_3d_y_series[segment_first_idx]
    first_rz = deform_keys.rotation_3d_z_series[segment_first_idx]

    last_tx = deform_keys.translation_x_series[segment_last_idx]
    last_ty = deform_keys.translation_y_series[segment_last_idx]
    last_tz = deform_keys.translation_z_series[segment_last_idx]
    last_rx = deform_keys.rotation_3d_x_series[segment_last_idx]
    last_ry = deform_keys.rotation_3d_y_series[segment_last_idx]
    last_rz = deform_keys.rotation_3d_z_series[segment_last_idx]

    logger.debug(f"   Schedules: frames {segment_first_idx}->{segment_last_idx}, "
                f"pos ({first_tx:.1f},{first_ty:.1f},{first_tz:.1f})->({last_tx:.1f},{last_ty:.1f},{last_tz:.1f}), "
                f"rot ({first_rx:.1f},{first_ry:.1f},{first_rz:.1f})->({last_rx:.1f},{last_ry:.1f},{last_rz:.1f})")

    def build_camera_pose(tx: float, ty: float, tz: float,
                         rx: float, ry: float, rz: float) -> np.ndarray:
        """Build camera extrinsic from Deforum schedules.

        Process:
        1. Start at scene centroid (where DA3 cameras are clustered)
        2. Apply Deforum translations (for movement)
        3. Apply Deforum rotations
        4. Build extrinsic matrix

        Note: We use scene_centroid directly as base, not offset by distance.
        DA3's cameras are already near the centroid, so we should be too.
        """
        # Base position: scene centroid (where DA3 cameras are)
        # Apply Deforum translations for movement
        position = scene_centroid + np.array([tx, ty, tz])

        # Apply Deforum rotations
        rotation = euler_to_rotation_matrix(rx, ry, rz)

        # Build extrinsic matrix
        extrinsic = build_extrinsic_matrix(position, rotation)

        return extrinsic

    # Generate poses for segment boundaries
    first_pose = build_camera_pose(first_tx, first_ty, first_tz, first_rx, first_ry, first_rz)
    last_pose = build_camera_pose(last_tx, last_ty, last_tz, last_rx, last_ry, last_rz)

    # Generate interpolated poses for tween frames
    tween_poses = []
    num_tweens = len(target_frame_indices)

    for i, tween_idx in enumerate(target_frame_indices):
        # Calculate interpolation weight (0.0 at first, 1.0 at last)
        t = (tween_idx - segment_first_idx) / (segment_last_idx - segment_first_idx)

        # Interpolate translations
        tx = first_tx + (last_tx - first_tx) * t
        ty = first_ty + (last_ty - first_ty) * t
        tz = first_tz + (last_tz - first_tz) * t

        # Interpolate rotations (linear interpolation of Euler angles)
        # For better results, could use SLERP, but linear is simpler
        rx = first_rx + (last_rx - first_rx) * t
        ry = first_ry + (last_ry - first_ry) * t
        rz = first_rz + (last_rz - first_rz) * t

        tween_pose = build_camera_pose(tx, ty, tz, rx, ry, rz)
        tween_poses.append(tween_pose)

    # Log camera positions for debugging
    def extract_cam_pos(extrinsic: np.ndarray) -> np.ndarray:
        """Extract camera position from extrinsic matrix."""
        R, t = extrinsic[:3, :3], extrinsic[:3, 3]
        return -R.T @ t

    first_cam_pos = extract_cam_pos(first_pose)
    last_cam_pos = extract_cam_pos(last_pose)
    logger.debug(f"   Camera positions: first=({first_cam_pos[0]:.1f},{first_cam_pos[1]:.1f},{first_cam_pos[2]:.1f}), "
                 f"last=({last_cam_pos[0]:.1f},{last_cam_pos[1]:.1f},{last_cam_pos[2]:.1f})")

    return first_pose, last_pose, tween_poses


def apply_camera_shakify(
    extrinsics: List[np.ndarray],
    shakify_pattern: Optional[dict] = None
) -> List[np.ndarray]:
    """Apply Camera Shakify pattern on top of existing camera poses.

    This adds subtle shake/noise to camera movement for realism.

    Args:
        extrinsics: List of 4x4 camera extrinsic matrices
        shakify_pattern: Dict with shake data (amplitude, frequency, etc.)

    Returns:
        List of modified extrinsics with shake applied
    """
    if shakify_pattern is None:
        return extrinsics  # No shake to apply

    # TODO: Implement Camera Shakify integration
    # For now, just return original poses
    logger.debug("   Camera Shakify integration not yet implemented")

    return extrinsics
