"""
Convert Deforum movement schedules to camera poses for 3DGS rendering.

This module bridges Deforum's animation schedules (translation_x/y/z, rotation_3d_x/y/z)
with 3D Gaussian Splatting rendering, allowing users to control camera movement through
Deforum's familiar scheduling system instead of relying on DA3's automatic pose estimation.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, List
from deforum.utils.system.logging import get_logger

logger = get_logger()


def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert Euler angles (degrees) to 3x3 rotation matrix.

    Uses Deforum's rotation convention: X-Y-Z extrinsic rotations (pitch-yaw-roll).

    Args:
        rx: Rotation around X axis in degrees (pitch)
        ry: Rotation around Y axis in degrees (yaw)
        rz: Rotation around Z axis in degrees (roll)

    Returns:
        3x3 rotation matrix
    """
    # Convert degrees to radians
    rx_rad = np.deg2rad(rx)
    ry_rad = np.deg2rad(ry)
    rz_rad = np.deg2rad(rz)

    # Create rotation using scipy (XYZ extrinsic = ZYX intrinsic)
    r = Rotation.from_euler('XYZ', [rx_rad, ry_rad, rz_rad])
    return r.as_matrix()


def deforum_pose_to_extrinsic(
    tx: float,
    ty: float,
    tz: float,
    rx: float,
    ry: float,
    rz: float,
    scene_centroid: np.ndarray,
    scene_scale: float = 1.0
) -> np.ndarray:
    """Convert Deforum translation/rotation to 4x4 camera extrinsic matrix (world-to-camera).

    Deforum convention:
    - Translation: X=right, Y=up, Z=forward (into scene)
    - Rotation: Euler angles in degrees (pitch, yaw, roll)

    3DGS/OpenGL convention (camera space):
    - X=right, Y=up, Z=backward (camera looks down -Z)
    - World-to-camera transform

    Args:
        tx, ty, tz: Camera position in world space
        rx, ry, rz: Additional Deforum rotation values (degrees)
        scene_centroid: Scene center to look at
        scene_scale: Scaling factor to match 3DGS scene coordinates

    Returns:
        4x4 extrinsic matrix [R|t] where camera_pos = viewmat @ world_pos
    """
    # Camera position in world space
    cam_pos_world = np.array([tx, ty, tz])

    # Calculate direction from camera to scene centroid (look-at vector)
    forward = scene_centroid - cam_pos_world
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    # World up vector
    world_up = np.array([0.0, 1.0, 0.0])

    # Calculate right vector (perpendicular to forward and world up)
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)

    # Calculate up vector (perpendicular to forward and right)
    up = np.cross(right, forward)

    # Build camera-to-world rotation matrix (camera looks down -Z, so negate forward)
    # Camera space: X=right, Y=up, Z=backward
    R_cam_to_world = np.column_stack([right, up, -forward])

    # Apply Deforum rotation on top of look-at rotation
    R_deforum = euler_to_rotation_matrix(rx, ry, rz)
    R_cam_to_world = R_cam_to_world @ R_deforum

    # Convert to world-to-camera extrinsic matrix
    R_world_to_cam = R_cam_to_world.T
    t = -R_world_to_cam @ cam_pos_world

    # Build 4x4 extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_world_to_cam
    extrinsic[:3, 3] = t

    return extrinsic


def estimate_scene_scale(
    scene_bbox_min: np.ndarray,
    scene_bbox_max: np.ndarray,
    deforum_translation_range: float = 10.0
) -> float:
    """Estimate appropriate scale factor to map Deforum movement to 3DGS scene.

    DA3's 3DGS scenes can have arbitrary scales (tens to hundreds of units).
    Deforum translations are typically in the range 0-10 units per frame.

    This estimates a scale factor so Deforum movement translates reasonably
    into the 3DGS coordinate system.

    Args:
        scene_bbox_min: Minimum coords of 3DGS scene [x, y, z]
        scene_bbox_max: Maximum coords of 3DGS scene [x, y, z]
        deforum_translation_range: Typical range of Deforum translation values

    Returns:
        Scale factor to multiply Deforum translations by
    """
    # Calculate scene extent (size in each dimension)
    scene_extent = scene_bbox_max - scene_bbox_min

    # Use Z-depth as primary scaling reference (most important for camera movement)
    # If scene is 500 units deep and Deforum uses 0-10, scale = 500/10 = 50
    scene_depth = scene_extent[2]

    # Conservative scaling: use 20% of scene depth as full Deforum range
    # This prevents movement from overshooting the scene
    scale_factor = (scene_depth * 0.2) / deforum_translation_range

    logger.info(f"   Scene scale estimation:")
    logger.info(f"      3DGS scene extent: ({scene_extent[0]:.1f}, {scene_extent[1]:.1f}, {scene_extent[2]:.1f})")
    logger.info(f"      Deforum translation range: {deforum_translation_range:.1f}")
    logger.info(f"      Computed scale factor: {scale_factor:.2f}x")

    return scale_factor


def create_camera_poses_from_deforum_schedules(
    frame_indices: List[int],
    deform_keys,
    scene_bbox_min: np.ndarray,
    scene_bbox_max: np.ndarray,
    scene_centroid: np.ndarray
) -> np.ndarray:
    """Create camera extrinsic matrices from Deforum animation schedules.

    This replaces DA3's automatic pose estimation with user-controlled movement
    based on Deforum's translation_x/y/z and rotation_3d_x/y/z schedules.

    Args:
        frame_indices: List of frame numbers to generate poses for
        deform_keys: Deforum animation keys with translation/rotation series
        scene_bbox_min: 3DGS scene bounding box minimum [x, y, z]
        scene_bbox_max: 3DGS scene bounding box maximum [x, y, z]
        scene_centroid: 3DGS scene center point [x, y, z]

    Returns:
        Camera extrinsics [N, 4, 4] in world-to-camera format
    """
    num_frames = len(frame_indices)

    # Estimate scale factor to map Deforum coords to 3DGS scene coords
    scene_scale = estimate_scene_scale(scene_bbox_min, scene_bbox_max)

    # Initialize output
    extrinsics = np.zeros((num_frames, 4, 4))

    # Get first frame's translation as baseline
    # Offset all movement so camera starts at scene centroid
    first_idx = frame_indices[0]
    tx_base = deform_keys.translation_x_series[first_idx]
    ty_base = deform_keys.translation_y_series[first_idx]
    tz_base = deform_keys.translation_z_series[first_idx]

    # Calculate camera starting position: Place camera in front of scene, not inside it
    # Use scene extent to position camera at a distance where it can see the whole scene
    scene_extent = scene_bbox_max - scene_bbox_min
    camera_distance = max(scene_extent) * 0.5  # Start 50% of max extent away from centroid

    # Camera starts in front of scene (negative Z in world space)
    # Since camera looks at centroid, we offset in NEGATIVE Z direction to be in front
    camera_start_pos = scene_centroid + np.array([0, 0, -camera_distance])

    logger.info(f"   Creating {num_frames} camera poses from Deforum schedules:")
    logger.info(f"      First frame baseline: tx={tx_base:.2f}, ty={ty_base:.2f}, tz={tz_base:.2f}")
    logger.info(f"      Scene centroid: ({scene_centroid[0]:.1f}, {scene_centroid[1]:.1f}, {scene_centroid[2]:.1f})")
    logger.info(f"      Camera start position: ({camera_start_pos[0]:.1f}, {camera_start_pos[1]:.1f}, {camera_start_pos[2]:.1f})")
    logger.info(f"      Camera distance from centroid: {camera_distance:.1f}")

    for i, frame_idx in enumerate(frame_indices):
        # Get Deforum translation/rotation for this frame
        tx = deform_keys.translation_x_series[frame_idx]
        ty = deform_keys.translation_y_series[frame_idx]
        tz = deform_keys.translation_z_series[frame_idx]
        rx = deform_keys.rotation_3d_x_series[frame_idx]
        ry = deform_keys.rotation_3d_y_series[frame_idx]
        rz = deform_keys.rotation_3d_z_series[frame_idx]

        # Convert to relative movement (delta from first frame)
        tx_rel = tx - tx_base
        ty_rel = ty - ty_base
        tz_rel = tz - tz_base

        # Position camera in front of scene and apply Deforum movement
        tx_world = camera_start_pos[0] + tx_rel * scene_scale
        ty_world = camera_start_pos[1] + ty_rel * scene_scale
        tz_world = camera_start_pos[2] + tz_rel * scene_scale

        # Create extrinsic matrix (camera looks at scene centroid)
        extrinsic = deforum_pose_to_extrinsic(
            tx_world, ty_world, tz_world,
            rx, ry, rz,
            scene_centroid=scene_centroid,
            scene_scale=1.0  # Already scaled translations above
        )

        extrinsics[i] = extrinsic

        # Log first and last frames
        if i == 0 or i == num_frames - 1:
            logger.debug(f"      Frame {frame_idx}: pos=({tx_world:.1f}, {ty_world:.1f}, {tz_world:.1f}), "
                        f"rot=({rx:.1f}, {ry:.1f}, {rz:.1f})")

    return extrinsics


def get_interpolated_poses(
    first_frame_idx: int,
    last_frame_idx: int,
    target_frame_indices: List[int],
    deform_keys,
    scene_bbox_min: np.ndarray,
    scene_bbox_max: np.ndarray,
    scene_centroid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Get camera poses for segment boundaries and interpolated tween frames.

    Args:
        first_frame_idx: First keyframe index
        last_frame_idx: Last keyframe index
        target_frame_indices: Tween frame indices to generate
        deform_keys: Deforum animation keys
        scene_bbox_min: 3DGS scene bounding box minimum
        scene_bbox_max: 3DGS scene bounding box maximum
        scene_centroid: 3DGS scene center point

    Returns:
        Tuple of (first_pose, last_pose, tween_poses)
    """
    # Create poses for boundary keyframes
    boundary_indices = [first_frame_idx, last_frame_idx]
    boundary_extrinsics = create_camera_poses_from_deforum_schedules(
        boundary_indices,
        deform_keys,
        scene_bbox_min,
        scene_bbox_max,
        scene_centroid
    )

    first_pose = boundary_extrinsics[0]
    last_pose = boundary_extrinsics[1]

    # Create poses for tween frames
    tween_poses = []
    if target_frame_indices:
        tween_extrinsics = create_camera_poses_from_deforum_schedules(
            target_frame_indices,
            deform_keys,
            scene_bbox_min,
            scene_bbox_max,
            scene_centroid
        )
        tween_poses = [tween_extrinsics[i] for i in range(len(target_frame_indices))]

    return first_pose, last_pose, tween_poses
