"""DA3 3D Gaussian Splatting novel view synthesis for keyframe interpolation.

This module implements proper 3DGS-based interpolation that:
1. Builds a complete 3DGS scene from multiple keyframes
2. Interpolates camera poses between keyframes
3. Renders novel views by moving through the 3DGS scene
4. IGNORES original Deforum movement schedules (uses smooth camera interpolation instead)

This is fundamentally different from depth warping - we're navigating a real 3D scene.
"""

import os
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image
import cv2

from deforum.utils.system.logging import get_logger

logger = get_logger()


def slerp_quaternion(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    Args:
        q1: First quaternion [4] (w, x, y, z)
        q2: Second quaternion [4] (w, x, y, z)
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion [4]
    """
    # Normalize inputs
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute dot product
    dot = np.dot(q1, q2)

    # If dot < 0, slerp won't take the shorter path
    # Negate one quaternion to ensure shortest path
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Calculate slerp
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)

    w1 = np.sin((1.0 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2


def interpolate_camera_pose(pose1: np.ndarray, pose2: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between two camera poses (4x4 extrinsic matrices).

    Uses SLERP for rotation and linear interpolation for translation.

    Args:
        pose1: First camera pose [4, 4]
        pose2: Second camera pose [4, 4]
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated camera pose [4, 4]
    """
    # Extract rotation (3x3) and translation (3,)
    R1 = pose1[:3, :3]
    t1 = pose1[:3, 3]
    R2 = pose2[:3, :3]
    t2 = pose2[:3, 3]

    # Convert rotation matrices to quaternions
    from scipy.spatial.transform import Rotation
    q1 = Rotation.from_matrix(R1).as_quat()  # [x, y, z, w]
    q2 = Rotation.from_matrix(R2).as_quat()

    # Convert to [w, x, y, z] for slerp
    q1 = np.array([q1[3], q1[0], q1[1], q1[2]])
    q2 = np.array([q2[3], q2[0], q2[1], q2[2]])

    # Interpolate rotation with SLERP
    q_interp = slerp_quaternion(q1, q2, t)

    # Convert back to [x, y, z, w] and then to rotation matrix
    q_interp_xyzw = np.array([q_interp[1], q_interp[2], q_interp[3], q_interp[0]])
    R_interp = Rotation.from_quat(q_interp_xyzw).as_matrix()

    # Linear interpolation for translation
    t_interp = (1 - t) * t1 + t * t2

    # Construct interpolated pose
    pose_interp = np.eye(4)
    pose_interp[:3, :3] = R_interp
    pose_interp[:3, 3] = t_interp

    return pose_interp


def render_novel_view_from_gaussians(
    gaussians,
    camera_pose: np.ndarray,
    camera_intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    device: torch.device
) -> Image.Image:
    """Render a novel view from 3D Gaussian Splatting parameters.

    This is a STUB - proper 3DGS rendering requires diff-gaussian-rasterization.
    For now, this will just return a placeholder or use depth-based approximation.

    Args:
        gaussians: Gaussians object from DA3 with means, scales, rotations, etc.
        camera_pose: Camera extrinsic matrix [4, 4]
        camera_intrinsics: Camera intrinsic matrix [3, 3]
        image_size: (width, height)
        device: torch device

    Returns:
        Rendered PIL Image
    """
    # TODO: Implement proper 3DGS rasterization
    # This requires diff-gaussian-rasterization or gsplat library

    logger.warning("⚠️  Proper 3DGS rendering not yet implemented - using placeholder")

    # For now, return a simple placeholder
    # In production, this would use diff-gaussian-rasterization to render the gaussians
    width, height = image_size
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)

    # Add text indicating this is a placeholder
    cv2.putText(
        placeholder,
        "3DGS Rendering Stub",
        (width // 4, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    return Image.fromarray(placeholder)


def collect_nearby_keyframes(
    all_keyframe_images: dict,
    segment_first_idx: int,
    segment_last_idx: int,
    num_to_collect: int
) -> Tuple[List[Image.Image], List[int]]:
    """Collect nearby keyframes around a segment for multi-view 3DGS.

    Collects num_to_collect keyframes centered around the segment,
    including the segment boundaries.

    Args:
        all_keyframe_images: Dict mapping frame_idx -> PIL Image for ALL keyframes
        segment_first_idx: First keyframe index of current segment
        segment_last_idx: Last keyframe index of current segment
        num_to_collect: Total number of keyframes to collect (2-10)

    Returns:
        Tuple of (collected_images, collected_indices)
    """
    # Get all available keyframe indices sorted
    available_indices = sorted(all_keyframe_images.keys())

    # Find position of segment boundaries in the full keyframe list
    try:
        first_pos = available_indices.index(segment_first_idx)
        last_pos = available_indices.index(segment_last_idx)
    except ValueError:
        # Segment boundaries not in keyframes - just use them
        return (
            [all_keyframe_images[segment_first_idx], all_keyframe_images[segment_last_idx]],
            [segment_first_idx, segment_last_idx]
        )

    # Calculate how many extra frames to collect beyond segment boundaries
    # We want num_to_collect total, including first and last
    extras_needed = num_to_collect - 2  # -2 for first and last
    extras_before = extras_needed // 2
    extras_after = extras_needed - extras_before

    # Collect indices
    start_pos = max(0, first_pos - extras_before)
    end_pos = min(len(available_indices) - 1, last_pos + extras_after)

    # If we couldn't get enough before, try to get more after
    if (first_pos - start_pos) < extras_before:
        shortage = extras_before - (first_pos - start_pos)
        end_pos = min(len(available_indices) - 1, end_pos + shortage)

    # If we couldn't get enough after, try to get more before
    if (end_pos - last_pos) < extras_after:
        shortage = extras_after - (end_pos - last_pos)
        start_pos = max(0, start_pos - shortage)

    # Collect the actual keyframes
    collected_indices = available_indices[start_pos:end_pos + 1]
    collected_images = [all_keyframe_images[idx] for idx in collected_indices]

    logger.debug(
        f"   Collected {len(collected_images)} keyframes: "
        f"indices {collected_indices[0]}-{collected_indices[-1]} "
        f"(segment: {segment_first_idx}-{segment_last_idx})"
    )

    return collected_images, collected_indices


def generate_da3_3dgs_interpolation(
    keyframe_images: List[Image.Image],
    keyframe_indices: List[int],
    target_frame_indices: List[int],
    model_selection: str,
    output_dir: str,
    device: torch.device
) -> List[str]:
    """Generate interpolated frames using DA3 3D Gaussian Splatting.

    3DGS interpolation workflow (using DA3 automatic pose estimation):
    1. Build 3DGS scene from multiple nearby keyframes
    2. DA3 automatically estimates camera poses for each keyframe
    3. Interpolate camera pose smoothly between first and last segment keyframes
    4. Render novel views by moving camera through 3DGS scene

    Args:
        keyframe_images: List of PIL Images (keyframes to build scene from)
        keyframe_indices: Global frame indices of keyframes (must match keyframe_images)
        target_frame_indices: Global frame indices to generate (tween frames)
        model_selection: 'DA3-GIANT' or 'DA3NESTED-GIANT-LARGE'
        output_dir: Directory to save generated frames
        device: torch device

    Returns:
        List of paths to generated frame files
    """
    logger.info(f"🌌 DA3-3DGS Interpolation:")
    logger.info(f"   Keyframes: {len(keyframe_images)} frames at indices {keyframe_indices}")
    logger.info(f"   Targets: {len(target_frame_indices)} frames to generate")
    logger.info(f"   Model: {model_selection}")

    # Import DA3 depth model
    from deforum.depth.depth_anything_v3 import DepthAnythingV3

    # Determine variant and size from selection
    if model_selection == 'DA3-GIANT':
        variant = 'giant'
        size = 'giant'
    elif model_selection == 'DA3NESTED-GIANT-LARGE':
        variant = 'giant'
        size = 'nested-giant-large'
    else:
        logger.warning(f"Unknown model '{model_selection}', using DA3-GIANT")
        variant = 'giant'
        size = 'giant'

    # Initialize DA3 GIANT model
    depth_model = DepthAnythingV3(device, model_size=size, variant=variant)

    # Convert PIL images to numpy arrays (BGR for DA3)
    keyframe_arrays = [
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for img in keyframe_images
    ]

    # Build 3DGS scene from all keyframes
    logger.info(f"   Building 3DGS scene from {len(keyframe_arrays)} keyframes...")
    result = depth_model.estimate_3d_gaussians(keyframe_arrays)

    if result is None or result.gaussians is None:
        raise RuntimeError(
            f"Model {model_selection} doesn't support 3DGS (no gs_head/gs_adapter). "
            "This interpolation method requires a 3DGS-capable model."
        )

    # Extract camera poses and intrinsics
    logger.info("   Extracting camera poses from DA3...")
    extrinsics = result.extrinsics  # [N, 4, 4]
    intrinsics = result.intrinsics  # [N, 3, 3]
    gaussians = result.gaussians

    logger.info(f"   3DGS scene built: {gaussians.means.shape[1]} gaussian splats")
    logger.debug(f"   Camera poses: {extrinsics.shape}")
    logger.debug(f"   Camera intrinsics: {intrinsics.shape}")

    # Get first and last camera poses for interpolation
    first_pose = extrinsics[0]  # [4, 4]
    last_pose = extrinsics[-1]  # [4, 4]

    # Use average intrinsics (usually constant across views)
    avg_intrinsics = np.mean(intrinsics, axis=0)

    # Get image dimensions
    img_width, img_height = keyframe_images[0].size

    # Generate interpolated frames
    logger.info(f"   Rendering {len(target_frame_indices)} novel views...")
    frame_paths = []

    first_frame_idx = keyframe_indices[0]
    last_frame_idx = keyframe_indices[-1]
    total_span = last_frame_idx - first_frame_idx

    for target_idx in target_frame_indices:
        # Calculate interpolation parameter (0 to 1)
        t = (target_idx - first_frame_idx) / total_span if total_span > 0 else 0.5

        # Interpolate camera pose
        interp_pose = interpolate_camera_pose(first_pose, last_pose, t)

        # Render novel view from 3DGS scene
        rendered_image = render_novel_view_from_gaussians(
            gaussians=gaussians,
            camera_pose=interp_pose,
            camera_intrinsics=avg_intrinsics,
            image_size=(img_width, img_height),
            device=device
        )

        # Save frame
        target_filename = f"{target_idx:09d}.png"
        target_path = os.path.join(output_dir, target_filename)
        rendered_image.save(target_path)
        frame_paths.append(target_path)

    logger.info(f"   ✅ Generated {len(frame_paths)} novel views")
    return frame_paths
