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

from deforum.utils.system.logging import get_logger, emoji_if_enabled
from deforum.utils.system.logging.log import HEX_BLUE, HEX_PURPLE
from deforum.utils.system.logging.themes import get_tqdm_color_for_theme
from deforum.rendering.options import get_log_theme

logger = get_logger()

# Global variable to store depth range for dynamic far plane calculation
_last_depth_range = None


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


def reorient_cameras_to_target(extrinsics: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reorient camera poses to look at a target point (e.g., point cloud centroid).

    Keeps camera positions unchanged but adjusts rotation matrices to point at target.

    Args:
        extrinsics: Camera extrinsics [N, 4, 4] in world-to-camera format
        target: Target position [3] to look at (e.g., point cloud center)

    Returns:
        Reoriented extrinsics [N, 4, 4] with updated rotation matrices
    """
    N = extrinsics.shape[0]
    reoriented = extrinsics.copy()

    for i in range(N):
        # Extract camera position from extrinsic matrix
        # Extrinsic is [R|t] where R is rotation, t is translation
        # Camera position in world coords: -R^T @ t
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        cam_pos = -R.T @ t

        # Calculate direction from camera to target
        forward = target - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Calculate right vector (perpendicular to forward and world up)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)

        # Calculate up vector (perpendicular to forward and right)
        up = np.cross(right, forward)

        # Build new rotation matrix (camera-to-world)
        # Convention: camera looks down -Z axis, Y is up, X is right
        R_new_c2w = np.column_stack([right, up, -forward])

        # Convert to world-to-camera (invert rotation)
        R_new = R_new_c2w.T

        # Recalculate translation for new rotation
        t_new = -R_new @ cam_pos

        # Update extrinsic matrix
        reoriented[i, :3, :3] = R_new
        reoriented[i, :3, 3] = t_new

    return reoriented


def densify_gaussians(gaussians, densification_factor: int, device):
    """Densify gaussians by subdividing each gaussian into multiple splats.

    Args:
        gaussians: Original gaussians object with means, scales, rotations, harmonics, opacities
        densification_factor: How many splats to create per original splat (2 or 3 recommended)
        device: torch device

    Returns:
        Densified gaussians object with increased splat count
    """
    import torch
    from types import SimpleNamespace

    # Extract original parameters
    means = gaussians.means  # [batch, N, 3]
    scales = gaussians.scales  # [batch, N, 3]
    rotations = gaussians.rotations  # [batch, N, 4]
    opacities = gaussians.opacities  # [batch, N] or [batch, N, 1, d_sh]
    sh_coeffs = gaussians.harmonics  # [batch, N, 3, d_sh]

    batch_size, N, _ = means.shape

    # Create subdivisions: for each gaussian, create densification_factor copies with slight offsets
    # Offsets are based on the gaussian's scale to create sub-splats within the original volume
    densified_means = []
    densified_scales = []
    densified_rotations = []
    densified_opacities = []
    densified_sh_coeffs = []

    # Generate offset pattern (e.g., for factor=2: [-0.25, +0.25] along each axis)
    offset_range = 0.3  # Offset as fraction of scale

    for i in range(densification_factor):
        # Random offset within gaussian volume
        offset = torch.randn(batch_size, N, 3, device=device) * offset_range
        offset_world = offset * scales  # Scale offset by gaussian scale

        new_means = means + offset_world
        # Reduce scale of sub-splats to avoid overlap artifacts
        new_scales = scales * (1.0 / (densification_factor ** 0.333))  # Cube root for 3D
        # Keep rotations the same
        new_rotations = rotations.clone()
        # Reduce opacity to account for multiple overlapping splats
        if opacities.dim() == 2:
            new_opacities = opacities / densification_factor
        else:
            new_opacities = opacities.clone()
        # Keep colors the same
        new_sh_coeffs = sh_coeffs.clone()

        densified_means.append(new_means)
        densified_scales.append(new_scales)
        densified_rotations.append(new_rotations)
        densified_opacities.append(new_opacities)
        densified_sh_coeffs.append(new_sh_coeffs)

    # Concatenate all subdivisions along the N dimension
    densified_gaussians = SimpleNamespace(
        means=torch.cat(densified_means, dim=1),  # [batch, N*factor, 3]
        scales=torch.cat(densified_scales, dim=1),
        rotations=torch.cat(densified_rotations, dim=1),
        opacities=torch.cat(densified_opacities, dim=1),
        harmonics=torch.cat(densified_sh_coeffs, dim=1)
    )

    logger.info(f"   Densified gaussians: {N:,} → {N * densification_factor:,} splats (×{densification_factor})")

    return densified_gaussians


def render_novel_view_from_gaussians(
    gaussians,
    camera_pose: np.ndarray,
    camera_intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    device: torch.device,
    densification_factor: int = 1,
    near_clip_distance: float = 0.1
) -> Image.Image:
    """Render a novel view from 3D Gaussian Splatting parameters using gsplat.

    Args:
        gaussians: Gaussians object from DA3 with means, scales, rotations, harmonics, opacities
        camera_pose: Camera extrinsic matrix [4, 4] (world-to-camera)
        camera_intrinsics: Camera intrinsic matrix [3, 3]
        image_size: (width, height)
        device: torch device
        densification_factor: Gaussian densification factor (1-4)
        near_clip_distance: Remove splats closer than this distance (default 0.5)

    Returns:
        Rendered PIL Image
    """
    try:
        import gsplat
        from gsplat import rasterization
    except ImportError:
        logger.warning("gsplat not installed - returning placeholder. Install with: pip install gsplat")
        width, height = image_size
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(placeholder, "gsplat not installed", (width // 4, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return Image.fromarray(placeholder)

    width, height = image_size

    # Apply gaussian densification if requested (subdivide each splat for higher quality)
    if densification_factor > 1:
        gaussians = densify_gaussians(gaussians, densification_factor, device)

    # Extract gaussian parameters (all should be torch tensors on device)
    means = gaussians.means  # [batch, N, 3] - world space positions
    scales = gaussians.scales  # [batch, N, 3] - scale in each axis
    rotations = gaussians.rotations  # [batch, N, 4] - quaternions (w,x,y,z)
    opacities = gaussians.opacities  # [batch, N] or [batch, N, 1, d_sh]
    sh_coeffs = gaussians.harmonics  # [batch, N, 3, d_sh] - spherical harmonics for color

    # gsplat expects batch dimension - squeeze if needed
    if means.dim() == 3 and means.shape[0] == 1:
        means = means.squeeze(0)  # [N, 3]
        scales = scales.squeeze(0)  # [N, 3]
        rotations = rotations.squeeze(0)  # [N, 4]
        sh_coeffs = sh_coeffs.squeeze(0)  # [N, 3, d_sh]

        # Handle opacities carefully - can be [batch, N] or [batch, N, 1, d_sh]
        if opacities.dim() == 4:
            # [batch, N, 1, d_sh] -> squeeze all extra dims -> [N]
            opacities = opacities.squeeze(0).squeeze(-1).squeeze(-1)  # [N]
        elif opacities.dim() == 3:
            # [batch, N, 1] -> [N]
            opacities = opacities.squeeze(0).squeeze(-1)  # [N]
        elif opacities.dim() == 2:
            # [batch, N] -> [N]
            opacities = opacities.squeeze(0)  # [N]

    # Convert camera pose to view matrix (camera-to-world → world-to-camera)
    # DA3 provides extrinsics as [4, 4], gsplat expects viewmat
    viewmat = torch.from_numpy(camera_pose).float().to(device)  # [4, 4]

    # Transform means to camera space to get depth (ALWAYS needed for far plane calculation)
    # viewmat is world-to-camera, so: cam_pos = viewmat @ world_pos
    means_homogeneous = torch.cat([means, torch.ones(means.shape[0], 1, device=device)], dim=1)  # [N, 4]
    means_cam = (viewmat @ means_homogeneous.T).T  # [N, 4]
    depth = means_cam[:, 2]  # Z coordinate in camera space (negative = in front of camera)

    # Debug: Log depth distribution (use INFO so it always shows)
    depth_np = depth.detach().cpu().numpy()
    logger.info(f"   Depth: min={depth_np.min():.2f}, max={depth_np.max():.2f}, "
                f"mean={depth_np.mean():.2f}, median={np.median(depth_np):.2f}")

    # Store depth range for dynamic far plane calculation
    global _last_depth_range
    _last_depth_range = (float(depth_np.min()), float(depth_np.max()))

    # Apply adaptive near-clip filtering to remove splats too close to camera
    # NOTE: near_clip_distance is now interpreted as a PERCENTILE (0.0-1.0), not absolute world units
    # This makes filtering work consistently across DA3's arbitrary scene scales
    if near_clip_distance > 0.0:

        # ADAPTIVE NEAR-CLIP: Use percentile-based filtering instead of absolute world units
        # near_clip_distance interpreted as percentile: 0.01 = remove closest 1% of splats
        # This adapts to DA3's arbitrary scene scale
        if near_clip_distance <= 1.0:
            # Percentile mode: remove closest N% of splats
            percentile = near_clip_distance * 100  # 0.01 -> 1%

            # Only consider negative depths (in front of camera)
            negative_depths = depth_np[depth_np < 0]

            if len(negative_depths) > 0:
                # Calculate threshold as Nth percentile of negative depths
                # Higher (less negative) values are closer to camera
                threshold = np.percentile(negative_depths, 100 - percentile)

                # Keep splats beyond (more negative than) threshold
                # Also keep all positive depths (behind camera) - they're not in front so can't cause artifacts
                depth_tensor = torch.from_numpy(depth_np).to(device)
                mask = (depth_tensor >= 0) | (depth_tensor < threshold)

                kept_pct = (mask.sum().item() / means.shape[0]) * 100
                removed = (~mask).sum().item()

                logger.info(f"   Adaptive near-clip (percentile={percentile:.1f}%): "
                           f"threshold={threshold:.4f}, keeping {kept_pct:.1f}% ({mask.sum()}/{means.shape[0]} splats)")
            else:
                # All splats behind camera, don't filter
                mask = torch.ones(means.shape[0], dtype=torch.bool, device=device)
                logger.info(f"   All splats behind camera, skipping near-clip filter")
        else:
            # Legacy absolute mode (if user sets value > 1.0)
            mask = depth < -near_clip_distance
            logger.info(f"   Absolute near-clip (world units={near_clip_distance:.2f})")

        # Safety check: don't filter out ALL splats (would cause black frame)
        if mask.sum() == 0:
            logger.warning(
                f"   Near-clip filter would remove ALL {means.shape[0]} splats! "
                f"Disabling filter for this frame."
            )
            # DO NOT apply the mask - keep all splats to avoid black frame
        elif mask.sum() < means.shape[0]:
            # Apply mask to all gaussian parameters
            means = means[mask]
            scales = scales[mask]
            rotations = rotations[mask]
            opacities = opacities[mask]
            sh_coeffs = sh_coeffs[mask]

    # Build projection matrix from intrinsics
    fx = float(camera_intrinsics[0, 0])
    fy = float(camera_intrinsics[1, 1])
    cx = float(camera_intrinsics[0, 2])
    cy = float(camera_intrinsics[1, 2])

    # Construct OpenGL-style projection matrix with dynamic far plane
    near = 0.01

    # Calculate far plane based on actual scene depth
    # If we have depth range from near-clip filtering, use it
    if '_last_depth_range' in globals() and _last_depth_range is not None:
        depth_min, depth_max = _last_depth_range
        # depths are negative in camera space (in front of camera)
        # far plane should be abs(depth_min) with some margin
        far = abs(depth_min) * 1.2  # 20% margin beyond furthest splat
        logger.debug(f"   Far plane: {far:.1f} (dynamic, depth {depth_min:.1f}..{depth_max:.1f})")
    else:
        far = 100.0  # Fallback
        logger.debug(f"   Far plane: {far:.1f} (default)")
    projmat = torch.zeros(4, 4, device=device)
    projmat[0, 0] = 2.0 * fx / width
    projmat[1, 1] = 2.0 * fy / height
    projmat[0, 2] = (2.0 * cx / width) - 1.0
    projmat[1, 2] = (2.0 * cy / height) - 1.0
    projmat[2, 2] = -(far + near) / (far - near)
    projmat[2, 3] = -2.0 * far * near / (far - near)
    projmat[3, 2] = -1.0

    # Rasterize gaussians
    # gsplat API: rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height)
    # For SH colors, we need to evaluate them first or use gsplat's SH evaluation

    # Convert SH coeffs to RGB colors (use DC term only for now - simpler)
    # DC term is the first coefficient (index 0) in the SH series
    colors_dc = sh_coeffs[:, :, 0]  # [N, 3] - RGB from DC term
    # Clamp to valid range
    colors_rgb = torch.sigmoid(colors_dc)  # [N, 3]

    # Prepare inputs for gsplat rasterization
    try:
        rendered_image, _, _ = rasterization(
            means=means.unsqueeze(0),  # [1, N, 3]
            quats=rotations.unsqueeze(0),  # [1, N, 4]
            scales=scales.unsqueeze(0),  # [1, N, 3]
            opacities=opacities.unsqueeze(0),  # [1, N]
            colors=colors_rgb.unsqueeze(0),  # [1, N, 3]
            viewmats=viewmat.unsqueeze(0).unsqueeze(0),  # [1, 1, 4, 4] - batch, cameras, 4, 4
            Ks=torch.from_numpy(camera_intrinsics).float().to(device).unsqueeze(0).unsqueeze(0),  # [1, 1, 3, 3]
            width=width,
            height=height,
        )

        # rendered_image is [1, C, H, W, 3] where C=1 (single camera)
        # Squeeze extra dimensions to get [H, W, 3]
        img_tensor = rendered_image.squeeze(0).squeeze(0)  # Remove batch and camera dims
        img_np = img_tensor.detach().cpu().numpy()  # [H, W, 3]
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        return Image.fromarray(img_np)

    except Exception as e:
        logger.error(f"gsplat rasterization failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())

        # Return error placeholder
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"Render failed: {str(e)[:30]}", (10, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return Image.fromarray(placeholder)


def collect_nearby_keyframes(
    all_keyframe_images: dict,
    segment_first_idx: int,
    segment_last_idx: int,
    num_neighbor_segments: int = 1
) -> Tuple[List[Image.Image], List[int]]:
    """Collect keyframes from current segment + neighboring segments for multi-view 3DGS.

    Uses ACTUAL keyframes from the prompt schedule, not an arbitrary count.
    Includes segment boundaries + keyframes from N neighboring segments on each side.

    Args:
        all_keyframe_images: Dict mapping frame_idx -> PIL Image for ALL keyframes
        segment_first_idx: First keyframe index of current segment
        segment_last_idx: Last keyframe index of current segment
        num_neighbor_segments: How many segments to include before/after (0-3)
                               0 = just segment boundaries
                               1 = include 1 segment before + 1 after (default)
                               2 = include 2 segments before + 2 after
                               3 = include 3 segments before + 3 after

    Returns:
        Tuple of (collected_images, collected_indices)

    Example:
        Keyframes: [0, 12, 22, 32, 43, 53]
        Segment: 12→22 (frames 12-22)
        num_neighbor_segments=1 → collect [0, 12, 22, 32] (1 before, segment, 1 after)
        num_neighbor_segments=0 → collect [12, 22] (just segment)
        num_neighbor_segments=2 → collect [0, 12, 22, 32, 43] (2 before, segment, 2 after - clamped to available)
    """
    # Get all available keyframe indices sorted
    available_indices = sorted(all_keyframe_images.keys())

    # Find position of segment boundaries in the full keyframe list
    try:
        first_pos = available_indices.index(segment_first_idx)
        last_pos = available_indices.index(segment_last_idx)
    except ValueError:
        # Segment boundaries not in keyframes - just use them
        logger.warning(f"Segment boundaries {segment_first_idx}-{segment_last_idx} not in keyframe schedule!")
        return (
            [all_keyframe_images[segment_first_idx], all_keyframe_images[segment_last_idx]],
            [segment_first_idx, segment_last_idx]
        )

    # Expand to include neighboring segments
    # Each segment is one keyframe, so N segments = N keyframes on each side
    start_pos = max(0, first_pos - num_neighbor_segments)
    end_pos = min(len(available_indices) - 1, last_pos + num_neighbor_segments)

    # Collect the actual keyframes from the schedule
    collected_indices = available_indices[start_pos:end_pos + 1]

    # Validate that all keyframes have images
    collected_images = []
    missing_keyframes = []
    for idx in collected_indices:
        if idx in all_keyframe_images and all_keyframe_images[idx] is not None:
            collected_images.append(all_keyframe_images[idx])
        else:
            missing_keyframes.append(idx)
            logger.warning(f"   Missing keyframe image at index {idx}, skipping from 3DGS scene")

    # Update collected_indices to only include valid keyframes
    collected_indices = [idx for idx in collected_indices if idx not in missing_keyframes]

    if missing_keyframes:
        logger.warning(f"   Skipped {len(missing_keyframes)} missing keyframes: {missing_keyframes}")

    logger.debug(
        f"   Collected {len(collected_images)} keyframes from schedule: "
        f"indices {collected_indices} "
        f"(segment: {segment_first_idx}-{segment_last_idx}, neighbors: {num_neighbor_segments})"
    )

    return collected_images, collected_indices


def get_da3_model_config(model_selection: str) -> tuple[str, str]:
    """Get DA3 model variant and size from selection name.

    Args:
        model_selection: Model name ('DA3-GIANT')

    Returns:
        Tuple of (variant, size) for model initialization
    """
    if model_selection == 'DA3-GIANT':
        return 'giant', 'giant'
    else:
        logger.warning(f"Unknown model '{model_selection}', using DA3-GIANT")
        return 'giant', 'giant'


def convert_pil_to_bgr(images: List[Image.Image]) -> List[np.ndarray]:
    """Convert PIL images to BGR numpy arrays for DA3.

    Args:
        images: List of PIL images in RGB format

    Returns:
        List of numpy arrays in BGR format
    """
    return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]


def convert_extrinsics_to_4x4(extrinsics: np.ndarray) -> np.ndarray:
    """Convert camera extrinsics from [N, 3, 4] to [N, 4, 4] format.

    Adds bottom row [0, 0, 0, 1] to create homogeneous transformation matrices.

    Args:
        extrinsics: Camera poses [N, 3, 4] or [N, 4, 4]

    Returns:
        Camera poses [N, 4, 4]
    """
    if extrinsics.shape[1:] == (3, 4):
        logger.debug(f"   Converting camera poses from (3, 4) to (4, 4)...")
        num_cameras = extrinsics.shape[0]
        bottom_row = np.array([0, 0, 0, 1], dtype=extrinsics.dtype).reshape(1, 1, 4)
        bottom_rows = np.tile(bottom_row, (num_cameras, 1, 1))  # [N, 1, 4]
        return np.concatenate([extrinsics, bottom_rows], axis=1)  # [N, 4, 4]
    return extrinsics


def get_segment_boundary_poses(
    extrinsics: np.ndarray,
    keyframe_indices: List[int],
    segment_first_idx: int,
    segment_last_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract camera poses for segment boundaries from collected keyframes.

    Args:
        extrinsics: All camera poses [N, 4, 4]
        keyframe_indices: Frame indices of collected keyframes
        segment_first_idx: Start frame index of segment
        segment_last_idx: End frame index of segment

    Returns:
        Tuple of (first_pose, last_pose) for segment boundaries
    """
    try:
        first_idx_pos = keyframe_indices.index(segment_first_idx)
        last_idx_pos = keyframe_indices.index(segment_last_idx)
        first_pose = extrinsics[first_idx_pos]
        last_pose = extrinsics[last_idx_pos]
        logger.debug(
            f"   Using segment boundary poses: "
            f"collected[{first_idx_pos}]={segment_first_idx}, "
            f"collected[{last_idx_pos}]={segment_last_idx}"
        )
        return first_pose, last_pose
    except ValueError:
        logger.warning(
            f"   Segment boundaries {segment_first_idx}-{segment_last_idx} "
            f"not in collected keyframes {keyframe_indices}, using first/last"
        )
        return extrinsics[0], extrinsics[-1]


def scale_intrinsics_to_resolution(
    intrinsics: np.ndarray,
    target_width: int,
    target_height: int
) -> np.ndarray:
    """Scale camera intrinsics from DA3's processing resolution to target resolution.

    DA3 processes images at an internal resolution (e.g., 518x518) and returns
    intrinsics for that resolution. This function infers the processing resolution
    from the principal point and scales all intrinsic parameters appropriately.

    Args:
        intrinsics: Average intrinsics matrix [3, 3] from DA3
        target_width: Target render width in pixels
        target_height: Target render height in pixels

    Returns:
        Scaled intrinsics matrix [3, 3]
    """
    # Extract intrinsics values
    fx_da3 = intrinsics[0, 0]
    fy_da3 = intrinsics[1, 1]
    cx_da3 = intrinsics[0, 2]
    cy_da3 = intrinsics[1, 2]

    logger.debug(
        f"   DA3 intrinsics: fx={fx_da3:.1f}, fy={fy_da3:.1f}, "
        f"cx={cx_da3:.1f}, cy={cy_da3:.1f}"
    )
    logger.debug(f"   Target render size: {target_width}x{target_height}")

    # Infer DA3's processing resolution from principal point
    # Principal point should be roughly at image center: cx ≈ width/2, cy ≈ height/2
    inferred_da3_width = cx_da3 * 2.0
    inferred_da3_height = cy_da3 * 2.0

    logger.debug(
        f"   Inferred DA3 processing size: "
        f"{inferred_da3_width:.0f}x{inferred_da3_height:.0f}"
    )

    # Scale intrinsics to target resolution
    scale_x = target_width / inferred_da3_width
    scale_y = target_height / inferred_da3_height

    fx_scaled = fx_da3 * scale_x
    fy_scaled = fy_da3 * scale_y
    cx_scaled = cx_da3 * scale_x
    cy_scaled = cy_da3 * scale_y

    logger.debug(f"   Scaling factors: x={scale_x:.3f}, y={scale_y:.3f}")
    logger.debug(
        f"   Scaled intrinsics: fx={fx_scaled:.1f}, fy={fy_scaled:.1f}, "
        f"cx={cx_scaled:.1f}, cy={cy_scaled:.1f}"
    )

    # Rebuild intrinsics matrix with scaled values
    return np.array([
        [fx_scaled, 0, cx_scaled],
        [0, fy_scaled, cy_scaled],
        [0, 0, 1]
    ], dtype=np.float32)


def render_3dgs_keyframes(
    gaussians,
    extrinsics: np.ndarray,
    keyframe_indices: List[int],
    segment_first_idx: int,
    segment_last_idx: int,
    avg_intrinsics: np.ndarray,
    image_size: tuple[int, int],
    output_dir: str,
    device: torch.device,
    densification_factor: int,
    near_clip_distance: float = 0.0,
    dashboard=None
) -> List[str]:
    """Render 3DGS versions of segment boundary keyframes for visual consistency.

    Args:
        gaussians: 3D gaussian splat scene
        extrinsics: Camera poses [N, 4, 4]
        keyframe_indices: Indices of collected keyframes
        segment_first_idx: First keyframe index of segment
        segment_last_idx: Last keyframe index of segment
        avg_intrinsics: Camera intrinsics matrix [3, 3]
        image_size: (width, height) for rendering
        output_dir: Directory to save rendered keyframes
        device: Torch device
        densification_factor: Gaussian densification factor

    Returns:
        List of paths to rendered keyframe images
    """
    from tqdm import tqdm
    import modules.shared as shared
    from deforum.rendering import options as opt_utils

    logger.info(f"   Rendering 3DGS keyframes for visual consistency...")

    img_width, img_height = image_size
    keyframe_paths = []

    # Find which collected keyframes match the segment boundaries
    keyframe_to_render = []
    if segment_first_idx in keyframe_indices:
        idx_pos = keyframe_indices.index(segment_first_idx)
        keyframe_to_render.append((segment_first_idx, extrinsics[idx_pos]))
    if segment_last_idx in keyframe_indices and segment_last_idx != segment_first_idx:
        idx_pos = keyframe_indices.index(segment_last_idx)
        keyframe_to_render.append((segment_last_idx, extrinsics[idx_pos]))

    # Use themed tqdm color (blue for keyframes, matching standard tween color)
    theme = get_log_theme()
    bar_color = get_tqdm_color_for_theme(HEX_BLUE, theme)

    # Check if ASCII preview is enabled
    show_ascii = opt_utils.is_dashboard_ascii_preview_enabled()

    # Disable tqdm when using dashboard (dashboard will show progress instead)
    use_dashboard = dashboard is not None

    # Conditionally wrap iterator with tqdm
    if use_dashboard:
        # No tqdm - dashboard shows progress
        iterator = keyframe_to_render
    else:
        # Use tqdm for progress
        iterator = tqdm(
            keyframe_to_render,
            desc="  Rendering 3DGS keyframes",
            unit="keyframe",
            colour=bar_color,
            dynamic_ncols=True,
            file=shared.progress_print_out,
            disable=shared.cmd_opts.disable_console_progressbars
        )

    for idx, (kf_idx, kf_pose) in enumerate(iterator):
        # Update dashboard if available
        if dashboard:
            dashboard.update_3dgs_keyframes(idx + 1, len(keyframe_to_render))
            dashboard.update_vram_from_torch()

        rendered_kf = render_novel_view_from_gaussians(
            gaussians=gaussians,
            camera_pose=kf_pose,
            camera_intrinsics=avg_intrinsics,
            image_size=(img_width, img_height),
            device=device,
            densification_factor=densification_factor,
            near_clip_distance=near_clip_distance
        )
        kf_filename = f"{kf_idx:09d}.png"
        kf_path = os.path.join(output_dir, kf_filename)
        rendered_kf.save(kf_path)
        keyframe_paths.append(kf_path)
        logger.debug(f"   Saved 3DGS keyframe: {kf_filename}")

        # Don't show ASCII preview for 3DGS keyframes (these are re-rendered, not original diffusion)
        # ASCII preview is only shown for original diffusion-generated keyframes in Phase 1

    return keyframe_paths


def render_tween_frames(
    gaussians,
    first_pose: np.ndarray,
    last_pose: np.ndarray,
    target_frame_indices: List[int],
    segment_first_idx: int,
    segment_last_idx: int,
    keyframe_indices: List[int],
    avg_intrinsics: np.ndarray,
    image_size: tuple[int, int],
    output_dir: str,
    device: torch.device,
    densification_factor: int,
    near_clip_distance: float = 0.0,
    dashboard=None,
    tween_poses: List[np.ndarray] = None
) -> List[str]:
    """Render interpolated tween frames between segment boundaries.

    Args:
        gaussians: 3D gaussian splat scene
        first_pose: Camera pose for first segment boundary [4, 4]
        last_pose: Camera pose for last segment boundary [4, 4]
        target_frame_indices: Frame indices to generate
        segment_first_idx: First keyframe index of segment
        segment_last_idx: Last keyframe index of segment
        keyframe_indices: Indices of collected keyframes (for fallback)
        avg_intrinsics: Camera intrinsics matrix [3, 3]
        image_size: (width, height) for rendering
        output_dir: Directory to save rendered frames
        device: Torch device
        densification_factor: Gaussian densification factor
        tween_poses: Precomputed camera poses for tweens (if None, will interpolate)

    Returns:
        List of paths to rendered frame images
    """
    from tqdm import tqdm
    import modules.shared as shared
    from deforum.rendering import options as opt_utils

    img_width, img_height = image_size
    frame_paths = []

    # CRITICAL: Use SEGMENT BOUNDARIES for span, not collected keyframe range
    # This ensures interpolation stays in sync with segment tweens
    if segment_first_idx is not None and segment_last_idx is not None:
        first_frame_idx = segment_first_idx
        last_frame_idx = segment_last_idx
    else:
        # Fallback: use collected keyframe range
        first_frame_idx = keyframe_indices[0]
        last_frame_idx = keyframe_indices[-1]

    total_span = last_frame_idx - first_frame_idx

    # Use themed tqdm color (purple for tweens, matching standard diffusion frame color)
    theme = get_log_theme()
    bar_color = get_tqdm_color_for_theme(HEX_PURPLE, theme)

    # Check if ASCII preview is enabled
    show_ascii = opt_utils.is_dashboard_ascii_preview_enabled()

    # Disable tqdm when using dashboard (dashboard will show progress instead)
    use_dashboard = dashboard is not None

    # Conditionally wrap iterator with tqdm
    if use_dashboard:
        # No tqdm - dashboard shows progress
        iterator = target_frame_indices
    else:
        # Use tqdm for progress
        iterator = tqdm(
            target_frame_indices,
            desc="  Rendering 3DGS tweens",
            unit="frame",
            colour=bar_color,
            dynamic_ncols=True,
            file=shared.progress_print_out,
            disable=shared.cmd_opts.disable_console_progressbars
        )

    for idx, target_idx in enumerate(iterator):
        # Update dashboard if available
        if dashboard:
            dashboard.update_3dgs_tweens(idx + 1, len(target_frame_indices))
            dashboard.update_vram_from_torch()

        # Get camera pose for this frame
        if tween_poses is not None:
            # Use precomputed pose from Deforum schedules
            interp_pose = tween_poses[idx]
        else:
            # Interpolate between segment boundaries (old behavior)
            t = (target_idx - first_frame_idx) / total_span if total_span > 0 else 0.5
            interp_pose = interpolate_camera_pose(first_pose, last_pose, t)

        # Render novel view from 3DGS scene
        rendered_image = render_novel_view_from_gaussians(
            gaussians=gaussians,
            camera_pose=interp_pose,
            camera_intrinsics=avg_intrinsics,
            image_size=(img_width, img_height),
            device=device,
            densification_factor=densification_factor,
            near_clip_distance=near_clip_distance
        )

        # Save frame
        target_filename = f"{target_idx:09d}.png"
        target_path = os.path.join(output_dir, target_filename)
        rendered_image.save(target_path)
        frame_paths.append(target_path)

        # Don't show ASCII preview for tweens (they're 3DGS-rendered, often poor quality)
        # ASCII preview is only shown for diffusion-generated keyframes

    return frame_paths


def generate_da3_3dgs_interpolation(
    keyframe_images: List[Image.Image],
    keyframe_indices: List[int],
    target_frame_indices: List[int],
    model_selection: str,
    output_dir: str,
    device: torch.device,
    render_keyframes: bool = False,
    segment_first_idx: int = None,
    segment_last_idx: int = None,
    densification_factor: int = 1,
    near_clip_distance: float = 0.0,
    dashboard=None,
    deform_keys=None
) -> List[str]:
    """Generate interpolated frames using DA3 3D Gaussian Splatting.

    3DGS interpolation workflow:
    1. Build 3DGS scene from multiple nearby keyframes (via DA3)
    2. Extract camera poses automatically from DA3's depth estimation
    3. Interpolate camera movement between segment boundary keyframes
    4. Render novel views by moving camera through 3DGS scene

    Args:
        keyframe_images: List of PIL Images (keyframes to build scene from)
        keyframe_indices: Global frame indices of keyframes (must match keyframe_images)
        target_frame_indices: Global frame indices to generate (tween frames)
        model_selection: 'DA3-GIANT'
        output_dir: Directory to save generated frames
        device: torch device
        deform_keys: Deforum animation keys (translation/rotation schedules)

    Returns:
        List of paths to generated frame files
    """
    logger.info(f"{emoji_if_enabled('🌌')} DA3-3DGS: {len(keyframe_images)} keyframes {keyframe_indices}, "
                f"{len(target_frame_indices)} targets, model={model_selection}, densify={densification_factor}x, near_clip={near_clip_distance}")

    # Initialize dashboard totals if available
    if dashboard:
        # Phase 2a: 3DGS Build = number of keyframes to process
        dashboard.update_3dgs_build(0, len(keyframe_images))
        # Phase 2b: 3DGS Keyframes = number of segment boundary keyframes (0-2)
        num_boundary_keyframes = 2 if render_keyframes and segment_first_idx != segment_last_idx else 0
        dashboard.update_3dgs_keyframes(0, num_boundary_keyframes)
        # Phase 2c: 3DGS Tweens = number of target frames
        dashboard.update_3dgs_tweens(0, len(target_frame_indices))

    # Log VRAM status
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024 ** 3)
        total_gb = total_mem / (1024 ** 3)
        used_gb = total_gb - free_gb
        logger.info(f"   VRAM: {used_gb:.2f}GB / {total_gb:.2f}GB used ({free_gb:.2f}GB free)")

    # Import DA3 depth model
    from deforum.depth.depth_anything_v3 import DepthAnythingV3

    # Determine variant and size from selection
    variant, size = get_da3_model_config(model_selection)

    # Initialize DA3 GIANT model
    depth_model = DepthAnythingV3(device, model_size=size, variant=variant)

    # Convert PIL images to numpy arrays (BGR for DA3)
    keyframe_arrays = convert_pil_to_bgr(keyframe_images)

    # Build 3DGS scene from all keyframes
    logger.info(f"   Building 3DGS scene from {len(keyframe_arrays)} keyframes...")
    result = depth_model.estimate_3d_gaussians(keyframe_arrays)

    # Update dashboard: scene build complete
    if dashboard:
        dashboard.update_3dgs_build(len(keyframe_images), len(keyframe_images))

    if result is None or result.gaussians is None:
        raise RuntimeError(
            f"Model {model_selection} doesn't support 3DGS (no gs_head/gs_adapter). "
            "This interpolation method requires a 3DGS-capable model."
        )

    # Extract camera poses and intrinsics
    logger.info("   Extracting camera poses from DA3...")
    extrinsics = result.extrinsics  # May be [N, 3, 4] or [N, 4, 4]
    intrinsics = result.intrinsics  # [N, 3, 3]
    gaussians = result.gaussians

    # Log intrinsics (compact)
    fx, fy = intrinsics[0, 0, 0], intrinsics[0, 1, 1]
    cx, cy = intrinsics[0, 0, 2], intrinsics[0, 1, 2]
    img_w, img_h = keyframe_images[0].size
    logger.debug(f"   Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, img={img_w}x{img_h}")

    # Convert [N, 3, 4] to [N, 4, 4] by adding bottom row [0, 0, 0, 1]
    extrinsics = convert_extrinsics_to_4x4(extrinsics)

    logger.info(f"   3DGS: {gaussians.means.shape[1]} splats, {len(extrinsics)} cameras")

    # Calculate point cloud centroid and bounding box
    means = gaussians.means[0].cpu().numpy()  # [N, 3]
    centroid = np.mean(means, axis=0)
    bbox_min = np.min(means, axis=0)
    bbox_max = np.max(means, axis=0)
    scene_extent = bbox_max - bbox_min
    logger.debug(f"   Scene: centroid=({centroid[0]:.1f},{centroid[1]:.1f},{centroid[2]:.1f}), "
                 f"extent=({scene_extent[0]:.1f},{scene_extent[1]:.1f},{scene_extent[2]:.1f})")

    # Log DA3 camera positions for comparison
    da3_cam_positions = []
    for i, ext in enumerate(extrinsics):
        R, t = ext[:3, :3], ext[:3, 3]
        cam_pos = -R.T @ t
        da3_cam_positions.append(cam_pos)
    logger.debug(f"   DA3 cameras: {[f'({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})' for p in da3_cam_positions]}")

    # Choose camera pose strategy: Deforum schedules OR DA3 automatic
    if deform_keys is not None:
        # Use Deforum movement schedules relative to scene center
        from deforum.rendering.deforum_camera_poses import generate_camera_poses_from_deforum_schedules

        # Calculate average distance from DA3 cameras to scene centroid
        # This gives us the "natural" viewing distance for this scene
        da3_distances = [np.linalg.norm(cam_pos - centroid) for cam_pos in da3_cam_positions]
        avg_da3_distance = np.mean(da3_distances)

        first_pose, last_pose, tween_poses_list = generate_camera_poses_from_deforum_schedules(
            keyframe_indices=keyframe_indices,
            segment_first_idx=segment_first_idx,
            segment_last_idx=segment_last_idx,
            target_frame_indices=target_frame_indices,
            deform_keys=deform_keys,
            scene_centroid=centroid,
            scene_bounds=(bbox_min, bbox_max),
            base_camera_distance=avg_da3_distance  # Use DA3's viewing distance
        )

        logger.info(f"   Camera: Deforum schedules at centroid=({centroid[0]:.1f},{centroid[1]:.1f},{centroid[2]:.1f}), "
                    f"{len(tween_poses_list)} poses generated")

    else:
        # Use DA3's automatic pose estimation from depth
        logger.info(f"   Camera: DA3 automatic (Deforum schedules not provided)")

        # CRITICAL: Get camera poses for SEGMENT BOUNDARIES, not collected keyframes
        # We may have collected extras (e.g., [0, 12, 22, 32, 43] for segment 12-22)
        # but we MUST interpolate between segment boundaries to stay in sync
        first_pose, last_pose = get_segment_boundary_poses(
            extrinsics, keyframe_indices, segment_first_idx, segment_last_idx
        )
        tween_poses_list = None  # Will be interpolated in render_tween_frames

    # Use average intrinsics (usually constant across views)
    avg_intrinsics = np.mean(intrinsics, axis=0)

    # Get image dimensions
    img_width, img_height = keyframe_images[0].size

    # CRITICAL FIX: DA3 may return intrinsics for a different resolution than our images
    # Scale intrinsics to match target render resolution
    avg_intrinsics = scale_intrinsics_to_resolution(avg_intrinsics, img_width, img_height)

    # Optionally render 3DGS versions of segment boundary keyframes for visual consistency
    # Note: Original diffusion keyframes are moved to _diffusion/ by the caller
    keyframe_paths = []
    if render_keyframes and segment_first_idx is not None and segment_last_idx is not None:
        keyframe_paths = render_3dgs_keyframes(
            gaussians=gaussians,
            extrinsics=extrinsics,
            keyframe_indices=keyframe_indices,
            segment_first_idx=segment_first_idx,
            segment_last_idx=segment_last_idx,
            avg_intrinsics=avg_intrinsics,
            image_size=(img_width, img_height),
            output_dir=output_dir,
            device=device,
            densification_factor=densification_factor,
            near_clip_distance=near_clip_distance,
            dashboard=dashboard
        )

    # Generate interpolated tween frames
    num_tweens = len(target_frame_indices)
    num_keyframes_rendered = len(keyframe_paths)
    logger.info(
        f"   Rendering {num_tweens} tween views"
        f"{f' + {num_keyframes_rendered} keyframes' if num_keyframes_rendered > 0 else ''}..."
    )

    frame_paths = render_tween_frames(
        gaussians=gaussians,
        first_pose=first_pose,
        last_pose=last_pose,
        target_frame_indices=target_frame_indices,
        segment_first_idx=segment_first_idx,
        segment_last_idx=segment_last_idx,
        keyframe_indices=keyframe_indices,
        avg_intrinsics=avg_intrinsics,
        image_size=(img_width, img_height),
        output_dir=output_dir,
        device=device,
        densification_factor=densification_factor,
        near_clip_distance=near_clip_distance,
        dashboard=dashboard,
        tween_poses=tween_poses_list  # Pass precomputed Deforum schedule poses
    )

    logger.info(
        f"   {emoji_if_enabled('✅')} Generated {len(frame_paths)} tween views"
        f"{f' + {num_keyframes_rendered} keyframes' if num_keyframes_rendered > 0 else ''}"
    )
    return frame_paths
