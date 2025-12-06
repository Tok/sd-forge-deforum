"""3D Gaussian Splatting tween generation using Depth Anything V3.

This module implements Phase 3 of the DA3 integration:
- Builds full 3D scene representation from keyframes using 3DGS
- Renders tweens from arbitrary camera positions via Gaussian splatting
- Enables complex camera movements (orbital, dramatic zooms, etc.)

Requires:
- Depth Anything V3 (any variant)
- gsplat library for Gaussian rendering
"""

import numpy as np
import torch
from typing import List, Any
from .base import BaseTweenGenerator
from deforum.utils.system.logging import get_logger

logger = get_logger()


class DA3GaussianTweenGenerator(BaseTweenGenerator):
    """Generate tweens using 3D Gaussian Splatting scene reconstruction.

    This is the most advanced tween generation mode:
    1. Collect all keyframes in current sequence
    2. Build 3D Gaussian scene representation using DA3
    3. For each tween, render from camera position defined by Deforum schedules
    4. Result: Perfect alignment with manual camera paths, geometric consistency

    This mode essentially hands tween rendering to DA3's 3DGS system while
    keeping Deforum's precise keyframe scheduling and camera control.
    """

    def __init__(self, depth_model, keyframes: List[np.ndarray] = None):
        """Initialize 3DGS tween generator.

        Args:
            depth_model: DepthModel instance (DA3 required)
            keyframes: List of keyframe images for building 3D scene
        """
        self.depth_model = depth_model
        self.keyframes = keyframes or []
        self.scene_3dgs = None  # Will hold 3D Gaussian scene once built
        self._validate_dependencies()

    def _validate_dependencies(self):
        """Check that DA3 and gsplat are available."""
        # Check DA3
        if not hasattr(self.depth_model, 'is_v3') or not self.depth_model.is_v3:
            logger.error(
                "DA3 Gaussian tween generation requires Depth Anything V3. "
                "Current model is DA2."
            )
            return False

        # Check gsplat
        try:
            import gsplat
            logger.debug(f"gsplat version {gsplat.__version__} available")
        except ImportError:
            logger.error(
                "gsplat library not found. Install with: pip install gsplat>=0.1.0"
            )
            return False

        return True

    def build_scene(self, keyframes: List[np.ndarray], camera_schedules: Any = None):
        """Build 3D Gaussian scene from keyframe images.

        This is called once per sequence to build the 3D representation.
        After building, render_tween() can be called multiple times for different positions.

        Args:
            keyframes: List of diffused keyframe images (BGR numpy arrays)
            camera_schedules: Optional Deforum camera schedules for alignment

        Returns:
            Dictionary with 3DGS parameters or None if failed
        """
        logger.info(f"Building 3D Gaussian scene from {len(keyframes)} keyframes...")

        try:
            # Use DepthModel wrapper method instead of calling depth_anything directly
            self.scene_3dgs = self.depth_model.estimate_3d_gaussians(keyframes)

            if self.scene_3dgs is None:
                logger.error("DA3 failed to estimate 3D Gaussians")
                return None

            # Check if Prediction object has gaussians attribute
            if hasattr(self.scene_3dgs, 'gaussians'):
                gaussian_count = len(self.scene_3dgs.gaussians) if self.scene_3dgs.gaussians is not None else 0
                if gaussian_count == 0:
                    logger.warning(
                        "3D Gaussian scene returned 0 Gaussians - DA3 3DGS feature not yet fully implemented. "
                        "Falling back to depth warp."
                    )
                    return None
                logger.info(
                    f"✓ 3D Gaussian scene built successfully: "
                    f"{gaussian_count} Gaussians from Prediction object"
                )
            elif isinstance(self.scene_3dgs, dict):
                # Legacy dict format
                gaussian_count = len(self.scene_3dgs.get('means', []))
                if gaussian_count == 0:
                    logger.warning(
                        "3D Gaussian scene returned 0 Gaussians - insufficient data. "
                        "Falling back to depth warp."
                    )
                    return None
                logger.info(
                    f"✓ 3D Gaussian scene built successfully: "
                    f"{gaussian_count} Gaussians from dict"
                )
            else:
                logger.info(f"✓ 3D Gaussian scene built (Prediction type: {type(self.scene_3dgs).__name__})")

            return self.scene_3dgs

        except Exception as e:
            logger.error(f"Failed to build 3D Gaussian scene: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def generate_tween(self, data: Any, tween_frame: Any, prev_image: np.ndarray,
                       image: np.ndarray, depth: Any = None) -> np.ndarray:
        """Generate tween by rendering from 3DGS scene.

        Args:
            data: RenderData with animation state
            tween_frame: Tween metadata (value=0-1, index, etc.)
            prev_image: Previous keyframe (not used - scene already built)
            image: Next keyframe (not used - scene already built)
            depth: Pre-computed depth (not used for 3DGS)

        Returns:
            Rendered tween frame from 3DGS (BGR numpy array)
        """
        # Check if scene is built
        if self.scene_3dgs is None:
            logger.warning(
                "3D Gaussian scene not built. Call build_scene() first. "
                "Falling back to linear blend."
            )
            return self._linear_blend_fallback(prev_image, image, tween_frame.value)

        try:
            # Extract camera parameters from Deforum schedules
            camera_params = self._extract_camera_params_from_deforum(
                data,
                tween_frame.i
            )

            # Render view from 3DGS scene
            tween_image = self._render_gaussian_view(
                self.scene_3dgs,
                camera_params,
                data.width(),
                data.height()
            )

            return tween_image

        except Exception as e:
            logger.error(f"3DGS tween rendering failed: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._linear_blend_fallback(prev_image, image, tween_frame.value)

    def _extract_camera_params_from_deforum(self, data, frame_idx):
        """Extract camera parameters from Deforum schedules for given frame.

        Args:
            data: RenderData with animation keys
            frame_idx: Frame index to extract parameters for

        Returns:
            Dictionary with camera parameters:
                - translation: [x, y, z]
                - rotation: [rx, ry, rz] in radians
                - fov: Field of view in degrees
                - aspect_ratio: Width/height
        """
        keys = data.animation_keys.deform_keys

        # Ensure frame_idx is within bounds
        if frame_idx >= len(keys.translation_x_series):
            frame_idx = len(keys.translation_x_series) - 1

        # Extract translation (in Deforum's coordinate system)
        translation_scale = 1.0 / 200.0  # Same as animation.py
        translation = [
            keys.translation_x_series[frame_idx] * translation_scale * -1.0,
            keys.translation_y_series[frame_idx] * translation_scale,
            keys.translation_z_series[frame_idx] * translation_scale * -1.0
        ]

        # Extract rotation (convert to radians)
        import math
        rotation = [
            math.radians(keys.rotation_3d_x_series[frame_idx]),
            math.radians(keys.rotation_3d_y_series[frame_idx]),
            math.radians(keys.rotation_3d_z_series[frame_idx])
        ]

        # Extract FOV and aspect ratio
        fov = keys.fov_series[frame_idx]
        aspect_ratio = keys.aspect_ratio_series[frame_idx]

        return {
            'translation': translation,
            'rotation': rotation,
            'fov': fov,
            'aspect_ratio': aspect_ratio,
            'near': keys.near_series[frame_idx],
            'far': keys.far_series[frame_idx]
        }

    def _render_gaussian_view(self, scene_3dgs, camera_params, width, height):
        """Render novel view from 3D Gaussian scene.

        Args:
            scene_3dgs: 3DGS parameters (means, rotations, scales, colors, opacities)
            camera_params: Camera parameters from Deforum schedules
            width: Output image width
            height: Output image height

        Returns:
            Rendered image (BGR numpy array)
        """
        try:
            import gsplat

            # Build camera matrices from Deforum parameters
            view_matrix = self._build_view_matrix(
                camera_params['translation'],
                camera_params['rotation']
            )

            proj_matrix = self._build_projection_matrix(
                camera_params['fov'],
                camera_params['aspect_ratio'],
                camera_params['near'],
                camera_params['far']
            )

            # Render using gsplat
            rendered = gsplat.rasterize_gaussians(
                means=scene_3dgs['means'],
                quats=scene_3dgs['rotations'],
                scales=scene_3dgs['scales'],
                opacities=scene_3dgs['opacities'],
                colors=scene_3dgs['colors'],
                viewmat=view_matrix,
                projmat=proj_matrix,
                width=width,
                height=height
            )

            # Convert to numpy BGR format
            rendered_rgb = rendered.cpu().numpy()
            rendered_bgr = rendered_rgb[:, :, ::-1]  # RGB -> BGR

            return (rendered_bgr * 255).astype(np.uint8)

        except Exception as e:
            logger.error(f"gsplat rendering failed: {str(e)}")
            raise

    def _build_view_matrix(self, translation, rotation):
        """Build OpenGL view matrix from translation and rotation.

        Args:
            translation: [tx, ty, tz]
            rotation: [rx, ry, rz] in radians

        Returns:
            4x4 view matrix tensor
        """
        import torch
        import math

        # Create rotation matrices for each axis
        rx, ry, rz = rotation

        # Rotation around X axis
        Rx = torch.tensor([
            [1, 0, 0, 0],
            [0, math.cos(rx), -math.sin(rx), 0],
            [0, math.sin(rx), math.cos(rx), 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

        # Rotation around Y axis
        Ry = torch.tensor([
            [math.cos(ry), 0, math.sin(ry), 0],
            [0, 1, 0, 0],
            [-math.sin(ry), 0, math.cos(ry), 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

        # Rotation around Z axis
        Rz = torch.tensor([
            [math.cos(rz), -math.sin(rz), 0, 0],
            [math.sin(rz), math.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

        # Combined rotation
        R = Rz @ Ry @ Rx

        # Translation
        T = torch.eye(4, dtype=torch.float32)
        T[0, 3] = -translation[0]
        T[1, 3] = -translation[1]
        T[2, 3] = -translation[2]

        # View matrix = T * R
        view_matrix = T @ R

        return view_matrix

    def _build_projection_matrix(self, fov, aspect_ratio, near, far):
        """Build OpenGL projection matrix.

        Args:
            fov: Field of view in degrees
            aspect_ratio: Width / height
            near: Near clipping plane
            far: Far clipping plane

        Returns:
            4x4 projection matrix tensor
        """
        import torch
        import math

        fov_rad = math.radians(fov)
        f = 1.0 / math.tan(fov_rad / 2.0)

        proj_matrix = torch.tensor([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=torch.float32)

        return proj_matrix

    def _linear_blend_fallback(self, prev_image, image, t):
        """Simple linear blend fallback when 3DGS unavailable.

        Args:
            prev_image: Previous frame
            image: Next frame
            t: Blend factor (0-1)

        Returns:
            Blended image
        """
        return (prev_image * (1.0 - t) + image * t).astype(np.uint8)

    def supports_mode(self, animation_mode: str) -> bool:
        """Check if 3DGS tweens support this animation mode.

        Args:
            animation_mode: '2D' or '3D'

        Returns:
            True if mode is supported
        """
        # 3DGS works best with 3D mode where camera movements are defined
        # In 2D mode, camera is static so 3DGS doesn't add much value
        return animation_mode == '3D'
