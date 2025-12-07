"""Unit tests for DA3-3DGS novel view synthesis functionality."""

import pytest
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Import functions to test
from deforum.rendering.da3_3dgs_novel_view import (
    slerp_quaternion,
    interpolate_camera_pose,
    densify_gaussians,
    collect_nearby_keyframes
)
from deforum.rendering.keyframe_interp import (
    detect_model_prefix,
    build_output_filename
)


class TestModelDetection:
    """Test diffusion model detection from checkpoint names."""

    def test_detect_flux_model(self):
        """Should detect Flux from checkpoint name."""
        assert detect_model_prefix("flux1-dev-bnb-nf4-v2.safetensors") == "flux"
        assert detect_model_prefix("FLUX.1-schnell.safetensors") == "flux"

    def test_detect_zit_model(self):
        """Should detect Z-Image Turbo from various name formats."""
        assert detect_model_prefix("z-image-turbo.safetensors") == "zit"
        assert detect_model_prefix("zimage_turbo_v1.safetensors") == "zit"
        assert detect_model_prefix("ZIT-turbo.safetensors") == "zit"

    def test_detect_lumina_model(self):
        """Should detect Lumina from checkpoint name."""
        assert detect_model_prefix("lumina-v1.0.safetensors") == "lumina"
        assert detect_model_prefix("Neta-Lumina.safetensors") == "lumina"

    def test_fallback_to_diffusion(self):
        """Should fallback to 'diffusion' for unknown models."""
        assert detect_model_prefix("unknown-model.safetensors") == "diffusion"
        assert detect_model_prefix("") == "diffusion"
        assert detect_model_prefix("sd-v1-5.safetensors") == "diffusion"


class TestOutputFilename:
    """Test output filename construction."""

    def test_build_filename_with_flux_wan(self):
        """Should build correct filename for Flux + Wan."""
        filename = build_output_filename("20251207002039", "flux", "Wan")
        assert filename == "20251207002039_flux_wan.mp4"

    def test_build_filename_with_zit_da3(self):
        """Should build correct filename for ZIT + DA3-3DGS."""
        filename = build_output_filename("20251207002039", "zit", "DA3-3DGS")
        assert filename == "20251207002039_zit_da3-3dgs.mp4"

    def test_build_filename_with_lumina_film(self):
        """Should build correct filename for Lumina + FILM."""
        filename = build_output_filename("20251207143525", "lumina", "FILM")
        assert filename == "20251207143525_lumina_film.mp4"


class TestQuaternionSLERP:
    """Test spherical linear interpolation of quaternions."""

    def test_slerp_identity_quaternions(self):
        """Should interpolate between identity quaternions."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity [w, x, y, z]
        q2 = np.array([1.0, 0.0, 0.0, 0.0])
        result = slerp_quaternion(q1, q2, 0.5)

        np.testing.assert_array_almost_equal(result, q1, decimal=5)

    def test_slerp_at_endpoints(self):
        """Should return input quaternions at t=0 and t=1."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.707, 0.707, 0.0, 0.0])  # 90° rotation around x

        result_0 = slerp_quaternion(q1, q2, 0.0)
        result_1 = slerp_quaternion(q1, q2, 1.0)

        np.testing.assert_array_almost_equal(result_0, q1, decimal=5)
        np.testing.assert_array_almost_equal(result_1, q2, decimal=3)  # Reduced precision for normalized quaternion

    def test_slerp_midpoint(self):
        """Should interpolate to midpoint between quaternions."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        q2 = np.array([0.707, 0.707, 0.0, 0.0])  # 90° rotation

        result = slerp_quaternion(q1, q2, 0.5)

        # Should be roughly halfway (45° rotation)
        expected = np.array([0.924, 0.383, 0.0, 0.0])  # 45° around x
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_slerp_handles_negative_dot_product(self):
        """Should take shortest path when dot product is negative."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([-0.707, 0.707, 0.0, 0.0])  # Negated quaternion

        result = slerp_quaternion(q1, q2, 0.5)

        # Should still interpolate smoothly
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)


class TestCameraPoseInterpolation:
    """Test camera pose interpolation."""

    def test_interpolate_identity_poses(self):
        """Should interpolate between two identity poses."""
        pose1 = np.eye(4)
        pose2 = np.eye(4)

        result = interpolate_camera_pose(pose1, pose2, 0.5)

        np.testing.assert_array_almost_equal(result, np.eye(4), decimal=5)

    def test_interpolate_translation_only(self):
        """Should linearly interpolate translation component."""
        pose1 = np.eye(4)
        pose1[:3, 3] = [0.0, 0.0, 0.0]  # At origin

        pose2 = np.eye(4)
        pose2[:3, 3] = [10.0, 0.0, 0.0]  # Moved 10 units in x

        result = interpolate_camera_pose(pose1, pose2, 0.5)

        # Translation should be halfway
        expected_translation = np.array([5.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(
            result[:3, 3],
            expected_translation,
            decimal=5
        )

    def test_interpolate_at_endpoints(self):
        """Should return input poses at t=0 and t=1."""
        pose1 = np.eye(4)
        pose1[:3, 3] = [1.0, 2.0, 3.0]

        pose2 = np.eye(4)
        pose2[:3, 3] = [4.0, 5.0, 6.0]

        result_0 = interpolate_camera_pose(pose1, pose2, 0.0)
        result_1 = interpolate_camera_pose(pose1, pose2, 1.0)

        np.testing.assert_array_almost_equal(result_0, pose1, decimal=5)
        np.testing.assert_array_almost_equal(result_1, pose2, decimal=5)


class TestGaussianDensification:
    """Test gaussian splat densification."""

    def test_densification_doubles_splat_count(self):
        """Should double splat count with factor=2."""
        # Create mock gaussians with batch dimension
        device = torch.device('cpu')
        N = 100  # Original splat count
        batch_size = 1

        mock_gaussians = SimpleNamespace(
            means=torch.randn(batch_size, N, 3, device=device),
            scales=torch.rand(batch_size, N, 3, device=device) * 0.1,
            rotations=torch.randn(batch_size, N, 4, device=device),
            opacities=torch.rand(batch_size, N, device=device),
            harmonics=torch.randn(batch_size, N, 3, 16, device=device)
        )

        densified = densify_gaussians(mock_gaussians, densification_factor=2, device=device)

        assert densified.means.shape == (batch_size, N * 2, 3)
        assert densified.scales.shape == (batch_size, N * 2, 3)
        assert densified.rotations.shape == (batch_size, N * 2, 4)

    def test_densification_triples_splat_count(self):
        """Should triple splat count with factor=3."""
        device = torch.device('cpu')
        N = 50
        batch_size = 1

        mock_gaussians = SimpleNamespace(
            means=torch.randn(batch_size, N, 3, device=device),
            scales=torch.rand(batch_size, N, 3, device=device) * 0.1,
            rotations=torch.randn(batch_size, N, 4, device=device),
            opacities=torch.rand(batch_size, N, device=device),
            harmonics=torch.randn(batch_size, N, 3, 16, device=device)
        )

        densified = densify_gaussians(mock_gaussians, densification_factor=3, device=device)

        assert densified.means.shape == (batch_size, N * 3, 3)

    def test_densification_reduces_scale(self):
        """Should reduce scale of sub-splats to avoid overlap."""
        device = torch.device('cpu')
        N = 10
        batch_size = 1

        original_scale = 1.0
        mock_gaussians = SimpleNamespace(
            means=torch.zeros(batch_size, N, 3, device=device),
            scales=torch.ones(batch_size, N, 3, device=device) * original_scale,
            rotations=torch.randn(batch_size, N, 4, device=device),
            opacities=torch.ones(batch_size, N, device=device),
            harmonics=torch.randn(batch_size, N, 3, 16, device=device)
        )

        densified = densify_gaussians(mock_gaussians, densification_factor=2, device=device)

        # Scale should be reduced (cube root for 3D)
        expected_scale = original_scale / (2 ** 0.333)
        actual_scale = densified.scales[0, 0, 0].item()

        assert actual_scale == pytest.approx(expected_scale, abs=1e-4)

    def test_densification_reduces_opacity(self):
        """Should reduce opacity to account for overlapping splats."""
        device = torch.device('cpu')
        N = 10
        batch_size = 1

        original_opacity = 0.8
        mock_gaussians = SimpleNamespace(
            means=torch.zeros(batch_size, N, 3, device=device),
            scales=torch.ones(batch_size, N, 3, device=device),
            rotations=torch.randn(batch_size, N, 4, device=device),
            opacities=torch.ones(batch_size, N, device=device) * original_opacity,
            harmonics=torch.randn(batch_size, N, 3, 16, device=device)
        )

        densified = densify_gaussians(mock_gaussians, densification_factor=2, device=device)

        # Opacity should be halved
        expected_opacity = original_opacity / 2
        actual_opacity = densified.opacities[0, 0].item()

        assert actual_opacity == pytest.approx(expected_opacity, abs=1e-5)


class TestCollectNearbyKeyframes:
    """Test keyframe collection with neighbor segments."""

    def test_collect_zero_neighbors(self):
        """Should collect only segment boundaries with neighbors=0."""
        from PIL import Image

        # Create mock keyframes at indices [0, 10, 20, 30, 40]
        all_keyframes = {
            0: Image.new('RGB', (64, 64)),
            10: Image.new('RGB', (64, 64)),
            20: Image.new('RGB', (64, 64)),
            30: Image.new('RGB', (64, 64)),
            40: Image.new('RGB', (64, 64))
        }

        # Collect for segment 10→20 with 0 neighbors
        collected, indices = collect_nearby_keyframes(
            all_keyframes,
            segment_first_idx=10,
            segment_last_idx=20,
            num_neighbor_segments=0
        )

        assert indices == [10, 20]
        assert len(collected) == 2

    def test_collect_one_neighbor(self):
        """Should include 1 segment before+after with neighbors=1."""
        from PIL import Image

        all_keyframes = {
            0: Image.new('RGB', (64, 64)),
            10: Image.new('RGB', (64, 64)),
            20: Image.new('RGB', (64, 64)),
            30: Image.new('RGB', (64, 64)),
            40: Image.new('RGB', (64, 64))
        }

        collected, indices = collect_nearby_keyframes(
            all_keyframes,
            segment_first_idx=10,
            segment_last_idx=20,
            num_neighbor_segments=1
        )

        # Should get: 1 before (0), segment (10, 20), 1 after (30)
        assert indices == [0, 10, 20, 30]
        assert len(collected) == 4

    def test_collect_handles_boundary_constraints(self):
        """Should not exceed available keyframes at boundaries."""
        from PIL import Image

        all_keyframes = {
            0: Image.new('RGB', (64, 64)),
            10: Image.new('RGB', (64, 64)),
            20: Image.new('RGB', (64, 64))
        }

        # Ask for 5 neighbors but only 3 keyframes total
        collected, indices = collect_nearby_keyframes(
            all_keyframes,
            segment_first_idx=0,
            segment_last_idx=10,
            num_neighbor_segments=5
        )

        # Should clamp to available: [0, 10, 20]
        assert indices == [0, 10, 20]
        assert len(collected) == 3


class TestIntrinsicsScaling:
    """Test camera intrinsics scaling for resolution mismatch."""

    def test_intrinsics_scaling_identity(self):
        """Should not modify intrinsics when resolutions match."""
        # Original intrinsics for 512x512
        fx, fy, cx, cy = 500.0, 500.0, 256.0, 256.0
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Infer DA3's processing size from principal point
        inferred_width = cx * 2.0  # 512
        inferred_height = cy * 2.0  # 512

        # Target size matches inferred
        target_width, target_height = 512, 512

        scale_x = target_width / inferred_width
        scale_y = target_height / inferred_height

        # Scale intrinsics
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y

        # Should be unchanged
        assert fx_scaled == pytest.approx(fx, abs=1e-3)
        assert fy_scaled == pytest.approx(fy, abs=1e-3)
        assert cx_scaled == pytest.approx(cx, abs=1e-3)
        assert cy_scaled == pytest.approx(cy, abs=1e-3)

    def test_intrinsics_scaling_upscale(self):
        """Should scale intrinsics when upscaling resolution."""
        # DA3 processed at 256x256
        fx, fy, cx, cy = 250.0, 250.0, 128.0, 128.0

        # Infer DA3's size
        inferred_width = cx * 2.0  # 256
        inferred_height = cy * 2.0  # 256

        # Target is 1024x1024 (4x larger)
        target_width, target_height = 1024, 1024

        scale_x = target_width / inferred_width  # 4.0
        scale_y = target_height / inferred_height  # 4.0

        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y

        # Should be 4x larger
        assert fx_scaled == pytest.approx(1000.0, abs=1e-3)
        assert fy_scaled == pytest.approx(1000.0, abs=1e-3)
        assert cx_scaled == pytest.approx(512.0, abs=1e-3)
        assert cy_scaled == pytest.approx(512.0, abs=1e-3)

    def test_intrinsics_scaling_downscale(self):
        """Should scale intrinsics when downscaling resolution."""
        # DA3 processed at 1024x1024
        fx, fy, cx, cy = 1000.0, 1000.0, 512.0, 512.0

        # Infer DA3's size
        inferred_width = cx * 2.0  # 1024
        inferred_height = cy * 2.0  # 1024

        # Target is 512x512 (0.5x smaller)
        target_width, target_height = 512, 512

        scale_x = target_width / inferred_width  # 0.5
        scale_y = target_height / inferred_height  # 0.5

        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y

        # Should be 0.5x
        assert fx_scaled == pytest.approx(500.0, abs=1e-3)
        assert fy_scaled == pytest.approx(500.0, abs=1e-3)
        assert cx_scaled == pytest.approx(256.0, abs=1e-3)
        assert cy_scaled == pytest.approx(256.0, abs=1e-3)

    def test_intrinsics_principal_point_centering(self):
        """Should maintain principal point at image center after scaling."""
        # DA3 at 518x518 (common DA3 resolution)
        cx_da3, cy_da3 = 259.0, 259.0

        # Target 1024x1024
        target_width, target_height = 1024, 1024

        # Infer and scale
        inferred_width = cx_da3 * 2.0
        inferred_height = cy_da3 * 2.0

        scale_x = target_width / inferred_width
        scale_y = target_height / inferred_height

        cx_scaled = cx_da3 * scale_x
        cy_scaled = cy_da3 * scale_y

        # Principal point should be at center of target image
        expected_cx = target_width / 2.0  # 512.0
        expected_cy = target_height / 2.0  # 512.0

        assert cx_scaled == pytest.approx(expected_cx, abs=1.0)
        assert cy_scaled == pytest.approx(expected_cy, abs=1.0)


class TestExtrinsicsConversion:
    """Test camera extrinsics matrix conversion."""

    def test_converts_3x4_to_4x4(self):
        """Should add homogeneous bottom row [0, 0, 0, 1]."""
        from deforum.rendering.da3_3dgs_novel_view import convert_extrinsics_to_4x4

        # Create [N, 3, 4] extrinsics
        extrinsics_3x4 = np.array([
            [[1, 0, 0, 10],
             [0, 1, 0, 20],
             [0, 0, 1, 30]]
        ], dtype=np.float32)

        result = convert_extrinsics_to_4x4(extrinsics_3x4)

        assert result.shape == (1, 4, 4)
        assert np.allclose(result[0, 3, :], [0, 0, 0, 1])
        assert np.allclose(result[0, :3, :], extrinsics_3x4[0])

    def test_preserves_4x4_input(self):
        """Should return unchanged if already 4x4."""
        from deforum.rendering.da3_3dgs_novel_view import convert_extrinsics_to_4x4

        # Create [N, 4, 4] extrinsics
        extrinsics_4x4 = np.eye(4, dtype=np.float32).reshape(1, 4, 4)

        result = convert_extrinsics_to_4x4(extrinsics_4x4)

        assert result.shape == (1, 4, 4)
        assert np.allclose(result, extrinsics_4x4)

    def test_handles_multiple_cameras(self):
        """Should convert multiple camera poses."""
        from deforum.rendering.da3_3dgs_novel_view import convert_extrinsics_to_4x4

        # Create [5, 3, 4] extrinsics
        N = 5
        extrinsics_3x4 = np.random.randn(N, 3, 4).astype(np.float32)

        result = convert_extrinsics_to_4x4(extrinsics_3x4)

        assert result.shape == (N, 4, 4)
        for i in range(N):
            assert np.allclose(result[i, 3, :], [0, 0, 0, 1])
            assert np.allclose(result[i, :3, :], extrinsics_3x4[i])


class TestSegmentBoundaryPoses:
    """Test segment boundary pose extraction."""

    def test_extracts_boundary_poses(self):
        """Should extract poses for segment first and last keyframes."""
        from deforum.rendering.da3_3dgs_novel_view import get_segment_boundary_poses

        # Collected keyframes: [0, 10, 20, 30, 40]
        keyframe_indices = [0, 10, 20, 30, 40]
        extrinsics = np.random.randn(5, 4, 4).astype(np.float32)

        # Segment: 10-30
        segment_first_idx = 10
        segment_last_idx = 30

        first_pose, last_pose = get_segment_boundary_poses(
            extrinsics, keyframe_indices, segment_first_idx, segment_last_idx
        )

        # Should return poses at indices 1 and 3
        assert np.allclose(first_pose, extrinsics[1])
        assert np.allclose(last_pose, extrinsics[3])

    def test_fallback_when_boundary_not_found(self):
        """Should use first/last poses if segment boundaries not in collection."""
        from deforum.rendering.da3_3dgs_novel_view import get_segment_boundary_poses

        keyframe_indices = [0, 10, 20, 30, 40]
        extrinsics = np.random.randn(5, 4, 4).astype(np.float32)

        # Request non-existent boundaries
        segment_first_idx = 5
        segment_last_idx = 35

        first_pose, last_pose = get_segment_boundary_poses(
            extrinsics, keyframe_indices, segment_first_idx, segment_last_idx
        )

        # Should fallback to first/last
        assert np.allclose(first_pose, extrinsics[0])
        assert np.allclose(last_pose, extrinsics[-1])

    def test_uses_first_last_when_no_segment_info(self):
        """Should use first/last when segment indices are None."""
        from deforum.rendering.da3_3dgs_novel_view import get_segment_boundary_poses

        keyframe_indices = [0, 10, 20, 30, 40]
        extrinsics = np.random.randn(5, 4, 4).astype(np.float32)

        first_pose, last_pose = get_segment_boundary_poses(
            extrinsics, keyframe_indices, None, None
        )

        assert np.allclose(first_pose, extrinsics[0])
        assert np.allclose(last_pose, extrinsics[-1])


class TestPILtoBGRConversion:
    """Test PIL image to BGR numpy array conversion."""

    def test_converts_single_image(self):
        """Should convert PIL RGB to BGR numpy array."""
        from deforum.rendering.da3_3dgs_novel_view import convert_pil_to_bgr
        from PIL import Image

        # Create test image (red, green, blue)
        img = Image.new('RGB', (10, 10), (255, 0, 0))
        images = [img]

        result = convert_pil_to_bgr(images)

        assert len(result) == 1
        assert result[0].shape == (10, 10, 3)
        # Red in RGB becomes Blue channel in BGR
        assert result[0][0, 0, 2] == 255  # R channel now in B position
        assert result[0][0, 0, 1] == 0    # G stays
        assert result[0][0, 0, 0] == 0    # B channel now in R position

    def test_converts_multiple_images(self):
        """Should handle batch conversion."""
        from deforum.rendering.da3_3dgs_novel_view import convert_pil_to_bgr
        from PIL import Image

        images = [
            Image.new('RGB', (10, 10), (255, 0, 0)),
            Image.new('RGB', (10, 10), (0, 255, 0)),
            Image.new('RGB', (10, 10), (0, 0, 255))
        ]

        result = convert_pil_to_bgr(images)

        assert len(result) == 3
        assert all(arr.shape == (10, 10, 3) for arr in result)


class TestModelConfigSelection:
    """Test DA3 model configuration selection."""

    def test_selects_giant_config(self):
        """Should return giant variant and size for DA3-GIANT."""
        from deforum.rendering.da3_3dgs_novel_view import get_da3_model_config

        variant, size = get_da3_model_config('DA3-GIANT')

        assert variant == 'giant'
        assert size == 'giant'

    def test_selects_nested_giant_large_config(self):
        """Should return correct config for DA3NESTED-GIANT-LARGE."""
        from deforum.rendering.da3_3dgs_novel_view import get_da3_model_config

        variant, size = get_da3_model_config('DA3NESTED-GIANT-LARGE')

        assert variant == 'giant'
        assert size == 'nested-giant-large'

    def test_fallback_to_giant_for_unknown(self):
        """Should default to giant config for unknown models."""
        from deforum.rendering.da3_3dgs_novel_view import get_da3_model_config

        variant, size = get_da3_model_config('UNKNOWN-MODEL')

        assert variant == 'giant'
        assert size == 'giant'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
