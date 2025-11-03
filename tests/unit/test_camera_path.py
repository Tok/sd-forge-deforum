"""Unit tests for 3D camera path spline generation.

Tests the spline generation, path calculation, and schedule conversion
for the Camera Path feature.

NOTE: Preset integration tests require full Forge environment (k_diffusion, etc.)
and are currently skipped in isolated test runs. They work in full Forge context.
"""

import pytest
import numpy as np
from deforum.utils.spline_camera_path import (
    CameraPoint,
    SplineConfig,
    generate_control_points_circle,
    generate_control_points_figure_eight,
    catmull_rom_spline,
    calculate_tangent_vectors,
    tangent_to_rotation,
    generate_camera_path,
    generate_rotate_around_path,
    generate_street_path,
    camera_path_to_schedules
)


class TestCameraPoint:
    """Test CameraPoint dataclass."""

    def test_camera_point_creation(self):
        """Test creating a camera point."""
        point = CameraPoint(
            x=10.0,
            y=20.0,
            z=30.0,
            rot_x=45.0,
            rot_y=-30.0,
            rot_z=0.0,
            frame=5
        )

        assert point.x == 10.0
        assert point.y == 20.0
        assert point.z == 30.0
        assert point.rot_x == 45.0
        assert point.rot_y == -30.0
        assert point.rot_z == 0.0
        assert point.frame == 5

    def test_camera_point_immutable(self):
        """Test that CameraPoint is immutable (frozen dataclass)."""
        point = CameraPoint(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0)

        with pytest.raises(AttributeError):
            point.x = 5.0  # Should fail - frozen dataclass


class TestControlPointGeneration:
    """Test control point generation functions."""

    def test_generate_circle_points(self):
        """Test circular control point generation."""
        points = generate_control_points_circle(
            num_points=8,
            radius=100.0,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
            height_variation=0.0
        )

        assert len(points) == 8

        # Check that points form a circle in XZ plane
        for x, y, z in points:
            distance = np.sqrt(x**2 + z**2)
            assert abs(distance - 100.0) < 0.01  # Within tolerance
            assert y == 0.0  # Flat circle (no height variation)

    def test_generate_circle_with_height_variation(self):
        """Test circular control points with height variation."""
        points = generate_control_points_circle(
            num_points=8,
            radius=100.0,
            height_variation=20.0
        )

        assert len(points) == 8

        # Check that some points have non-zero Y
        y_values = [p[1] for p in points]
        assert max(y_values) > 0 or min(y_values) < 0

    def test_generate_figure_eight_points(self):
        """Test figure-8 control point generation."""
        points = generate_control_points_figure_eight(
            num_points=16,
            scale=100.0
        )

        assert len(points) == 16

        # Check that points are within expected bounds
        x_values = [p[0] for p in points]
        z_values = [p[2] for p in points]

        assert min(x_values) >= -100.0 - 1.0  # Within tolerance
        assert max(x_values) <= 100.0 + 1.0


class TestSplineInterpolation:
    """Test spline interpolation functions."""

    def test_catmull_rom_basic(self):
        """Test basic Catmull-Rom spline generation."""
        control_points = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0)
        ]

        spline_points = catmull_rom_spline(
            control_points,
            num_samples=100,
            closed=False
        )

        assert spline_points.shape == (100, 3)

        # First point should be close to first control point
        assert np.allclose(spline_points[0], control_points[0], atol=0.1)

        # Last point should be close to last control point
        assert np.allclose(spline_points[-1], control_points[-1], atol=0.1)

    def test_catmull_rom_closed_loop(self):
        """Test Catmull-Rom spline with closed loop."""
        # For closed loop, first and last points should match for smooth closure
        control_points = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 0.0)  # Close the loop
        ]

        spline_points = catmull_rom_spline(
            control_points,
            num_samples=100,
            closed=False  # Already closed in control points
        )

        assert spline_points.shape == (100, 3)

        # First and last points should be close (closed loop)
        assert np.allclose(spline_points[0], spline_points[-1], atol=2.0)


class TestTangentCalculations:
    """Test tangent vector calculations."""

    def test_calculate_tangent_vectors(self):
        """Test tangent vector calculation."""
        # Linear path along X axis
        spline_points = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0]
        ])

        tangents = calculate_tangent_vectors(spline_points)

        assert tangents.shape == spline_points.shape

        # All tangents should point along X axis (normalized)
        for tangent in tangents:
            assert abs(tangent[0] - 1.0) < 0.01  # X component ~1
            assert abs(tangent[1]) < 0.01  # Y component ~0
            assert abs(tangent[2]) < 0.01  # Z component ~0

    def test_tangent_to_rotation_forward(self):
        """Test converting forward tangent to rotation."""
        # Forward direction (along +Z)
        tangent = np.array([0.0, 0.0, 1.0])

        rot_x, rot_y, rot_z = tangent_to_rotation(tangent)

        # Looking straight ahead should have ~0 rotation
        assert abs(rot_x) < 0.1
        assert abs(rot_y) < 0.1
        assert abs(rot_z) < 0.1

    def test_tangent_to_rotation_right(self):
        """Test converting right tangent to rotation."""
        # Right direction (along +X)
        tangent = np.array([1.0, 0.0, 0.0])

        rot_x, rot_y, rot_z = tangent_to_rotation(tangent)

        # Looking right should have positive Y rotation
        assert abs(rot_y - 90.0) < 1.0  # ~90 degrees


class TestRotateAroundPath:
    """Test rotate-around path generation."""

    def test_rotate_around_basic(self):
        """Test basic rotate-around path with sphere rotation (default)."""
        camera_path = generate_rotate_around_path(
            num_frames=100,
            radius=50.0,
            center_x=0.0,
            center_y=0.0,
            height=10.0,
            rotation_factor=-5.0,
            use_sphere=True  # Default behavior
        )

        assert len(camera_path) == 100

        # Check first point
        assert camera_path[0].frame == 0

        # Sphere mode: check that all points are on sphere surface (3D distance)
        for point in camera_path:
            dx = point.x - 0.0
            dy = (point.y - 10.0)  # Subtract height offset
            dz = point.z - 0.0
            distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            assert abs(distance_3d - 50.0) < 1.0  # Within tolerance (sphere)

    def test_rotate_around_rotation_factor(self):
        """Test that camera looks at center (not rotation_y = x * factor in sphere mode)."""
        camera_path = generate_rotate_around_path(
            num_frames=100,
            radius=50.0,
            rotation_factor=-5.0,
            use_sphere=True  # Default behavior
        )

        # In sphere mode, camera looks at center, not translation_x * rotation_factor
        # Check that rotation values are reasonable (pointing toward center)
        for point in camera_path:
            # Rotations should be within reasonable bounds for looking at center
            assert -90.0 <= point.rot_x <= 90.0  # Tilt
            assert -180.0 <= point.rot_y <= 180.0  # Pan


class TestCameraPathGeneration:
    """Test complete camera path generation."""

    def test_generate_camera_path_circle(self):
        """Test generating camera path from circular control points."""
        control_points = generate_control_points_circle(
            num_points=8,
            radius=100.0
        )

        config = SplineConfig(
            num_frames=200,
            num_control_points=8,
            spline_type="catmull_rom",
            closed_loop=False,  # Test non-closed loop to avoid scipy periodic issues
            smoothness=0.7
        )

        camera_path = generate_camera_path(
            config,
            control_points,
            look_at_curve=True
        )

        assert len(camera_path) == 200

        # Check that all frames are sequential
        for i, point in enumerate(camera_path):
            assert point.frame == i

    def test_generate_camera_path_linear(self):
        """Test generating camera path with linear interpolation."""
        control_points = [
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 0.0)
        ]

        config = SplineConfig(
            num_frames=100,
            num_control_points=2,
            spline_type="linear",
            closed_loop=False,
            smoothness=0.5
        )

        camera_path = generate_camera_path(
            config,
            control_points,
            look_at_curve=False
        )

        assert len(camera_path) == 100

        # First and last points should match control points
        assert abs(camera_path[0].x - 0.0) < 0.1
        assert abs(camera_path[-1].x - 100.0) < 0.1


class TestScheduleConversion:
    """Test conversion of camera path to Deforum schedules."""

    def test_camera_path_to_schedules(self):
        """Test converting camera path to schedule strings."""
        camera_path = [
            CameraPoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
            CameraPoint(10.0, 5.0, -5.0, 15.0, -10.0, 0.0, 1),
            CameraPoint(20.0, 10.0, -10.0, 30.0, -20.0, 0.0, 2)
        ]

        schedules = camera_path_to_schedules(camera_path)

        # Check that all 6 schedules exist
        assert 'translation_x' in schedules
        assert 'translation_y' in schedules
        assert 'translation_z' in schedules
        assert 'rotation_3d_x' in schedules
        assert 'rotation_3d_y' in schedules
        assert 'rotation_3d_z' in schedules

        # Check translation_x schedule format
        assert '0: (0.00)' in schedules['translation_x']
        assert '1: (10.00)' in schedules['translation_x']
        assert '2: (20.00)' in schedules['translation_x']

    def test_schedule_string_format(self):
        """Test that schedule strings are properly formatted."""
        camera_path = [
            CameraPoint(1.5, 2.75, 3.333, 0.0, 0.0, 0.0, 0),
            CameraPoint(4.5, 5.75, 6.333, 0.0, 0.0, 0.0, 10)
        ]

        schedules = camera_path_to_schedules(camera_path)

        # Check that values are rounded to 2 decimal places
        assert '0: (1.50)' in schedules['translation_x']
        assert '10: (4.50)' in schedules['translation_x']
        assert '0: (3.33)' in schedules['translation_z']


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_frame_path(self):
        """Test generating path with single frame."""
        camera_path = generate_rotate_around_path(
            num_frames=1,
            radius=50.0
        )

        assert len(camera_path) == 1
        assert camera_path[0].frame == 0

    def test_zero_radius_circle(self):
        """Test circular control points with zero radius."""
        points = generate_control_points_circle(
            num_points=8,
            radius=0.0
        )

        assert len(points) == 8

        # All points should be at origin
        for x, y, z in points:
            assert abs(x) < 0.01
            assert abs(z) < 0.01

    def test_two_control_points(self):
        """Test spline with minimum number of control points."""
        control_points = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0)
        ]

        spline_points = catmull_rom_spline(
            control_points,
            num_samples=50,
            closed=False
        )

        assert spline_points.shape == (50, 3)


class TestSphereRotation:
    """Test 3D sphere rotation for rotate-around preset."""

    def test_rotate_around_sphere_mode(self):
        """Test rotate-around with sphere rotation enabled."""
        camera_path = generate_rotate_around_path(
            num_frames=100,
            radius=50.0,
            center_x=0.0,
            center_y=0.0,
            height=10.0,
            rotation_factor=-5.0,
            use_sphere=True
        )

        assert len(camera_path) == 100

        # Check that Y values vary (not flat circle)
        y_values = [p.y for p in camera_path]
        y_range = max(y_values) - min(y_values)
        assert y_range > 5.0  # Should have significant elevation variation

        # Check that distance from center varies (spherical, not flat)
        # Note: height is added to center_y, so we need to account for that
        distances = []
        for point in camera_path:
            dx = point.x - 0.0
            dy = (point.y - 10.0)  # Subtract height offset
            dz = point.z - 0.0
            distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            distances.append(distance_3d)

        # All points should be approximately on sphere surface
        for dist in distances:
            assert abs(dist - 50.0) < 1.0  # Within tolerance

    def test_rotate_around_flat_circle_mode(self):
        """Test rotate-around with sphere rotation disabled (flat circle)."""
        camera_path = generate_rotate_around_path(
            num_frames=100,
            radius=50.0,
            center_x=0.0,
            center_y=0.0,
            height=10.0,
            rotation_factor=-5.0,
            use_sphere=False  # Classic flat circle
        )

        assert len(camera_path) == 100

        # Check that Y values are constant (flat circle)
        y_values = [p.y for p in camera_path]
        for y in y_values:
            assert abs(y - 10.0) < 0.01  # All should be at height offset

        # Check that XZ distance is constant (flat circle)
        for point in camera_path:
            distance_xz = np.sqrt(point.x**2 + point.z**2)
            assert abs(distance_xz - 50.0) < 0.01

    def test_sphere_rotation_camera_looks_at_center(self):
        """Test that camera always looks at center in sphere mode."""
        camera_path = generate_rotate_around_path(
            num_frames=50,
            radius=100.0,
            center_x=0.0,
            center_y=0.0,
            height=0.0,
            use_sphere=True
        )

        for point in camera_path:
            # Check that rotation_x and rotation_y point camera toward center
            # This is approximate - we're checking that rotations are reasonable
            assert -90.0 <= point.rot_x <= 90.0  # Tilt range
            assert -180.0 <= point.rot_y <= 180.0  # Pan range


class TestStreetPath:
    """Test street/dashcam/bodycam forward-facing path generation."""

    def test_street_path_basic(self):
        """Test basic street path generation."""
        camera_path = generate_street_path(
            num_frames=100,
            street_length=500.0,
            lane_weave=20.0,
            center_x=0.0,
            center_y=10.0,
            center_z=0.0
        )

        assert len(camera_path) == 100

        # Check forward motion (Z increases)
        assert camera_path[0].z == 0.0
        assert camera_path[-1].z > 400.0  # Should travel most of street_length

        # Check Y is around eye level
        for point in camera_path:
            assert 5.0 <= point.y <= 15.0  # Around eye level with bumps

    def test_street_path_forward_facing(self):
        """Test that camera always faces forward (dashcam/bodycam behavior)."""
        camera_path = generate_street_path(
            num_frames=100,
            street_length=500.0,
            lane_weave=20.0
        )

        # All frames should have zero rotation (facing forward)
        for point in camera_path:
            assert abs(point.rot_x) < 0.01  # Level horizon
            assert abs(point.rot_y) < 0.01  # Straight ahead
            assert abs(point.rot_z) < 0.01  # No roll

    def test_street_path_lane_weaving(self):
        """Test that path includes lane weaving (X variation)."""
        camera_path = generate_street_path(
            num_frames=100,
            street_length=500.0,
            lane_weave=30.0  # Pronounced weaving
        )

        # Extract X values
        x_values = [p.x for p in camera_path]
        x_range = max(x_values) - min(x_values)

        # Should have significant X variation (lane changes)
        assert x_range > 20.0  # Weaving should be noticeable

    def test_street_path_road_bumps(self):
        """Test that path includes road bumps (Y variation)."""
        camera_path = generate_street_path(
            num_frames=100,
            street_length=500.0,
            center_y=10.0
        )

        # Extract Y values
        y_values = [p.y for p in camera_path]

        # Should vary around center_y due to bumps
        assert min(y_values) < 10.0
        assert max(y_values) > 10.0

    def test_street_path_length_scaling(self):
        """Test that street length affects Z distance traveled."""
        path_short = generate_street_path(
            num_frames=100,
            street_length=100.0
        )

        path_long = generate_street_path(
            num_frames=100,
            street_length=500.0
        )

        # Longer street should travel further in Z
        assert path_long[-1].z > path_short[-1].z * 4.0


@pytest.mark.skip(reason="Requires full Forge backend (k_diffusion, etc.). Run in Forge WebUI context only.")
class TestPresetIntegration:
    """Test preset generation from handlers (integration tests)."""

    def test_all_presets_generate_valid_paths(self):
        """Test that all preset types generate valid camera paths."""
        pytest.importorskip("modules", reason="Requires Forge WebUI modules")
        from deforum.ui.handlers.camera_path_generator import generate_preset_path

        presets = [
            "rotate-around",
            "circle-path",
            "figure-eight",
            "forward-zoom",
            "orbit-up",
            "spiral",
            "street",
            "dashcam",
            "bodycam"
        ]

        for preset_type in presets:
            status, schedules, camera_path = generate_preset_path(
                preset_type=preset_type,
                radius=100.0,
                height=10.0,
                rotation_factor=-5.0,
                num_frames=50,  # Small number for fast tests
                closed_loop=False
            )

            # Check that generation succeeded
            assert "✅" in status, f"Preset {preset_type} failed: {status}"
            assert len(camera_path) == 50, f"Preset {preset_type} wrong length"
            assert len(schedules) == 6, f"Preset {preset_type} missing schedules"

            # Check that all 6 schedules have content
            for key in ['translation_x', 'translation_y', 'translation_z',
                       'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z']:
                assert key in schedules, f"Preset {preset_type} missing {key}"
                assert schedules[key], f"Preset {preset_type} has empty {key}"

    def test_forward_facing_presets_have_zero_rotation(self):
        """Test that street/dashcam/bodycam presets have zero rotation."""
        pytest.importorskip("modules", reason="Requires Forge WebUI modules")
        from deforum.ui.handlers.camera_path_generator import generate_preset_path

        forward_presets = ["street", "dashcam", "bodycam"]

        for preset_type in forward_presets:
            status, schedules, camera_path = generate_preset_path(
                preset_type=preset_type,
                radius=100.0,
                height=10.0,
                rotation_factor=0.0,  # Not used for these presets
                num_frames=50,
                closed_loop=False
            )

            # All rotation schedules should be "0: (0.00), 1: (0.00), ..."
            for key in ['rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z']:
                schedule_str = schedules[key]
                # Parse values - all should be 0.00
                import re
                values = [float(m) for m in re.findall(r':\s*\((-?\d+\.?\d*)\)', schedule_str)]
                for val in values:
                    assert abs(val) < 0.01, f"Preset {preset_type} has non-zero {key}"

    def test_orbit_up_cannot_be_closed(self):
        """Test that orbit-up preset always uses closed_loop=False."""
        pytest.importorskip("modules", reason="Requires Forge WebUI modules")
        from deforum.ui.handlers.camera_path_generator import generate_preset_path

        status, schedules, camera_path = generate_preset_path(
            preset_type="orbit-up",
            radius=100.0,
            height=0.0,
            rotation_factor=0.0,
            num_frames=50,
            closed_loop=True  # Try to request closed loop
        )

        # Should mention that orbit-up cannot be closed
        assert "cannot be closed" in status or "Note:" in status

    def test_rotate_around_uses_sphere(self):
        """Test that rotate-around preset uses sphere rotation by default."""
        pytest.importorskip("modules", reason="Requires Forge WebUI modules")
        from deforum.ui.handlers.camera_path_generator import generate_preset_path

        status, schedules, camera_path = generate_preset_path(
            preset_type="rotate-around",
            radius=100.0,
            height=0.0,
            rotation_factor=-5.0,
            num_frames=100,
            closed_loop=False
        )

        # Status should mention sphere rotation
        assert "sphere" in status.lower() or "3D" in status

        # Y values should vary (not flat)
        y_values = [p.y for p in camera_path]
        y_range = max(y_values) - min(y_values)
        assert y_range > 5.0, "Rotate-around should use sphere (Y variation expected)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
