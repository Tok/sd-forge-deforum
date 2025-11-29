"""Unit tests for deforum.utils.spline_camera_path module."""

import pytest
import numpy as np
from typing import List

from deforum.utils.spline_camera_path import (
    CameraPoint,
    SplineConfig,
    generate_rotate_around_path,
    camera_path_to_schedules,
    _calculate_frames_per_loop,
    _generate_orbital_positions,
    generate_control_points_circle,
    generate_control_points_figure_eight,
    catmull_rom_spline,
    calculate_tangent_vectors,
    tangent_to_rotation,
    generate_camera_path,
    generate_street_path,
    _normalize_angle_delta,
)


class TestCalculateFramesPerLoop:
    """Tests for _calculate_frames_per_loop helper function."""

    def test_explicit_frames_per_loop(self):
        """Should return explicit value when provided."""
        assert _calculate_frames_per_loop(120.0, True, 100) == 120.0
        assert _calculate_frames_per_loop(60.0, False, 100) == 60.0

    def test_closed_loop_default(self):
        """Should use num_frames for closed loop when not specified."""
        assert _calculate_frames_per_loop(None, True, 150) == 150.0

    def test_open_loop_default(self):
        """Should use 120.0 for open loop when not specified."""
        assert _calculate_frames_per_loop(None, False, 200) == 120.0


class TestGenerateOrbitalPositions:
    """Tests for _generate_orbital_positions helper function."""

    def test_circle_path_basic(self):
        """Should generate circular path positions."""
        positions = _generate_orbital_positions(
            num_frames=4,
            frames_per_loop=4.0,
            radius=10.0,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
            height=0.0,
            use_sphere=False
        )

        assert len(positions) == 4
        # First position should be at (radius, height, 0)
        assert positions[0] == pytest.approx((10.0, 0.0, 0.0), abs=0.01)
        # Y should be constant for circle
        assert all(abs(y) < 0.01 for _, y, _ in positions)

    def test_sphere_path_basic(self):
        """Should generate sphere path positions with varying Y."""
        positions = _generate_orbital_positions(
            num_frames=8,
            frames_per_loop=8.0,
            radius=10.0,
            center_x=0.0,
            center_y=5.0,
            center_z=0.0,
            height=0.0,
            use_sphere=True
        )

        assert len(positions) == 8
        # Sphere should have varying Y values
        y_values = [y for _, y, _ in positions]
        assert max(y_values) - min(y_values) > 0.1

    def test_center_offset(self):
        """Should apply center offset to all positions."""
        positions = _generate_orbital_positions(
            num_frames=2,
            frames_per_loop=2.0,
            radius=5.0,
            center_x=10.0,
            center_y=20.0,
            center_z=30.0,
            height=0.0,
            use_sphere=False
        )

        # All positions should be offset from origin
        for x, y, z in positions:
            assert abs(y - 20.0) < 0.01  # Y should be at center_y
            # X and Z should vary around center


class TestGenerateRotateAroundPath:
    """Tests for generate_rotate_around_path function."""

    def test_basic_circle_path(self):
        """Should generate basic circular camera path."""
        path = generate_rotate_around_path(
            num_frames=10,
            radius=5.0,
            use_sphere=False,
            rotation_mode="empirical"
        )

        assert len(path) == 10
        assert all(isinstance(p, CameraPoint) for p in path)

    def test_sphere_path(self):
        """Should generate sphere camera path."""
        path = generate_rotate_around_path(
            num_frames=20,
            radius=10.0,
            use_sphere=True,
            rotation_mode="empirical"
        )

        assert len(path) == 20
        # Sphere path should have varying Y positions
        y_values = [p.y for p in path]
        assert max(y_values) - min(y_values) > 0.1

    def test_quaternion_rotation_mode(self):
        """Should generate path with quaternion rotation mode."""
        path = generate_rotate_around_path(
            num_frames=15,
            radius=8.0,
            rotation_mode="quaternion",
            look_at_mode="center",
            stabilize_camera=True
        )

        assert len(path) == 15
        # Quaternion mode with stabilization should have zero roll
        assert all(abs(p.rot_z) < 0.01 for p in path)

    def test_empirical_rotation_mode(self):
        """Should generate path with empirical rotation mode."""
        path = generate_rotate_around_path(
            num_frames=12,
            radius=6.0,
            rotation_mode="empirical",
            rotation_factor=-8.0
        )

        assert len(path) == 12
        # Check that rotation values are applied
        assert any(abs(p.rot_y) > 0.01 for p in path)

    def test_closed_loop_vs_open_loop(self):
        """Should handle closed vs open loop correctly."""
        closed = generate_rotate_around_path(
            num_frames=10,
            radius=5.0,
            closed_loop=True,
            rotation_mode="empirical"
        )

        open_loop = generate_rotate_around_path(
            num_frames=10,
            radius=5.0,
            closed_loop=False,
            frames_per_loop=120.0,
            rotation_mode="empirical"
        )

        # Closed loop should complete full orbit
        # Open loop should be partial
        assert len(closed) == 10
        assert len(open_loop) == 10
        # Paths should differ
        assert closed[-1].x != pytest.approx(open_loop[-1].x, abs=0.1)

    def test_custom_center_and_height(self):
        """Should apply custom center and height offset."""
        path = generate_rotate_around_path(
            num_frames=8,
            radius=5.0,
            center_x=10.0,
            center_y=20.0,
            center_z=30.0,
            height=5.0,
            use_sphere=False,
            rotation_mode="empirical"
        )

        # All Y values should be around center_y + height
        assert all(abs(p.y - 25.0) < 0.01 for p in path)


class TestCameraPathToSchedules:
    """Tests for camera_path_to_schedules function."""

    def test_basic_schedule_generation(self):
        """Should generate schedules from camera path."""
        path = [
            CameraPoint(x=0.0, y=0.0, z=0.0, rot_x=0.0, rot_y=0.0, rot_z=0.0, frame=0),
            CameraPoint(x=1.0, y=0.0, z=0.0, rot_x=0.0, rot_y=5.0, rot_z=0.0, frame=1),
            CameraPoint(x=2.0, y=1.0, z=0.0, rot_x=0.0, rot_y=10.0, rot_z=0.0, frame=2),
        ]

        schedules = camera_path_to_schedules(path)

        # Should have all 6 schedule keys
        assert set(schedules.keys()) == {
            'translation_x', 'translation_y', 'translation_z',
            'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z'
        }

        # All schedules should be non-empty strings
        assert all(isinstance(v, str) and len(v) > 0 for v in schedules.values())

    def test_schedule_format(self):
        """Should generate schedules in correct format."""
        path = [
            CameraPoint(x=0.0, y=0.0, z=0.0, rot_x=0.0, rot_y=0.0, rot_z=0.0, frame=0),
            CameraPoint(x=5.0, y=0.0, z=0.0, rot_x=0.0, rot_y=0.0, rot_z=0.0, frame=1),
        ]

        schedules = camera_path_to_schedules(path)

        # Schedule should contain frame numbers and parentheses
        tx = schedules['translation_x']
        assert '0: (' in tx  # Format is "0: (value)" with space after colon
        assert '1: (' in tx
        assert ')' in tx

    def test_speed_multiplier(self):
        """Should apply speed multiplier to schedules."""
        path = [
            CameraPoint(x=0.0, y=0.0, z=0.0, rot_x=0.0, rot_y=0.0, rot_z=0.0, frame=0),
            CameraPoint(x=10.0, y=0.0, z=0.0, rot_x=0.0, rot_y=0.0, rot_z=0.0, frame=1),
        ]

        normal = camera_path_to_schedules(path, speed_multiplier=1.0)
        doubled = camera_path_to_schedules(path, speed_multiplier=2.0)

        # Doubled speed should produce different schedules
        assert normal['translation_x'] != doubled['translation_x']

    def test_single_point_path(self):
        """Should handle single point path."""
        path = [
            CameraPoint(x=5.0, y=10.0, z=15.0, rot_x=0.0, rot_y=0.0, rot_z=0.0, frame=0),
        ]

        schedules = camera_path_to_schedules(path)

        # Should still generate valid schedules
        assert all(isinstance(v, str) for v in schedules.values())


class TestCameraPoint:
    """Tests for CameraPoint dataclass."""

    def test_camera_point_creation(self):
        """Should create CameraPoint with all fields."""
        point = CameraPoint(
            x=1.0, y=2.0, z=3.0,
            rot_x=10.0, rot_y=20.0, rot_z=30.0,
            frame=42
        )

        assert point.x == 1.0
        assert point.y == 2.0
        assert point.z == 3.0
        assert point.rot_x == 10.0
        assert point.rot_y == 20.0
        assert point.rot_z == 30.0
        assert point.frame == 42

    def test_camera_point_immutable(self):
        """Should be frozen dataclass (immutable)."""
        point = CameraPoint(x=1.0, y=2.0, z=3.0, rot_x=0.0, rot_y=0.0, rot_z=0.0, frame=0)

        with pytest.raises(AttributeError):
            point.x = 5.0  # Should raise error (frozen dataclass)


class TestControlPoints:
    """Tests for control point generation functions."""

    def test_generate_control_points_circle(self):
        """Should generate circular control points."""
        points = generate_control_points_circle(radius=10.0, num_points=4)

        assert len(points) == 4
        # All points should be on a circle of radius 10
        for point in points:
            distance = np.sqrt(point[0]**2 + point[2]**2)
            assert distance == pytest.approx(10.0, abs=0.01)
            # Y should be 0 for circle
            assert point[1] == pytest.approx(0.0, abs=0.01)

    def test_generate_control_points_figure_eight(self):
        """Should generate figure-eight control points."""
        points = generate_control_points_figure_eight(num_points=8, scale=5.0)

        assert len(points) == 8
        # Figure-eight should have varying X and Z
        x_values = [p[0] for p in points]
        z_values = [p[2] for p in points]
        assert max(x_values) - min(x_values) > 0.1
        assert max(z_values) - min(z_values) > 0.1


class TestSplineGeneration:
    """Tests for spline generation functions."""

    def test_catmull_rom_spline_basic(self):
        """Should generate spline from control points."""
        control_points = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0]
        ])

        spline = catmull_rom_spline(control_points, num_samples=20)

        assert len(spline) == 20
        assert spline.shape == (20, 3)
        # Spline should pass through or near control points
        assert np.linalg.norm(spline[0] - control_points[0]) < 1.0

    def test_catmull_rom_spline_open_loop(self):
        """Should generate open loop spline."""
        control_points = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 10.0, 0.0]
        ])

        spline = catmull_rom_spline(control_points, num_samples=15, closed=False)

        assert len(spline) == 15
        # Open loop: first and last points should differ
        assert np.linalg.norm(spline[0] - spline[-1]) > 1.0


class TestTangentFunctions:
    """Tests for tangent vector and rotation functions."""

    def test_calculate_tangent_vectors(self):
        """Should calculate tangent vectors from spline points."""
        spline_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ])

        tangents = calculate_tangent_vectors(spline_points)

        assert tangents.shape == spline_points.shape
        # For straight line in X, tangents should point in X direction
        for tangent in tangents:
            assert tangent[0] > 0  # Pointing in positive X
            assert abs(tangent[1]) < 0.1  # Minimal Y component
            assert abs(tangent[2]) < 0.1  # Minimal Z component

    def test_tangent_to_rotation_forward(self):
        """Should convert forward tangent to rotation."""
        # Forward tangent (along +X axis)
        tangent = np.array([1.0, 0.0, 0.0])

        rot_x, rot_y, rot_z = tangent_to_rotation(tangent)

        # Forward motion should have minimal rotation
        assert abs(rot_x) < 1.0
        assert abs(rot_y - 90.0) < 1.0  # Looking forward (90° in Y)
        assert abs(rot_z) < 1.0

    def test_tangent_to_rotation_upward(self):
        """Should convert upward tangent to rotation."""
        # Upward tangent (along +Y axis)
        tangent = np.array([0.0, 1.0, 0.0])

        rot_x, rot_y, rot_z = tangent_to_rotation(tangent)

        # Upward motion should have pitch
        assert abs(rot_x + 90.0) < 5.0  # Pitched up (-90°)


class TestGenerateCameraPath:
    """Tests for generate_camera_path function."""

    def test_generate_camera_path_basic(self):
        """Should generate camera path from control points."""
        control_points = generate_control_points_circle(num_points=4, radius=5.0)
        config = SplineConfig(
            num_frames=20,
            num_control_points=4,
            spline_type="catmull_rom",
            closed_loop=False,
            smoothness=0.5
        )

        path = generate_camera_path(
            config=config,
            control_points=control_points
        )

        assert len(path) == 20
        assert all(isinstance(p, CameraPoint) for p in path)

    def test_generate_camera_path_figure_eight(self):
        """Should generate figure-eight camera path."""
        control_points = generate_control_points_figure_eight(num_points=8, scale=10.0)
        config = SplineConfig(
            num_frames=40,
            num_control_points=8,
            spline_type="catmull_rom",
            closed_loop=False,
            smoothness=0.5
        )

        path = generate_camera_path(
            config=config,
            control_points=control_points
        )

        assert len(path) == 40
        # Path should follow figure-eight pattern
        x_values = [p.x for p in path]
        assert max(x_values) - min(x_values) > 1.0


class TestGenerateStreetPath:
    """Tests for generate_street_path function."""

    def test_generate_street_path_basic(self):
        """Should generate street path."""
        path = generate_street_path(num_frames=30)

        assert len(path) == 30
        assert all(isinstance(p, CameraPoint) for p in path)

    def test_generate_street_path_with_params(self):
        """Should generate street path with custom parameters."""
        path = generate_street_path(
            num_frames=20,
            street_length=150.0,
            center_y=2.0
        )

        assert len(path) == 20
        # Path should have varying positions
        x_values = [p.x for p in path]
        z_values = [p.z for p in path]
        assert max(x_values) - min(x_values) > 0.1
        assert max(z_values) - min(z_values) > 0.1


class TestNormalizeAngleDelta:
    """Tests for _normalize_angle_delta helper function."""

    def test_normalize_small_angle(self):
        """Should not modify small angles."""
        assert _normalize_angle_delta(10.0) == 10.0
        assert _normalize_angle_delta(-10.0) == -10.0
        assert _normalize_angle_delta(0.0) == 0.0

    def test_normalize_large_positive_angle(self):
        """Should normalize large positive angles."""
        # 200° should become -160°
        assert _normalize_angle_delta(200.0) == pytest.approx(-160.0, abs=0.01)
        # 350° should become -10°
        assert _normalize_angle_delta(350.0) == pytest.approx(-10.0, abs=0.01)

    def test_normalize_large_negative_angle(self):
        """Should normalize large negative angles."""
        # -200° should become 160°
        assert _normalize_angle_delta(-200.0) == pytest.approx(160.0, abs=0.01)
        # -350° should become 10°
        assert _normalize_angle_delta(-350.0) == pytest.approx(10.0, abs=0.01)

    def test_normalize_full_rotation(self):
        """Should handle full rotations."""
        # 360° should become 0°
        assert _normalize_angle_delta(360.0) == pytest.approx(0.0, abs=0.01)
        # -360° should become 0°
        assert _normalize_angle_delta(-360.0) == pytest.approx(0.0, abs=0.01)
