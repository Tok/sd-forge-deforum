"""Unit tests for deforum.utils.spline_camera_path module."""

import pytest
import numpy as np
from typing import List

from deforum.utils.spline_camera_path import (
    CameraPoint,
    generate_rotate_around_path,
    camera_path_to_schedules,
    _calculate_frames_per_loop,
    _generate_orbital_positions,
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
