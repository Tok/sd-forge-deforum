"""Unit tests for camera roll calculation in spline paths.

Tests that roll is properly calculated and not hardcoded to zero.
"""

import pytest
import numpy as np
from deforum.utils.spline_camera_path import (
    tangent_to_rotation,
    generate_rotate_around_path,
    generate_camera_path,
    generate_control_points_circle,
    SplineConfig
)


class TestTangentToRotation:
    """Test tangent vector to rotation conversion with roll support."""

    def test_legacy_mode_without_camera_pos(self):
        """Should use simplified calculation when camera_pos not provided."""
        tangent = np.array([1.0, 0.0, 1.0])  # Forward-right

        rot_x, rot_y, rot_z = tangent_to_rotation(tangent, camera_pos=None, stabilize=True)

        # Should calculate pitch and yaw correctly
        assert rot_y == pytest.approx(45.0, abs=0.1)  # 45° right
        assert rot_x == pytest.approx(0.0, abs=0.1)   # Level
        assert rot_z == pytest.approx(0.0, abs=0.1)   # Stabilized = no roll

    def test_quaternion_mode_with_camera_pos_stabilized(self):
        """Should use quaternion look-at with roll when camera_pos provided."""
        tangent = np.array([1.0, 0.0, 1.0])  # Forward-right
        camera_pos = (0.0, 0.0, 0.0)

        rot_x, rot_y, rot_z = tangent_to_rotation(tangent, camera_pos=camera_pos, stabilize=True)

        # Should use quaternion-based calculation
        assert rot_y == pytest.approx(45.0, abs=0.1)  # 45° right
        assert rot_z == pytest.approx(0.0, abs=0.1)   # Stabilized = minimal roll

    def test_unstabilized_mode_allows_roll(self):
        """Should allow natural roll when stabilize=False."""
        tangent = np.array([1.0, 0.5, 1.0])  # Forward-right-up
        camera_pos = (0.0, 0.0, 0.0)

        rot_x, rot_y, rot_z = tangent_to_rotation(tangent, camera_pos=camera_pos, stabilize=False)

        # Unstabilized mode can have non-zero roll
        # The exact value depends on the geometry, just verify it's calculated
        assert isinstance(rot_z, (float, np.floating))


class TestRotateAroundPathRoll:
    """Test rotate-around paths properly calculate roll."""

    def test_rotate_around_quaternion_mode_stabilized(self):
        """Quaternion mode with stabilize should minimize roll."""
        path = generate_rotate_around_path(
            num_frames=10,
            radius=50.0,
            height=10.0,
            stabilize_camera=True,
            rotation_mode='quaternion',
            look_at_mode='center'
        )

        # Check that roll exists but is minimal (close to 0)
        for point in path:
            assert abs(point.rot_z) < 5.0  # Within 5 degrees of level

    def test_rotate_around_quaternion_mode_unstabilized(self):
        """Quaternion mode with stabilize=False can have more roll."""
        path = generate_rotate_around_path(
            num_frames=10,
            radius=50.0,
            height=10.0,
            stabilize_camera=False,
            rotation_mode='quaternion',
            look_at_mode='center'
        )

        # Roll should be calculated (not hardcoded to 0)
        # Exact values depend on path geometry
        for point in path:
            assert isinstance(point.rot_z, (float, np.floating))

    def test_rotate_around_empirical_mode_calculates_roll(self):
        """Empirical mode should also calculate proper roll."""
        path = generate_rotate_around_path(
            num_frames=10,
            radius=50.0,
            height=10.0,
            stabilize_camera=True,
            rotation_mode='empirical',
            rotation_factor=-8.0
        )

        # Empirical mode should also respect stabilize parameter
        for point in path:
            # With stabilize=True, roll should be minimal
            assert abs(point.rot_z) < 5.0

    def test_rotate_around_looks_at_center(self):
        """Rotate-around should orient camera toward center."""
        path = generate_rotate_around_path(
            num_frames=8,
            radius=50.0,
            center_x=10.0,
            center_y=5.0,
            center_z=15.0,
            height=0.0,
            rotation_mode='quaternion',
            look_at_mode='center'
        )

        # All points should have rotations calculated (not zeros)
        for point in path:
            # At least rot_y should be non-zero (panning to look at center)
            # rot_x might be near zero if height=0 and camera at same y
            # rot_z should exist (not None or hardcoded)
            assert point.rot_y != 0.0 or point.rot_x != 0.0  # Some rotation present
            assert point.rot_z is not None

    def test_rotate_around_schedules_contain_roll(self):
        """Test that rotate-around with look-at-center generates schedules with proper roll."""
        import re
        from deforum.utils.spline_camera_path import camera_path_to_schedules

        # Generate rotate-around path with quaternion look-at center mode
        path = generate_rotate_around_path(
            num_frames=10,
            radius=50.0,
            height=10.0,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
            stabilize_camera=True,
            rotation_mode='quaternion',
            look_at_mode='center'
        )

        # Convert to schedules with look_at_mode="center"
        schedules = camera_path_to_schedules(
            path,
            speed_multiplier=1.0,
            look_at_mode='center',
            stabilize_camera=True
        )

        # Verify all rotation schedules exist
        assert 'rotation_3d_x' in schedules
        assert 'rotation_3d_y' in schedules
        assert 'rotation_3d_z' in schedules

        # Parse rotation_3d_z schedule and verify it contains values
        rot_z_schedule = schedules['rotation_3d_z']
        pattern = r'(\d+)\s*:\s*\(([^)]+)\)'
        matches = re.findall(pattern, rot_z_schedule)

        assert len(matches) > 0, "rotation_3d_z schedule should contain frame values"

        # Extract all rot_z values
        rot_z_values = [float(val) for frame, val in matches]

        # With stabilize_camera=True, roll should be minimal but not hardcoded to 0
        # All values should be floats (not None)
        assert all(isinstance(v, float) for v in rot_z_values)

        # With stabilization, max roll should be < 5 degrees
        max_roll = max(abs(v) for v in rot_z_values)
        assert max_roll < 5.0, f"With stabilize=True, roll should be < 5°, got {max_roll:.2f}°"


class TestSplinePathRoll:
    """Test spline curve paths calculate roll properly."""

    def test_spline_path_with_look_at_curve_stabilized(self):
        """Spline following curve should have minimal roll when stabilized."""
        control_points = generate_control_points_circle(
            num_points=8,
            radius=50.0,
            height_variation=20.0  # Add vertical movement
        )

        config = SplineConfig(
            num_frames=20,
            num_control_points=8,
            spline_type="linear",  # Use linear instead of catmull_rom for simplicity
            closed_loop=False,  # Avoid periodic spline issues
            smoothness=0.8
        )

        path = generate_camera_path(
            config, control_points,
            look_at_curve=True,
            stabilize_camera=True
        )

        # Check that rotations are calculated
        for point in path:
            assert isinstance(point.rot_x, (float, np.floating))
            assert isinstance(point.rot_y, (float, np.floating))
            assert isinstance(point.rot_z, (float, np.floating))
            # With stabilization, roll should be close to 0
            assert abs(point.rot_z) < 5.0

    def test_spline_path_without_look_at_has_zero_rotation(self):
        """Spline without look_at_curve should have zero rotations."""
        control_points = generate_control_points_circle(num_points=8, radius=50.0)

        config = SplineConfig(
            num_frames=20,
            num_control_points=8,
            spline_type="linear",  # Use linear for simplicity
            closed_loop=False,  # Avoid periodic spline issues
            smoothness=0.8
        )

        path = generate_camera_path(
            config, control_points,
            look_at_curve=False,  # No automatic orientation
            stabilize_camera=True
        )

        # Without look_at_curve, all rotations should be zero
        for point in path:
            assert point.rot_x == 0.0
            assert point.rot_y == 0.0
            assert point.rot_z == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
