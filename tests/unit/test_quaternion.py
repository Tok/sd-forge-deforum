"""Unit Tests for Quaternion Math Utilities

Tests quaternion-based rotation calculations used for camera orientation.
Validates:
- Euler → Quaternion conversion
- Quaternion → Forward vector rotation
- Look-at target calculation
- Round-trip accuracy (position + target → angles → forward vector)
"""

import pytest
import numpy as np
from deforum.utils.math.quaternion import (
    Vector3,
    Quaternion,
    euler_to_forward_vector,
    look_at_target,
    validate_look_at
)


class TestVector3:
    """Test Vector3 dataclass operations."""

    def test_normalize_unit_vector(self):
        """Normalizing a unit vector returns same vector."""
        v = Vector3(1.0, 0.0, 0.0)
        normalized = v.normalize()
        assert abs(normalized.x - 1.0) < 1e-6
        assert abs(normalized.y - 0.0) < 1e-6
        assert abs(normalized.z - 0.0) < 1e-6

    def test_normalize_arbitrary_vector(self):
        """Normalizing arbitrary vector produces unit length."""
        v = Vector3(3.0, 4.0, 0.0)  # Length = 5
        normalized = v.normalize()
        assert abs(normalized.x - 0.6) < 1e-6
        assert abs(normalized.y - 0.8) < 1e-6
        length = np.sqrt(normalized.x**2 + normalized.y**2 + normalized.z**2)
        assert abs(length - 1.0) < 1e-6

    def test_normalize_zero_vector(self):
        """Normalizing zero vector returns default forward."""
        v = Vector3(0.0, 0.0, 0.0)
        normalized = v.normalize()
        assert abs(normalized.x - 0.0) < 1e-6
        assert abs(normalized.y - 0.0) < 1e-6
        assert abs(normalized.z - 1.0) < 1e-6

    def test_cross_product(self):
        """Cross product of X and Y axes gives Z axis."""
        x_axis = Vector3(1.0, 0.0, 0.0)
        y_axis = Vector3(0.0, 1.0, 0.0)
        result = x_axis.cross(y_axis)
        assert abs(result.x - 0.0) < 1e-6
        assert abs(result.y - 0.0) < 1e-6
        assert abs(result.z - 1.0) < 1e-6

    def test_dot_product_perpendicular(self):
        """Dot product of perpendicular vectors is zero."""
        x_axis = Vector3(1.0, 0.0, 0.0)
        y_axis = Vector3(0.0, 1.0, 0.0)
        result = x_axis.dot(y_axis)
        assert abs(result) < 1e-6

    def test_dot_product_parallel(self):
        """Dot product of parallel unit vectors is 1."""
        v1 = Vector3(1.0, 0.0, 0.0)
        v2 = Vector3(1.0, 0.0, 0.0)
        result = v1.dot(v2)
        assert abs(result - 1.0) < 1e-6


class TestQuaternion:
    """Test Quaternion operations."""

    def test_identity_quaternion(self):
        """Identity quaternion (no rotation) from zero euler angles."""
        q = Quaternion.from_euler_degrees(0.0, 0.0, 0.0)
        # Identity: w=1, xyz=0
        assert abs(q.w - 1.0) < 1e-6
        assert abs(q.x) < 1e-6
        assert abs(q.y) < 1e-6
        assert abs(q.z) < 1e-6

    def test_quaternion_90_deg_yaw(self):
        """90° yaw rotation (around Y axis)."""
        q = Quaternion.from_euler_degrees(0.0, 90.0, 0.0)
        # Rotate forward vector (0,0,1) by 90° yaw
        # Should point to (1,0,0) - right
        forward = Vector3(0.0, 0.0, 1.0)
        rotated = q.rotate_vector(forward)
        assert abs(rotated.x - 1.0) < 1e-6
        assert abs(rotated.y - 0.0) < 1e-6
        assert abs(rotated.z - 0.0) < 1e-6

    def test_quaternion_90_deg_pitch(self):
        """90° pitch rotation (around X axis)."""
        q = Quaternion.from_euler_degrees(90.0, 0.0, 0.0)
        # Rotate forward vector (0,0,1) by 90° pitch
        # Should point to (0,1,0) - up
        forward = Vector3(0.0, 0.0, 1.0)
        rotated = q.rotate_vector(forward)
        assert abs(rotated.x - 0.0) < 1e-6
        assert abs(rotated.y - 1.0) < 1e-6
        assert abs(rotated.z - 0.0) < 1e-6

    def test_quaternion_preserves_length(self):
        """Rotating a vector preserves its length."""
        q = Quaternion.from_euler_degrees(45.0, 30.0, 15.0)
        v = Vector3(3.0, 4.0, 5.0)
        original_length = np.sqrt(v.x**2 + v.y**2 + v.z**2)
        rotated = q.rotate_vector(v)
        rotated_length = np.sqrt(rotated.x**2 + rotated.y**2 + rotated.z**2)
        assert abs(rotated_length - original_length) < 1e-5


class TestEulerToForwardVector:
    """Test euler angles → forward vector conversion."""

    def test_zero_rotation_forward(self):
        """Zero rotation gives forward = +Z."""
        forward = euler_to_forward_vector(0.0, 0.0, 0.0)
        assert abs(forward.x - 0.0) < 1e-6
        assert abs(forward.y - 0.0) < 1e-6
        assert abs(forward.z - 1.0) < 1e-6

    def test_90_deg_yaw_right(self):
        """90° yaw looks right (+X)."""
        forward = euler_to_forward_vector(0.0, 90.0, 0.0)
        assert abs(forward.x - 1.0) < 1e-6
        assert abs(forward.y - 0.0) < 1e-6
        assert abs(forward.z - 0.0) < 1e-6

    def test_negative_90_deg_yaw_left(self):
        """-90° yaw looks left (-X)."""
        forward = euler_to_forward_vector(0.0, -90.0, 0.0)
        assert abs(forward.x - (-1.0)) < 1e-6
        assert abs(forward.y - 0.0) < 1e-6
        assert abs(forward.z - 0.0) < 1e-6

    def test_90_deg_pitch_up(self):
        """90° pitch looks up (+Y)."""
        forward = euler_to_forward_vector(90.0, 0.0, 0.0)
        assert abs(forward.x - 0.0) < 1e-6
        assert abs(forward.y - 1.0) < 1e-6
        assert abs(forward.z - 0.0) < 1e-6

    def test_negative_90_deg_pitch_down(self):
        """-90° pitch looks down (-Y)."""
        forward = euler_to_forward_vector(-90.0, 0.0, 0.0)
        assert abs(forward.x - 0.0) < 1e-6
        assert abs(forward.y - (-1.0)) < 1e-6
        assert abs(forward.z - 0.0) < 1e-6

    def test_45_deg_yaw_diagonal(self):
        """45° yaw looks diagonally (northeast)."""
        forward = euler_to_forward_vector(0.0, 45.0, 0.0)
        # Should be at 45° in XZ plane
        expected_x = np.sin(np.radians(45.0))  # ~0.707
        expected_z = np.cos(np.radians(45.0))  # ~0.707
        assert abs(forward.x - expected_x) < 1e-6
        assert abs(forward.y - 0.0) < 1e-6
        assert abs(forward.z - expected_z) < 1e-6


class TestLookAtTarget:
    """Test look-at target calculation."""

    def test_look_at_forward(self):
        """Camera at origin looking at point forward (+Z) = 0° rotation."""
        camera = (0.0, 0.0, 0.0)
        target = (0.0, 0.0, 100.0)
        pitch, yaw, roll = look_at_target(camera, target)
        assert abs(pitch - 0.0) < 1e-5
        assert abs(yaw - 0.0) < 1e-5
        assert abs(roll - 0.0) < 1e-5

    def test_look_at_right(self):
        """Camera at origin looking at point right (+X) = 90° yaw."""
        camera = (0.0, 0.0, 0.0)
        target = (100.0, 0.0, 0.0)
        pitch, yaw, roll = look_at_target(camera, target)
        assert abs(pitch - 0.0) < 1e-5
        assert abs(yaw - 90.0) < 1e-5
        assert abs(roll - 0.0) < 1e-5

    def test_look_at_left(self):
        """Camera at origin looking at point left (-X) = -90° yaw."""
        camera = (0.0, 0.0, 0.0)
        target = (-100.0, 0.0, 0.0)
        pitch, yaw, roll = look_at_target(camera, target)
        assert abs(pitch - 0.0) < 1e-5
        assert abs(yaw - (-90.0)) < 1e-5
        assert abs(roll - 0.0) < 1e-5

    def test_look_at_up(self):
        """Camera at origin looking at point up (+Y) = 90° pitch."""
        camera = (0.0, 0.0, 0.0)
        target = (0.0, 100.0, 0.0)
        pitch, yaw, roll = look_at_target(camera, target)
        assert abs(pitch - 90.0) < 1e-5
        # Yaw is undefined when looking straight up, but should be stable
        assert abs(roll - 0.0) < 1e-5

    def test_look_at_down(self):
        """Camera at origin looking at point down (-Y) = -90° pitch."""
        camera = (0.0, 0.0, 0.0)
        target = (0.0, -100.0, 0.0)
        pitch, yaw, roll = look_at_target(camera, target)
        assert abs(pitch - (-90.0)) < 1e-5
        assert abs(roll - 0.0) < 1e-5

    def test_look_at_diagonal(self):
        """Camera looking at diagonal point."""
        camera = (0.0, 0.0, 0.0)
        target = (100.0, 0.0, 100.0)  # 45° northeast
        pitch, yaw, roll = look_at_target(camera, target)
        assert abs(pitch - 0.0) < 1e-5
        assert abs(yaw - 45.0) < 1e-5
        assert abs(roll - 0.0) < 1e-5

    def test_look_at_same_position(self):
        """Camera looking at its own position = zero rotation."""
        camera = (50.0, 50.0, 50.0)
        target = (50.0, 50.0, 50.0)
        pitch, yaw, roll = look_at_target(camera, target)
        assert abs(pitch - 0.0) < 1e-5
        assert abs(yaw - 0.0) < 1e-5
        assert abs(roll - 0.0) < 1e-5

    def test_look_at_offset_camera(self):
        """Camera not at origin looking at target."""
        camera = (100.0, 50.0, 0.0)
        target = (200.0, 50.0, 0.0)  # Forward from camera (+X)
        pitch, yaw, roll = look_at_target(camera, target)
        assert abs(pitch - 0.0) < 1e-5
        assert abs(yaw - 90.0) < 1e-5
        assert abs(roll - 0.0) < 1e-5


class TestValidateLookAt:
    """Test look-at validation (round-trip accuracy)."""

    def test_validate_forward(self):
        """Forward look-at is valid."""
        camera = (0.0, 0.0, 0.0)
        target = (0.0, 0.0, 100.0)
        is_valid, error = validate_look_at(camera, target)
        assert is_valid
        assert error < 0.1

    def test_validate_right(self):
        """Right look-at is valid."""
        camera = (0.0, 0.0, 0.0)
        target = (100.0, 0.0, 0.0)
        is_valid, error = validate_look_at(camera, target)
        assert is_valid
        assert error < 0.1

    def test_validate_up(self):
        """Up look-at is valid."""
        camera = (0.0, 0.0, 0.0)
        target = (0.0, 100.0, 0.0)
        is_valid, error = validate_look_at(camera, target)
        assert is_valid
        assert error < 0.1

    def test_validate_diagonal(self):
        """Diagonal look-at is valid."""
        camera = (0.0, 0.0, 0.0)
        target = (100.0, 50.0, 100.0)
        is_valid, error = validate_look_at(camera, target)
        assert is_valid
        assert error < 0.1

    def test_validate_complex_position(self):
        """Complex camera/target positions are valid."""
        camera = (123.45, 67.89, -50.0)
        target = (-78.9, 123.4, 200.0)
        is_valid, error = validate_look_at(camera, target)
        assert is_valid
        assert error < 0.1

    def test_validate_same_position(self):
        """Camera at target position is valid (zero error)."""
        camera = (50.0, 50.0, 50.0)
        target = (50.0, 50.0, 50.0)
        is_valid, error = validate_look_at(camera, target)
        assert is_valid
        assert error == 0.0


class TestRotateAroundLookAt:
    """Test that rotate-around camera path correctly looks at center."""

    def test_rotate_around_circle_positions(self):
        """Test multiple positions on rotate-around circle."""
        from deforum.utils.spline_camera_path import generate_rotate_around_path

        radius = 100.0
        center = (0.0, 0.0, 0.0)

        # Generate path
        path = generate_rotate_around_path(
            num_frames=8,  # 8 positions around circle
            radius=radius,
            center_x=center[0],
            center_y=center[1],
            height=0.0,
            center_z=center[2],
            use_sphere=False,  # Flat circle for easier testing
            frames_per_loop=8.0  # 1 full rotation in 8 frames
        )

        # Verify each camera position looks at center
        for point in path:
            camera_pos = (point.x, point.y, point.z)

            # Validate look-at using quaternion utilities
            is_valid, error = validate_look_at(camera_pos, center, tolerance_degrees=1.0)

            # Also manually check: forward vector should point toward center
            forward = euler_to_forward_vector(point.rot_x, point.rot_y, point.rot_z)

            # Direction to center
            dx = center[0] - point.x
            dy = center[1] - point.y
            dz = center[2] - point.z
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            expected = Vector3(dx/dist, dy/dist, dz/dist) if dist > 0 else Vector3(0, 0, 1)

            # Dot product should be ~1 (parallel)
            dot = forward.dot(expected)

            assert is_valid, f"Frame {point.frame}: Look-at validation failed (error={error:.2f}°)"
            assert dot > 0.99, f"Frame {point.frame}: Forward not aligned with center (dot={dot:.3f})"
