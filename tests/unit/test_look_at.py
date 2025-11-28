"""Unit tests for look_at_target function.

These tests verify that the look-at calculation produces correct rotation angles
for cameras to face specific targets, with proper stabilization to minimize roll.
"""

import pytest
import numpy as np
from deforum.utils.math.quaternion import look_at_target, euler_to_forward_vector, Vector3


class TestLookAtBasicDirections:
    """Test look-at for cardinal directions."""

    def test_look_at_negative_x(self):
        """Camera at (100, 0, 0) looking at origin should face -X direction."""
        camera_pos = (100.0, 0.0, 0.0)
        target_pos = (0.0, 0.0, 0.0)

        pitch, yaw, roll = look_at_target(camera_pos, target_pos, stabilize=True)

        # Should look in -X direction: yaw=-90° or yaw=270°
        # Normalize yaw to [0, 360) range
        yaw_normalized = yaw % 360

        # Accept either -90° (270°) or close to it
        assert abs(yaw_normalized - 270.0) < 1.0 or abs(yaw + 90.0) < 1.0, (
            f"Expected yaw near -90° or 270°, got {yaw}° (normalized: {yaw_normalized}°)"
        )

        # Pitch should be 0 (horizontal)
        assert abs(pitch) < 1.0, f"Expected pitch near 0°, got {pitch}°"

        # Roll should be 0 (stabilized, camera level)
        assert abs(roll) < 1.0, f"Expected roll near 0°, got {roll}°"

    def test_look_at_positive_x(self):
        """Camera at (-100, 0, 0) looking at origin should face +X direction."""
        camera_pos = (-100.0, 0.0, 0.0)
        target_pos = (0.0, 0.0, 0.0)

        pitch, yaw, roll = look_at_target(camera_pos, target_pos, stabilize=True)

        # Should look in +X direction: yaw=90°
        assert abs(yaw - 90.0) < 1.0, f"Expected yaw near 90°, got {yaw}°"
        assert abs(pitch) < 1.0, f"Expected pitch near 0°, got {pitch}°"
        assert abs(roll) < 1.0, f"Expected roll near 0°, got {roll}°"

    def test_look_at_negative_z(self):
        """Camera at (0, 0, 100) looking at origin should face -Z direction."""
        camera_pos = (0.0, 0.0, 100.0)
        target_pos = (0.0, 0.0, 0.0)

        pitch, yaw, roll = look_at_target(camera_pos, target_pos, stabilize=True)

        # Should look in -Z direction: yaw=180° or -180°
        yaw_normalized = yaw % 360
        assert abs(yaw_normalized - 180.0) < 1.0, f"Expected yaw near 180°, got {yaw}° (normalized: {yaw_normalized}°)"
        assert abs(pitch) < 1.0, f"Expected pitch near 0°, got {pitch}°"
        assert abs(roll) < 1.0, f"Expected roll near 0°, got {roll}°"

    def test_look_at_positive_z(self):
        """Camera at (0, 0, -100) looking at origin should face +Z direction."""
        camera_pos = (0.0, 0.0, -100.0)
        target_pos = (0.0, 0.0, 0.0)

        pitch, yaw, roll = look_at_target(camera_pos, target_pos, stabilize=True)

        # Should look in +Z direction: yaw=0°
        assert abs(yaw) < 1.0, f"Expected yaw near 0°, got {yaw}°"
        assert abs(pitch) < 1.0, f"Expected pitch near 0°, got {pitch}°"
        assert abs(roll) < 1.0, f"Expected roll near 0°, got {roll}°"


class TestLookAtForwardVector:
    """Test that look-at produces correct forward vectors."""

    def test_forward_vector_matches_target_direction(self):
        """Forward vector from look-at angles should point toward target."""
        camera_pos = (100.0, 50.0, 30.0)
        target_pos = (0.0, 0.0, 0.0)

        # Get look-at angles
        pitch, yaw, roll = look_at_target(camera_pos, target_pos, stabilize=True)

        # Convert back to forward vector
        forward = euler_to_forward_vector(pitch, yaw, roll)

        # Calculate expected direction (normalized)
        dx = target_pos[0] - camera_pos[0]
        dy = target_pos[1] - camera_pos[1]
        dz = target_pos[2] - camera_pos[2]
        expected_length = np.sqrt(dx**2 + dy**2 + dz**2)
        expected = Vector3(dx / expected_length, dy / expected_length, dz / expected_length)

        # Forward vector should match expected direction (within tolerance)
        tolerance = 0.01  # 1% error tolerance
        assert abs(forward.x - expected.x) < tolerance, (
            f"Forward X mismatch: {forward.x} vs expected {expected.x}"
        )
        assert abs(forward.y - expected.y) < tolerance, (
            f"Forward Y mismatch: {forward.y} vs expected {expected.y}"
        )
        assert abs(forward.z - expected.z) < tolerance, (
            f"Forward Z mismatch: {forward.z} vs expected {expected.z}"
        )


class TestLookAtStabilization:
    """Test that stabilization minimizes roll."""

    def test_stabilized_has_zero_roll_horizontal(self):
        """Stabilized look-at should have zero roll for horizontal views."""
        camera_pos = (100.0, 0.0, 0.0)
        target_pos = (0.0, 0.0, 0.0)

        pitch, yaw, roll = look_at_target(camera_pos, target_pos, stabilize=True)

        # Roll should be 0 for horizontal view
        assert abs(roll) < 0.1, f"Expected roll near 0° for horizontal view, got {roll}°"

    def test_stabilized_has_zero_roll_elevated(self):
        """Stabilized look-at should have zero roll even with elevation."""
        camera_pos = (100.0, 50.0, 0.0)
        target_pos = (0.0, 0.0, 0.0)

        pitch, yaw, roll = look_at_target(camera_pos, target_pos, stabilize=True)

        # Roll should be 0 even with pitch
        assert abs(roll) < 0.1, f"Expected roll near 0° for elevated view, got {roll}°"


class TestRotateAroundOrbit:
    """Test look-at for circular orbit (rotate-around preset simulation)."""

    def test_orbit_camera_faces_center(self):
        """Camera orbiting around center should always face inward."""
        radius = 100.0
        center = (0.0, 0.0, 0.0)
        num_frames = 20

        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            camera_x = radius * np.cos(angle)
            camera_z = radius * np.sin(angle)
            camera_pos = (camera_x, 0.0, camera_z)

            pitch, yaw, roll = look_at_target(camera_pos, center, stabilize=True)

            # Calculate expected yaw (angle toward center)
            expected_yaw = np.degrees(np.arctan2(-camera_x, -camera_z))

            # Normalize both to [-180, 180]
            yaw_norm = ((yaw + 180) % 360) - 180
            expected_norm = ((expected_yaw + 180) % 360) - 180

            # Should match within 1 degree
            yaw_diff = abs(yaw_norm - expected_norm)
            if yaw_diff > 180:  # Handle wrap-around
                yaw_diff = 360 - yaw_diff

            assert yaw_diff < 1.0, (
                f"Frame {i}: Camera at ({camera_x:.1f}, 0, {camera_z:.1f}) "
                f"should face center with yaw {expected_norm:.1f}°, got {yaw_norm:.1f}° "
                f"(diff: {yaw_diff:.1f}°)"
            )

            # Pitch should be near 0 (horizontal orbit)
            assert abs(pitch) < 1.0, f"Frame {i}: Expected pitch near 0°, got {pitch}°"

            # Roll should be 0 (stabilized)
            assert abs(roll) < 1.0, f"Frame {i}: Expected roll near 0°, got {roll}°"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
