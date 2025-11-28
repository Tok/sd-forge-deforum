"""Quaternion Math Utilities

Provides quaternion-based rotations for camera orientation calculations.
Used for:
- Converting euler angles to forward/up vectors
- Look-at calculations (position + target → rotation angles)
- Proper 3D rotation composition for visualization

All angles are in degrees (converted to radians internally).
Rotation order: YXZ (yaw-pitch-roll, standard for cameras).
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class Vector3:
    """3D vector with x, y, z components."""
    x: float
    y: float
    z: float

    def normalize(self) -> 'Vector3':
        """Return normalized (unit length) vector."""
        magnitude = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        if magnitude < 1e-8:
            return Vector3(0.0, 0.0, 1.0)  # Default forward
        return Vector3(
            self.x / magnitude,
            self.y / magnitude,
            self.z / magnitude
        )

    def cross(self, other: 'Vector3') -> 'Vector3':
        """Cross product with another vector."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def dot(self, other: 'Vector3') -> float:
        """Dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z


@dataclass(frozen=True)
class Quaternion:
    """Quaternion for representing 3D rotations.

    Components: w (scalar), x, y, z (vector).
    Unit quaternions represent rotations (magnitude = 1).
    """
    w: float
    x: float
    y: float
    z: float

    @classmethod
    def from_euler_degrees(cls, pitch: float, yaw: float, roll: float) -> 'Quaternion':
        """Create quaternion from euler angles in degrees.

        Args:
            pitch: Rotation around X axis (degrees) - tilt up/down
                   Positive = look up, Negative = look down
            yaw: Rotation around Y axis (degrees) - pan left/right
                 Positive = look right, Negative = look left
            roll: Rotation around Z axis (degrees) - roll left/right

        Returns:
            Quaternion representing the rotation (YXZ order)
        """
        # Convert to radians
        # Negate pitch because Deforum uses inverted pitch convention
        pitch_rad = np.radians(-pitch)
        yaw_rad = np.radians(yaw)
        roll_rad = np.radians(roll)

        # Half angles for quaternion conversion
        cy = np.cos(yaw_rad * 0.5)
        sy = np.sin(yaw_rad * 0.5)
        cx = np.cos(pitch_rad * 0.5)
        sx = np.sin(pitch_rad * 0.5)
        cz = np.cos(roll_rad * 0.5)
        sz = np.sin(roll_rad * 0.5)

        # YXZ rotation order (standard for cameras)
        w = cy * cx * cz + sy * sx * sz
        x = cy * sx * cz + sy * cx * sz
        y = sy * cx * cz - cy * sx * sz
        z = cy * cx * sz - sy * sx * cz

        return cls(w, x, y, z)

    def rotate_vector(self, v: Vector3) -> Vector3:
        """Rotate a vector by this quaternion.

        Args:
            v: Vector to rotate

        Returns:
            Rotated vector
        """
        # Convert to quaternion multiplication: q * v * q^-1
        # For unit quaternions: q^-1 = q* (conjugate)

        # v as quaternion (w=0, xyz=vector)
        vx, vy, vz = v.x, v.y, v.z

        # First: q * v
        t_w = -self.x * vx - self.y * vy - self.z * vz
        t_x = self.w * vx + self.y * vz - self.z * vy
        t_y = self.w * vy + self.z * vx - self.x * vz
        t_z = self.w * vz + self.x * vy - self.y * vx

        # Second: (q * v) * q* (conjugate)
        rx = t_x * self.w + t_w * -self.x + t_y * -self.z - t_z * -self.y
        ry = t_y * self.w + t_w * -self.y + t_z * -self.x - t_x * -self.z
        rz = t_z * self.w + t_w * -self.z + t_x * -self.y - t_y * -self.x

        return Vector3(rx, ry, rz)


def euler_to_forward_vector(pitch: float, yaw: float, roll: float) -> Vector3:
    """Convert euler angles to forward direction vector.

    Args:
        pitch: Rotation around X axis (degrees) - tilt up/down
        yaw: Rotation around Y axis (degrees) - pan left/right
        roll: Rotation around Z axis (degrees) - roll (usually 0 for cameras)

    Returns:
        Normalized forward direction vector (where camera is looking)
    """
    # Create quaternion from euler angles
    q = Quaternion.from_euler_degrees(pitch, yaw, roll)

    # Default forward vector (down +Z axis in camera space)
    default_forward = Vector3(0.0, 0.0, 1.0)

    # Rotate by quaternion to get actual forward direction
    forward = q.rotate_vector(default_forward)

    return forward.normalize()


def look_at_target(
    camera_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    stabilize: bool = True
) -> Tuple[float, float, float]:
    """Calculate euler angles to make camera look at target.

    Args:
        camera_pos: Camera position (x, y, z)
        target_pos: Target position to look at (x, y, z)
        stabilize: If True, stabilize camera to minimize roll (default: True).
                   Uses world up vector (0, 1, 0) to keep camera level.

    Returns:
        (pitch, yaw, roll) in degrees - rotation angles to face target
    """
    # Direction from camera to target
    dx = target_pos[0] - camera_pos[0]
    dy = target_pos[1] - camera_pos[1]
    dz = target_pos[2] - camera_pos[2]

    # Handle zero direction (camera at target)
    if abs(dx) < 1e-8 and abs(dy) < 1e-8 and abs(dz) < 1e-8:
        return (0.0, 0.0, 0.0)

    if not stabilize:
        # Simple look-at without stabilization (legacy behavior)
        # Yaw: horizontal angle (rotation around Y axis)
        yaw = np.degrees(np.arctan2(dx, dz))
        # Pitch: vertical angle (rotation around X axis)
        horizontal_dist = np.sqrt(dx**2 + dz**2)
        pitch = np.degrees(np.arctan2(dy, horizontal_dist))
        # Roll: always 0 (but may introduce visual roll in curved paths)
        roll = 0.0
        return (pitch, yaw, roll)

    # Stabilized look-at: construct orthonormal basis to minimize roll
    # Forward direction (normalized)
    forward_length = np.sqrt(dx**2 + dy**2 + dz**2)
    forward = Vector3(dx / forward_length, dy / forward_length, dz / forward_length)

    # World up vector (preferred up direction)
    world_up = Vector3(0.0, 1.0, 0.0)

    # Calculate right vector (perpendicular to forward and world up)
    right = forward.cross(world_up)
    right_length = np.sqrt(right.x**2 + right.y**2 + right.z**2)

    # Handle case where forward is parallel to world up (looking straight up/down)
    if right_length < 1e-8:
        # Use fallback right vector (world forward × world up)
        world_forward = Vector3(0.0, 0.0, 1.0)
        right = world_forward.cross(world_up)
        right_length = np.sqrt(right.x**2 + right.y**2 + right.z**2)

    # Normalize right vector
    right = Vector3(right.x / right_length, right.y / right_length, right.z / right_length)

    # Calculate stabilized up vector (perpendicular to right and forward)
    # This ensures orthonormality: up is exactly perpendicular to both right and forward
    up = right.cross(forward)

    # Build rotation matrix from orthonormal basis
    # Camera space: X=right, Y=up, Z=-forward (camera looks down -Z in OpenGL convention)
    # Matrix columns: [right, up, -forward]
    m11, m12, m13 = right.x, up.x, -forward.x
    m21, m22, m23 = right.y, up.y, -forward.y
    m31, m32, m33 = right.z, up.z, -forward.z

    # Extract euler angles from rotation matrix (YXZ order)
    # Reference: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/
    # YXZ order: first Y (yaw), then X (pitch), then Z (roll)

    # Pitch (rotation around X axis)
    # sin(pitch) = -m23 (clamp to avoid arcsin domain errors)
    sin_pitch = max(-1.0, min(1.0, -m23))
    pitch = np.degrees(np.arcsin(sin_pitch))

    # Check for gimbal lock (pitch near ±90 degrees)
    if abs(abs(sin_pitch) - 1.0) < 1e-6:
        # Gimbal lock: set roll=0 and calculate yaw from m11 and m31
        roll = 0.0
        yaw = np.degrees(np.arctan2(-m31, m11))
    else:
        # Normal case: calculate yaw and roll
        yaw = np.degrees(np.arctan2(m13, m33))
        roll = np.degrees(np.arctan2(m21, m22))

    return (pitch, yaw, roll)


def validate_look_at(
    camera_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    tolerance_degrees: float = 0.1
) -> Tuple[bool, float]:
    """Validate that look_at calculation produces correct forward vector.

    Args:
        camera_pos: Camera position (x, y, z)
        target_pos: Target position to look at (x, y, z)
        tolerance_degrees: Maximum allowed angle error in degrees

    Returns:
        (is_valid, error_degrees) - whether look-at is valid and angle error
    """
    # Calculate look-at angles
    pitch, yaw, roll = look_at_target(camera_pos, target_pos)

    # Calculate forward vector from those angles
    forward = euler_to_forward_vector(pitch, yaw, roll)

    # Expected direction (camera → target, normalized)
    dx = target_pos[0] - camera_pos[0]
    dy = target_pos[1] - camera_pos[1]
    dz = target_pos[2] - camera_pos[2]
    expected_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)

    if expected_magnitude < 1e-8:
        # Camera at target - any direction is valid
        return (True, 0.0)

    expected = Vector3(
        dx / expected_magnitude,
        dy / expected_magnitude,
        dz / expected_magnitude
    )

    # Calculate angle between forward and expected
    dot_product = forward.dot(expected)
    # Clamp to [-1, 1] to avoid arccos domain errors from floating point
    dot_product = max(-1.0, min(1.0, dot_product))
    angle_error = np.degrees(np.arccos(dot_product))

    return (angle_error <= tolerance_degrees, angle_error)
