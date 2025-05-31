"""
Movement Analysis Utility for Deforum Schedule Translation
Translates xyz rotation/translation schedules into English descriptions for prompt enhancement
"""

import re
import math
from typing import Dict, List, Tuple, Optional

# Try to import DeformAnimKeys, but fall back to standalone parsing if not available
try:
    from ...animation_key_frames import DeformAnimKeys
    DEFORUM_AVAILABLE = True
except ImportError:
    DEFORUM_AVAILABLE = False


def parse_schedule_string(schedule_str: str, max_frames: int = 100) -> List[Tuple[int, float]]:
    """
    Parse Deforum schedule string into frame-value pairs
    
    Format: "0:(1.0), 30:(2.0), 60:(1.5)"
    Returns: [(0, 1.0), (30, 2.0), (60, 1.5)]
    """
    if not schedule_str or schedule_str.strip() == "":
        return [(0, 0.0)]
        
    # Clean the string
    schedule_str = schedule_str.strip()
    
    # Pattern to match frame:(value) pairs
    pattern = r'(\d+):\s*\(([^)]+)\)'
    matches = re.findall(pattern, schedule_str)
    
    if not matches:
        # Try to parse as single value
        try:
            value = float(schedule_str)
            return [(0, value)]
        except ValueError:
            return [(0, 0.0)]
    
    # Convert matches to frame-value pairs
    keyframes = []
    for frame_str, value_str in matches:
        try:
            frame = int(frame_str)
            value = float(value_str)
            keyframes.append((frame, value))
        except ValueError:
            continue
    
    # Sort by frame number
    keyframes.sort(key=lambda x: x[0])
    
    return keyframes if keyframes else [(0, 0.0)]


def interpolate_schedule(keyframes: List[Tuple[int, float]], max_frames: int) -> List[float]:
    """
    Interpolate schedule values across all frames
    """
    if not keyframes:
        return [0.0] * max_frames
    
    values = []
    
    for frame in range(max_frames):
        # Find surrounding keyframes
        prev_kf = None
        next_kf = None
        
        for kf in keyframes:
            if kf[0] <= frame:
                prev_kf = kf
            if kf[0] >= frame and next_kf is None:
                next_kf = kf
        
        if prev_kf is None:
            values.append(keyframes[0][1])
        elif next_kf is None:
            values.append(prev_kf[1])
        elif prev_kf[0] == next_kf[0]:
            values.append(prev_kf[1])
        else:
            # Linear interpolation
            t = (frame - prev_kf[0]) / (next_kf[0] - prev_kf[0])
            value = prev_kf[1] + t * (next_kf[1] - prev_kf[1])
            values.append(value)
    
    return values


class MovementAnalyzer:
    """Analyzes Deforum movement schedules and translates them to English descriptions"""
    
    def __init__(self, sensitivity: float = 1.0):
        """
        Initialize movement analyzer
        
        Args:
            sensitivity: Sensitivity for movement detection (0.1-2.0)
                        Higher values detect smaller movements
        """
        self.sensitivity = max(0.1, min(2.0, sensitivity))
    
    def analyze_translation(self, x_schedule: str, y_schedule: str, z_schedule: str, max_frames: int) -> Tuple[str, float]:
        """Analyze translation schedules and return description + strength"""
        
        # Parse schedules
        x_keyframes = parse_schedule_string(x_schedule, max_frames)
        y_keyframes = parse_schedule_string(y_schedule, max_frames)
        z_keyframes = parse_schedule_string(z_schedule, max_frames)
        
        # Interpolate values
        x_values = interpolate_schedule(x_keyframes, max_frames)
        y_values = interpolate_schedule(y_keyframes, max_frames)
        z_values = interpolate_schedule(z_keyframes, max_frames)
        
        # Calculate movement magnitudes
        x_range = max(x_values) - min(x_values)
        y_range = max(y_values) - min(y_values)
        z_range = max(z_values) - min(z_values)
        
        # Apply sensitivity
        x_range *= self.sensitivity
        y_range *= self.sensitivity
        z_range *= self.sensitivity
        
        # Determine primary movements
        movements = []
        total_movement = 0
        
        # X-axis (horizontal pan)
        if x_range > 5:
            x_delta = x_values[-1] - x_values[0]
            if x_delta > 0:
                movements.append("right pan")
            else:
                movements.append("left pan")
            total_movement += x_range
        
        # Y-axis (vertical pan) 
        if y_range > 5:
            y_delta = y_values[-1] - y_values[0]
            if y_delta > 0:
                movements.append("upward pan")
            else:
                movements.append("downward pan")
            total_movement += y_range
        
        # Z-axis (dolly)
        if z_range > 5:
            z_delta = z_values[-1] - z_values[0]
            if z_delta > 0:
                movements.append("forward dolly")
            else:
                movements.append("backward dolly")
            total_movement += z_range
        
        # Generate description
        if not movements:
            return "static camera", 0.0
        
        # Determine speed
        speed = "slow"
        if total_movement > 100:
            speed = "fast"
        elif total_movement > 50:
            speed = "medium"
        
        description = f"{speed} {', '.join(movements)}"
        strength = min(1.0, total_movement / 200.0)  # Normalize to 0-1
        
        return description, strength
    
    def analyze_rotation(self, x_rot: str, y_rot: str, z_rot: str, max_frames: int) -> Tuple[str, float]:
        """Analyze rotation schedules and return description + strength"""
        
        # Parse schedules
        x_keyframes = parse_schedule_string(x_rot, max_frames)
        y_keyframes = parse_schedule_string(y_rot, max_frames)
        z_keyframes = parse_schedule_string(z_rot, max_frames)
        
        # Interpolate values
        x_values = interpolate_schedule(x_keyframes, max_frames)
        y_values = interpolate_schedule(y_keyframes, max_frames)
        z_values = interpolate_schedule(z_keyframes, max_frames)
        
        # Calculate rotation ranges
        x_range = max(x_values) - min(x_values)
        y_range = max(y_values) - min(y_values)
        z_range = max(z_values) - min(z_values)
        
        # Apply sensitivity
        x_range *= self.sensitivity
        y_range *= self.sensitivity
        z_range *= self.sensitivity
        
        # Determine rotations
        rotations = []
        total_rotation = 0
        
        # X-axis rotation (pitch)
        if x_range > 1:
            x_delta = x_values[-1] - x_values[0]
            if x_delta > 0:
                rotations.append("upward pitch")
            else:
                rotations.append("downward pitch")
            total_rotation += x_range
        
        # Y-axis rotation (yaw)
        if y_range > 1:
            y_delta = y_values[-1] - y_values[0]
            if y_delta > 0:
                rotations.append("right yaw")
            else:
                rotations.append("left yaw")
            total_rotation += y_range
        
        # Z-axis rotation (roll)
        if z_range > 1:
            z_delta = z_values[-1] - z_values[0]
            if z_delta > 0:
                rotations.append("clockwise roll")
            else:
                rotations.append("counterclockwise roll")
            total_rotation += z_range
        
        # Generate description
        if not rotations:
            return "", 0.0
        
        # Determine speed
        speed = "slow"
        if total_rotation > 30:
            speed = "fast"
        elif total_rotation > 15:
            speed = "medium"
        
        description = f"{speed} {', '.join(rotations)}"
        strength = min(1.0, total_rotation / 60.0)  # Normalize to 0-1
        
        return description, strength
    
    def analyze_zoom(self, zoom_schedule: str, max_frames: int) -> Tuple[str, float]:
        """Analyze zoom schedule and return description + strength"""
        
        # Parse schedule
        zoom_keyframes = parse_schedule_string(zoom_schedule, max_frames)
        zoom_values = interpolate_schedule(zoom_keyframes, max_frames)
        
        if not zoom_values or len(zoom_values) < 2:
            return "", 0.0
        
        # Calculate zoom change
        zoom_start = zoom_values[0]
        zoom_end = zoom_values[-1]
        zoom_delta = zoom_end - zoom_start
        zoom_range = max(zoom_values) - min(zoom_values)
        
        # Apply sensitivity
        zoom_range *= self.sensitivity
        
        if abs(zoom_delta) < 0.1:
            return "", 0.0
        
        # Determine zoom direction and speed
        if zoom_delta > 0:
            direction = "zoom in"
        else:
            direction = "zoom out"
        
        # Determine speed
        speed = "slow"
        if zoom_range > 1.0:
            speed = "fast"
        elif zoom_range > 0.5:
            speed = "medium"
        
        description = f"{speed} {direction}"
        strength = min(1.0, zoom_range / 2.0)  # Normalize to 0-1
        
        return description, strength


def analyze_deforum_movement(anim_args, sensitivity: float = 1.0, max_frames: int = 100) -> Tuple[str, float]:
    """
    Analyze Deforum movement schedules and return English description + motion strength
    
    Args:
        anim_args: Object with Deforum animation arguments
        sensitivity: Movement detection sensitivity (0.1-2.0)
        max_frames: Maximum frames to analyze
    
    Returns:
        Tuple of (description_string, motion_strength_float)
    """
    
    analyzer = MovementAnalyzer(sensitivity)
    
    # Get schedule strings from anim_args and ensure they are strings
    translation_x = str(getattr(anim_args, 'translation_x', "0: (0)"))
    translation_y = str(getattr(anim_args, 'translation_y', "0: (0)"))
    translation_z = str(getattr(anim_args, 'translation_z', "0: (0)"))
    rotation_3d_x = str(getattr(anim_args, 'rotation_3d_x', "0: (0)"))
    rotation_3d_y = str(getattr(anim_args, 'rotation_3d_y', "0: (0)"))
    rotation_3d_z = str(getattr(anim_args, 'rotation_3d_z', "0: (0)"))
    zoom = str(getattr(anim_args, 'zoom', "0: (1.0)"))
    angle = str(getattr(anim_args, 'angle', "0: (0)"))
    
    # Debug output to help troubleshoot
    print(f"🔍 Movement Analysis Debug:")
    print(f"   translation_x: {translation_x}")
    print(f"   translation_y: {translation_y}")
    print(f"   translation_z: {translation_z}")
    print(f"   rotation_3d_x: {rotation_3d_x}")
    print(f"   rotation_3d_y: {rotation_3d_y}")
    print(f"   rotation_3d_z: {rotation_3d_z}")
    print(f"   zoom: {zoom}")
    print(f"   angle: {angle}")
    
    # Analyze each type of movement
    translation_desc, translation_strength = analyzer.analyze_translation(
        translation_x, translation_y, translation_z, max_frames
    )
    
    rotation_desc, rotation_strength = analyzer.analyze_rotation(
        rotation_3d_x, rotation_3d_y, rotation_3d_z, max_frames
    )
    
    zoom_desc, zoom_strength = analyzer.analyze_zoom(zoom, max_frames)
    
    # Handle 2D angle rotation if present
    angle_desc, angle_strength = "", 0.0
    if angle != "0: (0)" and angle != "0:(0)":
        angle_desc, angle_strength = analyzer.analyze_zoom(angle, max_frames)  # Reuse zoom logic
        if angle_desc:
            angle_desc = angle_desc.replace("zoom", "rotation")
    
    # Combine descriptions
    descriptions = []
    strengths = []
    
    if translation_desc and translation_desc != "static camera":
        descriptions.append(translation_desc)
        strengths.append(translation_strength)
    
    if rotation_desc:
        descriptions.append(rotation_desc)
        strengths.append(rotation_strength)
    
    if zoom_desc:
        descriptions.append(zoom_desc)
        strengths.append(zoom_strength)
    
    if angle_desc:
        descriptions.append(angle_desc)
        strengths.append(angle_strength)
    
    # Generate final description
    if not descriptions:
        return "static camera shot", 0.0
    
    combined_description = "camera movement with " + ", ".join(descriptions)
    combined_strength = max(strengths) if strengths else 0.0
    
    return combined_description, combined_strength


def generate_wan_motion_intensity_schedule(anim_args, max_frames: int = 100, sensitivity: float = 1.0) -> str:
    """
    Generate a Wan motion intensity schedule from Deforum movement schedules
    
    Args:
        anim_args: Object with Deforum animation arguments
        max_frames: Maximum frames to analyze
        sensitivity: Movement detection sensitivity
        
    Returns:
        Motion intensity schedule string (e.g., "0:(0.5), 50:(1.2), 100:(0.8)")
    """
    
    analyzer = MovementAnalyzer(sensitivity)
    
    # Get schedule strings from anim_args
    translation_x = str(getattr(anim_args, 'translation_x', "0: (0)"))
    translation_y = str(getattr(anim_args, 'translation_y', "0: (0)"))
    translation_z = str(getattr(anim_args, 'translation_z', "0: (0)"))
    rotation_3d_x = str(getattr(anim_args, 'rotation_3d_x', "0: (0)"))
    rotation_3d_y = str(getattr(anim_args, 'rotation_3d_y', "0: (0)"))
    rotation_3d_z = str(getattr(anim_args, 'rotation_3d_z', "0: (0)"))
    zoom = str(getattr(anim_args, 'zoom', "0: (1.0)"))
    angle = str(getattr(anim_args, 'angle', "0: (0)"))
    
    # Parse all movement schedules
    tx_keyframes = parse_schedule_string(translation_x, max_frames)
    ty_keyframes = parse_schedule_string(translation_y, max_frames)
    tz_keyframes = parse_schedule_string(translation_z, max_frames)
    rx_keyframes = parse_schedule_string(rotation_3d_x, max_frames)
    ry_keyframes = parse_schedule_string(rotation_3d_y, max_frames)
    rz_keyframes = parse_schedule_string(rotation_3d_z, max_frames)
    zoom_keyframes = parse_schedule_string(zoom, max_frames)
    angle_keyframes = parse_schedule_string(angle, max_frames)
    
    # Interpolate all movement values across frames
    tx_values = interpolate_schedule(tx_keyframes, max_frames)
    ty_values = interpolate_schedule(ty_keyframes, max_frames)
    tz_values = interpolate_schedule(tz_keyframes, max_frames)
    rx_values = interpolate_schedule(rx_keyframes, max_frames)
    ry_values = interpolate_schedule(ry_keyframes, max_frames)
    rz_values = interpolate_schedule(rz_keyframes, max_frames)
    zoom_values = interpolate_schedule(zoom_keyframes, max_frames)
    angle_values = interpolate_schedule(angle_keyframes, max_frames)
    
    # Calculate motion intensity for each frame
    motion_intensities = []
    
    for frame in range(max_frames):
        # Calculate movement deltas for this frame (compared to previous frame)
        if frame == 0:
            frame_intensity = 0.0
        else:
            # Translation deltas
            tx_delta = abs(tx_values[frame] - tx_values[frame-1]) * sensitivity
            ty_delta = abs(ty_values[frame] - ty_values[frame-1]) * sensitivity
            tz_delta = abs(tz_values[frame] - tz_values[frame-1]) * sensitivity
            
            # Rotation deltas
            rx_delta = abs(rx_values[frame] - rx_values[frame-1]) * sensitivity * 10  # Scale rotation
            ry_delta = abs(ry_values[frame] - ry_values[frame-1]) * sensitivity * 10
            rz_delta = abs(rz_values[frame] - rz_values[frame-1]) * sensitivity * 10
            
            # Zoom delta
            zoom_delta = abs(zoom_values[frame] - zoom_values[frame-1]) * sensitivity * 20  # Scale zoom
            
            # Angle delta
            angle_delta = abs(angle_values[frame] - angle_values[frame-1]) * sensitivity * 10
            
            # Combine all deltas into motion intensity
            total_delta = tx_delta + ty_delta + tz_delta + rx_delta + ry_delta + rz_delta + zoom_delta + angle_delta
            
            # Normalize to 0.0-2.0 range for Wan motion strength
            frame_intensity = min(2.0, total_delta / 20.0)  # Adjust divisor to tune sensitivity
        
        motion_intensities.append(frame_intensity)
    
    # Create keyframes for the motion intensity schedule
    # Sample at regular intervals to avoid huge schedules
    sample_interval = max(1, max_frames // 20)  # Max 20 keyframes
    keyframes = []
    
    for frame in range(0, max_frames, sample_interval):
        intensity = motion_intensities[frame]
        keyframes.append(f"{frame}:({intensity:.3f})")
    
    # Always include the last frame
    if (max_frames - 1) % sample_interval != 0:
        last_intensity = motion_intensities[-1]
        keyframes.append(f"{max_frames-1}:({last_intensity:.3f})")
    
    schedule_string = ", ".join(keyframes)
    
    print(f"📐 Generated Wan motion intensity schedule: {schedule_string}")
    
    return schedule_string 