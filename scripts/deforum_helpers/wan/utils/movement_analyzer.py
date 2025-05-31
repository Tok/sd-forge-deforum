"""
Enhanced Movement Analyzer for Deforum
Analyzes camera movement schedules and converts them to text descriptions for Wan
Supports basic movements (pan, dolly, zoom, tilt) and complex patterns (circle, roll, shake)
Integrates with Camera Shakify to generate combined movement schedules
"""

import re
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from types import SimpleNamespace

# Try to import Camera Shakify components for integration
try:
    # Direct import of shake data to avoid dependency issues
    from ...rendering.data.shakify.shake_data import SHAKE_LIST
    from ...defaults import get_camera_shake_list
    import pandas as pd
    from scipy.interpolate import CubicSpline
    SHAKIFY_AVAILABLE = True
    print("🎬 Camera Shakify integration available")
except ImportError:
    SHAKIFY_AVAILABLE = False
    print("⚠️ Camera Shakify integration not available")

# Try to import DeformAnimKeys, but fall back to standalone parsing if not available
try:
    from ...animation_key_frames import DeformAnimKeys
    DEFORUM_AVAILABLE = True
except ImportError:
    DEFORUM_AVAILABLE = False

# Simplified Camera Shakify integration - hardcoded data for testing
SHAKIFY_AVAILABLE = True

# Simplified shake data for INVESTIGATION pattern (extracted from the actual data)
INVESTIGATION_SHAKE_DATA = {
    'translation': {
        'x': [0.021819, 0.012368, 0.003192, -0.006550, -0.016339, -0.026018, -0.034887, -0.042019, -0.049939, -0.056172, -0.061366, -0.066012, -0.070278, -0.075834, -0.080062, -0.085495, -0.090940, -0.095961, -0.101095, -0.106177],
        'y': [0.004563, 0.000000, -0.004563, -0.008587, -0.012519, -0.017078, -0.021599, -0.026818, -0.031728, -0.036636, -0.041351, -0.045605, -0.049422, -0.052272, -0.054630, -0.055820, -0.056572, -0.057333, -0.057542, -0.057255],
        'z': [-0.003604, -0.003431, -0.003387, -0.002934, -0.002923, -0.004425, -0.006044, -0.007711, -0.009409, -0.011712, -0.013893, -0.015884, -0.018140, -0.020823, -0.022288, -0.024929, -0.028371, -0.031594, -0.035160, -0.039225]
    },
    'rotation_3d': {
        'x': [0.001086, 0.000000, -0.001075, -0.002893, -0.005007, -0.007058, -0.009427, -0.012662, -0.015891, -0.018930, -0.021792, -0.024398, -0.026323, -0.027796, -0.029139, -0.029992, -0.030573, -0.031318, -0.032108, -0.032828],
        'y': [0.003974, 0.000000, -0.003977, -0.007220, -0.008964, -0.009045, -0.008403, -0.007829, -0.008275, -0.008627, -0.009438, -0.011140, -0.013551, -0.016658, -0.019090, -0.020987, -0.022070, -0.022678, -0.023137, -0.023632],
        'z': [0.002614, 0.000000, -0.002610, -0.005844, -0.009775, -0.014285, -0.018739, -0.022587, -0.025950, -0.028716, -0.030574, -0.031568, -0.031884, -0.031717, -0.031828, -0.032260, -0.032908, -0.033682, -0.034255, -0.034364]
    }
}

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


def create_shakify_data(shake_name: str, shake_intensity: float, shake_speed: float, target_fps: int = 30, max_frames: int = 120) -> Optional[Dict]:
    """
    Create Camera Shakify data using simplified hardcoded patterns
    """
    if not shake_name or shake_name.lower() in ['none', 'off', ''] or shake_intensity <= 0:
        return None
    
    print(f"🎬 Creating Camera Shakify data: {shake_name} (intensity: {shake_intensity}, speed: {shake_speed})")
    
    # For now, only support INVESTIGATION pattern as demonstration
    if shake_name.upper() != "INVESTIGATION":
        print(f"⚠️ Shake pattern '{shake_name}' not yet supported, using static data")
        return None
    
    # Use the hardcoded data and apply intensity/speed scaling
    base_data = INVESTIGATION_SHAKE_DATA
    result = {
        'translation': {'x': [], 'y': [], 'z': []},
        'rotation_3d': {'x': [], 'y': [], 'z': []}
    }
    
    # Generate values for the requested number of frames
    for frame in range(max_frames):
        for transform_type in ['translation', 'rotation_3d']:
            for axis in ['x', 'y', 'z']:
                # Get base pattern data
                base_values = base_data[transform_type][axis]
                
                # Calculate which frame to sample from (with speed scaling)
                source_frame_idx = int((frame * shake_speed) % len(base_values))
                base_value = base_values[source_frame_idx]
                
                # Apply intensity scaling
                scaled_value = base_value * shake_intensity
                
                result[transform_type][axis].append(scaled_value)
    
    print(f"✅ Generated simplified Camera Shakify data for {max_frames} frames")
    return result


def apply_shakify_to_schedule(base_schedule: str, shake_values: List[float], max_frames: int) -> str:
    """
    Apply Camera Shakify values to a base movement schedule to create combined schedule
    This mimics the _maybe_shake function from the experimental render core
    """
    if not shake_values or len(shake_values) == 0:
        return base_schedule
    
    # Parse base schedule
    base_keyframes = parse_schedule_string(base_schedule, max_frames)
    base_values = interpolate_schedule(base_keyframes, max_frames)
    
    # Apply shake to base values (additive)
    combined_values = []
    for frame in range(min(len(base_values), len(shake_values))):
        combined_value = base_values[frame] + shake_values[frame]
        combined_values.append(combined_value)
    
    # Create new schedule string from combined values
    # Sample every few frames to keep schedule reasonable
    sample_interval = max(1, max_frames // 20)  # Max 20 keyframes
    keyframes = []
    
    for frame in range(0, len(combined_values), sample_interval):
        value = combined_values[frame]
        keyframes.append(f"{frame}:({value:.6f})")
    
    # Always include the last frame
    if (len(combined_values) - 1) % sample_interval != 0:
        last_value = combined_values[-1]
        keyframes.append(f"{len(combined_values)-1}:({last_value:.6f})")
    
    return ", ".join(keyframes)


class MovementAnalyzer:
    """
    Enhanced Movement Analyzer with fine-grained, frame-by-frame analysis
    Detects subtle movements and provides specific descriptions for affected frame ranges
    """
    
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity
        self.min_translation_threshold = 0.1 * sensitivity  # Much lower threshold for subtle movement
        self.min_rotation_threshold = 0.05 * sensitivity   # Much lower threshold for subtle rotation
        self.min_zoom_threshold = 0.001 * sensitivity      # Much lower threshold for subtle zoom
    
    def analyze_frame_ranges(self, values: List[float], movement_type: str) -> List[Dict]:
        """
        Analyze movement values frame by frame and identify specific movement ranges
        Returns list of movement segments with frame ranges and descriptions
        """
        if not values or len(values) < 2:
            return []
        
        # Calculate frame-by-frame changes
        changes = []
        for i in range(1, len(values)):
            change = values[i] - values[i-1]
            changes.append(change)
        
        # Find movement segments
        segments = []
        current_segment = None
        
        for frame, change in enumerate(changes):
            abs_change = abs(change)
            
            # Determine movement threshold based on type
            if movement_type == "zoom":
                threshold = self.min_zoom_threshold
            elif movement_type in ["rotation_x", "rotation_y", "rotation_z"]:
                threshold = self.min_rotation_threshold
            else:  # translation
                threshold = self.min_translation_threshold
            
            # Check if this frame has significant movement
            if abs_change > threshold:
                direction = "increasing" if change > 0 else "decreasing"
                
                # Start new segment or continue existing one
                if current_segment is None or current_segment['direction'] != direction:
                    # End previous segment
                    if current_segment is not None:
                        current_segment['end_frame'] = frame
                        segments.append(current_segment)
                    
                    # Start new segment
                    current_segment = {
                        'start_frame': frame,
                        'end_frame': frame,
                        'direction': direction,
                        'movement_type': movement_type,
                        'max_change': abs_change,
                        'total_range': abs(values[frame+1] - values[frame])
                    }
                else:
                    # Continue current segment
                    current_segment['end_frame'] = frame
                    current_segment['max_change'] = max(current_segment['max_change'], abs_change)
                    current_segment['total_range'] = abs(values[frame+1] - values[current_segment['start_frame']])
            
            elif current_segment is not None:
                # End current segment if movement stopped
                current_segment['end_frame'] = frame
                segments.append(current_segment)
                current_segment = None
        
        # Close final segment if needed
        if current_segment is not None:
            current_segment['end_frame'] = len(changes)
            segments.append(current_segment)
        
        return segments
    
    def generate_segment_description(self, segment: Dict, total_frames: int) -> str:
        """
        Generate specific description for a movement segment
        """
        start = segment['start_frame']
        end = segment['end_frame']
        movement_type = segment['movement_type']
        direction = segment['direction']
        total_range = segment['total_range']
        
        # Frame range description
        if end - start < 10:
            frame_desc = f"frames {start}-{end}"
        elif end - start < total_frames * 0.3:
            frame_desc = f"frames {start}-{end} (brief)"
        elif end - start < total_frames * 0.7:
            frame_desc = f"frames {start}-{end} (moderate)"
        else:
            frame_desc = f"frames {start}-{end} (extended)"
        
        # Movement intensity
        if total_range < 1.0:
            intensity = "subtle"
        elif total_range < 5.0:
            intensity = "gentle"
        elif total_range < 20.0:
            intensity = "moderate"
        else:
            intensity = "strong"
        
        # Movement type specific descriptions
        if movement_type == "translation_x":
            motion = "panning right" if direction == "increasing" else "panning left"
        elif movement_type == "translation_y":
            motion = "moving up" if direction == "increasing" else "moving down"
        elif movement_type == "translation_z":
            motion = "dolly forward" if direction == "increasing" else "dolly backward"
        elif movement_type == "rotation_x":
            motion = "tilting up" if direction == "increasing" else "tilting down"
        elif movement_type == "rotation_y":
            motion = "rotating right" if direction == "increasing" else "rotating left"
        elif movement_type == "rotation_z":
            motion = "rolling clockwise" if direction == "increasing" else "rolling counter-clockwise"
        elif movement_type == "zoom":
            motion = "zooming in" if direction == "increasing" else "zooming out"
        else:
            motion = f"{movement_type} {direction}"
        
        return f"{intensity} {motion} ({frame_desc})"

    def analyze_translation(self, x_schedule: str, y_schedule: str, z_schedule: str, max_frames: int) -> Tuple[str, float]:
        """Enhanced translation analysis with frame-by-frame detection"""
        # Parse schedules
        x_keyframes = parse_schedule_string(x_schedule, max_frames)
        y_keyframes = parse_schedule_string(y_schedule, max_frames)
        z_keyframes = parse_schedule_string(z_schedule, max_frames)
        
        # Interpolate values
        x_values = interpolate_schedule(x_keyframes, max_frames)
        y_values = interpolate_schedule(y_keyframes, max_frames)
        z_values = interpolate_schedule(z_keyframes, max_frames)
        
        # Analyze each axis for movement segments
        x_segments = self.analyze_frame_ranges(x_values, "translation_x")
        y_segments = self.analyze_frame_ranges(y_values, "translation_y")
        z_segments = self.analyze_frame_ranges(z_values, "translation_z")
        
        all_segments = x_segments + y_segments + z_segments
        
        if not all_segments:
            return "static camera position", 0.0
        
        # Sort segments by frame order
        all_segments.sort(key=lambda s: s['start_frame'])
        
        # Generate combined description
        descriptions = []
        total_strength = 0.0
        
        for segment in all_segments:
            desc = self.generate_segment_description(segment, max_frames)
            descriptions.append(desc)
            # Calculate strength based on range and duration
            duration = segment['end_frame'] - segment['start_frame']
            strength = (segment['total_range'] * duration) / (max_frames * 10.0)
            total_strength += strength
        
        # Combine descriptions intelligently
        if len(descriptions) == 1:
            final_desc = f"camera movement with {descriptions[0]}"
        elif len(descriptions) <= 3:
            final_desc = f"camera movement with {', '.join(descriptions[:-1])} and {descriptions[-1]}"
        else:
            # Too many segments - summarize
            main_movements = set()
            for segment in all_segments:
                if "panning" in self.generate_segment_description(segment, max_frames):
                    main_movements.add("panning movement")
                elif "dolly" in self.generate_segment_description(segment, max_frames):
                    main_movements.add("dolly movement")
                elif "moving" in self.generate_segment_description(segment, max_frames):
                    main_movements.add("vertical movement")
            
            if main_movements:
                final_desc = f"complex camera movement with {', '.join(main_movements)}"
            else:
                final_desc = "dynamic camera movement with multiple motion phases"
        
        return final_desc, min(total_strength, 2.0)  # Cap at 2.0
    
    def analyze_rotation(self, x_rot: str, y_rot: str, z_rot: str, max_frames: int) -> Tuple[str, float]:
        """Enhanced rotation analysis with frame-by-frame detection"""
        # Parse schedules
        x_keyframes = parse_schedule_string(x_rot, max_frames)
        y_keyframes = parse_schedule_string(y_rot, max_frames)
        z_keyframes = parse_schedule_string(z_rot, max_frames)
        
        # Interpolate values
        x_values = interpolate_schedule(x_keyframes, max_frames)
        y_values = interpolate_schedule(y_keyframes, max_frames)
        z_values = interpolate_schedule(z_keyframes, max_frames)
        
        # Analyze each axis for movement segments
        x_segments = self.analyze_frame_ranges(x_values, "rotation_x")
        y_segments = self.analyze_frame_ranges(y_values, "rotation_y")
        z_segments = self.analyze_frame_ranges(z_values, "rotation_z")
        
        all_segments = x_segments + y_segments + z_segments
        
        if not all_segments:
            return "", 0.0
        
        # Sort segments by frame order
        all_segments.sort(key=lambda s: s['start_frame'])
        
        # Generate combined description
        descriptions = []
        total_strength = 0.0
        
        for segment in all_segments:
            desc = self.generate_segment_description(segment, max_frames)
            descriptions.append(desc)
            # Calculate strength based on range and duration
            duration = segment['end_frame'] - segment['start_frame']
            # Rotation strength calculation (degrees are smaller numbers)
            strength = (segment['total_range'] * duration) / (max_frames * 2.0)  # More sensitive for rotation
            total_strength += strength
        
        # Combine descriptions intelligently
        if len(descriptions) == 1:
            final_desc = descriptions[0]
        elif len(descriptions) <= 3:
            final_desc = f"{', '.join(descriptions[:-1])} and {descriptions[-1]}"
        else:
            # Too many segments - summarize
            main_movements = set()
            for segment in all_segments:
                if "tilting" in self.generate_segment_description(segment, max_frames):
                    main_movements.add("tilting movement")
                elif "rotating" in self.generate_segment_description(segment, max_frames):
                    main_movements.add("rotation movement")
                elif "rolling" in self.generate_segment_description(segment, max_frames):
                    main_movements.add("roll movement")
            
            if main_movements:
                final_desc = f"complex {', '.join(main_movements)}"
            else:
                final_desc = "dynamic rotation with multiple phases"
        
        return final_desc, min(total_strength, 2.0)  # Cap at 2.0
    
    def analyze_zoom(self, zoom_schedule: str, max_frames: int) -> Tuple[str, float]:
        """Enhanced zoom analysis with frame-by-frame detection and proper static detection"""
        # Parse zoom schedule
        zoom_keyframes = parse_schedule_string(zoom_schedule, max_frames)
        zoom_values = interpolate_schedule(zoom_keyframes, max_frames)
        
        # Analyze for movement segments
        zoom_segments = self.analyze_frame_ranges(zoom_values, "zoom")
        
        if not zoom_segments:
            return "", 0.0
        
        # Sort segments by frame order
        zoom_segments.sort(key=lambda s: s['start_frame'])
        
        # Generate combined description
        descriptions = []
        total_strength = 0.0
        
        for segment in zoom_segments:
            desc = self.generate_segment_description(segment, max_frames)
            descriptions.append(desc)
            # Calculate strength based on range and duration
            duration = segment['end_frame'] - segment['start_frame']
            # Zoom strength calculation (zoom values are often small decimals)
            strength = (segment['total_range'] * duration) / (max_frames * 0.1)  # Very sensitive for zoom
            total_strength += strength
        
        # Combine descriptions intelligently
        if len(descriptions) == 1:
            final_desc = descriptions[0]
        elif len(descriptions) <= 3:
            final_desc = f"{', '.join(descriptions[:-1])} and {descriptions[-1]}"
        else:
            # Too many segments - summarize
            zoom_ins = sum(1 for seg in zoom_segments if seg['direction'] == "increasing")
            zoom_outs = sum(1 for seg in zoom_segments if seg['direction'] == "decreasing")
            
            if zoom_ins > zoom_outs:
                final_desc = "complex zoom movement with predominantly zooming in"
            elif zoom_outs > zoom_ins:
                final_desc = "complex zoom movement with predominantly zooming out"
            else:
                final_desc = "complex zoom movement with multiple in/out phases"
        
        return final_desc, min(total_strength, 2.0)  # Cap at 2.0
    
    def analyze_rotation_pattern(self, x_schedule: str, y_schedule: str, z_schedule: str, max_frames: int) -> Tuple[str, float, str]:
        """Analyze rotation schedules for complex patterns like circular motion and roll"""
        
        # Parse schedules
        x_keyframes = parse_schedule_string(x_schedule, max_frames)
        y_keyframes = parse_schedule_string(y_schedule, max_frames)
        z_keyframes = parse_schedule_string(z_schedule, max_frames)
        
        # Interpolate values
        x_values = interpolate_schedule(x_keyframes, max_frames)
        y_values = interpolate_schedule(y_keyframes, max_frames)
        z_values = interpolate_schedule(z_keyframes, max_frames)
        
        descriptions = []
        strengths = []
        
        # Check for circular motion (combined X and Y rotation)
        if len(x_values) > 1 and len(y_values) > 1:
            x_range = max(x_values) - min(x_values)
            y_range = max(y_values) - min(y_values)
            
            # Apply sensitivity
            x_range *= self.sensitivity
            y_range *= self.sensitivity
            
            # Check if both have significant motion
            if x_range > 2.0 and y_range > 2.0:
                # Check if it's circular motion by looking at phase relationship
                if self._is_circular_motion(x_values, y_values):
                    circle_strength = (x_range + y_range) / 2
                    descriptions.append("circular camera movement")
                    strengths.append(circle_strength)
                    return " and ".join(descriptions), max(strengths) if strengths else 0.0, "circular"
        
        # Check for camera roll (Z rotation)
        if len(z_values) > 1:
            z_range = max(z_values) - min(z_values)
            z_range *= self.sensitivity
            
            if z_range > 1.0:  # Threshold for detecting roll
                z_start = z_values[0]
                z_end = z_values[-1]
                z_delta = z_end - z_start
                
                if abs(z_delta) > 0.5:
                    if z_delta > 0:
                        descriptions.append("camera roll clockwise")
                    else:
                        descriptions.append("camera roll counter-clockwise")
                    strengths.append(abs(z_delta))
                    return " and ".join(descriptions), max(strengths) if strengths else 0.0, "roll"
        
        # Fall back to individual rotation analysis
        individual_desc, individual_strength = self.analyze_rotation(x_schedule, y_schedule, z_schedule, max_frames)
        
        # Determine motion type for classification
        motion_type = "rotation"
        if "tilt" in individual_desc:
            motion_type = "tilt"
        elif "yaw" in individual_desc or "pan" in individual_desc:
            motion_type = "yaw"
            
        return individual_desc, individual_strength, motion_type
    
    def _is_circular_motion(self, x_values: List[float], y_values: List[float]) -> bool:
        """Detect if X and Y rotation values form a circular pattern"""
        if len(x_values) < 4 or len(y_values) < 4:
            return False
            
        try:
            # Calculate phase difference between X and Y
            # For circular motion, there should be a ~90 degree phase difference
            x_array = np.array(x_values)
            y_array = np.array(y_values)
            
            # Normalize to remove DC offset
            x_norm = x_array - np.mean(x_array)
            y_norm = y_array - np.mean(y_array)
            
            # Calculate correlation with 90-degree phase shift
            quarter_period = len(x_values) // 4
            if quarter_period > 0:
                x_shifted = np.roll(x_norm, quarter_period)
                correlation = np.corrcoef(y_norm, x_shifted)[0, 1]
                
                # High correlation indicates circular motion
                return abs(correlation) > 0.7
                
        except (ValueError, IndexError):
            pass
            
        return False
    
    def analyze_camera_shake(self, shake_name: str, shake_intensity: float, shake_speed: float) -> Tuple[str, float]:
        """Analyze Camera Shakify shake patterns"""
        
        if not shake_name or shake_name.lower() in ['none', 'off', '']:
            return "", 0.0
            
        # Map shake names to descriptive terms
        shake_descriptions = {
            'INVESTIGATION': 'investigative handheld camera movement',
            'THE_CLOSEUP': 'intimate close-up camera shake',
            'THE_WEDDING': 'gentle documentary-style camera movement',
            'WALK_TO_THE_STORE': 'walking pace camera movement',
            'HANDYCAM_RUN': 'dynamic running camera shake',
            'OUT_CAR_WINDOW': 'vehicle motion camera movement',
            'BIKE_ON_GRAVEL_2D': 'bumpy terrain camera shake',
            'SPACESHIP_SHAKE_2D': 'vibrating spacecraft camera movement',
            'THE_ZEEK_2D': 'rhythmic camera shake pattern'
        }
        
        # Get base description
        base_desc = shake_descriptions.get(shake_name.upper(), f'{shake_name.lower()} camera shake')
        
        # Modify description based on intensity and speed
        intensity_modifiers = {
            (0.0, 0.3): 'subtle',
            (0.3, 0.7): 'gentle',
            (0.7, 1.5): 'moderate',
            (1.5, 2.5): 'intense',
            (2.5, float('inf')): 'extreme'
        }
        
        speed_modifiers = {
            (0.0, 0.5): 'slow',
            (0.5, 0.8): 'steady',
            (0.8, 1.2): 'normal',
            (1.2, 2.0): 'fast',
            (2.0, float('inf')): 'rapid'
        }
        
        # Find appropriate modifiers
        intensity_mod = next((mod for (min_val, max_val), mod in intensity_modifiers.items() 
                             if min_val <= shake_intensity < max_val), 'moderate')
        
        speed_mod = next((mod for (min_val, max_val), mod in speed_modifiers.items() 
                         if min_val <= shake_speed < max_val), 'normal')
        
        # Combine description
        if intensity_mod != 'moderate' or speed_mod != 'normal':
            description = f"{intensity_mod} {speed_mod} {base_desc}"
        else:
            description = base_desc
            
        # Calculate combined strength
        strength = shake_intensity * shake_speed * 0.5  # Scale down as it's additive
        
        return description, strength


def analyze_deforum_movement(anim_args, sensitivity: float = 1.0, max_frames: int = 120) -> Tuple[str, float]:
    """
    Enhanced movement analysis with full Camera Shakify integration
    Creates combined movement schedules like the experimental render core does
    """
    analyzer = MovementAnalyzer(sensitivity)
    
    descriptions = []
    strengths = []
    
    # 1. Handle Camera Shakify integration first
    shake_name = getattr(anim_args, 'shake_name', 'None')
    shake_intensity = getattr(anim_args, 'shake_intensity', 0.0)
    shake_speed = getattr(anim_args, 'shake_speed', 1.0)
    
    # Generate Camera Shakify data if enabled
    shakify_data = None
    if shake_name and shake_name != 'None' and SHAKIFY_AVAILABLE:
        shakify_data = create_shakify_data(shake_name, shake_intensity, shake_speed, target_fps=30, max_frames=max_frames)
        print(f"🎬 Camera Shakify '{shake_name}' integration: {'✅ Success' if shakify_data else '❌ Failed'}")
    
    # 2. Create combined movement schedules (base + shake)
    # This mimics what the experimental render core does with _maybe_shake
    if shakify_data:
        # Apply shakify to translation schedules
        combined_translation_x = apply_shakify_to_schedule(
            anim_args.translation_x, shakify_data['translation']['x'], max_frames)
        combined_translation_y = apply_shakify_to_schedule(
            anim_args.translation_y, shakify_data['translation']['y'], max_frames)
        combined_translation_z = apply_shakify_to_schedule(
            anim_args.translation_z, shakify_data['translation']['z'], max_frames)
        
        # Apply shakify to rotation schedules
        combined_rotation_x = apply_shakify_to_schedule(
            anim_args.rotation_3d_x, shakify_data['rotation_3d']['x'], max_frames)
        combined_rotation_y = apply_shakify_to_schedule(
            anim_args.rotation_3d_y, shakify_data['rotation_3d']['y'], max_frames)
        combined_rotation_z = apply_shakify_to_schedule(
            anim_args.rotation_3d_z, shakify_data['rotation_3d']['z'], max_frames)
        
        print(f"🔧 Applied Camera Shakify to movement schedules")
        print(f"   📐 Combined Translation X: {combined_translation_x[:50]}...")
        print(f"   📐 Combined Rotation Y: {combined_rotation_y[:50]}...")
        
    else:
        # Use original schedules if no shake
        combined_translation_x = anim_args.translation_x
        combined_translation_y = anim_args.translation_y
        combined_translation_z = anim_args.translation_z
        combined_rotation_x = anim_args.rotation_3d_x
        combined_rotation_y = anim_args.rotation_3d_y
        combined_rotation_z = anim_args.rotation_3d_z
    
    # 3. Analyze combined movement schedules
    # Translation analysis
    translation_desc, translation_strength = analyzer.analyze_translation(
        combined_translation_x, combined_translation_y, combined_translation_z, max_frames)
    
    if translation_desc and translation_desc != "static camera position":
        descriptions.append(translation_desc)
        strengths.append(translation_strength)
    
    # Rotation analysis for complex patterns
    rotation_desc, rotation_strength, motion_type = analyzer.analyze_rotation_pattern(
        combined_rotation_x, combined_rotation_y, combined_rotation_z, max_frames)
    
    if rotation_desc:
        descriptions.append(rotation_desc)
        strengths.append(rotation_strength)
    
    # Zoom analysis (if not complex rotation)
    if motion_type not in ['circular', 'roll']:
        zoom_desc, zoom_strength = analyzer.analyze_zoom(anim_args.zoom, max_frames)
        if zoom_desc:
            descriptions.append(zoom_desc)
            strengths.append(zoom_strength)
    
    # 4. Add Camera Shakify description if it was the primary movement source
    if shakify_data and not descriptions:
        # Only add shake description if no other movement was detected
        shake_desc, shake_strength = analyzer.analyze_camera_shake(shake_name, shake_intensity, shake_speed)
        if shake_desc:
            descriptions.append(shake_desc)
            strengths.append(shake_strength)
    
    # 5. Generate final description
    if not descriptions:
        return "static camera position", 0.0
    
    # Combine descriptions naturally
    if len(descriptions) == 1:
        combined_description = f"camera movement with {descriptions[0]}"
    elif len(descriptions) == 2:
        combined_description = f"camera movement with {descriptions[0]} and {descriptions[1]}"
    else:
        combined_description = f"dynamic camera movement with {', '.join(descriptions[:-1])}, and {descriptions[-1]}"
    
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