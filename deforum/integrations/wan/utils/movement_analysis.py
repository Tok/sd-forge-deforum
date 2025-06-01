#!/usr/bin/env python3
"""
Movement Analysis and Description Generation
High-level analysis and text description generation for camera movements
"""

import math
from typing import Dict, List, Tuple, Optional
from types import SimpleNamespace

# Import movement detection functions
from .movement_detection import (
    parse_schedule_string, interpolate_schedule, detect_movement_segments,
    detect_circular_motion, detect_pattern_type, calculate_movement_intensity,
    group_similar_segments
)

# Import movement utilities  
from .movement_utils import (
    create_shakify_data, apply_shakify_to_schedule
)


class MovementAnalyzer:
    """
    Enhanced movement analyzer for Deforum camera movement schedules
    Converts movement data into descriptive text for AI video generation
    """
    
    def __init__(self, sensitivity: float = 1.0):
        """
        Initialize the movement analyzer
        
        Args:
            sensitivity: Detection sensitivity (0.1 to 2.0, default 1.0)
        """
        self.sensitivity = max(0.1, min(2.0, sensitivity))
    
    def generate_segment_description(self, segment: Dict, total_frames: int) -> str:
        """
        Generate human-readable description for a movement segment
        
        Args:
            segment: Movement segment dictionary
            total_frames: Total frames in the sequence
            
        Returns:
            Text description of the movement
        """
        movement_type = segment['movement_type']
        duration = segment['end_frame'] - segment['start_frame']
        total_change = abs(segment['total_change'])
        
        # Determine intensity level
        if total_change < 2:
            intensity = "subtle"
        elif total_change < 10:
            intensity = "gentle"
        elif total_change < 25:
            intensity = "moderate"
        elif total_change < 50:
            intensity = "strong"
        else:
            intensity = "dramatic"
        
        # Determine duration description
        duration_ratio = duration / total_frames
        if duration_ratio < 0.2:
            duration_desc = "brief"
        elif duration_ratio < 0.4:
            duration_desc = "short"
        elif duration_ratio < 0.7:
            duration_desc = "extended"
        else:
            duration_desc = "sustained"
        
        # Generate movement-specific description
        if movement_type == 'translation_x':
            direction = "right" if segment['total_change'] > 0 else "left"
            return f"{intensity} panning {direction} ({duration_desc})"
        elif movement_type == 'translation_y':
            direction = "up" if segment['total_change'] > 0 else "down"
            return f"{intensity} tilting {direction} ({duration_desc})"
        elif movement_type == 'translation_z':
            direction = "forward" if segment['total_change'] < 0 else "backward"
            return f"{intensity} dolly {direction} ({duration_desc})"
        elif movement_type == 'rotation_3d_x':
            direction = "up" if segment['total_change'] > 0 else "down"
            return f"{intensity} pitching {direction} ({duration_desc})"
        elif movement_type == 'rotation_3d_y':
            direction = "right" if segment['total_change'] > 0 else "left"
            return f"{intensity} rotating {direction} ({duration_desc})"
        elif movement_type == 'rotation_3d_z':
            direction = "clockwise" if segment['total_change'] > 0 else "counterclockwise"
            return f"{intensity} rolling {direction} ({duration_desc})"
        elif movement_type == 'zoom':
            direction = "in" if segment['total_change'] > 0 else "out"
            return f"{intensity} zooming {direction} ({duration_desc})"
        else:
            return f"{intensity} {movement_type} ({duration_desc})"
    
    def analyze_translation(self, x_schedule: str, y_schedule: str, z_schedule: str, max_frames: int) -> Tuple[str, float]:
        """
        Analyze translation movement (pan, tilt, dolly)
        
        Args:
            x_schedule: X translation schedule string
            y_schedule: Y translation schedule string  
            z_schedule: Z translation schedule string
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Tuple of (description_text, intensity_score)
        """
        try:
            # Parse schedules
            x_values = interpolate_schedule(parse_schedule_string(x_schedule), max_frames)
            y_values = interpolate_schedule(parse_schedule_string(y_schedule), max_frames)
            z_values = interpolate_schedule(parse_schedule_string(z_schedule), max_frames)
            
            # Detect movement segments for each axis
            x_segments = detect_movement_segments(x_values, 'translation_x', self.sensitivity)
            y_segments = detect_movement_segments(y_values, 'translation_y', self.sensitivity)
            z_segments = detect_movement_segments(z_values, 'translation_z', self.sensitivity)
            
            all_segments = x_segments + y_segments + z_segments
            
            if not all_segments:
                return "static camera position", 0.0
            
            # Group similar segments
            segment_groups = self._group_similar_segments(all_segments, max_frames)
            
            # Generate descriptions for each group
            group_descriptions = []
            total_intensity = 0.0
            
            for group in segment_groups:
                group_desc, group_intensity = self._generate_group_description(group, max_frames)
                if group_desc:
                    group_descriptions.append(group_desc)
                    total_intensity += group_intensity
            
            if not group_descriptions:
                return "minimal camera movement", 0.1
            
            # Combine descriptions
            if len(group_descriptions) == 1:
                description = group_descriptions[0]
            elif len(group_descriptions) == 2:
                description = f"{group_descriptions[0]} and {group_descriptions[1]}"
            else:
                description = f"{', '.join(group_descriptions[:-1])}, and {group_descriptions[-1]}"
            
            # Normalize intensity
            avg_intensity = total_intensity / len(segment_groups)
            
            return description, min(avg_intensity, 1.0)
            
        except Exception as e:
            return "camera movement", 0.5
    
    def _group_similar_segments(self, segments: List[Dict], max_frames: int) -> List[List[Dict]]:
        """Group similar movement segments together"""
        return group_similar_segments(segments, max_frames, similarity_threshold=0.3)
    
    def _generate_group_description(self, group: List[Dict], max_frames: int) -> Tuple[str, float]:
        """
        Generate description for a group of similar segments
        
        Args:
            group: List of similar movement segments
            max_frames: Total frames
            
        Returns:
            Tuple of (description, intensity)
        """
        if not group:
            return "", 0.0
        
        # Get the primary segment (largest total change)
        primary_segment = max(group, key=lambda s: abs(s['total_change']))
        
        # Calculate total duration covered by this group
        total_duration = sum(seg['end_frame'] - seg['start_frame'] for seg in group)
        
        # Generate base description
        base_description = self.generate_segment_description(primary_segment, max_frames)
        
        # Calculate intensity from all segments in group
        total_intensity = sum(calculate_movement_intensity(
            [seg['start_value'], seg['end_value']], seg['movement_type']
        ) for seg in group)
        avg_intensity = total_intensity / len(group)
        
        # Add multiplicity if there are multiple segments
        if len(group) > 1:
            if len(group) == 2:
                base_description = f"repeated {base_description}"
            else:
                base_description = f"multiple {base_description}"
        
        return base_description, avg_intensity
    
    def analyze_rotation(self, x_rot: str, y_rot: str, z_rot: str, max_frames: int) -> Tuple[str, float]:
        """
        Analyze rotation movement (pitch, yaw, roll)
        
        Args:
            x_rot: X rotation schedule string (pitch)
            y_rot: Y rotation schedule string (yaw)
            z_rot: Z rotation schedule string (roll)
            max_frames: Maximum number of frames
            
        Returns:
            Tuple of (description_text, intensity_score)
        """
        try:
            # Parse rotation schedules
            x_values = interpolate_schedule(parse_schedule_string(x_rot), max_frames)
            y_values = interpolate_schedule(parse_schedule_string(y_rot), max_frames)
            z_values = interpolate_schedule(parse_schedule_string(z_rot), max_frames)
            
            # Detect segments for each rotation axis
            x_segments = detect_movement_segments(x_values, 'rotation_3d_x', self.sensitivity)
            y_segments = detect_movement_segments(y_values, 'rotation_3d_y', self.sensitivity)
            z_segments = detect_movement_segments(z_values, 'rotation_3d_z', self.sensitivity)
            
            all_segments = x_segments + y_segments + z_segments
            
            if not all_segments:
                return "stable camera orientation", 0.0
            
            # Group similar segments
            segment_groups = self._group_similar_segments(all_segments, max_frames)
            
            # Generate descriptions
            group_descriptions = []
            total_intensity = 0.0
            
            for group in segment_groups:
                group_desc, group_intensity = self._generate_rotation_group_description(group, max_frames)
                if group_desc:
                    group_descriptions.append(group_desc)
                    total_intensity += group_intensity
            
            if not group_descriptions:
                return "minimal camera rotation", 0.1
            
            # Combine descriptions
            if len(group_descriptions) == 1:
                description = group_descriptions[0]
            else:
                description = " and ".join(group_descriptions)
            
            avg_intensity = total_intensity / len(segment_groups) if segment_groups else 0.0
            
            return description, min(avg_intensity, 1.0)
            
        except Exception:
            return "camera rotation", 0.5
    
    def _generate_rotation_group_description(self, group: List[Dict], max_frames: int) -> Tuple[str, float]:
        """Generate description for rotation segment group"""
        if not group:
            return "", 0.0
        
        # Get primary segment
        primary_segment = max(group, key=lambda s: abs(s['total_change']))
        
        # Generate base description  
        base_description = self.generate_segment_description(primary_segment, max_frames)
        
        # Calculate intensity
        total_intensity = sum(calculate_movement_intensity(
            [seg['start_value'], seg['end_value']], seg['movement_type']
        ) for seg in group)
        avg_intensity = total_intensity / len(group)
        
        return base_description, avg_intensity
    
    def analyze_zoom(self, zoom_schedule: str, max_frames: int) -> Tuple[str, float]:
        """
        Analyze zoom movement
        
        Args:
            zoom_schedule: Zoom schedule string
            max_frames: Maximum number of frames
            
        Returns:
            Tuple of (description_text, intensity_score)
        """
        try:
            zoom_values = interpolate_schedule(parse_schedule_string(zoom_schedule), max_frames)
            zoom_segments = detect_movement_segments(zoom_values, 'zoom', self.sensitivity)
            
            if not zoom_segments:
                return "steady zoom level", 0.0
            
            # Group similar zoom segments
            segment_groups = group_similar_segments(zoom_segments, max_frames)
            
            descriptions = []
            total_intensity = 0.0
            
            for group in segment_groups:
                primary_segment = max(group, key=lambda s: abs(s['total_change']))
                desc = self.generate_segment_description(primary_segment, max_frames)
                descriptions.append(desc)
                
                # Calculate intensity
                intensity = calculate_movement_intensity(
                    [seg['start_value'] for seg in group] + [seg['end_value'] for seg in group],
                    'zoom'
                )
                total_intensity += intensity
            
            if not descriptions:
                return "minimal zoom adjustment", 0.1
            
            # Combine descriptions
            description = " and ".join(descriptions) if len(descriptions) > 1 else descriptions[0]
            avg_intensity = total_intensity / len(segment_groups) if segment_groups else 0.0
            
            return description, min(avg_intensity, 1.0)
            
        except Exception:
            return "zoom movement", 0.5
    
    def analyze_rotation_pattern(self, x_schedule: str, y_schedule: str, z_schedule: str, max_frames: int) -> Tuple[str, float, str]:
        """
        Analyze complex rotation patterns (orbiting, circling, etc.)
        
        Args:
            x_schedule: X rotation schedule
            y_schedule: Y rotation schedule  
            z_schedule: Z rotation schedule
            max_frames: Maximum frames
            
        Returns:
            Tuple of (description, intensity, pattern_type)
        """
        try:
            # Get interpolated values
            x_values = interpolate_schedule(parse_schedule_string(x_schedule), max_frames)
            y_values = interpolate_schedule(parse_schedule_string(y_schedule), max_frames)
            z_values = interpolate_schedule(parse_schedule_string(z_schedule), max_frames)
            
            # Detect pattern type
            pattern_type = detect_pattern_type(x_values, y_values, z_values)
            
            # Calculate overall intensity
            total_intensity = (
                calculate_movement_intensity(x_values, 'rotation_3d_x') +
                calculate_movement_intensity(y_values, 'rotation_3d_y') +
                calculate_movement_intensity(z_values, 'rotation_3d_z')
            ) / 3.0
            
            # Generate pattern-specific description
            if pattern_type == 'circular':
                if detect_circular_motion(x_values, y_values):
                    description = "circular camera rotation around subject"
                else:
                    description = "orbital camera movement"
            elif pattern_type == 'oscillating':
                description = "oscillating camera rotation"
            elif pattern_type == 'linear':
                description = "smooth directional rotation"
            elif pattern_type == 'static':
                description = "static camera orientation"
                total_intensity = 0.0
            else:  # complex
                description = "complex rotation pattern"
            
            return description, min(total_intensity, 1.0), pattern_type
            
        except Exception:
            return "camera rotation", 0.5, "unknown"
    
    def analyze_camera_shake(self, shake_name: str, shake_intensity: float, shake_speed: float) -> Tuple[str, float]:
        """
        Analyze Camera Shakify shake patterns
        
        Args:
            shake_name: Name of the shake pattern
            shake_intensity: Shake intensity multiplier
            shake_speed: Shake speed multiplier
            
        Returns:
            Tuple of (description_text, intensity_score)
        """
        if not shake_name or shake_name.lower() in ['none', 'off', ''] or shake_intensity <= 0:
            return "", 0.0
        
        # Map shake names to descriptive text
        shake_descriptions = {
            'DRONE': 'aerial drone camera movement',
            'INVESTIGATION': 'investigative handheld camera movement',
            'WALKING': 'walking handheld camera movement',
            'RUNNING': 'running handheld camera movement',
            'DRIVING': 'vehicle-mounted camera movement',
            'EARTHQUAKE': 'earthquake-like camera shake',
            'HANDHELD': 'handheld camera movement',
            'SUBTLE': 'subtle natural camera movement',
            'JITTER': 'jittery camera movement',
            'VIBRATION': 'vibrating camera movement'
        }
        
        # Get base description
        shake_key = shake_name.upper()
        base_description = shake_descriptions.get(shake_key, f"{shake_name.lower()} camera movement")
        
        # Determine intensity modifier
        if shake_intensity < 0.3:
            intensity_mod = "subtle"
        elif shake_intensity < 0.7:
            intensity_mod = "moderate"
        elif shake_intensity < 1.5:
            intensity_mod = "strong"
        else:
            intensity_mod = "intense"
        
        # Determine speed modifier
        if shake_speed < 0.5:
            speed_mod = "slow"
        elif shake_speed < 1.5:
            speed_mod = ""  # Normal speed, no modifier
        else:
            speed_mod = "fast"
        
        # Combine modifiers
        modifiers = [mod for mod in [intensity_mod, speed_mod] if mod]
        if modifiers:
            modifier_text = " ".join(modifiers)
            description = f"{modifier_text} {base_description}"
        else:
            description = base_description
        
        # Calculate final intensity score
        intensity_score = min(shake_intensity * 0.5, 1.0)  # Scale shake intensity to 0-1 range
        
        return description, intensity_score


def analyze_deforum_movement(anim_args, sensitivity: float = 1.0, max_frames: int = 120, frame_start: int = 0) -> Tuple[str, float]:
    """
    Main function to analyze Deforum movement arguments and generate description
    
    Args:
        anim_args: Deforum animation arguments object
        sensitivity: Detection sensitivity (0.1 to 2.0)
        max_frames: Maximum frames to analyze
        frame_start: Starting frame for frame-specific analysis
        
    Returns:
        Tuple of (movement_description, overall_intensity)
    """
    try:
        analyzer = MovementAnalyzer(sensitivity)
        
        # Extract movement schedules
        translation_x = getattr(anim_args, 'translation_x', "0:(0)")
        translation_y = getattr(anim_args, 'translation_y', "0:(0)")
        translation_z = getattr(anim_args, 'translation_z', "0:(0)")
        
        rotation_3d_x = getattr(anim_args, 'rotation_3d_x', "0:(0)")
        rotation_3d_y = getattr(anim_args, 'rotation_3d_y', "0:(0)")
        rotation_3d_z = getattr(anim_args, 'rotation_3d_z', "0:(0)")
        
        zoom = getattr(anim_args, 'zoom', "0:(1.0)")
        
        # Camera Shakify analysis (frame-specific)
        camera_shake_name = getattr(anim_args, 'camera_shake_name', None)
        camera_shake_intensity = getattr(anim_args, 'camera_shake_intensity', 0.0)
        camera_shake_speed = getattr(anim_args, 'camera_shake_speed', 1.0)
        
        # Analyze each type of movement
        translation_desc, translation_intensity = analyzer.analyze_translation(
            translation_x, translation_y, translation_z, max_frames
        )
        
        rotation_desc, rotation_intensity = analyzer.analyze_rotation(
            rotation_3d_x, rotation_3d_y, rotation_3d_z, max_frames
        )
        
        zoom_desc, zoom_intensity = analyzer.analyze_zoom(zoom, max_frames)
        
        # Analyze Camera Shakify (frame-specific)
        shake_desc, shake_intensity = analyzer.analyze_camera_shake(
            camera_shake_name, camera_shake_intensity, camera_shake_speed
        )
        
        # Apply Camera Shakify to existing schedules if enabled
        combined_schedules = {}
        if shake_desc and camera_shake_intensity > 0:
            # Create frame-specific shake data starting from frame_start
            shake_data = create_shakify_data(
                camera_shake_name, camera_shake_intensity, camera_shake_speed,
                target_fps=30, max_frames=max_frames, frame_start=frame_start
            )
            
            if shake_data:
                # Apply shake to translation schedules
                if shake_data['translation']['x']:
                    combined_schedules['translation_x'] = apply_shakify_to_schedule(
                        translation_x, shake_data['translation']['x'], max_frames
                    )
                if shake_data['translation']['y']:
                    combined_schedules['translation_y'] = apply_shakify_to_schedule(
                        translation_y, shake_data['translation']['y'], max_frames
                    )
                if shake_data['translation']['z']:
                    combined_schedules['translation_z'] = apply_shakify_to_schedule(
                        translation_z, shake_data['translation']['z'], max_frames
                    )
                
                # Apply shake to rotation schedules
                if shake_data['rotation_3d']['x']:
                    combined_schedules['rotation_3d_x'] = apply_shakify_to_schedule(
                        rotation_3d_x, shake_data['rotation_3d']['x'], max_frames
                    )
                if shake_data['rotation_3d']['y']:
                    combined_schedules['rotation_3d_y'] = apply_shakify_to_schedule(
                        rotation_3d_y, shake_data['rotation_3d']['y'], max_frames
                    )
                if shake_data['rotation_3d']['z']:
                    combined_schedules['rotation_3d_z'] = apply_shakify_to_schedule(
                        rotation_3d_z, shake_data['rotation_3d']['z'], max_frames
                    )
        
        # Combine movement descriptions
        movement_parts = []
        intensity_scores = []
        
        if translation_desc and translation_desc not in ["static camera position", "minimal camera movement"]:
            movement_parts.append(translation_desc)
            intensity_scores.append(translation_intensity)
        
        if rotation_desc and rotation_desc not in ["stable camera orientation", "minimal camera rotation"]:
            movement_parts.append(rotation_desc)
            intensity_scores.append(rotation_intensity)
        
        if zoom_desc and zoom_desc not in ["steady zoom level", "minimal zoom adjustment"]:
            movement_parts.append(zoom_desc)
            intensity_scores.append(zoom_intensity)
        
        if shake_desc:
            movement_parts.append(shake_desc)
            intensity_scores.append(shake_intensity)
        
        # Generate final description
        if not movement_parts:
            return "static camera", 0.0
        
        if len(movement_parts) == 1:
            final_description = movement_parts[0]
        elif len(movement_parts) == 2:
            final_description = f"{movement_parts[0]} and {movement_parts[1]}"
        else:
            final_description = f"{', '.join(movement_parts[:-1])}, and {movement_parts[-1]}"
        
        # Calculate overall intensity
        overall_intensity = sum(intensity_scores) / len(intensity_scores) if intensity_scores else 0.0
        
        return f"camera movement with {final_description}", min(overall_intensity, 1.0)
        
    except Exception as e:
        # Fallback for any errors
        return "camera movement", 0.5 