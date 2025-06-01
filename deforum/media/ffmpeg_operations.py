import os
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

log = logging.getLogger(__name__)

class FFmpegProcessor:
    """FFmpeg-based video processing utilities for upscaling, interpolation, and audio replacement"""
    
    # Common resolution presets
    RESOLUTION_PRESETS = {
        "720p": (1280, 720),
        "1080p": (1920, 1080), 
        "1440p": (2560, 1440),
        "4K": (3840, 2160),
        "8K": (7680, 4320)
    }
    
    # Common FPS presets for interpolation
    FPS_PRESETS = [24, 30, 60, 120]
    
    def __init__(self):
        self.ffmpeg_path = self._find_ffmpeg()
        self.ffprobe_path = self._find_ffprobe()
        
    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable in system PATH or common locations"""
        # Try system PATH first
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return 'ffmpeg'
        except:
            pass
            
        # Try common installation paths
        common_paths = [
            'ffmpeg.exe',
            'C:/ffmpeg/bin/ffmpeg.exe', 
            'C:/Program Files/ffmpeg/bin/ffmpeg.exe',
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg'
        ]
        
        for path in common_paths:
            try:
                result = subprocess.run([path, '-version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return path
            except:
                continue
                
        raise FileNotFoundError("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
    
    def _find_ffprobe(self) -> str:
        """Find FFprobe executable"""
        ffmpeg_dir = os.path.dirname(self.ffmpeg_path) if os.path.dirname(self.ffmpeg_path) else ''
        
        # Try same directory as ffmpeg
        if ffmpeg_dir:
            ffprobe_path = os.path.join(ffmpeg_dir, 'ffprobe.exe' if os.name == 'nt' else 'ffprobe')
            if os.path.exists(ffprobe_path):
                return ffprobe_path
        
        # Try system PATH
        try:
            result = subprocess.run(['ffprobe', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return 'ffprobe'
        except:
            pass
            
        # Fallback to ffmpeg directory assumption
        return self.ffmpeg_path.replace('ffmpeg', 'ffprobe')
    
    def get_video_info(self, video_path: str) -> Dict:
        """Extract comprehensive video information using ffprobe"""
        if not os.path.exists(video_path):
            return {"error": f"File not found: {video_path}"}
            
        try:
            cmd = [
                self.ffprobe_path,
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return {"error": f"FFprobe failed: {result.stderr}"}
                
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            audio_streams = []
            
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                elif stream.get('codec_type') == 'audio':
                    audio_streams.append(stream)
            
            if not video_stream:
                return {"error": "No video stream found"}
            
            # Calculate additional info
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(data.get('format', {}).get('duration', 0))
            fps_str = video_stream.get('r_frame_rate', '0/1')
            fps = self._parse_fps(fps_str)
            frame_count = int(float(video_stream.get('nb_frames', 0))) if video_stream.get('nb_frames') else int(duration * fps)
            
            # Determine orientation
            orientation = "Landscape" if width > height else "Portrait" if height > width else "Square"
            
            # File info
            file_size = os.path.getsize(video_path)
            filename = os.path.basename(video_path)
            
            return {
                "filename": filename,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "duration_seconds": round(duration, 2),
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "orientation": orientation,
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "video_codec": video_stream.get('codec_name', 'unknown'),
                "video_bitrate": int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else None,
                "audio_streams": len(audio_streams),
                "audio_codec": audio_streams[0].get('codec_name', 'none') if audio_streams else 'none',
                "format": data.get('format', {}).get('format_name', 'unknown'),
                "pixel_format": video_stream.get('pix_fmt', 'unknown')
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze video: {str(e)}"}
    
    def _parse_fps(self, fps_str: str) -> float:
        """Parse FPS from ffprobe fraction format"""
        try:
            if '/' in fps_str:
                num, den = fps_str.split('/')
                return float(num) / float(den) if float(den) != 0 else 0
            return float(fps_str)
        except:
            return 0
    
    def upscale_video(self, input_path: str, output_path: str, target_resolution: str, 
                     quality: str = "medium", progress_callback=None) -> bool:
        """
        Upscale video using FFmpeg
        
        Args:
            input_path: Source video file
            output_path: Output video file  
            target_resolution: Target resolution (e.g., "1080p", "4K" or "1920x1080")
            quality: Quality preset ("fast", "medium", "slow", "veryslow")
            progress_callback: Optional callback for progress updates
        """
        try:
            # Parse target resolution
            if target_resolution in self.RESOLUTION_PRESETS:
                width, height = self.RESOLUTION_PRESETS[target_resolution]
            elif 'x' in target_resolution:
                width, height = map(int, target_resolution.split('x'))
            else:
                raise ValueError(f"Invalid resolution format: {target_resolution}")
            
            # Quality to CRF mapping
            quality_map = {
                "fast": "28",
                "medium": "23", 
                "slow": "18",
                "veryslow": "15"
            }
            crf = quality_map.get(quality, "23")
            
            cmd = [
                self.ffmpeg_path,
                '-i', input_path,
                '-vf', f'scale={width}:{height}:flags=lanczos',
                '-c:v', 'libx264',
                '-crf', crf,
                '-preset', quality,
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-y',  # Overwrite output
                output_path
            ]
            
            return self._run_ffmpeg_with_progress(cmd, progress_callback, "Upscaling video")
            
        except Exception as e:
            log.error(f"Video upscaling failed: {e}")
            return False
    
    def interpolate_frames(self, input_path: str, output_path: str, target_fps: int,
                          method: str = "minterpolate", progress_callback=None) -> bool:
        """
        Interpolate video frames to increase FPS
        
        Args:
            input_path: Source video file
            output_path: Output video file
            target_fps: Target FPS (e.g., 60)
            method: Interpolation method ("minterpolate", "fps")
            progress_callback: Optional callback for progress updates
        """
        try:
            if method == "minterpolate":
                # More sophisticated motion interpolation
                filter_complex = f'minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'
            else:
                # Simple FPS filter (faster but lower quality)
                filter_complex = f'fps={target_fps}'
            
            cmd = [
                self.ffmpeg_path,
                '-i', input_path,
                '-vf', filter_complex,
                '-c:v', 'libx264',
                '-crf', '18',  # High quality for interpolation
                '-preset', 'medium',
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            
            return self._run_ffmpeg_with_progress(cmd, progress_callback, "Interpolating frames")
            
        except Exception as e:
            log.error(f"Frame interpolation failed: {e}")
            return False
    
    def replace_audio(self, video_path: str, audio_path: str, output_path: str,
                     audio_start_time: float = 0, progress_callback=None) -> bool:
        """
        Replace video audio track with new audio
        
        Args:
            video_path: Source video file
            audio_path: Audio file to use
            output_path: Output video file
            audio_start_time: Start time offset for audio in seconds
            progress_callback: Optional callback for progress updates
        """
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',  # Copy video without re-encoding
                '-c:a', 'aac',   # Re-encode audio to AAC
                '-map', '0:v:0', # Use video from first input
                '-map', '1:a:0', # Use audio from second input
                '-shortest',     # Finish when shortest stream ends
                '-y',
                output_path
            ]
            
            if audio_start_time > 0:
                cmd.insert(-2, '-ss')
                cmd.insert(-2, str(audio_start_time))
            
            return self._run_ffmpeg_with_progress(cmd, progress_callback, "Replacing audio")
            
        except Exception as e:
            log.error(f"Audio replacement failed: {e}")
            return False
    
    def _run_ffmpeg_with_progress(self, cmd: List[str], progress_callback, operation: str) -> bool:
        """Run FFmpeg command with progress monitoring"""
        try:
            if progress_callback:
                progress_callback(f"Starting {operation}...")
            
            # Add progress reporting for supported operations
            if '-progress' not in cmd:
                cmd.insert(-2, '-progress')
                cmd.insert(-2, 'pipe:1')
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor progress
            last_progress = ""
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output and progress_callback:
                    # Parse FFmpeg progress output
                    if "time=" in output:
                        time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', output)
                        if time_match:
                            current_progress = f"{operation} - {time_match.group(1)}"
                            if current_progress != last_progress:
                                progress_callback(current_progress)
                                last_progress = current_progress
            
            rc = process.poll()
            if rc == 0:
                if progress_callback:
                    progress_callback(f"{operation} completed successfully!")
                return True
            else:
                error_output = process.stderr.read()
                log.error(f"FFmpeg error: {error_output}")
                if progress_callback:
                    progress_callback(f"{operation} failed: {error_output[:200]}...")
                return False
                
        except Exception as e:
            log.error(f"FFmpeg execution failed: {e}")
            if progress_callback:
                progress_callback(f"{operation} failed: {str(e)}")
            return False
    
    def get_optimal_settings(self, video_info: Dict) -> Dict:
        """Get optimal processing settings based on video characteristics"""
        width = video_info.get('width', 0)
        height = video_info.get('height', 0)
        fps = video_info.get('fps', 0)
        
        recommendations = {
            "upscale_options": [],
            "interpolation_options": [],
            "quality_recommendation": "medium"
        }
        
        # Upscale recommendations
        current_pixels = width * height
        for preset, (w, h) in self.RESOLUTION_PRESETS.items():
            if w * h > current_pixels * 1.5:  # Only suggest meaningful upscales
                recommendations["upscale_options"].append(preset)
        
        # Interpolation recommendations  
        for target_fps in self.FPS_PRESETS:
            if target_fps > fps * 1.5:  # Only suggest meaningful interpolation
                recommendations["interpolation_options"].append(target_fps)
        
        # Quality recommendation based on resolution
        if current_pixels > 1920 * 1080:  # Above 1080p
            recommendations["quality_recommendation"] = "medium"
        elif current_pixels > 1280 * 720:  # Above 720p
            recommendations["quality_recommendation"] = "slow"
        else:
            recommendations["quality_recommendation"] = "veryslow"
        
        return recommendations 