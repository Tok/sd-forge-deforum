#!/usr/bin/env python3
"""
WAN Direct Integration - Use Official WAN Repository Directly
Instead of trying to reinvent the wheel, let's use the official WAN repo
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import tempfile
import shutil
from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()


class WanDirectIntegration:
    """Direct integration using official WAN repository"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.extension_root = Path(__file__).parent.parent.parent.parent  # Go up to extension root
        self.wan_repo_path = self.extension_root / "Wan2.1"
        self.discovered_models = []
        
    def setup_wan_repository(self) -> bool:
        """Ensure WAN repository is available and up to date"""
        if not self.wan_repo_path.exists():
            logger.info("📥 Cloning official WAN repository...")
            return self._clone_wan_repository()
        else:
            logger.info(f"{emoji_if_enabled("✅")} WAN repository already available")
            return True
    
    def _clone_wan_repository(self) -> bool:
        """Clone the official WAN repository"""
        try:
            logger.info("🌐 Cloning from https://github.com/Wan-Video/Wan2.1.git...")
            result = subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/Wan-Video/Wan2.1.git",
                str(self.wan_repo_path)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"{emoji_if_enabled("✅")} WAN repository cloned successfully")
                return True
            else:
                logger.error(f"Failed to clone: {result.stderr}", emoji='off')
                return False
                
        except Exception as e:
            logger.error(f"Error cloning repository: {e}", emoji='off')
            return False
    
    def discover_models(self) -> List[Dict]:
        """Discover WAN models using our discovery system"""
        try:
            from .wan_model_discovery import WanModelDiscovery
        except ImportError:
            # Handle running as standalone script
            sys.path.append(str(Path(__file__).parent))
            from wan_model_discovery import WanModelDiscovery
        
        discovery = WanModelDiscovery()
        self.discovered_models = discovery.discover_models()
        return self.discovered_models
    
    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model"""
        if not self.discovered_models:
            self.discover_models()
        return self.discovered_models[0] if self.discovered_models else None
    
    def generate_video_direct(self, 
                            prompt: str,
                            model_info: Dict,
                            output_dir: str,
                            width: int = 1280,
                            height: int = 720,
                            num_frames: int = 81,
                            steps: int = 20,
                            guidance_scale: float = 7.5,
                            seed: int = -1,
                            **kwargs) -> Optional[str]:
        """Generate video using official WAN repository directly"""
        
        if not self.setup_wan_repository():
            raise RuntimeError("Failed to setup WAN repository")
        
        logger.info(f"Generating video using official WAN repository...", emoji='movie_camera')
        logger.info(f"   📝 Prompt: {prompt}")
        logger.info(f"   📐 Size: {width}x{height}")
        logger.info(f"   {emoji_if_enabled("🎬")} Frames: {num_frames}", emoji='movie_camera')
        logger.info(f"   📁 Model: {model_info['name']} ({model_info['type']}, {model_info['size']})")
        
        # Determine the correct task based on model type and size
        task = self._determine_wan_task(model_info)
        
        # Prepare arguments for official WAN script
        wan_args = self._prepare_wan_arguments(
            task=task,
            prompt=prompt,
            model_path=model_info['path'],
            output_dir=output_dir,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        # Run official WAN generation
        try:
            output_file = self._run_wan_generation(wan_args)
            if output_file and Path(output_file).exists():
                logger.info(f"{emoji_if_enabled("✅")} Video generated successfully: {output_file}")
                return output_file
            else:
                logger.error("Video generation failed - no output file created", emoji='off')
                return None
                
        except Exception as e:
            logger.error(f"Video generation failed: {e}", emoji='off')
            return None
    
    def _determine_wan_task(self, model_info: Dict) -> str:
        """Determine the correct WAN task based on model info"""
        model_type = model_info['type'].lower()
        model_size = model_info['size'].lower()
        
        if 'vace' in model_type:
            if '1.3b' in model_size:
                return 'vace-1.3B'
            else:
                return 'vace-14B'
        elif 'i2v' in model_type:
            return 'i2v-14B'
        elif 't2v' in model_type:
            if '1.3b' in model_size:
                return 't2v-1.3B'
            else:
                return 't2v-14B'
        else:
            # Default to T2V based on size
            if '1.3b' in model_size:
                return 't2v-1.3B'
            else:
                return 't2v-14B'
    
    def _prepare_wan_arguments(self, 
                             task: str,
                             prompt: str,
                             model_path: str,
                             output_dir: str,
                             width: int,
                             height: int,
                             num_frames: int,
                             steps: int,
                             guidance_scale: float,
                             seed: int) -> List[str]:
        """Prepare arguments for official WAN generate.py script"""
        
        # Ensure num_frames follows WAN's 4n+1 rule
        if (num_frames - 1) % 4 != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
            logger.info(f"   {emoji_if_enabled("✅")} Adjusted frames to WAN requirement: {num_frames} (4n+1 rule)")
        
        args = [
            sys.executable,  # Python interpreter
            "generate.py",   # Official WAN script
            "--task", task,
            "--size", f"{width}*{height}",
            "--ckpt_dir", str(model_path),
            "--prompt", prompt,
            "--frame_num", str(num_frames),
            "--sampling_steps", str(steps),
            "--guide_scale", str(guidance_scale),
        ]
        
        # Add seed if specified
        if seed > 0:
            args.extend(["--base_seed", str(seed)])
        
        # Add output directory
        args.extend(["--output_dir", str(output_dir)])
        
        # Add memory optimization for consumer GPUs
        if 'cuda' in str(os.environ.get('CUDA_VISIBLE_DEVICES', '0')):
            args.extend(["--offload_model", "True", "--t5_cpu"])
        
        return args
    
    def _run_wan_generation(self, args: List[str]) -> Optional[str]:
        """Run the official WAN generation script"""
        
        # Change to WAN repository directory
        original_cwd = os.getcwd()
        
        try:
            os.chdir(self.wan_repo_path)
            logger.info(f"🚀 Running official WAN generation in {self.wan_repo_path}")
            logger.info(f"   {emoji_if_enabled("🔧")} Command: {' '.join(args)}", emoji='wrench')
            
            # Add WAN repo to Python path temporarily
            env = os.environ.copy()
            if str(self.wan_repo_path) not in env.get('PYTHONPATH', ''):
                env['PYTHONPATH'] = f"{self.wan_repo_path}:{env.get('PYTHONPATH', '')}"
            
            # Run the generation with a reasonable timeout
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                env=env
            )
            
            if result.returncode == 0:
                logger.info(f"{emoji_if_enabled("✅")} Official WAN generation completed successfully")
                
                # Parse output to find generated video file
                output_file = self._parse_wan_output(result.stdout, args)
                return output_file
            else:
                logger.error(f"WAN generation failed:", emoji='off')
                logger.info(f"   stdout: {result.stdout}")
                logger.info(f"   stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.info("WAN generation timed out after 10 minutes", emoji='off')
            return None
        except Exception as e:
            logger.error(f"Error running WAN generation: {e}", emoji='off')
            return None
        finally:
            os.chdir(original_cwd)
    
    def _parse_wan_output(self, stdout: str, args: List[str]) -> Optional[str]:
        """Parse WAN output to find the generated video file"""
        
        # Look for output directory in args
        output_dir = None
        for i, arg in enumerate(args):
            if arg == "--output_dir" and i + 1 < len(args):
                output_dir = Path(args[i + 1])
                break
        
        if not output_dir:
            # Default WAN output location
            output_dir = self.wan_repo_path / "results"
        
        # Look for the most recent video file
        if output_dir.exists():
            video_files = list(output_dir.glob("**/*.mp4"))
            if video_files:
                # Return the most recently created file
                latest_file = max(video_files, key=lambda p: p.stat().st_mtime)
                return str(latest_file)
        
        # Also check stdout for file mentions
        lines = stdout.split('\n')
        for line in lines:
            if '.mp4' in line and ('saved' in line.lower() or 'output' in line.lower()):
                # Try to extract file path from line
                import re
                matches = re.findall(r'[^\s]+\.mp4', line)
                if matches:
                    potential_file = Path(matches[0])
                    if potential_file.exists():
                        return str(potential_file)
        
        return None
    
    def install_wan_dependencies(self) -> bool:
        """Install WAN dependencies if needed"""
        if not self.setup_wan_repository():
            return False
        
        requirements_file = self.wan_repo_path / "requirements.txt"
        if not requirements_file.exists():
            logger.warning("⚠️ No requirements.txt found in WAN repository")
            return True
        
        try:
            logger.info("📦 Installing WAN dependencies...")
            
            # Install essential dependencies first (avoid conflicts)
            essential_deps = [
                "dashscope",
                "easydict", 
                "imageio",
                "imageio-ffmpeg",
                "ftfy"
            ]
            
            for dep in essential_deps:
                try:
                    logger.info(f"   📦 Installing {dep}...")
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", dep, "--quiet"
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        logger.info(f"   {emoji_if_enabled("✅")} {dep} installed")
                    else:
                        logger.warning(f"   ⚠️ {dep} installation warning: {result.stderr[:100]}")
                        
                except Exception as e:
                    logger.error(f"   ⚠️ {dep} installation error: {e}")
                    continue  # Continue with other dependencies
            
            logger.info(f"{emoji_if_enabled("✅")} WAN dependencies installation completed")
            return True
                
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}", emoji='off')
            return False

def generate_video_with_official_wan(prompt: str, 
                                   output_dir: str,
                                   width: int = 1280,
                                   height: int = 720,
                                   num_frames: int = 81,
                                   steps: int = 20,
                                   guidance_scale: float = 7.5,
                                   seed: int = -1,
                                   **kwargs) -> Optional[str]:
    """Convenience function to generate video using official WAN"""
    
    integration = WanDirectIntegration()
    
    # Find the best model
    best_model = integration.get_best_model()
    if not best_model:
        logger.info("No WAN models found", emoji='off')
        logger.info("Please download a WAN model first:", emoji='bulb')
        logger.info("   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./models/Deforum/wan")
        return None
    
    # Install dependencies if needed
    if not integration.install_wan_dependencies():
        logger.error("⚠️ Failed to install WAN dependencies, continuing anyway...")
    
    # Generate video
    return integration.generate_video_direct(
        prompt=prompt,
        model_info=best_model,
        output_dir=output_dir,
        width=width,
        height=height,
        num_frames=num_frames,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        **kwargs
    )

if __name__ == "__main__":
    # Test the direct integration
    logger.info("🧪 Testing WAN Direct Integration...")
    
    integration = WanDirectIntegration()
    
    # Test model discovery
    models = integration.discover_models()
    if models:
        logger.info(f"{emoji_if_enabled("✅")} Found {len(models)} model(s)")
        best = integration.get_best_model()
        logger.info(f"🏆 Best model: {best['name']} ({best['type']}, {best['size']})")
        
        # Test repository setup
        if integration.setup_wan_repository():
            logger.info(f"{emoji_if_enabled("✅")} WAN repository ready")
        else:
            logger.error("Failed to setup WAN repository", emoji='off')
    else:
        print("❌ No models found") 