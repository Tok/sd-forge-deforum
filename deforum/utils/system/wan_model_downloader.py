"""
Wan Model Downloader
Automatically downloads Wan 2.1 FLF2V model from HuggingFace when needed
"""

import os
import subprocess
from pathlib import Path
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


class WanModelDownloader:
    """Handles automatic downloading of Wan models"""

    def __init__(self):
        # Detect models directory
        self.models_dir = self._detect_models_directory()

        # Wan 2.1 FLF2V model (required for FLF2V interpolation workflow)
        self.flf2v_model = {
            "name": "Wan2.1-FLF2V-14B",
            "repo_id": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
            "local_dir": str(self.models_dir / "Deforum" / "wan" / "Wan2.1-FLF2V-14B"),
            "description": "Wan 2.1 FLF2V-14B (first-last-frame video interpolation)",
            "size_gb": 14,
            "gated": False,
            "required_for": "Flux + Interpolation mode with Wan FLF2V"
        }

    def _detect_models_directory(self) -> Path:
        """Detect the Forge models directory"""
        # Try to find webui root by going up from extension directory
        extension_root = Path(__file__).parent.parent.parent.parent

        # Option 1: webui/models (standard Forge installation)
        webui_models = extension_root.parent.parent / "models"
        if webui_models.exists():
            return webui_models

        # Option 2: Current working directory models
        local_models = Path("models")
        if local_models.exists():
            return local_models

        # Option 3: Create in current directory (fallback)
        local_models.mkdir(exist_ok=True)
        return local_models

    def is_flf2v_installed(self) -> bool:
        """Check if Wan 2.1 FLF2V model is already installed"""
        model_dir = Path(self.flf2v_model["local_dir"])

        # Check for key files that indicate model is present
        required_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
        ]

        # Check basic files
        if not all((model_dir / file).exists() for file in required_files):
            return False

        # Check for transformer model (either single file or sharded)
        transformer_dir = model_dir / "transformer"
        has_single_file = (transformer_dir / "diffusion_pytorch_model.safetensors").exists()
        has_sharded_index = (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").exists()

        return has_single_file or has_sharded_index

    def _download_with_python_api(self, model: dict) -> bool:
        """Try downloading using huggingface_hub Python API"""
        try:
            from huggingface_hub import snapshot_download

            local_dir = Path(model["local_dir"])
            local_dir.mkdir(parents=True, exist_ok=True)

            # Note: resume_download is deprecated in huggingface_hub 1.0.0+
            # Downloads always resume automatically when possible, so we don't need to specify it
            snapshot_download(
                repo_id=model["repo_id"],
                local_dir=str(local_dir),
                local_dir_use_symlinks=False
            )
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"Python API download failed: {e}")
            return False

    def _download_with_cli(self, model: dict) -> bool:
        """Try downloading using huggingface-cli subprocess"""
        try:
            # Check if CLI is available
            result = subprocess.run(
                ["huggingface-cli", "--version"],
                capture_output=True,
                timeout=10
            )
            if result.returncode != 0:
                return False

            local_dir = Path(model["local_dir"])
            local_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "huggingface-cli", "download",
                model["repo_id"],
                "--local-dir", str(local_dir),
                "--local-dir-use-symlinks", "False"
            ]

            result = subprocess.run(
                cmd,
                capture_output=False,  # Show download progress
                text=True,
                timeout=7200  # 2 hour timeout
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        except Exception as e:
            logger.warning(f"CLI download failed: {e}")
            return False

    def download_flf2v(self) -> bool:
        """Download Wan 2.1 FLF2V model (tries Python API first, then CLI fallback)

        Returns:
            True if downloaded successfully, False otherwise
        """
        # Check if already installed
        if self.is_flf2v_installed():
            logger.info(f"✓ {self.flf2v_model['name']} already installed")
            return True

        model = self.flf2v_model

        logger.info(f"Downloading {model['description']} (~{model['size_gb']}GB)...")
        logger.info(f"  From: {model['repo_id']}")
        logger.info(f"  To: {model['local_dir']}")
        logger.info(f"  Required for: {model['required_for']}")
        logger.info("")
        logger.info("This will take some time depending on your connection speed...")

        # Try Python API first (works on Forge Neo)
        logger.info("Attempting download with huggingface_hub Python API...")
        if self._download_with_python_api(model):
            logger.info("")
            logger.info(f"✓ Downloaded {model['name']} successfully!")
            logger.info("You can now use Flux + Interpolation mode with Wan FLF2V")
            return True

        # Fall back to CLI (works on classic Forge)
        logger.info("Python API failed, trying huggingface-cli...")
        if self._download_with_cli(model):
            logger.info("")
            logger.info(f"✓ Downloaded {model['name']} successfully!")
            logger.info("You can now use Flux + Interpolation mode with Wan FLF2V")
            return True

        # Both methods failed
        logger.error(f"Failed to download {model['name']}")
        logger.error("Neither huggingface_hub Python API nor huggingface-cli worked")
        logger.error("Please install manually or check your huggingface_hub installation")
        return False


def auto_download_wan_flf2v_if_needed() -> bool:
    """
    Auto-download Wan 2.1 FLF2V model if not present.

    Returns:
        True if Wan FLF2V is available (already present or just downloaded), False otherwise
    """
    try:
        downloader = WanModelDownloader()

        # Check if already installed
        if downloader.is_flf2v_installed():
            logger.info("Wan 2.1 FLF2V model already available")
            return True

        logger.info("Wan 2.1 FLF2V model not detected - starting auto-download...")
        logger.info("Note: This model is required for Flux + Interpolation mode with Wan FLF2V")
        logger.info("")

        result = downloader.download_flf2v()

        # If download failed, just log and continue (don't crash extension load)
        if not result:
            logger.info("Wan FLF2V auto-download skipped - you can manually download later")
            logger.info("Extension will load normally, Wan features available when model is installed")

        return result
    except Exception as e:
        # Catch any unexpected errors during download attempt
        logger.warning(f"Wan FLF2V auto-download error: {e}")
        logger.info("Extension will load normally, Wan features available when model is installed")
        return False
