"""
Flux Model Downloader
Automatically downloads Flux models and VAEs from HuggingFace when needed
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


class FluxModelDownloader:
    """Handles automatic downloading of Flux models"""

    def __init__(self):
        # Detect Forge models directory
        self.models_dir = self._detect_models_directory()

        # Use community mirrors with ungated access (no authentication required)
        # Kijai's mirrors are widely used and trusted in the community
        self.recommended_model = {
            "repo_id": "Kijai/flux-fp8",
            "filename": "flux1-dev-fp8.safetensors",
            "local_dir": str(self.models_dir / "Stable-diffusion"),
            "description": "Flux.1 Dev FP8 (quantized, ~17GB, community mirror)",
            "size_gb": 17,
        }
        self.recommended_vae = {
            "repo_id": "Kijai/flux-fp8",
            "filename": "ae.safetensors",
            "local_dir": str(self.models_dir / "VAE"),
            "description": "Flux.1 VAE (community mirror, ungated)",
            "size_gb": 0.3,
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

    def check_huggingface_cli(self) -> bool:
        """Check if huggingface-cli is available"""
        try:
            result = subprocess.run(
                ["huggingface-cli", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def install_huggingface_hub(self) -> bool:
        """Install huggingface_hub if not available"""
        try:
            logger.info("Installing huggingface_hub...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "huggingface_hub"],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                logger.info("✓ huggingface_hub installed successfully")
                return True
            else:
                logger.error(f"Failed to install huggingface_hub: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error installing huggingface_hub: {e}")
            return False

    def download_model(self, model_info: dict) -> bool:
        """Download a single model file using huggingface-cli"""
        try:
            repo_id = model_info["repo_id"]
            filename = model_info["filename"]
            local_dir = Path(model_info["local_dir"])

            # Create target directory
            local_dir.mkdir(parents=True, exist_ok=True)

            # Check if already downloaded
            target_file = local_dir / filename
            if target_file.exists():
                logger.info(f"✓ {filename} already exists, skipping download")
                return True

            logger.info(f"Downloading {model_info['description']} (~{model_info['size_gb']}GB)...")
            logger.info(f"  From: {repo_id}/{filename}")
            logger.info(f"  To: {local_dir}")

            # Use huggingface-cli download
            cmd = [
                "huggingface-cli", "download",
                repo_id,
                filename,
                "--local-dir", str(local_dir),
                "--local-dir-use-symlinks", "False"
            ]

            result = subprocess.run(
                cmd,
                capture_output=False,  # Show progress
                text=True,
                timeout=3600  # 1 hour timeout for large files
            )

            if result.returncode == 0:
                logger.info(f"✓ Downloaded {filename} successfully")
                return True
            else:
                logger.error(f"Failed to download {filename}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Download timed out for {filename}")
            return False
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return False

    def download_flux_and_vae(self) -> bool:
        """
        Download recommended Flux model and VAE.

        Returns:
            True if both downloaded successfully, False otherwise
        """
        # Check/install huggingface-cli
        if not self.check_huggingface_cli():
            logger.info("huggingface-cli not found, installing...")
            if not self.install_huggingface_hub():
                logger.error("Failed to install huggingface_hub")
                return False

        logger.info("Starting Flux model and VAE download...")
        logger.info(f"Models will be saved to: {self.models_dir}")

        # Download Flux model
        success_model = self.download_model(self.recommended_model)

        # Download VAE
        success_vae = self.download_model(self.recommended_vae)

        if success_model and success_vae:
            logger.info("✓ Flux model and VAE downloaded successfully!")
            logger.info(f"  Model: {self.models_dir / 'Stable-diffusion' / self.recommended_model['filename']}")
            logger.info(f"  VAE: {self.models_dir / 'VAE' / self.recommended_vae['filename']}")
            logger.info("Please restart Forge and select the Flux model from the checkpoint dropdown")
            return True
        else:
            logger.error("Failed to download Flux model and/or VAE")
            return False


def auto_download_flux_if_needed() -> bool:
    """
    Auto-download Flux model and VAE if not present.

    Returns:
        True if Flux is available (already present or just downloaded), False otherwise
    """
    from deforum.utils.system.flux_check import is_flux_available

    # Check if Flux is already available
    if is_flux_available():
        logger.info("Flux model already available")
        return True

    logger.info("Flux model not detected, starting auto-download...")

    downloader = FluxModelDownloader()
    return downloader.download_flux_and_vae()
