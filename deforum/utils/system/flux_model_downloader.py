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

        # Model options - try quantized first (lower VRAM), fall back to full model
        # Note: Place in Stable-diffusion/Flux/ subdirectory to keep organized
        flux_model_dir = str(self.models_dir / "Stable-diffusion" / "Flux")

        self.model_options = [
            {
                "name": "quantized",
                "repo_id": "lllyasviel/flux1-dev-bnb-nf4",
                "filename": "flux1-dev-bnb-nf4-v2.safetensors",
                "local_dir": flux_model_dir,
                "description": "Flux.1 Dev NF4 (quantized, ~5GB, works on 12GB VRAM)",
                "size_gb": 5,
                "gated": False,  # lllyasviel's repo is ungated
                "recommended": True
            },
            {
                "name": "full",
                "repo_id": "black-forest-labs/FLUX.1-dev",
                "filename": "flux1-dev.safetensors",
                "local_dir": flux_model_dir,
                "description": "Flux.1 Dev (official, ~24GB, requires 24GB+ VRAM & HF login)",
                "size_gb": 24,
                "gated": True,
                "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
                "recommended": False
            }
        ]

        # Text encoders go in text_encoder/, VAE goes in VAE/
        self.text_encoder_files = [
            {
                "name": "clip_l",
                "repo_id": "comfyanonymous/flux_text_encoders",
                "filename": "clip_l.safetensors",
                "local_dir": str(self.models_dir / "text_encoder"),
                "description": "CLIP-L text encoder",
                "size_gb": 0.25,
                "gated": False,
                "recommended": True
            },
            {
                "name": "t5xxl_fp16",
                "repo_id": "comfyanonymous/flux_text_encoders",
                "filename": "t5xxl_fp16.safetensors",
                "local_dir": str(self.models_dir / "text_encoder"),
                "description": "T5-XXL FP16 text encoder",
                "size_gb": 9.8,
                "gated": False,
                "recommended": True
            },
            {
                "name": "ae_vae",
                "repo_id": "black-forest-labs/FLUX.1-dev",
                "filename": "ae.safetensors",
                "local_dir": str(self.models_dir / "VAE"),
                "description": "Flux.1 VAE (autoencoder)",
                "size_gb": 0.3,
                "gated": True,
                "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
                "recommended": True
            }
        ]

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
            is_gated = model_info.get("gated", False)

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

            if is_gated:
                logger.info("  ⚠️  This model requires HuggingFace authentication")

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
                capture_output=True,  # Capture to check for gating errors
                text=True,
                timeout=3600  # 1 hour timeout for large files
            )

            if result.returncode == 0:
                logger.info(f"✓ Downloaded {filename} successfully")
                return True
            else:
                # Check for gating error
                if "GatedRepoError" in result.stderr or "Access to model" in result.stderr or "401" in result.stderr:
                    logger.error(f"❌ Access denied: {repo_id} is a gated model")
                    logger.error("")
                    logger.error("To download this model, you need to:")
                    logger.error(f"  1. Visit: {model_info.get('license_url', f'https://huggingface.co/{repo_id}')}")
                    logger.error("  2. Click 'Agree and access repository' to accept the license")
                    logger.error("  3. Login to HuggingFace CLI:")
                    logger.error("     huggingface-cli login")
                    logger.error("  4. Restart Forge to retry the download")
                    logger.error("")
                    return False
                else:
                    logger.error(f"Failed to download {filename}: {result.stderr}")
                    return False

        except subprocess.TimeoutExpired:
            logger.error(f"Download timed out for {filename}")
            return False
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return False

    def download_flux_and_vae(self) -> bool:
        """
        Download recommended Flux model, text encoders, and VAE.
        Tries quantized version first (lower VRAM), falls back to full model if needed.

        Returns:
            True if all files downloaded successfully, False otherwise
        """
        # Check/install huggingface-cli
        if not self.check_huggingface_cli():
            logger.info("huggingface-cli not found, installing...")
            if not self.install_huggingface_hub():
                logger.error("Failed to install huggingface_hub")
                return False

        logger.info("Starting Flux model, text encoders, and VAE download...")
        logger.info(f"Models will be saved to: {self.models_dir}")
        logger.info("")
        logger.info("This will download ~12GB across 4 files:")
        logger.info("  - flux1-dev-bnb-nf4-v2.safetensors (~5GB)")
        logger.info("  - clip_l.safetensors (~250MB)")
        logger.info("  - t5xxl_fp16.safetensors (~9.8GB)")
        logger.info("  - ae.safetensors (~300MB)")
        logger.info("")

        # Try downloading quantized model first (recommended for most users)
        success_model = False
        for model_option in self.model_options:
            if model_option.get("recommended", False):
                logger.info(f"[1/4] Downloading {model_option['description']}")
                success_model = self.download_model(model_option)
                if success_model:
                    break

        # If quantized failed, show instructions for full model
        if not success_model:
            logger.error("")
            logger.error("Failed to download recommended quantized model.")
            logger.error("Alternative: Download full Flux.1 Dev model (requires 24GB+ VRAM)")
            for model_option in self.model_options:
                if not model_option.get("recommended", False):
                    logger.error(f"  - {model_option['description']}")
                    if model_option.get("gated"):
                        logger.error(f"    Visit: {model_option.get('license_url', 'N/A')}")
            return False

        # Download all text encoders and VAE (all required)
        logger.info("")
        logger.info("Downloading text encoders and VAE...")
        text_encoder_success = []
        for idx, encoder_file in enumerate(self.text_encoder_files, start=2):
            logger.info(f"[{idx}/4] Downloading {encoder_file['description']}")
            success = self.download_model(encoder_file)
            text_encoder_success.append(success)

            # For gated files, provide instructions but continue
            if not success and encoder_file.get("gated"):
                logger.warning(f"Skipping {encoder_file['filename']} (requires HF authentication)")
                logger.warning("You can download it manually later if needed")

        # Check if we have at least the critical files
        all_success = success_model and all(text_encoder_success)

        if all_success:
            logger.info("")
            logger.info("✓ Flux model and all components downloaded successfully!")
            logger.info("Please restart Forge and select the Flux model from the checkpoint dropdown")
            return True
        else:
            # Check which files failed
            failed_files = []
            if not success_model:
                failed_files.append("flux1-dev-bnb-nf4-v2.safetensors")
            for idx, success in enumerate(text_encoder_success):
                if not success:
                    failed_files.append(self.text_encoder_files[idx]["filename"])

            logger.error("")
            logger.error(f"Failed to download some files: {', '.join(failed_files)}")
            logger.error("Flux may not work properly without all required files")
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
