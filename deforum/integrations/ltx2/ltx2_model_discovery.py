"""
LTX-2 Model Discovery and Management

Handles auto-discovery of LTX-2 models from the Deforum models directory
and provides download functionality via HuggingFace.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from deforum.utils.system.logging import get_logger

logger = get_logger()


class LTX2ModelDiscovery:
    """Discovers and manages LTX-2 models"""

    # LTX-2 model variants
    MODEL_VARIANTS = {
        # GGUF variants (recommended - better quantization, lower VRAM)
        "LTX-2-Q4_K_M-GGUF": {
            "huggingface_id": "unsloth/LTX-2-GGUF",
            "gguf_filename": "ltx-2-19b-dev-Q4_K_M.gguf",
            "description": "4K variant, Q4_K_M quantized (19B params, GGUF)",
            "vram_gb": 13,  # 12.8GB file size
            "max_resolution": "4K (3840x2160)",
            "max_fps": 50,
            "quantization": "gguf-q4_k_m",
            "recommended": True,
            "note": "Recommended for 14GB+ VRAM (RTX 3080 Ti, 4070 Ti, 4080, etc.) - best quality/VRAM balance"
        },
        "LTX-2-Q3_K_M-GGUF": {
            "huggingface_id": "unsloth/LTX-2-GGUF",
            "gguf_filename": "ltx-2-19b-dev-Q3_K_M.gguf",
            "description": "4K variant, Q3_K_M quantized (19B params, GGUF)",
            "vram_gb": 10,  # 10.1GB file size
            "max_resolution": "4K (3840x2160)",
            "max_fps": 50,
            "quantization": "gguf-q3_k_m",
            "recommended": False,
            "note": "For 11-13GB VRAM - slightly lower quality but fits in less VRAM"
        },
        "LTX-2-Q2_K-GGUF": {
            "huggingface_id": "unsloth/LTX-2-GGUF",
            "gguf_filename": "ltx-2-19b-dev-Q2_K.gguf",
            "description": "4K variant, Q2_K quantized (19B params, GGUF)",
            "vram_gb": 8,  # 8.1GB file size
            "max_resolution": "4K (3840x2160)",
            "max_fps": 50,
            "quantization": "gguf-q2_k",
            "recommended": False,
            "note": "For 9-11GB VRAM - lowest quality but smallest footprint"
        },

        # BitsAndBytes NF4 variants (fallback if GGUF doesn't work)
        "LTX-2-4K-NF4": {
            "huggingface_id": "Lightricks/LTX-2",
            "description": "4K variant, 4-bit quantized transformer + text encoder (19B params, NF4)",
            "vram_gb": 10,
            "max_resolution": "4K (3840x2160)",
            "max_fps": 50,
            "quantization": "nf4",
            "recommended": False,
            "note": "BitsAndBytes quantization fallback (if GGUF doesn't work)"
        },

        # Full precision variants (for high-end cards)
        "LTX-2-4K": {
            "huggingface_id": "Lightricks/LTX-2",
            "description": "Full 4K variant (19B params, FP16)",
            "vram_gb": 24,
            "max_resolution": "4K (3840x2160)",
            "max_fps": 50,
            "quantization": None,
            "recommended": False,
            "note": "Requires 24GB+ VRAM (RTX 4090, A6000, etc.)"
        },
    }

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize LTX-2 model discovery.

        Args:
            models_dir: Directory to search for models. If None, uses models/Deforum/ltx2/
        """
        if models_dir is None:
            try:
                import modules.paths as paths
                base_models_dir = Path(paths.models_path)
            except:
                base_models_dir = Path("models")

            models_dir = base_models_dir / "Deforum" / "ltx2"

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def discover_models(self) -> List[Dict]:
        """
        Discover all LTX-2 models in the models directory.

        Returns:
            List of dictionaries containing model info
        """
        discovered = []

        for variant_name, variant_info in self.MODEL_VARIANTS.items():
            model_path = self.models_dir / variant_name

            if model_path.exists() and self._is_valid_ltx2_model(model_path):
                discovered.append({
                    "name": variant_name,
                    "path": str(model_path),
                    "downloaded": True,
                    **variant_info
                })
                logger.debug(f"Found LTX-2 model: {variant_name} at {model_path}")
            else:
                discovered.append({
                    "name": variant_name,
                    "path": str(model_path),
                    "downloaded": False,
                    **variant_info
                })

        if not any(m["downloaded"] for m in discovered):
            logger.info("No LTX-2 models found - use download_all_models.sh to download")

        return discovered

    def _is_valid_ltx2_model(self, model_path: Path) -> bool:
        """
        Check if directory contains a valid LTX-2 model.

        Args:
            model_path: Path to check

        Returns:
            True if valid LTX-2 model
        """
        # Check for required files
        required_files = [
            "config.json",
            "model_index.json",
            "diffusion_pytorch_model.safetensors"  # Main model weights
        ]

        for filename in required_files:
            if not (model_path / filename).exists():
                return False

        return True

    def get_model_path(self, variant: str = "LTX-2-4K") -> Optional[Path]:
        """
        Get path to a specific LTX-2 model variant.

        Args:
            variant: Model variant name

        Returns:
            Path to model or None if not found
        """
        models = self.discover_models()

        for model in models:
            if model["name"] == variant and model["downloaded"]:
                return Path(model["path"])

        return None

    def get_recommended_variant(self, available_vram_gb: float) -> str:
        """
        Get recommended model variant based on available VRAM.

        Args:
            available_vram_gb: Available VRAM in GB

        Returns:
            Recommended variant name
        """
        # Prefer GGUF variants (better quantization quality and VRAM efficiency)
        if available_vram_gb >= 24:
            return "LTX-2-4K"  # Full precision for high-end cards
        elif available_vram_gb >= 13:
            return "LTX-2-Q4_K_M-GGUF"  # Best quality/VRAM balance (RTX 3080 Ti, 4070 Ti, 4080, etc.)
        elif available_vram_gb >= 10:
            return "LTX-2-Q3_K_M-GGUF"  # Lower VRAM, slightly reduced quality
        elif available_vram_gb >= 8:
            logger.warning(f"Only {available_vram_gb:.1f}GB VRAM available - using Q2_K (lowest quality)")
            return "LTX-2-Q2_K-GGUF"  # Minimum viable quantization
        else:
            logger.warning(f"Only {available_vram_gb:.1f}GB VRAM available - LTX-2 requires 8GB minimum (with Q2_K quantization)")
            return "LTX-2-Q2_K-GGUF"  # Return Q2_K as absolute minimum

    def is_any_model_available(self) -> bool:
        """Check if any LTX-2 model is downloaded."""
        models = self.discover_models()
        return any(m["downloaded"] for m in models)

    def download_model(self, variant: str = "LTX-2-4K", auto_download: bool = False) -> bool:
        """
        Download LTX-2 model from HuggingFace.

        Args:
            variant: Model variant to download
            auto_download: Whether to automatically download

        Returns:
            True if download succeeded or already exists
        """
        if variant not in self.MODEL_VARIANTS:
            logger.error(f"Unknown LTX-2 variant: {variant}")
            return False

        model_path = self.models_dir / variant

        # Check if already downloaded
        if self._is_valid_ltx2_model(model_path):
            logger.info(f"LTX-2 model already exists: {variant}")
            return True

        if not auto_download:
            logger.info(f"LTX-2 model not found: {variant}")
            logger.info(f"Download manually: huggingface-cli download {self.MODEL_VARIANTS[variant]['huggingface_id']} --local-dir {model_path}")
            return False

        # Auto-download using huggingface_hub
        try:
            from huggingface_hub import snapshot_download

            logger.info(f"Downloading LTX-2 model: {variant} (this may take several minutes, ~19GB)...")

            snapshot_download(
                repo_id=self.MODEL_VARIANTS[variant]["huggingface_id"],
                local_dir=str(model_path),
                local_dir_use_symlinks=False
            )

            logger.info(f"Successfully downloaded LTX-2 model: {variant}")
            return True

        except Exception as e:
            logger.error(f"Error downloading LTX-2 model: {e}")
            return False
