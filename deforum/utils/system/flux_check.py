"""
Flux availability check for Deforum extension.

All Deforum modes now require Flux, so we check if it's properly configured.
"""

import modules.shared as shared
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()



def is_flux_available() -> bool:
    """
    Check if Flux model is available in Forge.

    Returns:
        True if Flux appears to be configured or files exist on disk, False otherwise
    """
    try:
        import os
        import glob
        from pathlib import Path

        # Method 1: Check if currently selected checkpoint is Flux
        if hasattr(shared, 'opts') and hasattr(shared.opts, 'sd_model_checkpoint'):
            checkpoint = shared.opts.sd_model_checkpoint or ""
            if "flux" in checkpoint.lower():
                return True

        # Method 2: Check if loaded model is Flux
        if hasattr(shared, 'sd_model') and shared.sd_model is not None:
            if hasattr(shared.sd_model, 'sd_model_checkpoint'):
                model_name = getattr(shared.sd_model, 'sd_model_checkpoint', '').lower()
                if "flux" in model_name:
                    return True

        # Method 3: Check if Flux model files exist on disk (most reliable during startup)
        try:
            import modules.paths as ph
            models_dir = Path(ph.models_path)

            # Check common Flux model locations
            flux_locations = [
                models_dir / "Stable-diffusion" / "Flux" / "flux*.safetensors",
                models_dir / "Stable-diffusion" / "flux*.safetensors",
            ]

            for pattern in flux_locations:
                matches = glob.glob(str(pattern))
                if matches:
                    logger.debug(f"Found Flux model on disk: {matches[0]}")
                    return True
        except:
            pass

        return False

    except Exception as e:
        logger.error(f"Could not check Flux availability: {e}")
        # If we can't check, assume it's available to not block the user
        return True


def get_flux_setup_message() -> str:
    """
    Get HTML message instructing user how to set up Flux.

    Returns:
        HTML-formatted setup instructions
    """
    return """
    <div style="padding: 20px; background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 8px; margin: 20px 0;">
        <h2 style="color: #856404; margin-top: 0;">⚠️ Flux Model Required</h2>
        <p style="color: #856404; font-size: 14px; line-height: 1.6;">
            All Deforum modes now require Flux to be configured in Forge.<br><br>

            <strong>Setup Steps:</strong><br>
            1. Download Flux model (e.g., flux1-dev-bnb-nf4-v2.safetensors)<br>
            2. Place it in <code>models/Stable-diffusion/</code><br>
            3. Select Flux from the Forge checkpoint dropdown at the top<br>
            4. Reload this page<br><br>

            <strong>Recommended Model:</strong><br>
            • <code>flux1-dev-bnb-nf4-v2.safetensors</code> (quantized, lower VRAM)<br>
            • Plus corresponding VAE files<br><br>

            See the main README.md for detailed installation instructions.
        </p>
    </div>
    """


def should_show_flux_blocker() -> bool:
    """
    Determine if we should show the Flux setup blocker message.

    DEPRECATED: Blocker removed - we now auto-download Flux instead.
    This function always returns False to never block the UI.

    Returns:
        False - blocker disabled
    """
    # Blocker removed - auto-download handles missing models
    return False
