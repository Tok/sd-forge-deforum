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
        True if Flux appears to be configured, False otherwise
    """
    try:
        # Check if a checkpoint is selected
        if not hasattr(shared, 'opts') or not hasattr(shared.opts, 'sd_model_checkpoint'):
            return False

        checkpoint = shared.opts.sd_model_checkpoint or ""
        checkpoint_lower = checkpoint.lower()

        # Check if checkpoint name contains "flux"
        if "flux" in checkpoint_lower:
            return True

        # Check if sd_model exists and has flux in the name
        if hasattr(shared, 'sd_model') and shared.sd_model is not None:
            if hasattr(shared.sd_model, 'sd_model_checkpoint'):
                model_name = getattr(shared.sd_model, 'sd_model_checkpoint', '').lower()
                if "flux" in model_name:
                    return True

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

    Returns:
        True if blocker should be shown, False if Deforum UI should load normally
    """
    return not is_flux_available()
