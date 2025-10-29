"""Lumina Compatibility Patch

Ensures dynamic_args["num_tokens"] is populated before sampling.

The issue: Lumina's forward method requires num_tokens parameter from dynamic_args,
but it's not always set correctly when going through Deforum's img2img pipeline.

Root cause: GemmaTextProcessingEngine.process_tokens() sets dynamic_args["num_tokens"]
but only during get_learned_conditioning(). For img2img with CFG, we need both
conditional and unconditional passes to have num_tokens set.

Solution: Patch the sampling to ensure num_tokens is available before Lumina's
forward method tries to access it.
"""

from typing import Optional
from deforum.utils.system.logging import get_logger
from deforum.utils.model_detection import is_lumina_model

logger = get_logger()

_lumina_patch_applied = False


def ensure_num_tokens_for_lumina(p) -> bool:
    """Ensure dynamic_args["num_tokens"] is set before sampling for Lumina models.

    Args:
        p: StableDiffusionProcessing object with prompt/negative_prompt

    Returns:
        True if patch was applied, False if not needed
    """
    global _lumina_patch_applied

    if not is_lumina_model():
        return False

    try:
        from backend.args import dynamic_args
        import modules.shared as shared

        # Check if num_tokens is already set
        if "num_tokens" in dynamic_args and isinstance(dynamic_args["num_tokens"], list):
            if len(dynamic_args["num_tokens"]) >= 2:  # Has both cond and uncond
                logger.debug(f"num_tokens already set: {dynamic_args['num_tokens']}")
                return False

        # Force conditioning calculation to populate num_tokens
        if hasattr(shared, 'sd_model'):
            model = shared.sd_model

            # Get prompts
            prompts = [p.prompt] if isinstance(p.prompt, str) else p.prompt
            negative_prompts = [p.negative_prompt] if isinstance(p.negative_prompt, str) else p.negative_prompt

            logger.debug(f"Forcing conditioning for Lumina to populate num_tokens")
            logger.debug(f"Prompts: {len(prompts)}, Negative: {len(negative_prompts)}")

            # Calculate conditioning (this should populate dynamic_args["num_tokens"])
            # Positive conditioning
            if hasattr(model, 'get_learned_conditioning'):
                cond = model.get_learned_conditioning(prompts)
                logger.debug(f"Positive conditioning shape: {cond.shape if hasattr(cond, 'shape') else 'unknown'}")

            # Negative conditioning
            if hasattr(model, 'get_learned_conditioning') and negative_prompts:
                uncond = model.get_learned_conditioning(negative_prompts)
                logger.debug(f"Negative conditioning shape: {uncond.shape if hasattr(uncond, 'shape') else 'unknown'}")

            # Check if num_tokens is now set
            if "num_tokens" in dynamic_args:
                logger.debug(f"✓ num_tokens populated: {dynamic_args['num_tokens']}")
                _lumina_patch_applied = True
                return True
            else:
                logger.warning("Failed to populate num_tokens via get_learned_conditioning()")
                return False

    except Exception as e:
        logger.error(f"Failed to ensure num_tokens for Lumina: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

    return False


def apply_lumina_patch_if_needed(p) -> bool:
    """Check if Lumina is loaded and apply compatibility patch if needed.

    This should be called before process_images() to ensure num_tokens is set.

    Args:
        p: StableDiffusionProcessing object

    Returns:
        True if patch was applied, False otherwise
    """
    if not is_lumina_model():
        return False

    logger.debug("Lumina model detected - applying compatibility patch")
    return ensure_num_tokens_for_lumina(p)
