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

            # Get prompts - handle both string and list
            if isinstance(p.prompt, str):
                prompts = [p.prompt]
            elif isinstance(p.prompt, list):
                prompts = p.prompt
            else:
                prompts = [str(p.prompt)]

            if isinstance(p.negative_prompt, str):
                negative_prompts = [p.negative_prompt]
            elif isinstance(p.negative_prompt, list):
                negative_prompts = p.negative_prompt
            else:
                negative_prompts = [str(p.negative_prompt)] if p.negative_prompt else [""]

            logger.debug(f"Forcing conditioning for Lumina to populate num_tokens")
            logger.debug(f"Prompts: {prompts}")
            logger.debug(f"Negative: {negative_prompts}")

            # Calculate conditioning (this should populate dynamic_args["num_tokens"])
            # The actual calculation happens inside process_images(), but we need
            # to trigger the text encoder's tokenization to get the real num_tokens

            # For Lumina, we need to call the text processing engine directly
            if hasattr(model, 'text_processing_engine_gemma'):
                try:
                    engine = model.text_processing_engine_gemma
                    logger.debug(f"Found Gemma text processing engine")

                    # Tokenize to get actual token counts
                    if hasattr(engine, 'tokenize'):
                        tokens = engine.tokenize(prompts)
                        actual_count = len(tokens[0]) if tokens else 256
                        logger.debug(f"Actual token count from tokenization: {actual_count}")

                        # Set num_tokens to actual count (duplicated for cond/uncond)
                        dynamic_args["num_tokens"] = [actual_count, actual_count]
                        logger.debug(f"Set num_tokens to actual: {dynamic_args['num_tokens']}")
                    else:
                        logger.warning("Gemma engine has no tokenize method")
                except Exception as e:
                    logger.warning(f"Failed to use Gemma engine directly: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())

            # Check if num_tokens is now set
            if "num_tokens" in dynamic_args:
                num_tokens_list = dynamic_args["num_tokens"]
                logger.debug(f"✓ num_tokens populated: {num_tokens_list}")

                # Ensure we have at least 2 entries (cond and uncond)
                # If CFG=1.0, unconditional is skipped, so we only have 1 entry
                # Duplicate it so both indices work
                if isinstance(num_tokens_list, list) and len(num_tokens_list) == 1:
                    dynamic_args["num_tokens"] = num_tokens_list + num_tokens_list
                    logger.debug(f"CFG=1.0 detected - duplicated num_tokens: {dynamic_args['num_tokens']}")

                _lumina_patch_applied = True
                return True
            else:
                logger.warning("Failed to populate num_tokens via get_learned_conditioning()")
                # Set a fallback value to prevent KeyError
                dynamic_args["num_tokens"] = [256, 256]  # Default token count
                logger.warning(f"Using fallback num_tokens: {dynamic_args['num_tokens']}")
                return True  # Return True anyway to prevent crashes

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
