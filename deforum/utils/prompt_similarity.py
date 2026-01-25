"""
Prompt Similarity Utilities for Adaptive Keyframe Strength

Uses CLIP text embeddings to measure semantic similarity between prompts,
enabling adaptive strength calculation for I2V chaining.

Higher similarity → Higher strength (preserve more)
Lower similarity → Lower strength (change more)
"""

from typing import Tuple, Optional
import torch
import numpy as np
from deforum.utils.system.logging import get_logger

logger = get_logger()


def calculate_prompt_similarity(
    prompt1: str,
    prompt2: str,
    clip_model = None,
    clip_processor = None
) -> float:
    """
    Calculate semantic similarity between two prompts using CLIP embeddings.

    Args:
        prompt1: First prompt text
        prompt2: Second prompt text
        clip_model: Optional pre-loaded CLIP model (will auto-load if None)
        clip_processor: Optional pre-loaded CLIP processor (will auto-load if None)

    Returns:
        Similarity score from 0.0 (completely different) to 1.0 (identical)

    Note:
        Uses Forge's loaded CLIP model when available, otherwise loads transformers CLIP.
    """
    # Handle empty or None prompts
    if not prompt1 or not prompt2:
        return 0.0

    # Identical prompts
    if prompt1.strip() == prompt2.strip():
        return 1.0

    try:
        # Try to use Forge's loaded CLIP model first (already in memory)
        if clip_model is None:
            try:
                from modules import shared
                from modules import sd_hijack

                # Get the current model's CLIP encoder
                if hasattr(shared.sd_model, 'cond_stage_model'):
                    clip_model = shared.sd_model.cond_stage_model
                    logger.debug("Using Forge's loaded CLIP model for prompt similarity")

                    # Encode prompts using Forge's CLIP
                    with torch.no_grad():
                        # Use the same encoding method as Forge
                        emb1 = sd_hijack.model_hijack.get_prompt_lengths([prompt1])[0]
                        emb2 = sd_hijack.model_hijack.get_prompt_lengths([prompt2])[0]

                        # Get actual embeddings (not just lengths)
                        tokens1 = clip_model.tokenize([prompt1])
                        tokens2 = clip_model.tokenize([prompt2])

                        with torch.autocast('cuda', enabled=True):
                            enc1 = clip_model.encode_from_tokens(tokens1)
                            enc2 = clip_model.encode_from_tokens(tokens2)

                        # Normalize and compute cosine similarity
                        enc1_norm = torch.nn.functional.normalize(enc1, dim=-1)
                        enc2_norm = torch.nn.functional.normalize(enc2, dim=-1)

                        # Cosine similarity (already normalized, so just dot product)
                        similarity = (enc1_norm * enc2_norm).sum().item()

                        # Clamp to [0, 1] range
                        similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))

                        logger.debug(f"Prompt similarity (Forge CLIP): {similarity:.3f}")
                        return similarity
            except Exception as e:
                logger.debug(f"Could not use Forge CLIP (will fallback): {e}")

        # Fallback: Use transformers CLIP
        logger.debug("Using transformers CLIP for prompt similarity")
        from transformers import CLIPTokenizer, CLIPTextModel

        # Load model and tokenizer (cached after first load)
        if clip_model is None or clip_processor is None:
            model_id = "openai/clip-vit-base-patch32"
            clip_processor = CLIPTokenizer.from_pretrained(model_id)
            clip_model = CLIPTextModel.from_pretrained(model_id)

            # Move to GPU if available
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()

            clip_model.eval()

        # Tokenize and encode
        with torch.no_grad():
            inputs1 = clip_processor(prompt1, return_tensors="pt", padding=True, truncation=True)
            inputs2 = clip_processor(prompt2, return_tensors="pt", padding=True, truncation=True)

            if torch.cuda.is_available():
                inputs1 = {k: v.cuda() for k, v in inputs1.items()}
                inputs2 = {k: v.cuda() for k, v in inputs2.items()}

            # Get embeddings (pooled output)
            outputs1 = clip_model(**inputs1)
            outputs2 = clip_model(**inputs2)

            emb1 = outputs1.pooler_output
            emb2 = outputs2.pooler_output

            # Normalize embeddings
            emb1_norm = torch.nn.functional.normalize(emb1, dim=-1)
            emb2_norm = torch.nn.functional.normalize(emb2, dim=-1)

            # Cosine similarity
            similarity = (emb1_norm * emb2_norm).sum().item()

            # Clamp to [0, 1] range (cosine similarity is [-1, 1])
            similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))

            logger.debug(f"Prompt similarity (transformers CLIP): {similarity:.3f}")
            return similarity

    except Exception as e:
        logger.warning(f"Error calculating prompt similarity: {e}")
        # Fallback to simple text comparison
        return simple_text_similarity(prompt1, prompt2)


def simple_text_similarity(prompt1: str, prompt2: str) -> float:
    """
    Simple text-based similarity fallback using Jaccard similarity of words.

    Args:
        prompt1: First prompt text
        prompt2: Second prompt text

    Returns:
        Similarity score from 0.0 to 1.0
    """
    # Normalize and tokenize
    words1 = set(prompt1.lower().split())
    words2 = set(prompt2.lower().split())

    # Handle empty sets
    if not words1 or not words2:
        return 0.0

    # Jaccard similarity: intersection / union
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    similarity = intersection / union if union > 0 else 0.0

    logger.debug(f"Prompt similarity (Jaccard fallback): {similarity:.3f}")
    return similarity


def adaptive_keyframe_strength(
    base_strength: float,
    prev_prompt: str,
    curr_prompt: str,
    min_strength: float = 0.10,
    max_strength: float = 0.30,
    clip_model = None,
    clip_processor = None
) -> Tuple[float, float]:
    """
    Calculate adaptive keyframe strength based on prompt similarity.

    Higher similarity → Higher strength (preserve more from previous keyframe)
    Lower similarity → Lower strength (allow more change)

    Args:
        base_strength: Base/fallback strength value
        prev_prompt: Previous keyframe's prompt
        curr_prompt: Current keyframe's prompt
        min_strength: Minimum allowed strength (maximum change)
        max_strength: Maximum allowed strength (maximum preservation)
        clip_model: Optional pre-loaded CLIP model
        clip_processor: Optional pre-loaded CLIP processor

    Returns:
        Tuple of (adaptive_strength, similarity_score)

    Example:
        # Similar prompts ("city day" → "city dusk"):
        # High similarity (0.85) → strength 0.265 (preserve most)

        # Different prompts ("city" → "forest"):
        # Low similarity (0.15) → strength 0.13 (allow change)
    """
    # Calculate semantic similarity
    similarity = calculate_prompt_similarity(
        prev_prompt,
        curr_prompt,
        clip_model,
        clip_processor
    )

    # Map similarity [0, 1] to strength [min_strength, max_strength]
    # High similarity → high strength (more preservation)
    adaptive_strength = min_strength + (max_strength - min_strength) * similarity

    logger.debug(
        f"Adaptive strength: similarity={similarity:.3f} → strength={adaptive_strength:.3f} "
        f"(range [{min_strength:.2f}, {max_strength:.2f}])"
    )

    return adaptive_strength, similarity


def batch_calculate_similarities(prompts: list[str]) -> list[float]:
    """
    Calculate similarities between consecutive prompts in a sequence.

    Args:
        prompts: List of prompt strings

    Returns:
        List of similarity scores (length = len(prompts) - 1)

    Example:
        prompts = ["city day", "city dusk", "forest night"]
        similarities = batch_calculate_similarities(prompts)
        # [0.85, 0.15]  # city→city high, city→forest low
    """
    if len(prompts) < 2:
        return []

    similarities = []
    for i in range(len(prompts) - 1):
        sim = calculate_prompt_similarity(prompts[i], prompts[i + 1])
        similarities.append(sim)

    return similarities
