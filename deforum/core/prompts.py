"""Core prompt scheduling and composition logic.

This module provides a high-level interface for managing prompt evolution
across animation frames. It combines the pure utility functions from
deforum.utils.prompt_utils into a stateful scheduler that handles:

- Parsing animation prompt dictionaries (format: "0: prompt1, 30: prompt2")
- Interpolating between prompts at keyframes
- Evaluating mathematical expressions in prompts
- Handling weighted subprompts (format: "(word:weight)")
- Splitting positive/negative prompts ("--neg" separator)

The PromptScheduler provides a cleaner API than calling individual utility
functions, making it easier to manage complex prompt animations.

Classes:
    PromptScheduler: Stateful prompt scheduler for animations

Functions:
    prepare_prompt: Legacy function for printing prompt info (IMPURE)
"""

from typing import Dict
import pandas as pd

# Import pure utilities
from deforum.utils.generation.prompts import (
    interpolate_prompts,
    substitute_prompt_expressions,
    split_prompt_into_pos_neg,
)
# Import keyframe parser for parsing prompt strings
from deforum.core.keyframes import FrameInterpolater
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


# Optional imports for console output
try:
    from deforum.utils.system.logging.log import (
        RED, GREEN, PURPLE, RESET_COLOR
    )
except ImportError:
    # Fallback for unit tests
    RED = "\033[91m"
    GREEN = "\033[92m"
    PURPLE = "\033[95m"
    RESET_COLOR = "\033[0m"


__all__ = [
    'PromptScheduler',
    'prepare_prompt',
]


class PromptScheduler:
    """Stateful prompt scheduler for animation frames.

    Manages prompt evolution across animation frames by parsing keyframe
    prompts and interpolating between them. Provides a clean interface
    for the common pattern of:
    1. Parse animation prompt string
    2. Get interpolated prompt for current frame
    3. Evaluate expressions
    4. Split positive/negative prompts

    Attributes:
        max_frames: Total number of frames in animation
        prompt_dict: Parsed dictionary of frame→prompt mappings
        prompt_series: Pandas Series with interpolated prompts

    Examples:
        >>> # Simple prompt animation
        >>> scheduler = PromptScheduler(
        ...     prompts="0: a cat, 30: a dog, 60: a bird",
        ...     max_frames=100
        ... )
        >>> scheduler.get_prompt(0)
        'a cat'
        >>> scheduler.get_prompt(15)  # Interpolated between cat and dog
        'a cat and a dog'
        >>> scheduler.get_prompt(30)
        'a dog'

        >>> # With mathematical expressions
        >>> scheduler = PromptScheduler(
        ...     prompts="0: (cat:1.0), 50: (cat:{sin(t/10)+1})",
        ...     max_frames=100
        ... )
        >>> prompt = scheduler.get_prompt(25)  # Expression evaluated at t=25

        >>> # Positive and negative prompts
        >>> pos, neg = scheduler.get_prompt_split(0)
    """

    def __init__(self, prompts: str | Dict[int, str], max_frames: int):
        """Initialize prompt scheduler.

        Args:
            prompts: Either animation prompt string ("0: prompt1, 30: prompt2")
                    or pre-parsed dictionary {frame: prompt}
            max_frames: Total number of frames in animation

        Examples:
            >>> # From string
            >>> scheduler = PromptScheduler(
            ...     "0: a cat, 30: a dog",
            ...     max_frames=100
            ... )

            >>> # From dict
            >>> scheduler = PromptScheduler(
            ...     {0: "a cat", 30: "a dog"},
            ...     max_frames=100
            ... )
        """
        self.max_frames = max_frames

        # Parse prompts if string, otherwise use dict directly
        if isinstance(prompts, str):
            # Use FrameInterpolater to parse simple format "0: cat, 30: dog"
            parser = FrameInterpolater(max_frames=max_frames)
            self.prompt_dict = parser.parse_key_frames(prompts)
        else:
            # Ensure integer keys
            self.prompt_dict = {int(k): v for k, v in prompts.items()}

        # Create interpolated series (interpolate_prompts needs string keys)
        prompt_dict_str_keys = {str(k): v for k, v in self.prompt_dict.items()}
        self.prompt_series = interpolate_prompts(prompt_dict_str_keys, max_frames)

    def get_prompt(self, frame_idx: int, evaluate_expressions: bool = True) -> str:
        """Get interpolated prompt for given frame.

        Args:
            frame_idx: Frame number to get prompt for
            evaluate_expressions: If True, evaluate math expressions

        Returns:
            Interpolated prompt string for the frame

        Examples:
            >>> scheduler = PromptScheduler("0: cat, 30: dog", max_frames=100)
            >>> scheduler.get_prompt(0)
            'cat'
            >>> scheduler.get_prompt(15)
            'cat and dog'
            >>> scheduler.get_prompt(30)
            'dog'
        """
        if frame_idx < 0 or frame_idx >= self.max_frames:
            raise ValueError(
                f"frame_idx {frame_idx} out of range [0, {self.max_frames})"
            )

        prompt = self.prompt_series[frame_idx]

        if evaluate_expressions:
            prompt = substitute_prompt_expressions(prompt, frame_idx, self.max_frames)

        return prompt

    def get_prompt_split(
        self,
        frame_idx: int,
        evaluate_expressions: bool = True
    ) -> tuple[str, str]:
        """Get positive and negative prompts for given frame.

        Splits prompt on "--neg" separator into positive and negative parts.

        Args:
            frame_idx: Frame number to get prompt for
            evaluate_expressions: If True, evaluate math expressions

        Returns:
            Tuple of (positive_prompt, negative_prompt)

        Examples:
            >>> scheduler = PromptScheduler(
            ...     "0: a cat --neg blurry, 30: a dog --neg low quality",
            ...     max_frames=100
            ... )
            >>> pos, neg = scheduler.get_prompt_split(0)
            >>> pos
            'a cat'
            >>> neg
            'blurry'
        """
        full_prompt = self.get_prompt(frame_idx, evaluate_expressions)
        return split_prompt_into_pos_neg(full_prompt)

    def update_prompts(self, prompts: str | Dict[int, str]):
        """Update prompt schedule with new prompts.

        Useful for dynamically changing animation prompts without
        creating a new scheduler.

        Args:
            prompts: New animation prompt string or dictionary

        Examples:
            >>> scheduler = PromptScheduler("0: cat", max_frames=100)
            >>> scheduler.update_prompts("0: cat, 50: dog, 100: bird")
            >>> scheduler.get_prompt(50)
            'dog'
        """
        if isinstance(prompts, str):
            # Use FrameInterpolater to parse simple format "0: cat, 30: dog"
            parser = FrameInterpolater(max_frames=self.max_frames)
            self.prompt_dict = parser.parse_key_frames(prompts)
        else:
            # Ensure integer keys
            self.prompt_dict = {int(k): v for k, v in prompts.items()}

        # Create interpolated series (interpolate_prompts needs string keys)
        prompt_dict_str_keys = {str(k): v for k, v in self.prompt_dict.items()}
        self.prompt_series = interpolate_prompts(prompt_dict_str_keys, self.max_frames)

    def get_keyframes(self) -> Dict[int, str]:
        """Get dictionary of keyframe prompts.

        Returns:
            Dictionary mapping frame numbers to prompt strings

        Examples:
            >>> scheduler = PromptScheduler("0: cat, 30: dog", max_frames=100)
            >>> scheduler.get_keyframes()
            {0: 'cat', 30: 'dog'}
        """
        return self.prompt_dict.copy()

    def has_negative_prompts(self) -> bool:
        """Check if any prompts contain negative prompt separator.

        Returns:
            True if any keyframe has "--neg" separator

        Examples:
            >>> scheduler = PromptScheduler("0: cat --neg blurry", max_frames=100)
            >>> scheduler.has_negative_prompts()
            True
        """
        return any("--neg" in prompt for prompt in self.prompt_dict.values())

    def __repr__(self) -> str:
        """String representation of scheduler."""
        keyframe_count = len(self.prompt_dict)
        return (
            f"PromptScheduler(max_frames={self.max_frames}, "
            f"keyframes={keyframe_count})"
        )


# ============================================================================
# LEGACY FUNCTION (IMPURE: console output)
# ============================================================================

def prepare_prompt(prompt_series: str, max_frames: int, seed: int, frame_idx: int) -> str:
    """Evaluate prompt expressions and print formatted prompt with seed info.

    Legacy function that evaluates expressions in a prompt and prints
    colored output to console. Maintained for backward compatibility with
    existing rendering pipeline.

    Args:
        prompt_series: Prompt string (may contain expressions)
        max_frames: Total frames for expression evaluation
        seed: Seed value to display
        frame_idx: Current frame for expression evaluation

    Returns:
        Evaluated prompt string (with --neg separator if present)

    Side Effects:
        - Prints seed in green
        - Prints positive prompt in purple
        - Prints negative prompt in red (if present)

    Examples:
        >>> # With console output
        >>> prepare_prompt("a cat", max_frames=100, seed=42, frame_idx=0)
        # Prints: Seed: 42
        # Prints: Prompt: a cat
        'a cat'

        >>> # With negative prompt
        >>> prepare_prompt("a cat --neg blurry", max_frames=100, seed=42, frame_idx=0)
        # Prints: Seed: 42
        # Prints: Prompt: a cat
        # Prints: Neg Prompt: blurry
        'a cat --neg blurry'
    """
    prompt_parsed = substitute_prompt_expressions(prompt_series, frame_idx, max_frames)

    prompt_to_print, *after_neg = prompt_parsed.strip().split("--neg")
    prompt_to_print = prompt_to_print.strip()
    after_neg = "".join(after_neg).strip()

    logger.info(f"{GREEN}Seed: {RESET_COLOR}{seed}")
    logger.info(f"{PURPLE}Prompt: {RESET_COLOR}{prompt_to_print}")
    if after_neg and after_neg.strip():
        logger.info(f"{RED}Neg Prompt: {RESET_COLOR}{after_neg}")
        prompt_to_print += f" --neg {after_neg}"

    return prompt_to_print
