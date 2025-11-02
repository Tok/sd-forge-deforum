from enum import Enum
from typing import List

from deforum.utils.system.logging import log as log_utils


class KeyFrameDistribution(Enum):
    OFF = "Off"
    KEYFRAMES_ONLY = "Keyframes Only"  # cadence is ignored. all other frames are handled as tweens.
    REDISTRIBUTED_CADENCE = "Redistributed Cadence"  # exact keyframes + cadence frames distributed between them

    @staticmethod
    def from_UI_tab(data):
        distribution = data.args.anim_args.keyframe_distribution
        match distribution:
            case "Off":
                return KeyFrameDistribution.OFF
            case "Keyframes Only":
                return KeyFrameDistribution.KEYFRAMES_ONLY
            case "Redistributed Cadence" | "Redistributed":  # Accept old name for migration
                return KeyFrameDistribution.REDISTRIBUTED_CADENCE
            case _:
                default = KeyFrameDistribution.default()
                log_utils.warning(f"Invalid keyframe_distribution from UI: '{distribution}'. Falling back to '{default}'.")
                return default

    @staticmethod
    def default():
        return KeyFrameDistribution.OFF

    def calculate(self, data, start_index, diffusion_frame_count) -> List[int]:
        max_frames = data.args.anim_args.max_frames
        match self:
            case KeyFrameDistribution.OFF:
                # To get here on purpose, override `is_use_new_render_core` in render.py
                log_utils.warning("Called new core without keyframe distribution. Using uniform from cadence'.")
                return self.uniform_indexes(start_index, max_frames, diffusion_frame_count)
            case KeyFrameDistribution.KEYFRAMES_ONLY:
                return self.select_keyframes(data)
            case KeyFrameDistribution.REDISTRIBUTED_CADENCE:
                return self._redistributed_cadence(data, start_index, max_frames, diffusion_frame_count)
            case _:
                raise ValueError(f"Invalid KeyFrameDistribution: {self}")

    @staticmethod
    def uniform_indexes(start_index, max_frames, diffusion_frame_count):
        # max_frames is count (e.g., 333), last valid index is max_frames - 1 (e.g., 332)
        return [start_index + int(n * (max_frames - 1 - start_index) / (diffusion_frame_count - 1))
                for n in range(diffusion_frame_count)]

    @staticmethod
    def _redistributed_cadence(data, start_index, max_frames, diffusion_frame_count):
        """Start with EXACT keyframes, then distribute cadence frames between them.

        Priority:
        1. EXACT keyframe placement (non-negotiable)
        2. Approximate desired cadence between keyframes
        3. Drop cadence frames if too close to keyframes or each other

        Algorithm:
        - Get exact keyframes from prompts
        - Calculate remaining budget for cadence frames
        - Distribute cadence frames between keyframe sections
        - Ensure minimum spacing (avoid back-to-back diffusions)
        """
        # Get EXACT keyframes (these are non-negotiable)
        keyframes = KeyFrameDistribution.select_keyframes(data)
        keyframes_set = set(keyframes)

        # Calculate how many cadence frames we can add
        num_keyframes = len(keyframes)
        cadence_budget = diffusion_frame_count - num_keyframes

        if cadence_budget <= 0:
            # No room for cadence frames, just use keyframes
            log_utils.info(f"Redistributed: Using {num_keyframes} keyframes only (no cadence frames)")
            return sorted(keyframes)

        # Get desired cadence from settings
        desired_cadence = data.args.anim_args.diffusion_cadence
        min_spacing = max(1, desired_cadence // 2)  # Minimum frames between diffusions

        # Distribute cadence frames between keyframes
        cadence_frames = []
        keyframes_sorted = sorted(keyframes)

        for i in range(len(keyframes_sorted) - 1):
            section_start = keyframes_sorted[i]
            section_end = keyframes_sorted[i + 1]
            section_length = section_end - section_start

            if section_length <= min_spacing:
                # Section too short for cadence frames
                continue

            # Calculate how many cadence frames fit in this section
            # Use desired cadence to approximate spacing
            num_cadence_in_section = max(0, (section_length - min_spacing) // desired_cadence)

            if num_cadence_in_section == 0:
                continue

            # Distribute evenly within section
            for j in range(1, num_cadence_in_section + 1):
                cadence_frame = section_start + int(j * section_length / (num_cadence_in_section + 1))

                # Ensure minimum spacing from keyframes and other cadence frames
                too_close = False
                for kf in keyframes_sorted:
                    if abs(cadence_frame - kf) < min_spacing:
                        too_close = True
                        break

                if not too_close and cadence_frame not in keyframes_set:
                    cadence_frames.append(cadence_frame)

        # Combine keyframes and cadence frames
        all_frames = sorted(list(keyframes_set) + cadence_frames)

        # If we have too many frames, drop cadence frames (keep keyframes!)
        if len(all_frames) > diffusion_frame_count:
            log_utils.warning(
                f"Redistributed: Too many frames ({len(all_frames)} > {diffusion_frame_count}). "
                f"Dropping {len(all_frames) - diffusion_frame_count} cadence frames."
            )
            # Keep all keyframes, trim cadence frames
            cadence_frames = [f for f in all_frames if f not in keyframes_set]
            cadence_frames = cadence_frames[:diffusion_frame_count - num_keyframes]
            all_frames = sorted(list(keyframes_set) + cadence_frames)

        # If we have too few frames, add more cadence frames
        while len(all_frames) < diffusion_frame_count:
            # Find largest gap and add a frame in the middle
            largest_gap_start = 0
            largest_gap_size = 0

            for i in range(len(all_frames) - 1):
                gap_size = all_frames[i + 1] - all_frames[i]
                if gap_size > largest_gap_size:
                    largest_gap_size = gap_size
                    largest_gap_start = all_frames[i]

            if largest_gap_size <= 1:
                # No more gaps to fill
                break

            # Add frame in middle of largest gap
            new_frame = largest_gap_start + largest_gap_size // 2
            if new_frame not in all_frames:
                all_frames.append(new_frame)
                all_frames.sort()

        log_utils.info(
            f"Redistributed Cadence: {num_keyframes} exact keyframes + {len(all_frames) - num_keyframes} cadence frames "
            f"= {len(all_frames)} total (target: {diffusion_frame_count})"
        )

        assert len(all_frames) <= diffusion_frame_count, \
            f"Too many frames: {len(all_frames)} > {diffusion_frame_count}"

        return all_frames

    @staticmethod
    def select_keyframes(data):
        return (KeyFrameDistribution.select_parseq_keyframes(data)
                if data.parseq_adapter.use_parseq
                else KeyFrameDistribution.select_deforum_keyframes(data))

    @staticmethod
    def select_parseq_keyframes(data):
        # Parseq keyframes are 0-indexed (frame 0 is first frame)
        keyframes = data.parseq_adapter.parseq_json["keyframes"]
        return list(map(lambda _: _["frame"], keyframes))

    @staticmethod
    def select_deforum_keyframes(data):
        # Prompt at 0 is always meant to be defined in prompts, but the last frame is not, so we just take max_frames - 1.
        prompt_keyframes = list(map(int, data.args.root.prompt_keyframes))
        last_frame = [data.args.anim_args.max_frames - 1]  # Last valid index with 0-based indexing
        keyframes = list(set(prompt_keyframes + last_frame))
        keyframes.sort()
        keyframes[0] = 0  # Makes sure 1st frame is always 0 (0-indexed).

        max_frames = data.args.anim_args.max_frames
        # Filter out frames >= max_frames or < 0 and log warning if any are removed.
        original_count = len(keyframes)
        keyframes = list(filter(lambda _: 0 <= _ < max_frames, keyframes))
        if len(keyframes) < original_count:
            log_utils.warning(f"Removed at least one prompt because its index is not between 0 and {max_frames - 1}. "
                           f"Original count: {original_count}, New count: {len(keyframes)}")
        return keyframes

    @staticmethod
    def is_deforum_keyframe(data, i):
        return i in KeyFrameDistribution.select_deforum_keyframes(data)

    @staticmethod
    def is_parseq_keyframe(data, i):
        return i in KeyFrameDistribution.select_parseq_keyframes(data)

    @staticmethod
    def is_keyframe(data, i):
        return (KeyFrameDistribution.is_parseq_keyframe(data, i)
                if data.parseq_adapter.use_parseq
                else KeyFrameDistribution.is_deforum_keyframe(data, i))
