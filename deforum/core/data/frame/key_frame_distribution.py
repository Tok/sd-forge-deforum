from enum import Enum
from typing import List

from ...util import log_utils


class KeyFrameDistribution(Enum):
    OFF = "Off"
    KEYFRAMES_ONLY = "Keyframes Only"  # cadence is ignored. all other frames are handled as tweens.
    ADDITIVE = "Additive"  # both cadence and parseq keyframes are used.
    REDISTRIBUTED = "Redistributed"  # similar to uniform, but keyframe diffusion is enforced.

    @staticmethod
    def from_UI_tab(data):
        distribution = data.args.anim_args.keyframe_distribution
        match distribution:
            case "Off":
                return KeyFrameDistribution.OFF
            case "Keyframes Only":
                return KeyFrameDistribution.KEYFRAMES_ONLY
            case "Additive":
                return KeyFrameDistribution.ADDITIVE
            case "Redistributed":
                return KeyFrameDistribution.REDISTRIBUTED
            case _:
                default = KeyFrameDistribution.default()
                log_utils.warn(f"Invalid keyframe_distribution from UI: '{distribution}'. Falling back to '{default}'.")
                return default

    @staticmethod
    def default():
        return KeyFrameDistribution.OFF

    def calculate(self, data, start_index, diffusion_frame_count) -> List[int]:
        max_frames = data.args.anim_args.max_frames
        match self:
            case KeyFrameDistribution.OFF:
                # To get here on purpose, override `is_use_new_render_core` in render.py
                log_utils.warn("Called new core without keyframe distribution. Using uniform from cadence'.")
                return self.uniform_indexes(start_index, max_frames, diffusion_frame_count)
            case KeyFrameDistribution.KEYFRAMES_ONLY:
                return self.select_keyframes(data)
            case KeyFrameDistribution.ADDITIVE:
                return self._additive(data, start_index, max_frames)
            case KeyFrameDistribution.REDISTRIBUTED:
                return self._redistributed(data, start_index, max_frames, diffusion_frame_count)
            case _:
                raise ValueError(f"Invalid KeyFrameDistribution: {self}")

    @staticmethod
    def uniform_indexes(start_index, max_frames, diffusion_frame_count):
        return [1 + start_index + int(n * (max_frames - 1 - start_index) / (diffusion_frame_count - 1))
                for n in range(diffusion_frame_count)]

    @staticmethod
    def _additive(data, start_index, max_frames):
        """Calculates uniform indices according to cadence and adds keyframes defined by Parseq or Deforum prompt."""
        temp_diffusion_frame_count = 1 + int((data.args.anim_args.max_frames - start_index) / data.cadence())
        uniform_indices = KeyFrameDistribution.uniform_indexes(start_index, max_frames, temp_diffusion_frame_count)
        keyframes = KeyFrameDistribution.select_keyframes(data)
        return KeyFrameDistribution._merge_with_uniform(uniform_indices, keyframes)

    @staticmethod
    def _redistributed(data, start_index, max_frames, diffusion_frame_count):
        """Calculates uniform indices according to cadence, but keyframes replace the closest cadence frame."""
        uniform_indices = KeyFrameDistribution.uniform_indexes(start_index, max_frames, diffusion_frame_count)
        keyframes = KeyFrameDistribution.select_keyframes(data)
        keyframes_set = set(uniform_indices)  # set for faster membership checks

        # Insert keyframes from Parseq or Deforum prompt while maintaining keyframe count according to cadence
        for current_frame in keyframes:
            if current_frame not in keyframes_set:
                # Find the closest index in the set to replace (1st and last frame excluded)
                closest_index = min(list(keyframes_set)[1:-1], key=lambda _: abs(_ - current_frame))
                keyframes_set.remove(closest_index)
                keyframes_set.add(current_frame)

        key_frames = list(keyframes_set)
        key_frames.sort()
        assert len(key_frames) == diffusion_frame_count
        return key_frames

    @staticmethod
    def _merge_with_uniform(uniform_indices, key_frames):
        key_frames = list(set(set(uniform_indices) | set(key_frames)))
        key_frames.sort()
        return key_frames

    @staticmethod
    def select_keyframes(data):
        return (KeyFrameDistribution.select_parseq_keyframes(data)
                if data.parseq_adapter.use_parseq
                else KeyFrameDistribution.select_deforum_keyframes(data))

    @staticmethod
    def select_parseq_keyframes(data):
        # Parseq keyframe indices are shifted 1 up before they used.
        keyframes = data.parseq_adapter.parseq_json["keyframes"]
        return list(map(lambda _: _["frame"] + 1, keyframes))

    @staticmethod
    def select_deforum_keyframes(data):
        # Prompt at 0 is always meant to be defined in prompts, but the last frame is not, so we just take max_frames.
        prompt_keyframes = list(map(int, data.args.root.prompt_keyframes))
        last_frame = [data.args.anim_args.max_frames]
        keyframes = list(set(prompt_keyframes + last_frame))
        keyframes.sort()
        keyframes[0] = 1  # Makes sure 1st frame is always 1.

        max_frames = data.args.anim_args.max_frames
        # Filter out frames > max_frames or < 1 and log warning if any are removed.
        original_count = len(keyframes)
        keyframes = list(filter(lambda _: 1 <= _ <= max_frames, keyframes))
        if len(keyframes) < original_count:
            log_utils.warn(f"Removed at least one prompt because its index is not between 0 and {max_frames}. "
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
