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
        redistribution = data.args.parseq_args.keyframe_redistribution
        match redistribution:
            case "Off":
                return KeyFrameDistribution.OFF
            case "Keyframes Only":
                return KeyFrameDistribution.KEYFRAMES_ONLY
            case "Additive":
                return KeyFrameDistribution.ADDITIVE
            case "Redistributed":
                return KeyFrameDistribution.REDISTRIBUTED
            case _:
                raise ValueError(f"Invalid keyframe_redistribution from UI: {redistribution}")

    @staticmethod
    def default():
        return KeyFrameDistribution.OFF

    def calculate(self, data, start_index, max_frames, num_key_steps) -> List[int]:
        match self:
            case KeyFrameDistribution.KEYFRAMES_ONLY:  # same as UNIFORM_SPACING, if no Parseq keys are present.
                return self.select_keyframes(data)
            case KeyFrameDistribution.ADDITIVE:
                return self._additive(data, start_index, max_frames)
            case KeyFrameDistribution.REDISTRIBUTED:
                return self._redistributed(data, start_index, max_frames, num_key_steps)
            case KeyFrameDistribution.OFF:
                log_utils.warn("Called new core without keyframe redistribution. Using 'KEYFRAMES_ONLY'.")
                return self.select_keyframes(data)
            case _:
                raise ValueError(f"Invalid KeyFrameDistribution: {self}")

    @staticmethod
    def uniform_indexes(start_index, max_frames, num_key_steps):
        return [1 + start_index + int(n * (max_frames - 1 - start_index) / (num_key_steps - 1))
                for n in range(num_key_steps)]

    @staticmethod
    def _additive(data, start_index, max_frames):
        """Calculates uniform indices according to cadence and adds keyframes defined by Parseq or Deforum prompt."""
        temp_num_key_steps = 1 + int((data.args.anim_args.max_frames - start_index) / data.cadence())
        uniform_indices = KeyFrameDistribution.uniform_indexes(start_index, max_frames, temp_num_key_steps)
        keyframes = KeyFrameDistribution.select_keyframes(data)
        return KeyFrameDistribution._merge_with_uniform(uniform_indices, keyframes)

    @staticmethod
    def _redistributed(data, start_index, max_frames, num_key_steps):
        """Calculates uniform indices according to cadence, but keyframes replace the closest cadence frame."""
        uniform_indices = KeyFrameDistribution.uniform_indexes(start_index, max_frames, num_key_steps)
        keyframes = KeyFrameDistribution.select_keyframes(data)
        keyframes_set = set(uniform_indices)  # set for faster membership checks

        # Insert keyframes from Parseq or Deforum prompt while maintaining keyframe count according to cadence
        for current_frame in keyframes:
            if current_frame not in keyframes_set:
                # Find the closest index in the set to replace (1st and last frame excluded)
                closest_index = min(list(keyframes_set)[1:-1], key=lambda x: abs(x - current_frame))
                keyframes_set.remove(closest_index)
                keyframes_set.add(current_frame)

        key_frames = list(keyframes_set)
        key_frames.sort()
        assert len(key_frames) == num_key_steps
        return key_frames

    @staticmethod
    def _merge_with_uniform(uniform_indices, key_frames):
        key_frames = list(set(set(uniform_indices) | set(key_frames)))
        key_frames.sort()
        return key_frames

    @staticmethod
    def select_keyframes(data):
        return KeyFrameDistribution._select_parseq_keyframes(data) if data.parseq_adapter.use_parseq \
            else KeyFrameDistribution._select_deforum_keyframes(data)

    @staticmethod
    def _select_parseq_keyframes(data):
        return [keyframe["frame"] + 1 for keyframe in data.parseq_adapter.parseq_json["keyframes"]]

    @staticmethod
    def _select_deforum_keyframes(data):
        return [int(frame) for frame in data.args.root.prompt_keyframes]
