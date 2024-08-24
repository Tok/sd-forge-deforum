import random
from enum import Enum
from typing import List

from ...util import log_utils


class KeyFrameDistribution(Enum):
    OFF = "Off"
    PARSEQ_ONLY = "Parseq Only"  # cadence is ignored. all frames not present in the Parseq table are handled as tweens.
    UNIFORM_WITH_PARSEQ = "Uniform with Parseq"  # similar to uniform, but parseq key frame diffusion is enforced.

    @staticmethod
    def from_UI_tab(data):
        redistribution = data.args.parseq_args.parseq_key_frame_redistribution
        match redistribution:
            case "Off":
                return KeyFrameDistribution.OFF
            case "Parseq Only (no cadence)":
                return KeyFrameDistribution.PARSEQ_ONLY
            case "Uniform with Parseq (pseudo-cadence)":
                return KeyFrameDistribution.UNIFORM_WITH_PARSEQ
            case _:
                raise ValueError(f"Invalid parseq_key_frame_redistribution from UI: {redistribution}")

    @staticmethod
    def default():
        return KeyFrameDistribution.OFF

    def calculate(self, key_frames, start_index, max_frames, num_key_steps, parseq_adapter) -> List[int]:
        key_indices: List[int] = self._calculate(start_index, max_frames, num_key_steps, parseq_adapter)
        for i, key_step in enumerate(key_indices):
            key_frames[i].i = key_indices[i]
        return key_frames

    def _calculate(self, start_index, max_frames, num_key_steps, parseq_adapter) -> List[int]:
        match self:
            case KeyFrameDistribution.PARSEQ_ONLY:  # same as UNIFORM_SPACING, if no Parseq keys are present.
                return self._parseq_only_indexes(start_index, max_frames, num_key_steps, parseq_adapter)
            case KeyFrameDistribution.UNIFORM_WITH_PARSEQ:
                return self._uniform_with_parseq_indexes(start_index, max_frames, num_key_steps, parseq_adapter)
            case KeyFrameDistribution.OFF:
                log_utils.warn("Called new core without key frame redistribution. Using 'PARSEQ_ONLY'.")
                return self._parseq_only_indexes(start_index, max_frames, num_key_steps, parseq_adapter)
            case _:
                raise ValueError(f"Invalid KeyFrameDistribution: {self}")

    @staticmethod
    def _uniform_indexes(start_index, max_frames, num_key_steps):
        return [1 + start_index + int(n * (max_frames - 1 - start_index) / (num_key_steps - 1))
                for n in range(num_key_steps)]

    @staticmethod
    def _parseq_only_indexes(start_index, max_frames, num_key_steps, parseq_adapter):
        """Only Parseq key frames are used. Cadence settings are ignored."""
        if not parseq_adapter.use_parseq:
            log_utils.warn("PARSEQ_ONLY, but Parseq is not active, using UNIFORM_SPACING instead.")
            return KeyFrameDistribution._uniform_indexes(start_index, max_frames, num_key_steps)

        parseq_key_frames = [keyframe["frame"] for keyframe in parseq_adapter.parseq_json["keyframes"]]
        shifted_parseq_frames = [frame + 1 for frame in parseq_key_frames]
        return shifted_parseq_frames

    @staticmethod
    def _uniform_with_parseq_indexes(start_index, max_frames, num_key_steps, parseq_adapter):
        """Calculates uniform indices according to cadence, but parseq key frames replace the closest deforum key."""
        uniform_indices = KeyFrameDistribution._uniform_indexes(start_index, max_frames, num_key_steps)
        if not parseq_adapter.use_parseq:
            log_utils.warn("UNIFORM_WITH_PARSEQ, but Parseq is not active, using UNIFORM_SPACING instead.")
            return uniform_indices

        parseq_key_frames = [keyframe["frame"] for keyframe in parseq_adapter.parseq_json["keyframes"]]
        shifted_parseq_frames = [frame + 1 for frame in parseq_key_frames]
        key_frames_set = set(uniform_indices)  # set for faster membership checks

        # Insert parseq keyframes while maintaining keyframe count
        for current_frame in shifted_parseq_frames:
            if current_frame not in key_frames_set:
                # Find the closest index in the set to replace (1st and last frame excluded)
                closest_index = min(list(key_frames_set)[1:-1], key=lambda x: abs(x - current_frame))
                key_frames_set.remove(closest_index)
                key_frames_set.add(current_frame)

        key_frames = list(key_frames_set)
        key_frames.sort()
        assert len(key_frames) == num_key_steps
        return key_frames
