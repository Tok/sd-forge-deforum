from dataclasses import dataclass
from typing import Any

from ....animation_key_frames import DeformAnimKeys


@dataclass(init=True, frozen=True, repr=False, eq=False)
class KeyFrameData:
    noise: Any = None
    strength: Any = None
    scale: Any = None
    contrast: Any = None
    kernel: int = 0
    sigma: Any = None
    amount: Any = None
    threshold: Any = None
    cadence_flow_factor: Any = None
    redo_flow_factor: Any = None
    hybrid_comp_schedules: Any = None

    def kernel_size(self) -> tuple[int, int]:
        return self.kernel, self.kernel

    def flow_factor(self):
        return self.hybrid_comp_schedules['flow_factor']

    def has_strength(self):
        return self.strength > 0

    @staticmethod
    def create(data):
        i = data.indexes.frame.i
        keys: DeformAnimKeys = data.animation_keys.deform_keys
        return KeyFrameData(
            keys.noise_schedule_series[i],
            keys.strength_schedule_series[i],
            keys.cfg_scale_schedule_series[i],
            keys.contrast_schedule_series[i],
            int(keys.kernel_schedule_series[i]),
            keys.sigma_schedule_series[i],
            keys.amount_schedule_series[i],
            keys.threshold_schedule_series[i],
            keys.cadence_flow_factor_schedule_series[i],
            keys.redo_flow_factor_schedule_series[i],
            KeyFrameData._hybrid_comp_args(keys, i))

    @staticmethod
    def _hybrid_comp_args(keys, i):
        return {
            "alpha": keys.hybrid_comp_alpha_schedule_series[i],
            "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[i],
            "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[i],
            "mask_auto_contrast_cutoff_low": int(keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[i]),
            "mask_auto_contrast_cutoff_high": int(keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[i]),
            "flow_factor": keys.hybrid_flow_factor_schedule_series[i]}
