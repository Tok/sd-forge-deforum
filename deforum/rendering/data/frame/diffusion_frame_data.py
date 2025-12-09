from dataclasses import dataclass
from typing import Any

from deforum.core.keyframes import DeformAnimKeys


@dataclass(init=True, frozen=True, repr=False, eq=False)
class DiffusionFrameData:
    noise: Any = None
    cfg_scale: Any = None  # defaults to 1.0 for Flux.1, but is typically used at 5.0-15.0 for other models
    distilled_cfg_scale: Any = None  # defaults to 3.5 for Flux.1, may be ignored for other models
    shift: Any = None  # defaults to 3.0 for Z-Image, controls FlowMatch timestep scaling
    contrast: Any = None
    kernel: int = 0
    sigma: Any = None
    amount: Any = None
    threshold: Any = None
    cadence_flow_factor: Any = None
    redo_flow_factor: Any = None

    def kernel_size(self) -> tuple[int, int]:
        return self.kernel, self.kernel

    @staticmethod
    def create(data, i):
        keys: DeformAnimKeys = data.animation_keys.deform_keys
        return DiffusionFrameData(
            keys.noise_schedule_series[i],
            keys.cfg_scale_schedule_series[i],
            keys.distilled_cfg_scale_schedule_series[i],
            keys.shift_schedule_series[i],
            keys.contrast_schedule_series[i],
            int(keys.kernel_schedule_series[i]),
            keys.sigma_schedule_series[i],
            keys.amount_schedule_series[i],
            keys.threshold_schedule_series[i],
            keys.cadence_flow_factor_schedule_series[i],
            keys.redo_flow_factor_schedule_series[i])
