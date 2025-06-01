from dataclasses import dataclass
from typing import Any, Optional

from ...keyframe_animation import DeformAnimKeys


@dataclass
class DiffusionFrameData:
    """Immutable collection of less essential frame data."""
    contrast: float
    amount: float
    kernel: int
    sigma: float
    threshold: float
    cadence_flow_factor: float
    redo_flow_factor: float
    # Note: hybrid_comp_schedules removed
    
    def flow_factor(self):
        # Note: hybrid flow factor functionality removed
        return 1.0

    @staticmethod
    def create(data: RenderData, i):
        keys = data.keys
        return DiffusionFrameData(
            contrast=keys.contrast_schedule_series[i],
            amount=keys.amount_schedule_series[i],
            kernel=int(keys.kernel_schedule_series[i]),
            sigma=keys.sigma_schedule_series[i],
            threshold=keys.threshold_schedule_series[i],
            cadence_flow_factor=keys.cadence_flow_factor_schedule_series[i],
            redo_flow_factor=keys.redo_flow_factor_schedule_series[i]
            # Note: hybrid schedules removed
        )
