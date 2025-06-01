import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ....utils import opt_utils
from ....depth import DepthModel
from ....integrations.raft.raft_integration import RAFT
# Note: hybrid video imports removed - functionality not available


@dataclass(init=True, frozen=False, repr=False, eq=False)
class AnimationMode:
    has_video_input: bool = False
    # Note: hybrid functionality removed
    prev_flow: Any | None = None
    is_keep_in_vram: bool = False
    depth_model: Any = None
    raft_model: Any = None

    def is_predicting_depths(self) -> bool:
        return self.depth_model is not None

    def is_raft_active(self) -> bool:
        return self.raft_model is not None

    def unload_raft_and_depth_model(self):
        if self.is_predicting_depths() and not self.is_keep_in_vram:
            self.depth_model.delete_model()  # handles adabins too
        if self.is_raft_active():
            self.raft_model.delete_model()

    @staticmethod
    def _has_video_input(anim_args) -> bool:
        return anim_args.animation_mode == 'Video Input'

    @staticmethod
    def _is_2d_or_3d_mode(anim_args):
        return anim_args.animation_mode in ['2D', '3D']

    @staticmethod
    def _is_load_depth_model_for_3d(args, anim_args):
        is_depth_warped_3d = anim_args.animation_mode == '3D' and anim_args.use_depth_warping
        is_depth_used = is_depth_warped_3d or anim_args.save_depth_maps
        return is_depth_used and not args.motion_preview_mode

    @staticmethod
    def load_raft_if_active(anim_args, args):
        is_cadenced_raft = anim_args.optical_flow_cadence == "RAFT" and int(anim_args.diffusion_cadence) > 1
        is_raft_redo = anim_args.optical_flow_redo_generation == "RAFT"
        is_load_raft = (is_cadenced_raft or is_raft_redo) and not args.motion_preview_mode
        if is_load_raft:
            print("Loading RAFT model...")
        return RAFT() if is_load_raft else None

    @staticmethod
    def load_depth_model_if_active(args, anim_args):
        return AnimationMode._is_load_depth_model_for_3d(args, anim_args) \
            if opt_utils.keep_3d_models_in_vram() else None

    @staticmethod
    def create(step_args):
        sa = step_args  # RenderInitArgs
        # Note: hybrid functionality removed
        previous_flow = None
        return AnimationMode(
            AnimationMode._has_video_input(sa.anim_args),
            previous_flow,
            opt_utils.keep_3d_models_in_vram(),
            AnimationMode.load_depth_model_if_active(sa.args, sa.anim_args),
            AnimationMode.load_raft_if_active(sa.anim_args, sa.args))
