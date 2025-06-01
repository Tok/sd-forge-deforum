import os

from . import filename_utils, memory_utils
from ..depth import DepthModel


def is_depth_required_for_hybrid_composite_mask(anim_args):
    # Note: hybrid composite functionality removed
    return False


def generate_and_save_depth_map_if_active(data, opencv_image, i):
    if data.args.anim_args.save_depth_maps:
        memory_utils.handle_vram_before_depth_map_generation(data)
        depth = data.depth_model.predict(opencv_image, data.args.anim_args.midas_weight,
                                         data.args.root.half_precision)
        depth_filename = filename_utils.depth_frame(data, i)
        data.depth_model.save(os.path.join(data.output_directory, depth_filename), depth)
        memory_utils.handle_vram_after_depth_map_generation(data)
        return depth
    return None


def create_depth_model_and_enable_depth_map_saving_if_active(anim_mode, root, anim_args, args):
    # Note: hybrid composite functionality removed
    return (DepthModel(root.models_path,
                       memory_utils.select_depth_device(root),
                       root.half_precision,
                       keep_in_vram=anim_mode.is_keep_in_vram,
                       depth_algorithm=anim_args.depth_algorithm,
                       Width=args.W, Height=args.H,
                       midas_weight=anim_args.midas_weight)
            if anim_mode.is_predicting_depths else None)
