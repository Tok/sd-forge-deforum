import os

from deforum.rendering.helpers import filename as filename_utils
from deforum.rendering.helpers import memory as memory_utils
from deforum.depth import DepthModel
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


def generate_and_save_depth_map_if_active(data, opencv_image, i):
    # Always save depth maps in 3D mode for preview, regardless of save_depth_maps setting
    # They will be cleaned up later if user doesn't want to keep them
    if data.depth_model is not None:
        memory_utils.handle_vram_before_depth_map_generation(data)
        depth = data.depth_model.predict(
            opencv_image,
            use_ray_pose=data.args.anim_args.da3_use_ray_pose,
            conf_thresh_percentile=data.args.anim_args.da3_conf_thresh_percentile
        )
        # Ensure depth-maps subdirectory exists
        depth_dir = os.path.join(data.output_directory, "depth-maps")
        os.makedirs(depth_dir, exist_ok=True)

        depth_filename = filename_utils.depth_frame(data, i)
        data.depth_model.save(os.path.join(data.output_directory, depth_filename), depth)
        memory_utils.handle_vram_after_depth_map_generation(data)
        return depth


def create_depth_model_and_enable_depth_map_saving_if_active(anim_mode, root, anim_args, args):
    """Create depth model with automatic DA3 upgrade for Gaussian Scene mode."""
    # Don't override user's save_depth_maps setting - we handle saving and cleanup separately
    if not anim_mode.is_predicting_depths:
        return None

    # Auto-override to DA3 for Gaussian Scene mode
    depth_algorithm = anim_args.depth_algorithm
    render_mode = getattr(args, 'render_mode', 'New 3D')
    tween_mode = getattr(anim_args, 'tween_generation_mode', 'depth_warp')

    # Check if Gaussian Scene mode or da3_gaussian tween mode requires DA3
    needs_da3 = (render_mode == 'Gaussian Scene' or tween_mode == 'da3_gaussian')
    is_da2 = 'v2' in depth_algorithm.lower()

    if needs_da3 and is_da2:
        # Auto-upgrade to DA3 AnyView (keep same size as user selected)
        original_algorithm = depth_algorithm
        # Extract size from original (Small/Base/Large)
        size = 'Small'  # Default
        for s in ['Small', 'Base', 'Large']:
            if s in depth_algorithm:
                size = s
                break
        depth_algorithm = f'Depth-Anything-V3-AnyView-{size}'
        logger.warning(
            f"Gaussian Scene/da3_gaussian mode requires Depth Anything V3 AnyView. "
            f"Auto-upgrading from '{original_algorithm}' to '{depth_algorithm}'"
        )

    return DepthModel(
        root.models_path,
        memory_utils.select_depth_device(root),
        root.half_precision,
        keep_in_vram=anim_mode.is_keep_in_vram,
        depth_algorithm=depth_algorithm,
        Width=args.W,
        Height=args.H
    )
