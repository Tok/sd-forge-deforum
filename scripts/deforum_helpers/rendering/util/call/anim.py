from ....animation import anim_frame_warp


def call_anim_frame_warp(data, i, image, depth):
    ia = data.args
    return anim_frame_warp(image, ia.args, ia.anim_args, data.animation_keys.deform_keys, i, data.depth_model,
                           depth=depth, device=ia.root.device, half_precision=ia.root.half_precision)
