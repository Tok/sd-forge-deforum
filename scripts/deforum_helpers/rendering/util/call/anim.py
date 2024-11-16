from ....animation import anim_frame_warp


def call_anim_frame_warp(data, i, image, depth_prediction):
    ia = data.args
    # If depth_prediction isn't provided, anim_frame_warp will generate, use and return a new one.
    return anim_frame_warp(image, ia.args, ia.anim_args, data.animation_keys.deform_keys, i,
                           data.depth_model, depth=depth_prediction, device=ia.root.device,
                           half_precision=ia.root.half_precision, shaker=data.shaker)
