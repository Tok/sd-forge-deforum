from ....hybrid_video import (
    # Functions related to flow calculation
    get_flow_for_hybrid_motion,
    get_flow_for_hybrid_motion_prev,

    # Functions related to matrix calculation
    get_matrix_for_hybrid_motion,
    get_matrix_for_hybrid_motion_prev,

    # Other hybrid functions
    hybrid_composite)


def call_get_flow_for_hybrid_motion_prev(data, i, image):
    mode = data.animation_mode
    aa = data.args.anim_args
    return get_flow_for_hybrid_motion_prev(
        i, data.dimensions(),
        mode.hybrid_input_files,
        mode.hybrid_frame_path,
        mode.prev_flow,
        image,
        aa.hybrid_flow_method,
        mode.raft_model,
        aa.hybrid_flow_consistency,
        aa.hybrid_consistency_blur,
        aa.hybrid_comp_save_extra_frames)


def call_get_flow_for_hybrid_motion(data, i):
    mode = data.animation_mode
    args = data.args.anim_args
    return get_flow_for_hybrid_motion(
        i, data.dimensions(), mode.hybrid_input_files, mode.hybrid_frame_path,
        mode.prev_flow, args.hybrid_flow_method, mode.raft_model,
        args.hybrid_flow_consistency, args.hybrid_consistency_blur, args)


def call_get_matrix_for_hybrid_motion_prev(data, i, image):
    return get_matrix_for_hybrid_motion_prev(
        i, data.dimensions(), data.animation_mode.hybrid_input_files,
        image, data.args.anim_args.hybrid_motion)


def call_get_matrix_for_hybrid_motion(data, i):
    return get_matrix_for_hybrid_motion(
        i, data.dimensions(), data.animation_mode.hybrid_input_files,
        data.args.anim_args.hybrid_motion)


def call_hybrid_composite(data, i, image, hybrid_comp_schedules):
    ia = data.args
    return hybrid_composite(
        ia.args, ia.anim_args, i, image,
        data.depth_model, hybrid_comp_schedules, data.args.root)
