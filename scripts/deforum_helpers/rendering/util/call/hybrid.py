from ....hybrid_video import (
    # Functions related to flow calculation
    get_flow_for_hybrid_motion,
    get_flow_for_hybrid_motion_prev,

    # Functions related to matrix calculation
    get_matrix_for_hybrid_motion,
    get_matrix_for_hybrid_motion_prev,

    # Other hybrid functions
    hybrid_composite)


def call_get_flow_for_hybrid_motion_prev(init, i, image):
    mode = init.animation_mode
    aa = init.args.anim_args
    return get_flow_for_hybrid_motion_prev(
        i, init.dimensions(),
        mode.hybrid_input_files,
        mode.hybrid_frame_path,
        mode.prev_flow,
        image,
        aa.hybrid_flow_method,
        mode.raft_model,
        aa.hybrid_flow_consistency,
        aa.hybrid_consistency_blur,
        aa.hybrid_comp_save_extra_frames)


def call_get_flow_for_hybrid_motion(init, i):
    mode = init.animation_mode
    args = init.args.anim_args
    return get_flow_for_hybrid_motion(
        i, init.dimensions(), mode.hybrid_input_files, mode.hybrid_frame_path,
        mode.prev_flow, args.hybrid_flow_method, mode.raft_model,
        args.hybrid_flow_consistency, args.hybrid_consistency_blur, args)


def call_get_matrix_for_hybrid_motion_prev(init, i, image):
    return get_matrix_for_hybrid_motion_prev(
        i, init.dimensions(), init.animation_mode.hybrid_input_files,
        image, init.args.anim_args.hybrid_motion)


def call_get_matrix_for_hybrid_motion(init, i):
    return get_matrix_for_hybrid_motion(
        i, init.dimensions(), init.animation_mode.hybrid_input_files,
        init.args.anim_args.hybrid_motion)


def call_hybrid_composite(init, i, image, hybrid_comp_schedules):
    ia = init.args
    return hybrid_composite(
        ia.args, ia.anim_args, i, image,
        init.depth_model, hybrid_comp_schedules, init.args.root)
