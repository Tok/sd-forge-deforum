from ...generate import generate


def call_generate(data, frame: 'DiffusionFrame', redo_seed: int = None):
    # TODO rename things, data.args.args.strength is actually "denoise", so strength is subtracted from 1.0 when passed.
    ia = data.args
    ia.args.strength = 1.0 - frame.strength  # update denoise for current diffusion from pre-generated frame
    ia.args.seed = frame.seed if redo_seed is None else redo_seed  # update seed with precalculated value from frame
    ia.root.subseed = frame.subseed
    ia.root.subseed_strength = frame.subseed_strength
    index = frame.i - 1
    return generate(ia.args, data.animation_keys.deform_keys, ia.anim_args, ia.loop_args, ia.controlnet_args,
                    ia.freeu_args, ia.kohya_hrfix_args, ia.root, data.parseq_adapter, index,
                    sampler_name=frame.schedule.sampler_name, scheduler_name=frame.schedule.scheduler_name)


def generate_frame(data, frame_idx):
    """
    Generate a frame for the experimental core.
    This is a simplified wrapper around the main generation pipeline.
    """
    # For experimental core, we need to implement frame generation
    # This is a placeholder that would need proper frame object creation
    # For now, we'll use the basic generate function with minimal parameters
    
    # TODO: Implement proper frame object creation and management
    # This is experimental code and needs proper DiffusionFrame object
    import warnings
    warnings.warn("generate_frame is experimental and not fully implemented")
    
    ia = data.args
    return generate(
        ia.args, 
        data.animation_keys.deform_keys if hasattr(data, 'animation_keys') else None,
        ia.anim_args, 
        ia.loop_args, 
        ia.controlnet_args,
        ia.freeu_args if hasattr(ia, 'freeu_args') else None,
        ia.kohya_hrfix_args if hasattr(ia, 'kohya_hrfix_args') else None,
        ia.root, 
        data.parseq_adapter if hasattr(data, 'parseq_adapter') else None,
        frame_idx
    )
