from ....generate import generate


def call_generate(data, frame):
    # TODO rename things, data.args.args.strength is actually "denoise", so strength is subtracted from 1.0 when passed.
    data.args.args.strength = 1.0 - frame.strength  # update denoise for current diffusion from pre-generated frame
    ia = data.args
    return generate(ia.args, data.animation_keys.deform_keys, ia.anim_args, ia.loop_args, ia.controlnet_args,
                    ia.freeu_args, ia.kohya_hrfix_args, ia.root, data.parseq_adapter, data.indexes.frame.i,
                    sampler_name=frame.schedule.sampler_name, scheduler_name=frame.schedule.scheduler_name)
