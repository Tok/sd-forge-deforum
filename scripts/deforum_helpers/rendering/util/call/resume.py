from ....resume import get_resume_vars


def call_get_resume_vars(data, turbo):
    return get_resume_vars(folder=data.args.args.outdir,
                           timestring=data.args.anim_args.resume_timestring,
                           cadence=turbo.cadence)
