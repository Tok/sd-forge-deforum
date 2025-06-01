from dataclasses import dataclass

from ....animation_key_frames import DeformAnimKeys, LooperAnimKeys


@dataclass(init=True, frozen=True, repr=False, eq=False)
class AnimationKeys:
    deform_keys: DeformAnimKeys
    looper_keys: LooperAnimKeys

    @staticmethod
    def create(step_args, parseq_adapter, seed):
        is_use_parseq = parseq_adapter.use_parseq

        def _choose_keys(default_keys, parseq_keys):
            return parseq_keys if is_use_parseq else default_keys

        # Creating AnimKeys may be expensive operations,
        # as it includes parsing all the in-between values.
        dak = (DeformAnimKeys(step_args.anim_args, seed)
               if not is_use_parseq else None)
        lak = (LooperAnimKeys(step_args.loop_args, step_args.anim_args, seed)
               if not is_use_parseq else None)

        return AnimationKeys(
            deform_keys=_choose_keys(dak, parseq_adapter.anim_keys),
            looper_keys=_choose_keys(lak, parseq_adapter.anim_keys))
