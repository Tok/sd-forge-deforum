from dataclasses import dataclass

from ....animation_key_frames import DeformAnimKeys, LooperAnimKeys


@dataclass(init=True, frozen=True, repr=False, eq=False)
class AnimationKeys:
    deform_keys: DeformAnimKeys
    looper_keys: LooperAnimKeys

    @staticmethod
    def _choose_default_or_parseq_keys(default_keys, parseq_keys, parseq_adapter):
        return default_keys if not parseq_adapter.use_parseq else parseq_keys

    @staticmethod
    def from_args(step_args, parseq_adapter, seed):
        ada = parseq_adapter

        def _choose(default_keys):
            return AnimationKeys._choose_default_or_parseq_keys(default_keys, ada.anim_keys, ada)

        # Parseq keys are decorated, see ParseqAnimKeysDecorator and ParseqLooperKeysDecorator
        return AnimationKeys(_choose(DeformAnimKeys(step_args.anim_args, seed)),
                             _choose(LooperAnimKeys(step_args.loop_args, step_args.anim_args, seed)))
