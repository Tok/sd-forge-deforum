from dataclasses import dataclass
from typing import Optional, Any

from .render_data import RenderData
from ...args import DeforumAnimArgs, DeforumArgs


@dataclass(init=True, frozen=True, repr=False, eq=False)
class Schedule:
    seed: int
    steps: int
    sampler_name: str
    scheduler_name: str
    clipskip: int
    noise_multiplier: float
    eta_ddim: float
    eta_ancestral: float
    mask: Optional[Any]
    noise_mask: Optional[Any]

    @staticmethod
    def create(data: RenderData, seed, i):
        """Create a new Schedule instance based on the provided parameters."""
        args: DeforumArgs = data.args.args
        anim_args: DeforumAnimArgs = data.args.anim_args
        keys = data.animation_keys.deform_keys
        (steps, sampler_name, schedule_name, clipskip, noise_multiplier,
         eta_ddim, eta_ancestral, mask, noise_mask) = (
            Schedule.schedule_steps(keys, i, anim_args, args),
            Schedule.schedule_sampler(keys, i, anim_args),
            Schedule.schedule_scheduler(keys, i, anim_args),
            Schedule.schedule_clipskip(keys, i, anim_args),
            Schedule.schedule_noise_multiplier(keys, i, anim_args),
            Schedule.schedule_ddim_eta(keys, i, anim_args),
            Schedule.schedule_ancestral_eta(keys, i, anim_args),
            Schedule.schedule_mask(keys, i, args),
            (Schedule.schedule_noise_mask(keys, i, anim_args)
             if data.is_use_mask and not data.args.anim_args.use_noise_mask
             else None))
        return Schedule(seed, steps, sampler_name, schedule_name, clipskip, noise_multiplier,
                        eta_ddim, eta_ancestral, mask, noise_mask)

    @staticmethod
    def _has_schedule(keys, i):
        # Ensure i is within bounds of the series
        if i >= len(keys.steps_schedule_series):
            return False
        return keys.steps_schedule_series[i] is not None

    @staticmethod
    def _has_mask_schedule(keys, i):
        # Ensure i is within bounds of the series
        if i >= len(keys.mask_schedule_series):
            return False
        return keys.mask_schedule_series[i] is not None

    @staticmethod
    def _has_noise_mask_schedule(keys, i):
        # Ensure i is within bounds of the series
        if i >= len(keys.noise_mask_schedule_series):
            return False
        return keys.noise_mask_schedule_series[i] is not None

    @staticmethod
    def _use_on_cond_if_scheduled(keys, i, value, cond):
        return value if cond and Schedule._has_schedule(keys, i) else None

    @staticmethod
    def schedule_steps(keys, i, anim_args, deforum_args):
        # Ensure i is within bounds of the series
        if i >= len(keys.steps_schedule_series):
            return deforum_args.steps
        
        steps = Schedule._use_on_cond_if_scheduled(keys, i, int(keys.steps_schedule_series[i]),
                                                   anim_args.enable_steps_scheduling)
        if steps is None:
            steps = deforum_args.steps
        return steps

    @staticmethod
    def schedule_sampler(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, keys.sampler_schedule_series[i].casefold(),
                                                  anim_args.enable_sampler_scheduling)

    @staticmethod
    def schedule_scheduler(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, keys.scheduler_schedule_series[i].casefold(),
                                                  anim_args.enable_scheduler_scheduling)

    @staticmethod
    def schedule_clipskip(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, int(keys.clipskip_schedule_series[i]),
                                                  anim_args.enable_clipskip_scheduling)

    @staticmethod
    def schedule_noise_multiplier(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, float(keys.noise_multiplier_schedule_series[i]),
                                                  anim_args.enable_noise_multiplier_scheduling)

    @staticmethod
    def schedule_ddim_eta(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, float(keys.ddim_eta_schedule_series[i]),
                                                  anim_args.enable_ddim_eta_scheduling)

    @staticmethod
    def schedule_ancestral_eta(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, float(keys.ancestral_eta_schedule_series[i]),
                                                  anim_args.enable_ancestral_eta_scheduling)

    @staticmethod
    def schedule_mask(keys, i, args):
        return keys.mask_schedule_series[i] \
            if args.use_mask and Schedule._has_mask_schedule(keys, i) else None

    @staticmethod
    def schedule_noise_mask(keys, i, anim_args):
        return keys.noise_mask_schedule_series[i] \
            if anim_args.use_noise_mask and Schedule._has_noise_mask_schedule(keys, i) else None
