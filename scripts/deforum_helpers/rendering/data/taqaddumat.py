import gc

# noinspection PyUnresolvedReferences
import modules.shared as shared
from tqdm import tqdm

from ..util import log_utils
from ..util.log_utils import HEX_BLUE, HEX_GREEN, HEX_ORANGE, HEX_RED, HEX_PURPLE


class Taqaddumat:
    # Only used by experimental core. Stable core provides its own TQDM in deforum_tqdm.py
    NO_ETA_RBAR = "| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
    NO_ETA_BAR_FORMAT = "{l_bar}{bar}" + f"{NO_ETA_RBAR}"
    DEFAULT_BAR_FORMAT = "{l_bar}{bar}{r_bar}"  # see 'bar_format' at https://tqdm.github.io/docs/tqdm/

    def __init__(self):
        self.tweens = None
        self.total_frames = None
        self.steps = None
        self.total_steps = None
        self.total_animation_cycles = None

    def reset(self, data, frames):
        def create(iterable, position, color, description, unit, bar_format=Taqaddumat.NO_ETA_BAR_FORMAT):
            return tqdm(iterable, position=position, desc=description, unit=unit, dynamic_ncols=True,
                        file=shared.progress_print_out, bar_format=bar_format,
                        disable=shared.cmd_opts.disable_console_progressbars, colour=color)

        # Positions greater than 0 are assigned where bars are meant to show up directly after each other and
        # need to be updated at the same time. 'Tweens' is paired with 'Total Frames' and 'Steps' with 'Total Steps'.
        # Those are intended to be updated together and always be displayed at the same time.
        # The global step counter tqdm provided by Forge keeps reattaching itself after the highest position used here.
        # This dictates how much space the total tqdm will take up, even when positions are reassigned here later.

        second_frame_tweens_count = len(frames[1].tweens)  # DF 0 can not have Tweens, so we initialize at 1.
        self.tweens = create(
            range(second_frame_tweens_count), 0, HEX_BLUE,
            "Current Tweens", "tween")

        total_frames = sum(len(frame.tweens) for frame in frames)
        self.total_frames = create(
            range(total_frames), 1, HEX_GREEN,
            "Total Frames", "frame")

        first_frame_steps_count = frames[0].schedule.steps
        self.steps = create(
            range(first_frame_steps_count), 0, HEX_ORANGE,
            "Current Diffusion Steps", "step")

        total_steps = sum(frame.actual_steps(data) for frame in frames)
        self.total_steps = create(
            range(total_steps), 1, HEX_RED,
            "Total Diffusion Steps", "step")

        self.total_animation_cycles = create(
            frames, 0, HEX_PURPLE,
            "Total Animation Cycles", "cycle",
            Taqaddumat.DEFAULT_BAR_FORMAT)

        self.clear_all()

    def update(self):
        # CombinedTQDM is assigned to 'shared.total_tqdm', causing this method to be called from Forge once every step.
        # See 'shared.total_tqdm.update()' call in 'Sampler' callback at <sd-webui>/modules/sd_samplers_common.py
        self.increment_step_count()

    def updateTotal(self, total_steps):
        # May be called from '<sd-webui>/modules/processing.py' in some circumstances when using txt2img pipes,
        # but doesn't seem to be currently relevant for Deforum (untested, use "update" instead).
        self.increment_step_count()

    def increment_tween_count(self):
        self.tweens.update()
        self.tweens.refresh()
        self.total_frames.update()
        self.total_frames.refresh()
        if Taqaddumat.is_last_iteration(self.tweens):
            print("\n")

    def increment_step_count(self):
        if self.steps.n == 0:
            Taqaddumat.disable_all_tqdm_without_a_description()
            log_utils.clear_next_n_lines(2)  # depends on cursor resting at the end of the line with the 0th tqdm.
        self.steps.update()
        self.steps.refresh()
        self.total_steps.update()
        self.total_steps.refresh()
        if Taqaddumat.is_last_iteration(self.steps):
            print("\n")

    def increment_animation_cycle_count(self):
        # Calls to tqdm.update() without an argument increment it by 1.
        self.total_animation_cycles.update()
        self.total_animation_cycles.refresh()
        print("")

    def reset_tween_count(self, n):
        if n == 0:
            return
        self.tweens.reset()
        self.tweens.clear()
        self.tweens.total = n

    def reset_step_count(self, n):
        self.steps.reset()
        self.steps.clear()
        self.steps.total = n

    def clear_all(self):
        self.tweens.clear()
        self.steps.clear()
        self.total_steps.clear()
        self.total_frames.clear()
        self.total_animation_cycles.clear()
        print("\n\n\n\n")

    @staticmethod
    def is_last_iteration(taqaddum):
        return taqaddum.n == taqaddum.total

    @staticmethod
    def disable_all_tqdm_without_a_description():
        # Forge provides its own step counter which inserts itself at the next position after the bars defined here.
        # It basically shows the same info as the per-frame step counter defined here, but it's unlabeled
        # and has a plain ASCII progress bar with a different style and format.
        # TODO try to work around it by forcing shared.cmd_opts.disable_console_progressbars True after 1st reset here?
        def mute(tq):  # FIXME doesn't really work as intended.
            tq.position = 0  # global counter just reinserts itself after the last position
            tq.leave = False
            tq.disable = True
            tq.bar_format = '{l_bar:0}{bar:0}{r_bar:0}'  # another attempt at shutting the bar...
            tq.clear()
            tq.close()

        # Since Forge step counter is continuously reinstantiated and not exposed in 'modules.shared',
        # we just use the GC to access and continuously mute all taqaddumat that don't have any description.
        # TODO? find a way to do this without accessing the GC.
        list(map(lambda _: mute(_),
                 filter(lambda _: not _.desc,
                        filter(lambda _: isinstance(_, tqdm),
                               gc.get_objects()))))
