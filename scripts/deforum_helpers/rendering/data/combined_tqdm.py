import gc

# noinspection PyUnresolvedReferences
import modules.shared as shared
from tqdm import tqdm

from ..util import log_utils
from ..util.log_utils import HEX_RED, HEX_ORANGE, HEX_GREEN, HEX_BLUE, HEX_PURPLE


class CombinedTqdm:
    # Only used by experimental core. Stable core provides its own TQDM in deforum_tqdm.py
    R_BAR = '{desc:<16}: {percentage:>07.3f}%'
    L_BAR = '{n_fmt:>05}/{total_fmt:>05} [{elapsed:>8}<{remaining:>8}, {rate_fmt:>18}{postfix}]'
    BAR = ' |{bar}| '
    FIXED_FORMAT = f"{R_BAR}{BAR}{L_BAR}"  # see 'bar_format' at https://tqdm.github.io/docs/tqdm/
    DEFAULT_FORMAT = "{l_bar}{bar}{r_bar}"

    def __init__(self):
        self.tweens_tqdm = None
        self.steps_tqdm = None
        self.total_steps_tqdm = None
        self.total_frames_tqdm = None
        self.total_diffusions_tqdm = None

    def reset(self, diffusion_frames):
        out = shared.progress_print_out
        is_disable_bars = shared.cmd_opts.disable_console_progressbars
        bar_form = CombinedTqdm.DEFAULT_FORMAT

        # Positions greater than 0 are assigned where bars are meant to show up directly after each other and
        # need to be updated at the same time. 'Tweens' is paired with 'Total Frames' and 'Steps' with 'Total Steps'.
        # Those are intended to be updated together and always be displayed at the same time.
        # The global step counter tqdm provided by Forge keeps reattaching itself after the highest position used here.
        # This dictates how much space the total tqdm will take up, even when positions are reassigned here later.

        second_frame_tweens_count = len(diffusion_frames[1].tweens)  # DF 0 can not have Tweens, so we start at 1.
        self.tweens_tqdm = tqdm(range(second_frame_tweens_count), desc="Tweens", position=0,
                                file=out, unit="tween", bar_format=bar_form, dynamic_ncols=True,
                                disable=is_disable_bars, colour=HEX_BLUE)

        total_frames = sum(len(frame.tweens) for frame in diffusion_frames)
        self.total_frames_tqdm = tqdm(range(total_frames), desc="Total Frames", position=1,
                                      file=out, unit="frame", bar_format=bar_form, dynamic_ncols=True,
                                      disable=is_disable_bars, colour=HEX_GREEN)

        first_frame_steps_count = diffusion_frames[0].schedule.steps
        self.steps_tqdm = tqdm(range(first_frame_steps_count), desc="Steps", position=0,
                               file=out, unit="step", bar_format=bar_form, dynamic_ncols=True,
                               disable=is_disable_bars, colour=HEX_ORANGE)

        total_steps = sum(frame.actual_steps() for frame in diffusion_frames)
        self.total_steps_tqdm = tqdm(range(total_steps), desc="Total Steps", position=1,
                                     file=out, unit="step", bar_format=bar_form, dynamic_ncols=True,
                                     disable=is_disable_bars, colour=HEX_RED)

        self.total_diffusions_tqdm = tqdm(diffusion_frames, desc="Total Diffusions", position=0,
                                          file=out, unit="diffusion", bar_format=bar_form, dynamic_ncols=True,
                                          disable=is_disable_bars, colour=HEX_PURPLE)

        shared.total_tqdm.clear_all()

    def update(self):
        # CombinedTQDM is assigned to 'shared.total_tqdm', causing this method to be called from Forge once every step.
        # See 'shared.total_tqdm.update()' call in 'Sampler' callback at <sd-webui>/modules/sd_samplers_common.py
        self.increment_step_count()

    def updateTotal(self, total_steps):
        # May be called from '<sd-webui>/modules/processing.py' in some circumstances when using txt2img pipes,
        # but doesn't seem to be currently relevant for Deforum (untested, use "update" instead).
        self.increment_step_count()

    def increment_tween_count(self):
        self.tweens_tqdm.update()
        self.tweens_tqdm.refresh()
        self.total_frames_tqdm.update()
        self.total_frames_tqdm.refresh()
        if CombinedTqdm.is_last_iteration(self.tweens_tqdm):
            print("\n")

    def increment_step_count(self):
        if self.steps_tqdm.n == 0:
            CombinedTqdm.disable_all_taqaddumat_without_a_description()
        log_utils.clear_next_n_lines(2)
        self.steps_tqdm.update()
        self.steps_tqdm.refresh()
        self.total_steps_tqdm.update()
        self.total_steps_tqdm.refresh()
        if CombinedTqdm.is_last_iteration(self.steps_tqdm):
            print("\n")

    def increment_diffusion_frame_count(self):
        # Calls to tqdm.update() without an argument increment it by 1.
        self.total_diffusions_tqdm.update()
        self.total_diffusions_tqdm.refresh()
        print("")

    def reset_tween_count(self, n):
        if n == 0:
            return
        self.tweens_tqdm.reset()
        self.tweens_tqdm.clear()
        self.tweens_tqdm.total = n

    def reset_step_count(self, n):
        self.steps_tqdm.reset()
        self.steps_tqdm.clear()
        self.steps_tqdm.total = n

    def clear_all(self):
        self.tweens_tqdm.clear()
        self.steps_tqdm.clear()
        self.total_steps_tqdm.clear()
        self.total_frames_tqdm.clear()
        self.total_diffusions_tqdm.clear()
        print("\n\n\n\n")

    @staticmethod
    def is_last_iteration(taqaddum):
        return taqaddum.n == taqaddum.total

    @staticmethod
    def disable_all_taqaddumat_without_a_description():
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
