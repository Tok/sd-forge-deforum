import gc

# noinspection PyUnresolvedReferences
import modules.shared as shared
from tqdm import tqdm

from deforum.utils.system.logging import log as log_utils
from deforum.utils.system.logging.log import HEX_BLUE, HEX_GREEN, HEX_ORANGE, HEX_RED, HEX_PURPLE
from deforum.utils.system.logging import get_logger
from deforum.utils.system.logging.themes import get_tqdm_color_for_theme
from deforum.rendering.options import get_log_theme

# Initialize logger
logger = get_logger()



class Taqaddumat:
    # Progress bar helper for the render core
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
        """Initialize all progress bars with correct totals.

        Args:
            data: RenderData containing animation settings
            frames: List of DiffusionFrame objects (in GENERATION order)

        Note:
            In reverse generation mode, frames are already reversed by core.py
            before calling this method, so initialization is always correct.

        Progress Bar Strategy:
            - All bars except final: leave=False (disappear when animation completes)
            - Final progress bar (diffusion frames): leave=True (shows completion)
            - Position reuse allows bars to update in-place during rendering
        """
        def create(iterable, position, color, description, unit, leave=False, bar_format=Taqaddumat.NO_ETA_BAR_FORMAT):
            # Get themed color based on current theme
            themed_color = Taqaddumat._get_themed_color(color)
            return tqdm(iterable, position=position, desc=description, unit=unit, dynamic_ncols=True,
                        file=shared.progress_print_out, bar_format=bar_format, leave=leave,
                        disable=shared.cmd_opts.disable_console_progressbars, colour=themed_color)

        # Positions greater than 0 are assigned where bars are meant to show up directly after each other and
        # need to be updated at the same time. 'Tweens' is paired with 'Total Frames' and 'Steps' with 'Total Steps'.
        # Those are intended to be updated together and always be displayed at the same time.
        # The global step counter tqdm provided by Forge keeps reattaching itself after the highest position used here.
        # This dictates how much space the total tqdm will take up, even when positions are reassigned here later.

        # Get initial tween count safely (handle edge cases)
        initial_tween_count = 0
        if len(frames) > 1 and len(frames[1].tweens) > 0:
            initial_tween_count = len(frames[1].tweens)
        elif len(frames) > 0 and len(frames[0].tweens) > 0:
            initial_tween_count = len(frames[0].tweens)

        # Transient bar: resets for each diffusion frame, don't leave in log
        self.tweens = create(
            range(initial_tween_count), 0, HEX_BLUE,
            "Current Tweens", "tween", leave=False)

        # Accumulator bar: grows throughout animation, don't leave (prevents log clutter)
        total_frames = sum(len(frame.tweens) for frame in frames)
        self.total_frames = create(
            range(total_frames), 1, HEX_GREEN,
            "Total Frames", "frame", leave=False)

        # Transient bar: resets for each diffusion frame, don't leave in log
        initial_steps_count = frames[0].schedule.steps if len(frames) > 0 else 20
        self.steps = create(
            range(initial_steps_count), 0, HEX_ORANGE,
            "Current Diffusion Steps", "step", leave=False)

        # Accumulator bar: grows throughout animation, don't leave (prevents log clutter)
        total_steps = sum(frame.actual_steps(data) for frame in frames)
        self.total_steps = create(
            range(total_steps), 1, HEX_RED,
            "Total Diffusion Steps", "step", leave=False)

        # Final progress bar: shows overall completion, KEEP in log when done
        # Renamed from "Total Animation Cycles" to "Diffusion Frames" for clarity
        num_diffusion_frames = len(frames)
        self.total_animation_cycles = create(
            range(num_diffusion_frames), 0, HEX_PURPLE,
            "Diffusion Frames" + (" (Reverse)" if data.args.anim_args.reverse_generation else ""),
            "frame",
            leave=True,  # Keep final bar visible in log
            bar_format=Taqaddumat.DEFAULT_BAR_FORMAT)

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
        # Don't print newline on completion - bars use leave=False and should disappear cleanly

    def increment_step_count(self):
        if self.steps.n == 0:
            Taqaddumat.disable_all_tqdm_without_a_description()
            log_utils.clear_next_n_lines(2)  # depends on cursor resting at the end of the line with the 0th tqdm.
        self.steps.update()
        self.steps.refresh()
        self.total_steps.update()
        self.total_steps.refresh()
        # Don't print newline on completion - bars use leave=False and should disappear cleanly

    def increment_animation_cycle_count(self):
        # Calls to tqdm.update() without an argument increment it by 1.
        self.total_animation_cycles.update()
        self.total_animation_cycles.refresh()
        logger.info("")

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
        logger.info("\n\n\n\n")

    @staticmethod
    def _get_themed_color(classic_color):
        """Map classic tqdm color to current theme's color.

        Args:
            classic_color: Original color hex (HEX_BLUE, HEX_GREEN, etc.)

        Returns:
            Themed color hex based on current theme setting
        """
        theme = get_log_theme()
        return get_tqdm_color_for_theme(classic_color, theme)

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
        def safe_isinstance_check(obj):
            """Check isinstance with ReferenceError protection for weakly-referenced objects."""
            try:
                return isinstance(obj, tqdm)
            except ReferenceError:
                return False

        def safe_desc_check(obj):
            """Check desc attribute with ReferenceError protection."""
            try:
                return not obj.desc
            except ReferenceError:
                return False

        list(map(lambda _: mute(_),
                 filter(safe_desc_check,
                        filter(safe_isinstance_check,
                               gc.get_objects()))))
