import time

from deforum.api.api import JobStatusTracker
from modules.shared import state
from deforum.utils.system.logging import get_logger

# Defer logger initialization to avoid module-level opts access
_logger_instance = None


def _get_logger():
    """Get logger instance - deferred to avoid module-level opts access."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = get_logger()
    return _logger_instance


WEB_UI_SLEEP_DELAY = 0.1


def init_job(data):
    state.job_count = data.args.anim_args.max_frames


def update_job(data, i):
    frame = i + 1
    max_frames = data.args.anim_args.max_frames
    state.job = f"frame {frame}/{max_frames}"
    state.job_no = frame + 1
    if state.skipped:
        _get_logger().info("\n** PAUSED **")
        state.skipped = False
        while not state.skipped:
            time.sleep(WEB_UI_SLEEP_DELAY)
        _get_logger().info("** RESUMING **")


def update_status_tracker(data, i):
    progress = i / data.args.anim_args.max_frames
    JobStatusTracker().update_phase(data.args.root.job_id, phase="GENERATING", progress=progress)


def update_progress_during_cadence(data, i):
    state.job = f"frame {i + 1}/{data.args.anim_args.max_frames}"
    state.job_no = i + 1
