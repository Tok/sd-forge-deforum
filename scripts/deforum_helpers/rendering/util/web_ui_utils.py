import time

# noinspection PyUnresolvedReferences
from deforum_api import JobStatusTracker
# noinspection PyUnresolvedReferences
from modules.shared import state

WEB_UI_SLEEP_DELAY = 0.1


def init_job(data):
    state.job_count = data.args.anim_args.max_frames


def update_job(data, i):
    frame = i + 1
    max_frames = data.args.anim_args.max_frames
    state.job = f"frame {frame}/{max_frames}"
    state.job_no = frame + 1
    if state.skipped:
        print("\n** PAUSED **")
        state.skipped = False
        while not state.skipped:
            time.sleep(WEB_UI_SLEEP_DELAY)
        print("** RESUMING **")


def update_status_tracker(data, i):
    progress = i / data.args.anim_args.max_frames
    JobStatusTracker().update_phase(data.args.root.job_id, phase="GENERATING", progress=progress)


def update_progress_during_cadence(data, i):
    state.job = f"frame {i + 1}/{data.args.anim_args.max_frames}"
    state.job_no = i + 1
