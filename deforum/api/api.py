# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import os
import atexit
import json
import random
import tempfile
import traceback
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
from deforum.api.models import (
    Batch, DeforumJobErrorType, DeforumJobStatusCategory, DeforumJobPhase, DeforumJobStatus,
    BatchSubmitResponse, BatchCancelResponse, JobCancelResponse, ErrorResponse, VersionResponse,
    SimpleRunResponse
)
from contextlib import contextmanager
from scripts.deforum_extend_paths import deforum_sys_extend

import gradio as gr
from deforum.config.args import (DeforumAnimArgs, DeforumArgs,
                                  DeforumOutputArgs, LoopArgs, ParseqArgs,
                                  RootArgs, get_component_names)
from deforum.utils.system.opts_overrider import A1111OptionsOverrider
from fastapi import FastAPI, Response, status, HTTPException

from modules.shared import cmd_opts, opts, state


log = logging.getLogger(__name__)
log_level = os.environ.get("DEFORUM_API_LOG_LEVEL") or os.environ.get("SD_WEBUI_LOG_LEVEL") or "INFO"
log.setLevel(log_level)
logging.basicConfig(
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def make_ids(job_count: int):
    batch_id = f"batch({random.randint(0, int(1e9))})"
    job_ids = [f"{batch_id}-{i}" for i in range(job_count)]
    return [batch_id, job_ids]


def get_default_value(name:str):
    allArgs = RootArgs() | DeforumAnimArgs() | DeforumArgs() | LoopArgs() | ParseqArgs() | DeforumOutputArgs()
    if name in allArgs and isinstance(allArgs[name], dict):
        return allArgs[name].get("value", None)
    elif name in allArgs:
        return allArgs[name]
    else:
        return None


def run_deforum_batch(batch_id: str, job_ids: [str], deforum_settings_files: List[Any], opts_overrides: Dict[str, Any] = None):
    log.info(f"Starting batch {batch_id} in thread {threading.get_ident()}.")
    try:
        with A1111OptionsOverrider(opts_overrides):

            # Fill deforum args with default values.
            # We are overriding everything with the batch files, but some values are eagerly validated, so must appear valid.
            component_names = get_component_names()
            prefixed_gradio_args = 2
            expected_arg_count = prefixed_gradio_args + len(component_names)
            run_deforum_args = [None] * expected_arg_count
            for idx, name in enumerate(component_names):
                run_deforum_args[prefixed_gradio_args + idx] = get_default_value(name)

            # For some values, defaults don't pass validation...
            # Note: This placeholder is immediately overridden by batch settings, never used for generation
            run_deforum_args[prefixed_gradio_args + component_names.index('animation_prompts')] = '{"0":"placeholder prompt"}'
            run_deforum_args[prefixed_gradio_args + component_names.index('animation_prompts_negative')] = ''
            run_deforum_args[prefixed_gradio_args + component_names.index('animation_prompts_positive')] = ''

            # Arg 0 is a UID for the batch
            run_deforum_args[0] = batch_id

            # Setup batch override
            run_deforum_args[prefixed_gradio_args + component_names.index('override_settings_with_file')] = True
            run_deforum_args[prefixed_gradio_args + component_names.index('custom_settings_file')] = deforum_settings_files

            # Cleanup old state from previously cancelled jobs
            # WARNING: not thread safe because state is global. If we ever run multiple batches in parallel, this will need to be reworked.
            state.skipped = False
            state.interrupted = False

            # Invoke deforum with appropriate args
            from deforum.orchestration.run_deforum import run_deforum 
            run_deforum(*run_deforum_args)

    except Exception as e:
        log.error(f"Batch {batch_id} failed: {e}")
        traceback.print_exc()
        for job_id in job_ids:
            # Mark all jobs in this batch as failed
            JobStatusTracker().fail_job(job_id, 'TERMINAL', str(e))


# API to allow a batch of jobs to be submitted to the deforum pipeline.
# A batch is settings object OR a list of settings objects. 
# A settings object is the JSON structure you can find in your saved settings.txt files.
# 
# Request format:
# {
#   "deforum_settings": [
#       { ... settings object ... },
#       { ... settings object ... },
#   ]
# }
# OR:
# {
#   "deforum_settings": { ... settings object ... }
# }
#
# Each settings object in the request represents a job to run as part of the batch.
# Each submitted batch will be given a batch ID which the user can use to query the status of all jobs in the batch.
#
def deforum_api(_: gr.Blocks, app: FastAPI):  

    deforum_sys_extend()

    apiState = ApiState()

    @app.post(
        "/deforum_api/batches",
        response_model=BatchSubmitResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Batches"],
        summary="Submit a new batch of Deforum generation jobs",
        responses={
            202: {"description": "Batch accepted and queued for processing"},
            400: {"model": ErrorResponse, "description": "Invalid settings provided"}
        }
    )
    async def run_batch(batch: Batch, response: Response):
        """Submit one or more Deforum generation jobs as a batch.

        Each job in the batch will generate a complete animation based on the provided settings.
        Jobs are queued and processed sequentially (currently limited to 1 concurrent job).

        **Request Body:**
        - `deforum_settings`: Single settings object or list of settings objects
        - `options_overrides`: Optional Forge settings to override (e.g., save_gen_info_as_srt)

        **Returns:**
        - `batch_id`: Unique identifier for this batch
        - `job_ids`: List of job IDs (one per settings object)

        **Example:**
        ```python
        {
            "deforum_settings": {
                "animation_mode": "3D",
                "max_frames": 120,
                "prompts": {"0": "a beautiful landscape"},
                "W": 512,
                "H": 512
            }
        }
        ```
        """
        # Extract the settings files from the request
        deforum_settings_data = batch.deforum_settings
        if not deforum_settings_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No settings files provided. Please provide 'deforum_settings' in the request."
            )

        if not isinstance(deforum_settings_data, list):
            # Allow input deforum_settings to be top-level object as well as single object list
            deforum_settings_data = [deforum_settings_data]

        deforum_settings_tempfiles = []
        for data in deforum_settings_data:
            temp_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
            json.dump(data, temp_file)
            temp_file.close()
            deforum_settings_tempfiles.append(temp_file)

        job_count = len(deforum_settings_tempfiles)
        [batch_id, job_ids] = make_ids(job_count)
        apiState.submit_job(batch_id, job_ids, deforum_settings_tempfiles, batch.options_overrides)

        for idx, job_id in enumerate(job_ids):
            JobStatusTracker().accept_job(
                batch_id=batch_id,
                job_id=job_id,
                deforum_settings=deforum_settings_data[idx],
                options_overrides=batch.options_overrides
            )

        response.status_code = status.HTTP_202_ACCEPTED
        return BatchSubmitResponse(
            message="Job(s) accepted",
            batch_id=batch_id,
            job_ids=job_ids
        )

    @app.get(
        "/deforum_api/batches",
        response_model=Dict[str, List[str]],
        tags=["Batches"],
        summary="List all batches and their job IDs"
    )
    async def list_batches():
        """Get a dictionary mapping batch IDs to their job IDs.

        **Returns:**
        Dictionary where keys are batch IDs and values are lists of job IDs.
        """
        return JobStatusTracker().batches

    @app.get(
        "/deforum_api/batches/{id}",
        response_model=List[DeforumJobStatus],
        tags=["Batches"],
        summary="Get detailed status of all jobs in a batch",
        responses={
            200: {"description": "Batch found and job statuses returned"},
            404: {"model": ErrorResponse, "description": "Batch not found"}
        }
    )
    async def get_batch(id: str, response: Response):
        """Get detailed status information for all jobs in a specific batch.

        **Parameters:**
        - `id`: Batch identifier

        **Returns:**
        List of DeforumJobStatus objects for each job in the batch.
        """
        jobsForBatch = JobStatusTracker().batches.get(id)
        if not jobsForBatch:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch {id} not found"
            )
        return [JobStatusTracker().get(job_id) for job_id in jobsForBatch]

    @app.delete(
        "/deforum_api/batches/{id}",
        response_model=BatchCancelResponse,
        tags=["Batches"],
        summary="Cancel all jobs in a batch",
        responses={
            200: {"description": "Jobs cancelled successfully"},
            404: {"model": ErrorResponse, "description": "Batch not found"}
        }
    )
    async def cancel_batch(id: str, response: Response):
        """Cancel all jobs in a specific batch.

        Jobs that are currently generating will be interrupted.
        Queued jobs will be marked as cancelled.

        **Parameters:**
        - `id`: Batch identifier

        **Returns:**
        List of cancelled job IDs and count.
        """
        jobsForBatch = JobStatusTracker().batches.get(id)
        cancelled_jobs = []
        if not jobsForBatch:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch {id} not found"
            )

        for job_id in jobsForBatch:
            try:
                cancelled = _cancel_job(job_id)
                if cancelled:
                    cancelled_jobs.append(job_id)
            except:
                log.warning(f"Failed to cancel job {job_id}")

        return BatchCancelResponse(
            ids=cancelled_jobs,
            message=f"{len(cancelled_jobs)} job(s) cancelled."
        )

    @app.get(
        "/deforum_api/jobs",
        response_model=Dict[str, DeforumJobStatus],
        tags=["Jobs"],
        summary="List all jobs across all batches"
    )
    async def list_jobs():
        """Get status information for all jobs.

        **Returns:**
        Dictionary mapping job IDs to their DeforumJobStatus objects.
        """
        return JobStatusTracker().statuses

    @app.get(
        "/deforum_api/jobs/{id}",
        response_model=DeforumJobStatus,
        tags=["Jobs"],
        summary="Get detailed status of a specific job",
        responses={
            200: {"description": "Job found and status returned"},
            404: {"model": ErrorResponse, "description": "Job not found"}
        }
    )
    async def get_job(id: str, response: Response):
        """Get detailed status information for a specific job.

        **Parameters:**
        - `id`: Job identifier

        **Returns:**
        DeforumJobStatus object with current job state, progress, and output info.
        """
        jobStatus = JobStatusTracker().get(id)
        if not jobStatus:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {id} not found"
            )
        return jobStatus

    @app.delete(
        "/deforum_api/jobs/{id}",
        response_model=JobCancelResponse,
        tags=["Jobs"],
        summary="Cancel a specific job",
        responses={
            200: {"description": "Job cancelled successfully"},
            400: {"model": ErrorResponse, "description": "Job not in cancellable state"},
            404: {"model": ErrorResponse, "description": "Job not found"}
        }
    )
    async def cancel_job(id: str, response: Response):
        """Cancel a specific job.

        Jobs currently generating will be interrupted.
        Queued jobs will be marked as cancelled.
        Completed jobs cannot be cancelled.

        **Parameters:**
        - `id`: Job identifier

        **Returns:**
        Confirmation message with job ID.
        """
        try:
            if _cancel_job(id):
                return JobCancelResponse(id=id, message="Job cancelled.")
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Job {id} not in a cancellable state. Has it already finished?"
                )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {id} not found."
            )

    # Shared logic for job cancellation
    def _cancel_job(job_id:str):
        jobStatus = JobStatusTracker().get(job_id)
        if not jobStatus:
            raise FileNotFoundError(f"Job {job_id} not found.")
        
        if jobStatus.status != DeforumJobStatusCategory.ACCEPTED:
             # Ignore jobs in completed state (error or success)
            return False

        if job_id in ApiState().submitted_jobs:
            # Remove job from queue
            ApiState().submitted_jobs[job_id].cancel()
        if jobStatus.phase != DeforumJobPhase.QUEUED and jobStatus.phase != DeforumJobPhase.DONE:
            # Job must be actively running - interrupt it.
            # WARNING:
            #   - Possible race condition: if job_id just finished after the check and another started, we'll interrupt the wrong job.
            #   - Not thread safe because State object is global. Will break with concurrent jobs.
            state.interrupt()
        JobStatusTracker().cancel_job(job_id, "Cancelled due to user request.")
        return True
    
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# Maintains persistent state required by API, e.g. thread pook, list of submitted jobs.
class ApiState(metaclass=Singleton):
    
    ## Locking concurrency to 1. Concurrent generation does seem to work, but it's not clear if it's safe.
    ## TODO: more experimentation required.
    deforum_api_executor = ThreadPoolExecutor(max_workers=1)
    submitted_jobs : Dict[str, Any] = {}

    @staticmethod
    def cleanup():
        ApiState().deforum_api_executor.shutdown(wait=False)    

    def submit_job(self, batch_id: str, job_ids: [str], deforum_settings: List[Any], opts_overrides: Dict[str, Any]):
        log.debug(f"Submitting batch {batch_id} to threadpool.")
        future = self.deforum_api_executor.submit(lambda: run_deforum_batch(batch_id, job_ids, deforum_settings, opts_overrides))
        self.submitted_jobs[batch_id] = future

atexit.register(ApiState.cleanup)

# Maintains state that tracks status of submitted jobs, 
# so that clients can query job status.
class JobStatusTracker(metaclass=Singleton):
    statuses: Dict[str, DeforumJobStatus] = {}
    batches: Dict[str, List[str]] = {}

    def accept_job(self, batch_id : str, job_id: str, deforum_settings : List[Dict[str, Any]] , options_overrides : Dict[str, Any]):
        if batch_id in self.batches:
            self.batches[batch_id].append(job_id)
        else:
            self.batches[batch_id] = [job_id]

        now = datetime.now().timestamp()
        self.statuses[job_id] = DeforumJobStatus(
            id=job_id,
            status= DeforumJobStatusCategory.ACCEPTED,
            phase=DeforumJobPhase.QUEUED,
            error_type=DeforumJobErrorType.NONE,
            phase_progress=0.0,
            started_at=now,
            last_updated=now,
            execution_time=0,
            update_interval_time=0,
            updates=0,
            message=None,
            outdir=None,
            timestring=None,
            deforum_settings=deforum_settings,
            options_overrides=options_overrides,
        )

    def update_phase(self, job_id: str, phase: DeforumJobPhase, progress: float = 0):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            now = datetime.now().timestamp()
            new_status = current_status.model_copy(update={
                'phase': phase,
                'phase_progress': progress,
                'last_updated': now,
                'execution_time': now-current_status.started_at,
                'update_interval_time': now-current_status.last_updated,
                'updates': current_status.updates+1
            })
            self.statuses[job_id] = new_status

    def update_output_info(self, job_id: str, outdir: str, timestring: str):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            now = datetime.now().timestamp()
            new_status = current_status.model_copy(update={
                'outdir': outdir,
                'timestring': timestring,
                'last_updated': now,
                'execution_time': now-current_status.started_at,
                'update_interval_time': now-current_status.last_updated,
                'updates': current_status.updates+1
            })
            self.statuses[job_id] = new_status

    def complete_job(self, job_id: str):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            now = datetime.now().timestamp()
            new_status = current_status.model_copy(update={
                'status': DeforumJobStatusCategory.SUCCEEDED,
                'phase': DeforumJobPhase.DONE,
                'phase_progress': 1.0,
                'last_updated': now,
                'execution_time': now-current_status.started_at,
                'update_interval_time': now-current_status.last_updated,
                'updates': current_status.updates+1
            })
            self.statuses[job_id] = new_status

    def fail_job(self, job_id: str, error_type: str, message: str):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            now = datetime.now().timestamp()
            new_status = current_status.model_copy(update={
                'status': DeforumJobStatusCategory.FAILED,
                'error_type': error_type,
                'message': message,
                'last_updated': now,
                'execution_time': now-current_status.started_at,
                'update_interval_time': now-current_status.last_updated,
                'updates': current_status.updates+1
            })
            self.statuses[job_id] = new_status

    def cancel_job(self, job_id: str, message: str):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            now = datetime.now().timestamp()
            new_status = current_status.model_copy(update={
                'status': DeforumJobStatusCategory.CANCELLED,
                'message': message,
                'last_updated': now,
                'execution_time': now-current_status.started_at,
                'update_interval_time': now-current_status.last_updated,
                'updates': current_status.updates+1
            })
            self.statuses[job_id] = new_status


    def get(self, job_id:str):
        return self.statuses[job_id] if job_id in self.statuses else None

def deforum_init_batch(_: gr.Blocks, app: FastAPI):
    deforum_sys_extend()
    settings_files = [open(filename, 'r') for filename in cmd_opts.deforum_run_now.split(",")]
    [batch_id, job_ids] = make_ids(len(settings_files))
    log.info(f"Starting init batch {batch_id} with job(s) {job_ids}...")

    run_deforum_batch(batch_id, job_ids, settings_files, None)

    if cmd_opts.deforum_terminate_after_run_now:
        import os
        os._exit(0)

# A simplified, but safe version of Deforum's API
def deforum_simple_api(_: gr.Blocks, app: FastAPI):
    deforum_sys_extend()
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi import FastAPI, Query, Request, UploadFile
    from fastapi.encoders import jsonable_encoder
    from deforum.utils.general import get_deforum_version
    import uuid, pathlib

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )

    @app.get(
        "/deforum/api_version",
        response_model=VersionResponse,
        tags=["Simple API"],
        summary="Get Deforum Simple API version"
    )
    async def deforum_api_version():
        """Get the version of the Deforum Simple API.

        **Returns:**
        API version string.
        """
        return VersionResponse(version='1.0')

    @app.get(
        "/deforum/version",
        response_model=VersionResponse,
        tags=["Simple API"],
        summary="Get Deforum extension version"
    )
    async def deforum_version():
        """Get the version of the Deforum extension.

        **Returns:**
        Deforum version string.
        """
        return VersionResponse(version=get_deforum_version())
    
    @app.post(
        "/deforum/run",
        response_model=SimpleRunResponse,
        tags=["Simple API"],
        summary="Run a Deforum generation with limited parameters (safer)",
        responses={
            200: {"description": "Generation started successfully"},
            500: {"model": ErrorResponse, "description": "Error processing video"}
        }
    )
    async def deforum_run(settings_json: str, allowed_params: str = ""):
        """Run a Deforum generation using the simplified API.

        This endpoint is safer than the full API because it only allows specific
        parameters to be overridden. All other parameters use defaults.

        **Parameters:**
        - `settings_json`: JSON string with Deforum settings
        - `allowed_params`: Semicolon-delimited list of parameter names that can be overridden

        **How it works:**
        1. Loads default settings from config/default_settings.txt
        2. Only parameters in `allowed_params` are overridden with values from `settings_json`
        3. All other parameters remain at their default values
        4. Generates a unique batch_name automatically

        **Example:**
        ```
        POST /deforum/run
        settings_json='{"max_frames": 120, "prompts": {"0": "landscape"}}'
        allowed_params='max_frames;prompts;W;H'
        ```

        **Returns:**
        Output directory path where generated files will be saved.
        """
        try:
            allowed_params_list = allowed_params.split(';')
            deforum_settings = json.loads(settings_json)
            with open(
                os.path.join(pathlib.Path(__file__).parent.parent.absolute(), 'config', 'default_settings.txt'),
                'r',
                encoding='utf-8'
            ) as f:
                default_settings = json.loads(f.read())

            for k, _ in default_settings.items():
                if k in deforum_settings and k in allowed_params_list:
                    default_settings[k] = deforum_settings[k]

            deforum_settings = default_settings
            run_id = uuid.uuid4().hex
            deforum_settings['batch_name'] = run_id
            deforum_settings_str = json.dumps(deforum_settings, indent=4, ensure_ascii=False)
            settings_file = f"{run_id}.txt"

            with open(settings_file, 'w', encoding='utf-8') as f:
                f.write(deforum_settings_str)

            class SettingsWrapper:
                def __init__(self, filename):
                    self.name = filename

            [batch_id, job_ids] = make_ids(1)
            outdir = os.path.join(os.getcwd(), opts.outdir_samples or opts.outdir_img2img_samples, str(run_id))
            run_deforum_batch(batch_id, job_ids, [SettingsWrapper(settings_file)], None)
            return SimpleRunResponse(outdir=outdir)
        except Exception as e:
            log.error(f"Simple API error: {e}")
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"detail": "An error occurred while processing the video."}
            )

# Setup A1111 initialisation hooks
try:
    import modules.script_callbacks as script_callbacks

    # Auto-enable deforum_api if tuning mode is requested
    if getattr(cmd_opts, 'deforum_run_tuning', False):
        if not getattr(cmd_opts, 'deforum_api', False):
            log.info("[Deforum] Tuning mode enabled - auto-enabling Deforum API")
            cmd_opts.deforum_api = True

    if cmd_opts.deforum_api:
        script_callbacks.on_app_started(deforum_api)
    if cmd_opts.deforum_api or cmd_opts.deforum_simple_api:
        script_callbacks.on_app_started(deforum_simple_api)
    if cmd_opts.deforum_run_now:
        script_callbacks.on_app_started(deforum_init_batch)
    # Register tuning API if tuning mode enabled
    if getattr(cmd_opts, 'deforum_run_tuning', False) or cmd_opts.deforum_api:
        from deforum.api.tuning_api import tuning_api
        script_callbacks.on_app_started(tuning_api)
        log.info("Registered tuning API endpoints")
except:
    pass
