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

import inspect
from pathlib import Path
from tenacity import retry, stop_after_delay, wait_fixed
import requests
from deforum.api.models import DeforumJobStatus, DeforumJobStatusCategory, DeforumJobPhase

SERVER_BASE_URL = "http://localhost:7860"
API_ROOT = "/deforum_api"
API_BASE_URL = SERVER_BASE_URL + API_ROOT

# Dedicated test output directory (not mixed with production outputs)
TEST_OUTPUT_DIR = str(Path(__file__).parent.parent.parent.parent.parent / "outputs" / "deforum-tests")

def get_test_batch_name(test_name: str) -> str:
    """Get a batch name that includes the test module and test name for easier identification.

    Automatically detects the calling test module using inspect to create hierarchical
    directory names like: test_api-simple_settings_{timestring}

    Args:
        test_name: Name of the test function (e.g., 'test_simple_settings')

    Returns:
        Batch name pattern like 'test_api-simple_settings_{timestring}'

    Example:
        # Called from tests/integration/api_test.py
        deforum_settings['batch_name'] = get_test_batch_name('test_simple_settings')
        # Results in output directory: outputs/deforum-tests/api_test-simple_settings_20251025123456/
    """
    # Get the calling module name from the stack
    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_module = inspect.getmodule(frame.f_back)
        if caller_module and caller_module.__name__:
            # Extract just the module name without package prefix
            # e.g., 'tests.integration.api_test' -> 'api_test'
            module_name = caller_module.__name__.split('.')[-1]
        else:
            module_name = "unknown"
    else:
        module_name = "unknown"

    # Remove 'test_' prefix from test name for cleaner names
    clean_test_name = test_name.replace('test_', '', 1) if test_name.startswith('test_') else test_name

    # Format: module-testname_{timestring}
    return f"{module_name}-{clean_test_name}_{{timestring}}"

def get_test_options_overrides():
    """Get standard options_overrides for integration tests.

    Returns dict with:
    - outdir_samples: Dedicated test output directory

    This ensures test outputs don't pollute production outputs.

    Note: To make test outputs easier to identify, also set batch_name in deforum_settings:
        deforum_settings['batch_name'] = get_test_batch_name('test_simple_settings')
    """
    return {
        "outdir_samples": TEST_OUTPUT_DIR,
    }

def cleanup_test_output_dir():
    """Clean the test output directory before test run.

    Removes all contents to ensure clean state for each test run.
    Creates the directory if it doesn't exist.
    """
    import shutil
    import os

    test_dir = Path(TEST_OUTPUT_DIR)

    # Remove existing directory and all contents
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Recreate empty directory
    os.makedirs(test_dir, exist_ok=True)

@retry(wait=wait_fixed(2), stop=stop_after_delay(600))
def wait_for_job_to_complete(id : str):
    """Poll job status until it reaches SUCCEEDED or FAILED state.

    Args:
        id: Job ID to wait for

    Returns:
        Final DeforumJobStatus when job completes

    Raises:
        AssertionError: If job status is FAILED or CANCELLED
    """
    response = requests.get(
        f"{API_BASE_URL}/jobs/{id}",
        headers={"accept": "application/json"}
    )
    response.raise_for_status()

    # Parse JSON manually instead of using PydanticSession
    # This gives better error messages if validation fails
    try:
        jobStatus = DeforumJobStatus.model_validate(response.json())
    except Exception as e:
        print(f"Failed to parse job status response: {e}")
        print(f"Raw response: {response.text}")
        raise

    # Only log status changes and completion
    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        print(f"[POLL] Job {id} FAILED: {jobStatus.message}")
        raise AssertionError(f"Job {id} failed: {jobStatus.message}")
    if jobStatus.status == DeforumJobStatusCategory.CANCELLED:
        print(f"[POLL] Job {id} CANCELLED")
        raise AssertionError(f"Job {id} was cancelled")

    # Keep retrying until job succeeds (no spam)
    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {id} still running (status={jobStatus.status}, phase={jobStatus.phase})"

    return jobStatus
    
@retry(wait=wait_fixed(1), stop=stop_after_delay(120))
def wait_for_job_to_enter_phase(id : str, phase : DeforumJobPhase):
    response = requests.get(
        f"{API_BASE_URL}/jobs/{id}",
        headers={"accept": "application/json"}
    )
    response.raise_for_status()

    try:
        jobStatus = DeforumJobStatus.model_validate(response.json())
    except Exception as e:
        print(f"Failed to parse job status response: {e}")
        print(f"Raw response: {response.text}")
        raise

    print(f"Waiting for job {id} to enter phase {phase}. Currently: status={jobStatus.status}; phase={jobStatus.phase}; execution_time:{jobStatus.execution_time}s")
    assert jobStatus.phase != phase
    return jobStatus
    
@retry(wait=wait_fixed(1), stop=stop_after_delay(120))
def wait_for_job_to_enter_status(id : str, status : DeforumJobStatusCategory):
    response = requests.get(
        f"{API_BASE_URL}/jobs/{id}",
        headers={"accept": "application/json"}
    )
    response.raise_for_status()

    try:
        jobStatus = DeforumJobStatus.model_validate(response.json())
    except Exception as e:
        print(f"Failed to parse job status response: {e}")
        print(f"Raw response: {response.text}")
        raise

    print(f"Waiting for job {id} to enter status {status}. Currently: status={jobStatus.status}; phase={jobStatus.phase}; execution_time:{jobStatus.execution_time}s")
    assert jobStatus.status == status
    return jobStatus


def gpu_disabled():
    """Check if GPU is disabled on the server.

    Returns True if GPU is disabled, False if GPU is available.
    If the check fails (server error), assumes GPU is available (False).
    """
    try:
        response = requests.get(SERVER_BASE_URL+"/sdapi/v1/cmd-flags", timeout=5)
        response.raise_for_status()
        cmd_flags = response.json()
        return cmd_flags.get("use_cpu") == ["all"]
    except (requests.exceptions.RequestException, KeyError, ValueError):
        # If we can't check GPU status, assume GPU is available
        # This prevents tests from being skipped due to server issues
        return False

        


