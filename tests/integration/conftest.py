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

import pytest
import subprocess
import sys
import os
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path
from tenacity import retry, stop_after_delay, wait_fixed
import threading
import requests
from .utils import cleanup_test_output_dir

def pytest_addoption(parser):
    parser.addoption("--start-server", action="store_true", help="start the server before the test run (if not specified, you must start the server manually)")
    parser.addoption("--run-slow", action="store_true", help="run slow calibration/timing tests (skipped by default)")

@pytest.fixture(scope="session", autouse=True)
def clean_test_outputs():
    """Clean test output directory before test session starts.

    This ensures each test run starts with a clean slate.
    Runs automatically for all tests (autouse=True).
    """
    cleanup_test_output_dir()
    yield  # Tests run here
    # Could add cleanup after tests too if desired

@pytest.fixture(scope="function", autouse=True)
def cleanup_gpu_memory():
    """Clean up GPU memory after each test to prevent OOM errors.

    This is especially important for tests that use large models like
    FILM interpolation or upscaling, which can leave VRAM allocated.
    """
    yield  # Test runs here
    # Cleanup after test
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except Exception:
        pass  # Silent fail if torch not available

@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--start-server")

@retry(wait=wait_fixed(5), stop=stop_after_delay(60))
def wait_for_service(url):
    response = requests.get(url, timeout=(5, 5))
    print(f"Waiting for server to respond 200 at {url} (response: {response.status_code})...")
    assert response.status_code == 200

@pytest.fixture(scope="session", autouse=True)
def start_server(request):
    # Check if --start-server option exists (may not be registered with certain pytest flags)
    try:
        should_start_server = request.config.getoption("--start-server")
    except ValueError:
        # Option not registered (e.g., when using --snapshot-update)
        # Assume server is externally managed by run-api-tests.sh
        should_start_server = False

    if should_start_server:

        # Kick off server subprocess
        script_directory = os.path.dirname(__file__)
        a1111_directory = Path(script_directory).parent.parent.parent.parent  # extensions/sd-forge-deforum/tests/integration/ -> stable-diffusion-webui-forge
        venv_python = str(a1111_directory / "venv" / "bin" / "python")
        print(f"Starting server in {a1111_directory}...")
        print(f"Using Python: {venv_python}")
        proc = Popen([venv_python, "-m", "coverage", "run", "--data-file=.coverage.server", "launch.py",
                      "--skip-prepare-environment", "--skip-torch-cuda-test", "--test-server", "--no-half",
                      "--disable-opt-split-attention", "--use-cpu", "all", "--add-stop-route", "--api", "--deforum-api", "--listen"],
            cwd=a1111_directory,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True)
        
        # ensure server is killed at the end of the test run
        request.addfinalizer(proc.kill)

        # Spin up separate thread to capture the server output to file and stdout
        def server_console_manager():
            with proc.stdout, open('serverlog.txt', 'ab') as logfile:
                for line in proc.stdout:
                    sys.stdout.write(f"[SERVER LOG] {line}")
                    sys.stdout.flush()
                    logfile.write(line.encode('utf-8'))
                    logfile.flush()
                proc.wait()
        
        threading.Thread(target=server_console_manager).start()
        
        # Wait for deforum API to respond
        wait_for_service('http://localhost:7860/deforum_api/jobs/')
       
    else:
        print("Checking server is already running / waiting for it to come up...")
        wait_for_service('http://localhost:7860/deforum_api/jobs/')