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

import glob
import json
import os
from pathlib import Path

import pytest
import requests
from moviepy.editor import VideoFileClip
from .utils import (API_BASE_URL, get_test_options_overrides, gpu_disabled, wait_for_job_to_complete, get_test_batch_name)

from deforum.api.models import (DeforumJobPhase,
                                        DeforumJobStatusCategory)
from deforum.media.subtitle_handler import get_user_values


# Path to testdata directory relative to this test file
TESTDATA_DIR = Path(__file__).parent / 'testdata'


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_post_process_FILM(snapshot):
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_post_process_FILM')

    deforum_settings["frame_interpolation_engine"] = "FILM"
    deforum_settings["frame_interpolation_x_amount"] = 3
    deforum_settings["frame_interpolation_slow_mo_enabled"] = False

    options_overrides = get_test_options_overrides()
    options_overrides.update({
        "deforum_save_gen_info_as_srt": True,
        "deforum_save_gen_info_as_srt_params": get_user_values(),
    })

    response = requests.post(API_BASE_URL+"/batches", json={
        "deforum_settings":[deforum_settings],
        "options_overrides": options_overrides
        })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, f"Job {job_id} failed: {jobStatus}"

    # Ensure parameters used at each frame have not regressed
    srt_filenname =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.srt")
    with open(srt_filenname, 'r') as srt_file:
        assert srt_file.read() == snapshot

    # Ensure interpolated video format is as expected
    video_filenames = glob.glob(f'{jobStatus.outdir}/*FILM*.mp4', recursive=True)
    assert len(video_filenames) == 1, "Expected one FILM video to be generated"

    interpolated_video_filename =  video_filenames[0]
    clip = VideoFileClip(interpolated_video_filename)
    assert clip.fps == deforum_settings['fps'] * deforum_settings["frame_interpolation_x_amount"]  , "Video FPS does not match input settings (fps * interpolation amount)"
    # FILM interpolation: N frames + (N-1) gaps × (x-1) interpolated frames per gap
    expected_frame_count = deforum_settings['max_frames'] + (deforum_settings['max_frames'] - 1) * (deforum_settings["frame_interpolation_x_amount"] - 1)
    actual_frame_count = round(clip.duration * clip.fps)
    assert actual_frame_count == expected_frame_count, f"Video frame count does not match FILM interpolation formula (expected {expected_frame_count}, got {actual_frame_count})"
    assert clip.size == [deforum_settings['W'], deforum_settings['H']] , "Video dimensions are not as expected"    

@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_post_process_RIFE(snapshot):
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_post_process_RIFE')

    deforum_settings["frame_interpolation_engine"] = "RIFE v4.15"
    deforum_settings["frame_interpolation_x_amount"] = 3
    deforum_settings["frame_interpolation_slow_mo_enabled"] = False

    options_overrides = get_test_options_overrides()
    options_overrides.update({
        "deforum_save_gen_info_as_srt": True,
        "deforum_save_gen_info_as_srt_params": get_user_values(),
    })

    response = requests.post(API_BASE_URL+"/batches", json={
        "deforum_settings":[deforum_settings],
        "options_overrides": options_overrides
        })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, f"Job {job_id} failed: {jobStatus}"

    # Ensure parameters used at each frame have not regressed
    srt_filenname =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.srt")
    with open(srt_filenname, 'r') as srt_file:
        assert srt_file.read() == snapshot

    # Ensure interpolated video format is as expected
    video_filenames = glob.glob(f'{jobStatus.outdir}/*RIFE*.mp4', recursive=True)
    assert len(video_filenames) == 1, "Expected one RIFE video to be generated"

    interpolated_video_filename =  video_filenames[0]
    clip = VideoFileClip(interpolated_video_filename)
    assert clip.fps == deforum_settings['fps'] * deforum_settings["frame_interpolation_x_amount"]  , "Video FPS does not match input settings (fps * interpolation amount)"
    # RIFE interpolation: N frames + (N-1) gaps × (x-1) interpolated frames per gap
    expected_frame_count = deforum_settings['max_frames'] + (deforum_settings['max_frames'] - 1) * (deforum_settings["frame_interpolation_x_amount"] - 1)
    actual_frame_count = round(clip.duration * clip.fps)
    assert actual_frame_count == expected_frame_count, f"Video frame count does not match RIFE interpolation formula (expected {expected_frame_count}, got {actual_frame_count})"
    assert clip.size == [deforum_settings['W'], deforum_settings['H']] , "Video dimensions are not as expected"        

@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_post_process_UPSCALE(snapshot):
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_post_process_UPSCALE')

    deforum_settings["r_upscale_video"] = True
    deforum_settings["r_upscale_factor"] = "x4"
    deforum_settings["r_upscale_model"] = "realesrgan-x4plus"

    options_overrides = get_test_options_overrides()
    options_overrides.update({
        "deforum_save_gen_info_as_srt": True,
        "deforum_save_gen_info_as_srt_params": get_user_values(),
    })

    response = requests.post(API_BASE_URL+"/batches", json={
        "deforum_settings":[deforum_settings],
        "options_overrides": options_overrides
        })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, f"Job {job_id} failed: {jobStatus}"

    # Ensure parameters used at each frame have not regressed
    srt_filenname =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.srt")
    with open(srt_filenname, 'r') as srt_file:
        assert srt_file.read() == snapshot

    # Ensure interpolated video format is as expected
    video_filenames = glob.glob(f'{jobStatus.outdir}/*Upscaled*.mp4', recursive=True)
    assert len(video_filenames) == 1, "Expected one upscaled video to be generated"

    interpolated_video_filename =  video_filenames[0]
    clip = VideoFileClip(interpolated_video_filename)
    assert clip.fps == deforum_settings['fps'] , "Video FPS does not match input settings"
    assert clip.duration * clip.fps == deforum_settings['max_frames'], "Video frame count does not match input settings"
    assert clip.size == [deforum_settings['W']*4, deforum_settings['H']*4] , "Video dimensions are not as expected (including upscaling)"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_post_process_UPSCALE_FILM(snapshot):
    # Use smaller resolution to avoid OOM when running FILM on upscaled frames
    with open(TESTDATA_DIR / 'simple_small.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_post_process_UPSCALE_FILM')

    deforum_settings["r_upscale_video"] = True
    deforum_settings["r_upscale_factor"] = "x4"
    deforum_settings["r_upscale_model"] = "realesrgan-x4plus"
    deforum_settings["frame_interpolation_engine"] = "FILM"
    deforum_settings["frame_interpolation_x_amount"] = 3
    deforum_settings["frame_interpolation_slow_mo_enabled"] = False
    deforum_settings["frame_interpolation_use_upscaled"] = True

    options_overrides = get_test_options_overrides()
    options_overrides.update({
        "deforum_save_gen_info_as_srt": True,
        "deforum_save_gen_info_as_srt_params": get_user_values(),
    })

    response = requests.post(API_BASE_URL+"/batches", json={
        "deforum_settings":[deforum_settings],
        "options_overrides": options_overrides
        })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, f"Job {job_id} failed: {jobStatus}"

    # Ensure parameters used at each frame have not regressed
    srt_filenname =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.srt")
    with open(srt_filenname, 'r') as srt_file:
        assert srt_file.read() == snapshot

    # Ensure interpolated video format is as expected
    video_filenames = glob.glob(f'{jobStatus.outdir}/*FILM*upscaled*.mp4', recursive=True)
    assert len(video_filenames) == 1, "Expected one upscaled video to be generated"

    interpolated_video_filename =  video_filenames[0]
    clip = VideoFileClip(interpolated_video_filename)
    assert clip.fps == deforum_settings['fps'] * deforum_settings["frame_interpolation_x_amount"]  , "Video FPS does not match input settings (fps * interpolation amount)"
    # FILM may use different interpolation strategy - check actual frame count
    # For now, use same formula as RIFE: N frames + (N-1) gaps × (x-1) interpolated frames per gap
    expected_frame_count = deforum_settings['max_frames'] + (deforum_settings['max_frames'] - 1) * (deforum_settings["frame_interpolation_x_amount"] - 1)
    actual_frame_count = round(clip.duration * clip.fps)
    assert actual_frame_count == expected_frame_count, f"Video frame count does not match interpolation formula (expected {expected_frame_count}, got {actual_frame_count})"
    assert clip.size == [deforum_settings['W']*4, deforum_settings['H']*4] , "Video dimensions are not as expected (including upscaling)"    
