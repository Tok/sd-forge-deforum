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

import concurrent.futures
import glob
import math
import os
import re
import requests
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
from torch.hub import download_url_to_file

# noinspection PyUnresolvedReferences
from modules.shared import state, opts
from pkg_resources import resource_filename

from .general_utils import checksum, clean_gradio_path_strings, debug_print
from .http_client import get_http_client
from .rich import console

SUPPORTED_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "webp"]
SUPPORTED_VIDEO_EXTENSIONS = ["mov", "mpeg", "mp4", "m4v", "avi", "mpg", "webm"]

def convert_image(input_path, output_path):
    extension = get_extension_if_valid(input_path, SUPPORTED_IMAGE_EXTENSIONS)

    if not extension:
        return

    # Read the input image
    if input_path.startswith('http://') or input_path.startswith('https://'):
        resp = get_http_client().get(input_path, allow_redirects=True)
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
    else: 
        img = cv2.imread(input_path)

    # Convert the image to the specified output format
    if extension == "png":
        cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    elif extension == "jpg" or extension == "jpeg":
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 99])
    else:
        cv2.imwrite(output_path, img)


def get_ffmpeg_params(): # get ffmpeg params from webui's settings -> deforum tab. actual opts are set in deforum.py
    f_location = opts.data.get("deforum_ffmpeg_location", find_ffmpeg_binary())
    f_crf = opts.data.get("deforum_ffmpeg_crf", 17)
    f_preset = opts.data.get("deforum_ffmpeg_preset", 'slow')

    return [f_location, f_crf, f_preset]

def get_ffmpeg_paths(outdir, timestring, anim_args, video_args, output_suffix=''):
    image_path = os.path.join(outdir, f"{timestring}_%09d.png")
    mp4_path = os.path.join(outdir, f"{timestring}{output_suffix}.mp4")
    
    real_audio_track = None
    if video_args.add_soundtrack != 'None':
        real_audio_track = anim_args.video_init_path if video_args.add_soundtrack == 'Init Video' else video_args.soundtrack_path

    srt_path = None
    if opts.data.get("deforum_save_gen_info_as_srt", False) and opts.data.get("deforum_embed_srt", False):
        srt_path = os.path.join(outdir, f"{timestring}.srt")
        
    return [image_path, mp4_path, real_audio_track, srt_path]

# e.g gets 'x2' returns just 2 as int
def extract_number(string):
    return int(string[1:]) if len(string) > 1 and string[1:].isdigit() else -1
    
def save_frame(image, file_path):
    cv2.imwrite(file_path, image)

def vid2frames(video_path, video_in_frame_path, n=1, overwrite=True, extract_from_frame=0, extract_to_frame=-1, out_img_format='jpg', numeric_files_output = False):
    start_time = time.time()
    if (extract_to_frame <= extract_from_frame) and extract_to_frame != -1:
        raise RuntimeError('Error: extract_to_frame can not be higher than extract_from_frame')

    if n < 1: n = 1 #HACK Gradio interface does not currently allow min/max in gr.Number(...) 

    video_path = clean_gradio_path_strings(video_path)
    # check vid path using a function and only enter if we get True
    if get_extension_if_valid(video_path, SUPPORTED_VIDEO_EXTENSIONS):

        name = get_frame_name(video_path)

        if not (video_path.startswith('http://') or video_path.startswith('https://')):
            video_path = os.path.realpath(video_path)

        vidcap = cv2.VideoCapture(video_path)
        video_fps = vidcap.get(cv2.CAP_PROP_FPS)

        input_content = []
        if os.path.exists(video_in_frame_path) :
            input_content = os.listdir(video_in_frame_path)

        # check if existing frame is the same video, if not we need to erase it and repopulate
        if len(input_content) > 0 and numeric_files_output is False:
            #get the name of the existing frame
            content_name = get_frame_name(input_content[0])
            if not content_name.startswith(name):
                overwrite = True

        # grab the frame count to check against existing directory len 
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) 

        # raise error if the user wants to skip more frames than exist
        if n >= frame_count : 
            raise RuntimeError('Skipping more frames than input video contains. extract_nth_frames larger than input frames')

        expected_frame_count = math.ceil(frame_count / n) 
        # Check to see if the frame count is matches the number of files in path
        if overwrite or expected_frame_count != len(input_content):
            shutil.rmtree(video_in_frame_path)
            os.makedirs(video_in_frame_path, exist_ok=True) # just deleted the folder so we need to make it again
            input_content = os.listdir(video_in_frame_path)

        print(f"Trying to extract frames from video with input FPS of {video_fps}. Please wait patiently.")
        if len(input_content) == 0:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, extract_from_frame) # Set the starting frame
            success,image = vidcap.read()
            count = extract_from_frame
            t=0
            success = True
            max_workers = int(max(1, (os.cpu_count() / 2) - 1)) # set max threads to cpu cores halved, minus 1. minimum is 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                while success:
                    if state.interrupted:
                        return
                    if (count <= extract_to_frame or extract_to_frame == -1) and count % n == 0:
                        if numeric_files_output == True:
                            file_name = f"{t:09}.{out_img_format}"
                        else:
                            file_name = f"{name}{t:09}.{out_img_format}"
                        file_path = os.path.join(video_in_frame_path, file_name)
                        executor.submit(save_frame, image, file_path)
                        t += 1
                    count += 1
                    success, image = vidcap.read()
            print(f"Extracted {count} frames from video in {time.time() - start_time:.2f} seconds!")
        else:
            print("Frames already unpacked")
        vidcap.release()
        return video_fps

# Make sure  path_to_check provided is an existing local file or a web URL with a supported file extension
# Check Content-Disposition if necessary.
# If so, return the extension. If not, raise an error.
def get_extension_if_valid(path_to_check, acceptable_extensions: list[str] ) -> str:

    if path_to_check.startswith('http://') or path_to_check.startswith('https://'):
        extension = path_to_check.rsplit('?', 1)[0].rsplit('.', 1)[-1] # remove query string before checking file format extension.
        if extension in acceptable_extensions:
            return extension
        

        # Path is actually a URL. Make sure it resolves and has a valid file extension.
        response = get_http_client().head(path_to_check, allow_redirects=True)
        if response.status_code != 200:
            # Pre-signed URLs for GET requests might not like HEAD requests. Try a 0 range GET request.
            debug_print("Failed  HEAD request, trying 0 range GET. Status code: " + str(response.status_code))
            response = get_http_client().get(path_to_check, headers={'Range': 'bytes=0-0'}, allow_redirects=True)
            if response.status_code != 200:
                debug_print("Also failed 0 range GET. Status code: " + str(response.status_code))
                raise ConnectionError(f"URL {path_to_check} is not valid. Response status code: {response.status_code}")        

        content_disposition_extension = None
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition:
            match = re.search(r'filename="?(?P<filename>[^"]+)"?', content_disposition)
            if match:
                content_disposition_extension = match.group('filename').rsplit('.', 1)[-1].lower()
        
        if content_disposition_extension in acceptable_extensions:
            return content_disposition_extension
        

        raise ValueError(f"File {path_to_check} has format '{extension}' (from URL) or '{content_disposition_extension}' (from content disposition), which are not supported. Supported formats are: {acceptable_extensions}")
    
    else:
        path_to_check = os.path.realpath(path_to_check)
        extension = path_to_check.rsplit('.', 1)[-1].lower()
        
        if not os.path.exists(path_to_check):
            raise RuntimeError(f"Path does not exist: {path_to_check}")
        if extension in acceptable_extensions:
            return extension
            
    raise ValueError(f"File {path_to_check} has format '{extension}', which is not supported. Supported formats are: {acceptable_extensions}")



# quick-retreive frame count, FPS and H/W dimensions of a video (local or URL-based)
def get_quick_vid_info(vid_path):
    vidcap = cv2.VideoCapture(vid_path)
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    video_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vidcap.release()
    if video_fps.is_integer():
        video_fps = int(video_fps)

    return video_fps, video_frame_count, (video_width, video_height)


def download_audio(audio_path):
    # TODO? check if file already exists and don't DL again if it does.
    audio_path = clean_gradio_path_strings(audio_path)
    if audio_path.startswith(('http://', 'https://')):
        url = audio_path
        print(f"Downloading audio file from: {url}")
        response = get_http_client().get(url, stream=True)
        response.raise_for_status()
        _, ext = os.path.splitext(url)  # Extract the file extension from the URL..
        ext = ext if ext else '.mp3'  # ..or default to .mp3 if no extension is present
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    temp_file.write(chunk)
            audio_path = temp_file.name
            print(f"Audio saved to: {audio_path}")
        temp_file.close()
    return audio_path


# Stitch images to a h264 mp4 video using ffmpeg
def ffmpeg_stitch_video(ffmpeg_location=None, fps=None, outmp4_path=None, stitch_from_frame=0, stitch_to_frame=None,
                        imgs_path=None, add_soundtrack=None, audio_path=None, crf=17, preset='veryslow', srt_path=None):
    start_time = time.time()

    # Download audio at the beginning if soundtrack is enabled
    downloaded_audio_path = None
    if add_soundtrack != 'None' and audio_path is not None:
        try:
            print(f"Downloading audio file from: {audio_path}")
            downloaded_audio_path = download_audio(audio_path)
            print(f"Audio downloaded to: {downloaded_audio_path}")
        except Exception as e:
            print(f"Error downloading audio: {e}")
    
    print(f"Got a request to stitch frames to video using FFmpeg.\nFrames:\n{imgs_path}\nTo Video:\n{outmp4_path}")
    msg_to_print = f"Stitching *video*..."
    console.print(msg_to_print, style="blink yellow", end="") 
    if stitch_to_frame == -1:
        stitch_to_frame = 999999999
    try:
        cmd = [
            ffmpeg_location,
            '-y',
            '-r', str(float(fps)),
            '-start_number', str(stitch_from_frame),
            '-i', imgs_path,
            '-frames:v', str(stitch_to_frame),
            '-c:v', 'libx264',
            '-vf',
            f'fps={float(fps)}',
            '-pix_fmt', 'yuv420p',
            '-crf', str(crf),
            '-preset', preset,
            '-pattern_type', 'sequence',
            outmp4_path
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        
        # Check if the process was successful
        if process.returncode != 0:
            print(f"FFmpeg stderr: {stderr}")
            print(f"FFmpeg stdout: {stdout}")
            raise RuntimeError(f"FFmpeg failed with return code {process.returncode}: {stderr}")
            
    except FileNotFoundError:
        print("\r" + " " * len(msg_to_print), end="", flush=True)
        print(f"\r{msg_to_print}", flush=True)
        raise FileNotFoundError("FFmpeg not found. Please make sure you have a working ffmpeg path under 'ffmpeg_location' parameter.")
    except Exception as e:
        print("\r" + " " * len(msg_to_print), end="", flush=True)
        print(f"\r{msg_to_print}", flush=True)
        raise Exception(f'Error stitching frames to video. Actual runtime error:{e}')
    
    add_soundtrack_status = None
    add_soundtrack_success = None
    temp_file = None
    if add_soundtrack != 'None':
        try:
            # Use the previously downloaded audio path
            if downloaded_audio_path is not None:
                audio_path = downloaded_audio_path
            else:
                # Fallback to download now if it wasn't downloaded earlier
                audio_path = download_audio(audio_path)
                
            audio_add_start_time = time.time()
            cmd = [
                ffmpeg_location,
                '-i',
                outmp4_path,
                '-i',
                audio_path,
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', 'copy',
                '-shortest',
                outmp4_path+'.temp.mp4'
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(stderr)
            os.replace(outmp4_path+'.temp.mp4', outmp4_path)
            add_soundtrack_status = f"\rFFmpeg audio embedding \033[0;32mdone\033[0m in {time.time() - audio_add_start_time:.2f} seconds!"
            add_soundtrack_success = True
        except Exception as e:
            add_soundtrack_status = f"\rError adding audio to video: {e}"
            add_soundtrack_success = False
        finally:
            if temp_file is not None:
                file_path = Path(temp_file.name)
                file_path.unlink(missing_ok=True)
            
    add_srt = opts.data.get("deforum_save_gen_info_as_srt", False) and opts.data.get("deforum_embed_srt", False) and srt_path is not None
    add_srt_status = None
    add_srt_success = None
    if add_srt:
        try:
            srt_add_start_time = time.time()
            cmd = [
                ffmpeg_location,
                '-i', outmp4_path,
                '-i', srt_path,
                '-c', 'copy',
                '-c:s', 'mov_text',
                '-metadata:s:s:0', 'title=Deforum Data',
                outmp4_path+'.temp.mp4'
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(stderr)
            os.replace(outmp4_path+'.temp.mp4', outmp4_path)
            add_srt_status = f"\rFFmpeg subtitle embedding \033[0;32mdone\033[0m in {time.time() - srt_add_start_time:.2f} seconds!"
            add_srt_success = True
        except Exception as e:
            add_srt_status = f"\rError adding subtitles to video: {e}"
            add_srt_success = False

    print("\r" + " " * len(msg_to_print), end="", flush=True)
    print(f"\r{msg_to_print}", flush=True)

    status_summary = f"\rVideo stitching \033[0;32mdone\033[0m in {time.time() - start_time:.2f} seconds!"
    if add_soundtrack_status:
        print(add_soundtrack_status, flush=True)
        status_summary += " Audio embedded successfully." if add_soundtrack_success else " Sorry, no audio - see above for errors."
    if add_srt_status:
        print(add_srt_status, flush=True)
        status_summary += " Subtitles embedded successfully." if add_srt_success else " Sorry, no subtitles - see above for errors."

    print(status_summary, flush=True)

def get_frame_name(path):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    return name
    
def get_next_frame(outdir, video_path, frame_idx, mask=False):
    frame_path = 'inputframes'
    if (mask): frame_path = 'maskframes'
    return os.path.join(outdir, frame_path, get_frame_name(video_path) + f"{frame_idx:09}.jpg")
     
def find_ffmpeg_binary():
    try:
        import google.colab
        return 'ffmpeg'
    except:
        pass
    for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
        try:
            package_path = resource_filename(package, 'binaries')
            files = [os.path.join(package_path, f) for f in os.listdir(package_path) if f.startswith("ffmpeg-")]
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return files[0] if files else 'ffmpeg'
        except:
            return 'ffmpeg'
          
# These 2 functions belong to "stitch frames to video" in Output tab
def get_manual_frame_to_vid_output_path(input_path):
    dir_name = os.path.dirname(input_path)
    folder_name = os.path.basename(dir_name)
    output_path = os.path.join(dir_name, f"{folder_name}.mp4")
    i = 1
    while os.path.exists(output_path):
        output_path = os.path.join(dir_name, f"{folder_name}_{i}.mp4")
        i += 1
    return output_path

def direct_stitch_vid_from_frames(image_path, fps, add_soundtrack, audio_path):
    f_location, f_crf, f_preset = get_ffmpeg_params()
    matching_files = glob.glob(re.sub(r'%\d*d', '*', image_path))
    min_id = None
    for file in matching_files:
        try:
            id = int(re.search(r'(\d+)(?=\.\w+$)', file).group(1))
            min_id = min(min_id, id) if min_id is not None else id
        except (AttributeError, ValueError):
            pass
    if min_id is None or not all(os.path.isfile(image_path % (min_id + i)) for i in range(2)):
        print("Couldn't find images that match the provided path/ pattern. At least 2 matched images are required.")
        return
    out_mp4_path = get_manual_frame_to_vid_output_path(image_path)
    ffmpeg_stitch_video(ffmpeg_location=f_location, fps=fps, outmp4_path=out_mp4_path, stitch_from_frame=min_id,
                        stitch_to_frame=-1, imgs_path=image_path, add_soundtrack=add_soundtrack, audio_path=audio_path,
                        crf=f_crf, preset=f_preset)


# end of 2 stitch frame to video funcs

# returns True if filename (could be also media URL) contains an audio stream, othehrwise False
def media_file_has_audio(filename, ffmpeg_location):
    result = subprocess.run([ffmpeg_location, "-i", filename, "-af", "volumedetect", "-f", "null", "-"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    output = result.stderr.decode()
    return True if "Stream #0:1: Audio: " in output or "Stream #0:1(und): Audio" in output else False

# download gifski binaries if needed - linux and windows only atm (apple users won't even see the option)
def check_and_download_gifski(models_folder, current_user_os):
    if current_user_os == 'Windows':
        file_name = 'gifski.exe'
        checksum_value = 'b0dd261ad021c31c7fdb99db761b45165e6b2a7e8e09c5d070a2b8064b575d7a4976c364d8508b28a6940343119b16a23e9f7d76f1f3d5ff02289d3068b469cf'
        download_url = 'https://github.com/hithereai/d/releases/download/giski-windows-bin/gifski.exe'
    elif current_user_os == 'Linux':
        file_name = 'gifski'
        checksum_value = 'e65bf9502bca520a7fd373397e41078d5c73db12ec3e9b47458c282d076c04fa697adecb5debb5d37fc9cbbee0673bb95e78d92c1cf813b4f5cc1cabe96880ff'
        download_url = 'https://github.com/hithereai/d/releases/download/gifski-linux-bin/gifski'
    elif current_user_os == 'Mac':
        file_name = 'gifski'
        checksum_value = '622a65d25609677169ed2c1c53fd9aa496a98b357cf84d0c3627ae99c85a565d61ca42cdc4d24ed6d60403bb79b6866ce24f3c4b6fff58c4d27632264a96353c'
        download_url = 'https://github.com/hithereai/d/releases/download/gifski-mac-bin/gifski'
    else: # who are you then?
        raise Exception(f"No support for OS type: {current_user_os}")
        
    file_path = os.path.join(models_folder, file_name)
    
    if not os.path.exists(file_path):
        download_url_to_file(download_url, models_folder)
        if current_user_os in ['Linux','Mac']:
            os.chmod(file_path, 0o755)
            if current_user_os == 'Mac':
                # enable running the exec for mac users
                os.system(f'xattr -d com.apple.quarantine "{file_path}"')
        if checksum(file_path) != checksum_value:
            raise Exception(f"Error while downloading {file_name}. Please download from: {download_url} and place in: {models_folder}")
           
# create a gif using gifski - limited to up to 30 fps (from the ui; if users wanna try to hack it, results are not good, but possible up to 100 fps theoretically)   
def make_gifski_gif(imgs_raw_path, imgs_batch_id, fps, models_folder, current_user_os):
    msg_to_print = f"Stitching *gif* from frames using Gifski..."
    # blink the msg in the cli until action is done
    console.print(msg_to_print, style="blink yellow", end="") 
    start_time = time.time()
    gifski_location = os.path.join(models_folder, 'gifski' + ('.exe' if current_user_os == 'Windows' else ''))
    final_gif_path = os.path.join(imgs_raw_path, imgs_batch_id + '.gif')
    if current_user_os == "Linux":
        input_img_pattern = imgs_batch_id + '_0*.png'
        input_img_files = [os.path.join(imgs_raw_path, file) for file in sorted(glob.glob(os.path.join(imgs_raw_path, input_img_pattern)))]
        cmd = [gifski_location, '-o', final_gif_path] + input_img_files + ['--fps', str(fps), '--quality', str(95)]
    elif current_user_os == "Windows":
        input_img_pattern_for_gifski = os.path.join(imgs_raw_path, imgs_batch_id + '_0*.png')
        cmd = [gifski_location, '-o', final_gif_path, input_img_pattern_for_gifski, '--fps', str(fps), '--quality', str(95)]
    else: # should never this else as we check before, but just in case
        print("\r" + " " * len(msg_to_print), end="", flush=True)
        print(f"\r{msg_to_print}", flush=True)
        raise Exception(f"No support for OS type: {current_user_os}")
        
    check_and_download_gifski(models_folder, current_user_os)

    try:
        process = subprocess.run(cmd, capture_output=True, check=True, text=True, cwd=(models_folder if current_user_os == 'Mac' else None))
        print("\r" + " " * len(msg_to_print), end="", flush=True)
        print(f"\r{msg_to_print}", flush=True)
        print(f"GIF stitching \033[0;32mdone\033[0m in {time.time() - start_time:.2f} seconds!")
    except Exception as e:
        print("\r" + " " * len(msg_to_print), end="", flush=True)
        print(f"\r{msg_to_print}", flush=True)
        print(f"GIF stitching *failed* with error:\n{e}")
        
def handle_imgs_deletion(vid_path=None, imgs_folder_path=None, batch_id=None):
    try:
        total_imgs_to_delete = count_matching_frames(imgs_folder_path, batch_id)
        if total_imgs_to_delete is None or total_imgs_to_delete == 0:
            return
        print("Deleting raw images, as requested:")
        _, fcount, _ = get_quick_vid_info(vid_path)
        if fcount == total_imgs_to_delete:
            total_imgs_deleted = delete_matching_frames(imgs_folder_path, batch_id)
            print(f"Deleted {total_imgs_deleted} out of {total_imgs_to_delete} imgs!")
        else:
            print("Did not delete imgs as there was a mismatch between # of frames in folder, and # of frames in actual video. Please check and delete manually. ")
    except Exception as e:
        print(f"Error deleting raw images. Please delete them manually if you want. Actual error:\n{e}")

# handle deletion of inputframes created by video frame extraction
def handle_input_frames_deletion(imgs_folder_path=None):
    try:
        total_imgs_to_delete = count_matching_frames(imgs_folder_path, None)
        if total_imgs_to_delete is None or total_imgs_to_delete == 0:
            return
        print("Deleting input frames, as requested:")
        total_imgs_deleted = delete_input_frames(imgs_folder_path)
        print(f"Deleted {total_imgs_deleted} out of {total_imgs_to_delete} inputframes!")
        os.rmdir(imgs_folder_path)
    except Exception as e:
        print(f"Error deleting input frames. Please delete them manually if you want. Actual error:\n{e}")

def handle_cn_frames_deletion(cn_input_frames_list):
    try:
        for cn_inputframes_folder in cn_input_frames_list:
            if os.path.exists(cn_inputframes_folder):
                total_cn_imgs_to_delete = count_matching_frames(cn_inputframes_folder, None)
                if total_cn_imgs_to_delete is None or total_cn_imgs_to_delete == 0:
                    continue
                total_imgs_deleted = delete_input_frames(cn_inputframes_folder)
                print(f"Deleted {total_imgs_deleted} CN inputframes out of {total_cn_imgs_to_delete}!")
                os.rmdir(cn_inputframes_folder)
    except Exception as e:
        print(f"Error deleting CN input frames. Please delete them manually if you want. Actual error:\n{e}")

def delete_matching_frames(from_folder, img_batch_id):
    return sum(1 for f in os.listdir(from_folder) if get_matching_frame(f, img_batch_id) and os.remove(os.path.join(from_folder, f)) is None)

# delete inputframes
def delete_input_frames(from_folder):
    return sum(1 for f in os.listdir(from_folder) if os.remove(os.path.join(from_folder, f)) is None)
    
def count_matching_frames(from_folder, img_batch_id):
    if str(from_folder).endswith("inputframes"):
        return sum(1 for f in os.listdir(from_folder))
    return sum(1 for f in os.listdir(from_folder) if get_matching_frame(f, img_batch_id))

def get_matching_frame(f, img_batch_id=None):
    return ('png' in f or 'jpg' in f) and '-' not in f and '_depth_' not in f and ((img_batch_id is not None and f.startswith(img_batch_id) or img_batch_id is None))

def render_preview(args, anim_args, video_args, root, frame_idx, last_preview_frame):
    is_preview_on = "on" in opts.data.get("deforum_preview", "off").lower()
    preview_interval_frames = opts.data.get("deforum_preview_interval_frames", 50)
    is_preview_frame = (frame_idx % preview_interval_frames) == 0 or (frame_idx - last_preview_frame) >= preview_interval_frames
    is_close_to_end = frame_idx >= (anim_args.max_frames-1)

    debug_print(f"render preview video: frame_idx={frame_idx} preview_interval_frames={preview_interval_frames} anim_args.max_frames={anim_args.max_frames} is_preview_on={is_preview_on} is_preview_frame={is_preview_frame} is_close_to_end={is_close_to_end} ")

    if not is_preview_on or not is_preview_frame or is_close_to_end:
        debug_print(f"No preview video on frame {frame_idx}.")
        return last_preview_frame
    
    f_location, f_crf, f_preset = get_ffmpeg_params() # get params for ffmpeg exec
    image_path, mp4_temp_path, real_audio_track, srt_path = get_ffmpeg_paths(args.outdir, root.timestring, anim_args, video_args, "_preview__rendering__")
    mp4_preview_path = mp4_temp_path.replace("_preview__rendering__", "_preview")
    def task():
        if os.path.exists(mp4_temp_path):
            print(f"--! Skipping preview video on frame {frame_idx} (previous preview still rendering to {mp4_temp_path}...")            
        else:
            print(f"--> Rendering preview video up to frame {frame_idx} to {mp4_preview_path}...")
            try:
                ffmpeg_stitch_video(ffmpeg_location=f_location, fps=video_args.fps, outmp4_path=mp4_temp_path, stitch_from_frame=0, stitch_to_frame=frame_idx, imgs_path=image_path, add_soundtrack=video_args.add_soundtrack, audio_path=real_audio_track, crf=f_crf, preset=f_preset, srt_path=srt_path)
            finally:
                shutil.move(mp4_temp_path, mp4_preview_path)
        
    if "concurrent" in opts.data.get("deforum_preview", "off").lower():
        Thread(target=task).start()
    else:
        task()

    return frame_idx
