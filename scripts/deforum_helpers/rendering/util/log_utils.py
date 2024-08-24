from . import opt_utils

ESC = "\033["  # ANSI escape character, same as "\x1b["
TERM = "m"  # ANSI terminator

EIGHT_BIT = "38;5;"
TEXT = "38;2;"
BACKGROUND = "48;2;"

COLOUR_RGB = f"{ESC}{TEXT}%d;%d;%d{TERM}"
BG_COLOUR_RGB = f"{ESC}{BACKGROUND}%d;%d;%d{TERM}"
RESET_COLOR = f"{ESC}0{TERM}"

RED = f"{ESC}31{TERM}"
ORANGE = f"{ESC}{EIGHT_BIT}208{TERM}"
YELLOW = f"{ESC}33{TERM}"
GREEN = f"{ESC}32{TERM}"
CYAN = f"{ESC}36{TERM}"
BLUE = f"{ESC}34{TERM}"
INDIGO = f"{ESC}{EIGHT_BIT}66{TERM}"
VIOLET = f"{ESC}{EIGHT_BIT}130{TERM}"
BLACK = f"{ESC}30{TERM}"
WHITE = f"{ESC}37{TERM}"

BOLD = f"{ESC}1{TERM}"
UNDERLINE = f"{ESC}4{TERM}"


def clear_previous_line():
    print(f"{ESC}F{ESC}K", end="")  # "F" is cursor up, "K" is clear line.


def print_tween_frame_from_to_info(key_step, is_disabled=True):
    if not is_disabled:  # replaced with prog bar, but value info print may be useful
        tween_values = key_step.tween_values
        start_i = key_step.tweens[0].i()
        end_i = key_step.tweens[-1].i()
        if end_i > 0:
            formatted_values = [f"{val:.2f}" for val in tween_values]
            count = end_i - start_i + 1
            print(f"{ORANGE}Creating in-between: {RESET_COLOR}{count} frames ({start_i}-->{end_i}){formatted_values}")


def print_animation_frame_info(i, max_frames):
    print("")
    print(f"{CYAN}Animation frame: {RESET_COLOR}{i}/{max_frames}")


def print_tween_frame_info(data, indexes, cadence_flow, tween, is_disabled=True):
    if not is_disabled:  # disabled because it's spamming the cli on high cadence settings.
        msg_flow_name = '' if cadence_flow is None else data.args.anim_args.optical_flow_cadence + ' optical flow '
        msg_frame_info = f"cadence frame: {indexes.tween.i}; tween: {tween:0.2f};"
        print(f"Creating in-between {msg_flow_name}{msg_frame_info}")


def print_init_frame_info(init_frame):
    print(f"Using video init frame {init_frame}")


def print_optical_flow_info(data, optical_flow_redo_generation):
    msg_start = "Optical flow redo is diffusing and warping using"
    msg_end = "optical flow before generation."
    print(f"{msg_start} {optical_flow_redo_generation} and seed {data.args.args.seed} {msg_end}")


def print_redo_generation_info(data, n):
    print(f"Redo generation {n + 1} of {int(data.args.anim_args.diffusion_redo)} before final generation")


def print_tween_step_creation_info(key_steps, index_dist):
    tween_count = sum(len(ks.tweens) for ks in key_steps)
    msg_start = f"Created {len(key_steps)} key frames with {tween_count} tweens."
    msg_end = f"Key frame index distribution: '{index_dist.name}'."
    info(f"{msg_start} {msg_end}")


def print_key_step_debug_info_if_verbose(key_steps):
    for i, ks in enumerate(key_steps):
        tween_indices = [t.i() for t in ks.tweens]
        debug(f"Key frame {ks.i} has {len(tween_indices)} tweens: {tween_indices}")


def print_warning_generate_returned_no_image():
    print(f"{YELLOW}Warning: {RESET_COLOR}Generate returned no image. Skipping to next iteration.")


def print_cuda_memory_state(cuda):
    for i in range(cuda.device_count()):
        print(f"CUDA memory allocated on device {i}: {cuda.memory_allocated(i)} of {cuda.max_memory_allocated(i)}")
        print(f"CUDA memory reserved on device {i}: {cuda.memory_reserved(i)} of {cuda.max_memory_reserved(i)}")


def info(s: str):
    print(f"Info: {s}")


def warn(s: str):
    eye_catcher = "###"
    print(f"{ORANGE}{BOLD}{eye_catcher} Warning: {RESET_COLOR}{s}")


def debug(s: str):
    if opt_utils.is_verbose():
        eye_catcher = "###"
        print(f"{YELLOW}{BOLD}{eye_catcher} Debug: {RESET_COLOR}{s}")
