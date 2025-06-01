from . import opt_utils

ESC = "\033["  # ANSI escape character with bracket. Same as "\x1b[".
TERM = "m"  # ANSI terminator

EIGHT_BIT = "38;5;"
TEXT = "38;2;"
BACKGROUND = "48;2;"

COLOR_RGB = f"{ESC}{TEXT}%d;%d;%d{TERM}"
BG_COLOR_RGB = f"{ESC}{BACKGROUND}%d;%d;%d{TERM}"
RESET_COLOR = f"{ESC}0{TERM}"

HEX_RED = '#FE797B'
HEX_ORANGE = '#FFB750'
HEX_YELLOW = '#FFEA56'
HEX_GREEN = '#8FE968'
HEX_BLUE = '#36CEDC'
HEX_PURPLE = '#A587CA'


def from_hex_color(hex_color):
    def _hex_to_rgb(color):
        color = color.lstrip('#')  # Remove '#' if present
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    r, g, b = _hex_to_rgb(hex_color)
    return f"{ESC}{TEXT}{r};{g};{b}{TERM}"


RED = from_hex_color(HEX_RED)
ORANGE = from_hex_color(HEX_ORANGE)
YELLOW = from_hex_color(HEX_YELLOW)
GREEN = from_hex_color(HEX_GREEN)
BLUE = from_hex_color(HEX_BLUE)
PURPLE = from_hex_color(HEX_PURPLE)

BOLD = f"{ESC}1{TERM}"
UNDERLINE = f"{ESC}4{TERM}"


def clear_next_n_lines(n):
    clear_line = f'{ESC}2K{ESC}0G'  # Clear the entire line and move cursor to beginning
    print(f'{ESC}B{clear_line}' * n, end="")  # Move cursor down with B and clear the line n times.
    print(f'{ESC}F' * n, end="", flush=True)  # Move cursor back up n times.


def print_tween_frame_from_to_info(frame, is_disabled=True):
    if not is_disabled:  # replaced with prog bar, but value info print may be useful
        start_i = frame.tweens[0].i
        end_i = frame.tweens[-1].i
        if end_i > 0:
            formatted_values = [f"{val:.2f}" for val in frame.tween_values()]
            count = end_i - start_i + 1
            print(f"{ORANGE}Creating in-between: {RESET_COLOR}{count} frames ({start_i}-->{end_i}){formatted_values}")


def print_animation_frame_info(i, max_frames):
    print("")
    print(f"{BLUE}Animation frame: {RESET_COLOR}{BOLD}{i}{RESET_COLOR}/{max_frames}")


def print_tween_frame_info(data, i, cadence_flow, tween, is_disabled=True):
    if not is_disabled:  # disabled because it's spamming the cli on high cadence settings.
        msg_flow_name = '' if cadence_flow is None else data.args.anim_args.optical_flow_cadence + ' optical flow '
        msg_frame_info = f"cadence frame: {i}; tween: {tween:0.2f};"
        print(f"Creating in-between {msg_flow_name}{msg_frame_info}")


def print_init_frame_info(init_frame):
    print(f"Using video init frame {init_frame}")


def print_optical_flow_info(data, optical_flow_redo_generation, random_seed):
    msg_start = "Optical flow redo is diffusing and warping using"
    msg_end = "optical flow before generation."
    print(f"{msg_start} {optical_flow_redo_generation} and seed {random_seed} {msg_end}")


def print_redo_generation_info(data, n):
    print(f"Redo generation {n + 1} of {int(data.args.anim_args.diffusion_redo)} before final generation")


def print_tween_frame_creation_info(key_frames, index_dist):
    tween_count = sum(len(ks.tweens) for ks in key_frames)
    msg_start = f"Created {len(key_frames)} key frames with {tween_count} tweens."
    msg_end = f"Key frame index distribution: '{index_dist.name}'."
    info(f"{msg_start} {msg_end}")


def print_key_frame_debug_info_if_verbose(diffusion_frames):
    for i, df in enumerate(diffusion_frames):
        tween_indices = [t.i for t in df.tweens]
        frame_type = "Key Frame" if df.is_keyframe else "    Frame"
        tween_count = len(tween_indices)
        if tween_count > 6:  # Limit to first 3 and last 3
            first_three = [f"{index:05}" for index in tween_indices[:3]]
            last_three = [f"{index:05}" for index in tween_indices[-3:]]
            displayed_tweens = f"[{', '.join(first_three)}] ... [{', '.join(last_three)}]"
        else:
            displayed_tweens = [f"{index:05}" for index in tween_indices]
            displayed_tweens = f"[{', '.join(displayed_tweens)}]"
        debug(f"{frame_type} {df.i:05} has {tween_count:03} tweens: {displayed_tweens}")


def print_warning_generate_returned_no_image():
    print(f"{YELLOW}Warning: {RESET_COLOR}Generate returned no image. Skipping to next iteration.")


def print_cuda_memory_state(cuda):
    for i in range(cuda.device_count()):
        print(f"CUDA memory allocated on device {i}: {cuda.memory_allocated(i)} of {cuda.max_memory_allocated(i)}")
        print(f"CUDA memory reserved on device {i}: {cuda.memory_reserved(i)} of {cuda.max_memory_reserved(i)}")


def info(s: str, color: str = None):
    message = f"{color}{s}{RESET_COLOR}" if color else s
    print(f"{BLUE}{BOLD}Info: {RESET_COLOR}{message}")


def error(s: str):
    print(f"{RED}{BOLD}Error: {RESET_COLOR}{s}")


def warn(s: str):
    print(f"{ORANGE}{BOLD}Warning: {RESET_COLOR}{s}")


def debug(s: str):
    if opt_utils.is_verbose():
        print(f"{YELLOW}{BOLD}Debug: {RESET_COLOR}{s}")
