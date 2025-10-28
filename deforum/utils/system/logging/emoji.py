def _select(emoji):
    # Check if emojis are enabled before returning
    try:
        from deforum.rendering.options import is_emojis_enabled
        if not is_emojis_enabled():
            return ''
    except:
        # During early initialization, settings might not be available
        # Default to no emoji (matches UI default)
        return ''

    return emoji


# Use emojis sparingly to catch attention to essential items.
_suffix = '\U0000FE0F'

# essentials, shouldn't be turned off.
refresh = '\U0001f504'  # 🔄
info = f'\U00002139{_suffix}'  # ℹ️
warn = f'\U000026A0{_suffix}'  # ⚠️


def bulb():
    return _select('\U0001F4A1')  # 💡


def run():
    return _select(f'\U0001F3CE{_suffix}')  # 🏎️


def key():
    return _select('\U0001F511')  # 🔑


def frame():
    return _select(f'\U0001F5BC{_suffix}')  # 🖼️


def control():
    return _select(f'\U0001F39B{_suffix}')  # 🎛️


def net():
    return _select(f'\U0001F945{_suffix}')  # 🥅


def web():
    return _select(f'\U0001F578{_suffix}')  # 🕸️


def prompts():
    return _select(f'\U0000270D{_suffix}')  # ✍️


def cadence():
    return _select(f'\U000023F1{_suffix}')  # ⏱️


def off():
    return _select(f'\U0000274C')  # ❌


def distribution():
    return _select('\U0001F4CA')  # 📊


def strength():
    return _select('\U0001F4AA')  # 💪


def scale():
    return _select('\U0001F4CF')  # 📏




def video_camera():
    return _select('\U0001F4F9')  # 📹


def wan_video():
    return _select('\U0001F3A5')  # 🎥


def document():
    return _select('\U0001F4C4')  # 📄


def steps():
    return _select('\U0001F463')  # 👣


def numbers():
    return _select('\U0001F522')  # 🔢


def sound():
    return _select('\U0001F3B5')  # 🎵


def music():
    return _select('\U0001F3B6')  # 🎶


def frames():
    return _select(f'\U0001F39E{_suffix}')  # 🎞️


def up():
    return _select('\U0001F199')  # 🆙


def seed():
    return _select('\U0001F330')  # 🌰


def subseed():
    return _select('\U0001F95C')  # 🥜


def leaf():
    return _select('\U0001F343')  # 🍃


def bicycle():
    return _select('\U0001F6B2')  # 🚲


def hole():
    return _select(f'\U0001F573{_suffix}')  # 🕳️


def palette():
    return _select('\U0001F3A8')  # 🎨


def wave():
    return _select('\U0001F30A')  # 🌊


def broom():
    return _select('\U0001F9F9')  # 🧹


def masking():
    return _select('\U0001F3AD')  # 🎭


def gear():
    return _select(f'\U00002699{_suffix}')  # ⚙️


def wrench():
    return _select('\U0001F527')  # 🔧


def stopwatch():
    return _select(f'\U000023F1{_suffix}')  # ⏱️


def tools():
    return _select(f'\U0001F6E0{_suffix}')  # 🛠️


def movie_camera():
    return _select('\U0001F3AC')  # 🎬


def dice():
    return _select('\U0001F3B2')  # 🎲


def folder():
    return _select('\U0001F4C1')  # 📁


# Slopcore minimal emojis
def blue_square():
    return _select('\U0001F7E6')  # 🟦


def purple_square():
    return _select('\U0001F7EA')  # 🟪


# ============================================================================
# Theme-based Emoji Mapping
# ============================================================================

def get_themed_emoji(emoji_name: str, theme: str = 'classic') -> str:
    """Get emoji based on theme.

    Args:
        emoji_name: Name of emoji function (e.g., 'run', 'key', 'frame')
        theme: One of 'slopcore', 'classic', 'simple'

    Returns:
        Emoji string based on theme (caller is responsible for checking if emojis are enabled)
    """
    if theme == 'simple':
        # Simple theme: use emojis but keep them minimal
        if emoji_name in globals():
            emoji_val = globals()[emoji_name]
            return emoji_val() if callable(emoji_val) else emoji_val
        return ''

    elif theme == 'slopcore':
        # Slopcore theme: map most emojis to 🟦 or 🟪
        # Use 🟦 for inputs/processing, 🟪 for outputs/completion
        SLOPCORE_BLUE_OPS = [
            'run', 'key', 'frame', 'control', 'prompts', 'cadence',
            'steps', 'numbers', 'sound', 'music', 'seed', 'subseed',
            'leaf', 'bicycle', 'gear', 'wrench', 'stopwatch', 'tools',
        ]
        SLOPCORE_PURPLE_OPS = [
            'video_camera', 'wan_video', 'document', 'frames',
            'movie_camera', 'distribution', 'strength', 'scale',
        ]

        if emoji_name in SLOPCORE_BLUE_OPS:
            return blue_square()
        elif emoji_name in SLOPCORE_PURPLE_OPS:
            return purple_square()
        else:
            # Default fallback for unmapped emojis
            return blue_square()

    else:  # classic theme
        # Classic theme: use full emoji set
        if emoji_name in globals():
            emoji_val = globals()[emoji_name]
            return emoji_val() if callable(emoji_val) else emoji_val
        return ''
