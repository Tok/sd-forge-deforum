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

# Module constants (legacy - prefer function versions below for emoji toggle support)
refresh = '\U0001f504'  # 🔄
info = f'\U00002139{_suffix}'  # ℹ️
warn = f'\U000026A0{_suffix}'  # ⚠️


# Function versions that respect emoji toggle
def refresh_icon():
    """Refresh/reload icon - respects emoji toggle."""
    return _select('\U0001f504')  # 🔄


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


def rocket():
    return _select('\U0001F680')  # 🚀


def download():
    return _select('\U0001F4E5')  # 📥


def sparkles():
    return _select('\U00002728')  # ✨


def target():
    return _select('\U0001F3AF')  # 🎯


def magnifying_glass():
    return _select('\U0001F50D')  # 🔍


def lock():
    return _select('\U0001F510')  # 🔐


def save():
    return _select('\U0001F4BE')  # 💾


def trash():
    return _select(f'\U0001F5D1{_suffix}')  # 🗑️


def fire():
    return _select('\U0001F525')  # 🔥


def sleeping():
    return _select('\U0001F4A4')  # 💤


def clipboard():
    return _select('\U0001F4CB')  # 📋


def pencil():
    return _select(f'\U0000270F{_suffix}')  # ✏️


def link():
    return _select('\U0001F517')  # 🔗


def brain():
    return _select('\U0001F9E0')  # 🧠


def robot():
    return _select('\U0001F916')  # 🤖


def party():
    return _select('\U0001F389')  # 🎉


def eyes():
    return _select('\U0001F440')  # 👀


def globe():
    return _select(f'\U0001F310')  # 🌐


def microscope():
    return _select('\U0001F52C')  # 🔬


def stop():
    return _select(f'\U000023F9{_suffix}')  # ⏹️


# Slopcore minimal emojis
def blue_square():
    return _select('\U0001F7E6')  # 🟦


def purple_square():
    return _select('\U0001F7EA')  # 🟪


# Status indicators - theme-aware via emoji_if_enabled()
# Named "maybe_*" because they return empty string when emojis are disabled
def maybe_check():
    """Green check mark - theme-aware, returns empty string if emojis disabled.

    In slopcore: ✅ → ✓ (monochrome)
    In classic: ✅ → ✅ (colored)
    Disabled: '' (empty string)
    """
    from deforum.utils.system.logging import emoji_if_enabled
    return emoji_if_enabled('\U00002705')  # ✅


def maybe_cross():
    """Red X mark - theme-aware, returns empty string if emojis disabled.

    In slopcore: ❌ → ✗ (monochrome)
    In classic: ❌ → ❌ (colored)
    Disabled: '' (empty string)
    """
    from deforum.utils.system.logging import emoji_if_enabled
    return emoji_if_enabled('\U0000274C')  # ❌


def maybe_warning():
    """Warning triangle - theme-aware, returns empty string if emojis disabled.

    In slopcore: ⚠️ → ⚠ (no variation selector)
    In classic: ⚠️ → ⚠️ (with variation selector)
    Disabled: '' (empty string)
    """
    from deforum.utils.system.logging import emoji_if_enabled
    return emoji_if_enabled('\U000026A0\U0000FE0F')  # ⚠️


def maybe_alert():
    """Alert/siren - theme-aware, returns empty string if emojis disabled.

    In slopcore: 🚨 → ⚠ (mapped to warning)
    In classic: 🚨 → 🚨 (siren emoji)
    Disabled: '' (empty string)
    """
    from deforum.utils.system.logging import emoji_if_enabled
    return emoji_if_enabled('\U0001F6A8')  # 🚨


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
