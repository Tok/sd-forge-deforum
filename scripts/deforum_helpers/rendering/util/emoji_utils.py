# noinspection PyUnresolvedReferences
from modules.shared import opts


def _is_emojis_disabled():
    return opts.data.get("deforum_disable_nonessential_emojis", False)


def _select(emoji):
    return '' if _is_emojis_disabled() else emoji


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
    return _select(f'\U0001F578{_suffix}')  # 🕸️


def prompts():
    return _select(f'\U0000270D{_suffix}')  # ✍️


def cadence():
    return _select(f'\U000023F1{_suffix}')  # ⏱️


def distribution():
    return _select('\U0001F4CA')  # 📊


def strength():
    return _select('\U0001F4AA')  # 💪


def scale():
    return _select('\U0001F4CF')  # 📏


def hybrid():
    return _select('\U0001F3AC')  # 🎬


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
