"""
Simplified emoji utilities for Deforum UI
Provides central emoji configuration with easy on/off toggle
"""

from . import opt_utils


def _emoji(char, essential=False):
    """Return emoji character or empty string based on settings
    
    Args:
        char: The emoji character (direct unicode, not code)
        essential: If True, always show even when emojis are disabled
    """
    if essential:
        return char
    return '' if opt_utils.is_emojis_disabled() else char


# Essential emojis (always shown for critical UI elements)
refresh = _emoji('🔄', essential=True)
info = _emoji('ℹ️', essential=True) 
warn = _emoji('⚠️', essential=True)

# UI Tab emojis (respectful of user's emoji preferences)
def run(): return _emoji('🏎️')
def key(): return _emoji('🔑')
def prompts(): return _emoji('✍️')
def wan_ai(): return _emoji('🤖')
def setup(): return _emoji('🎯')
def animation(): return _emoji('🎬')
def advanced(): return _emoji('⚙️')
def output(): return _emoji('📤')
def post_process(): return _emoji('🎬')

# Content type emojis
def distribution(): return _emoji('📊')
def strength(): return _emoji('💪')
def scale(): return _emoji('📏')
def seed(): return _emoji('🌰')
def subseed(): return _emoji('🥜')
def motion(): return _emoji('🚲')
def noise(): return _emoji('🌊')
def coherence(): return _emoji('🎨')
def anti_blur(): return _emoji('🧹')
def depth(): return _emoji('🕳️')
def video_camera(): return _emoji('📹')
def frames(): return _emoji('🎞️')
def upscaling(): return _emoji('🆙')
def document(): return _emoji('📄')

# Legacy aliases for backward compatibility
def bicycle(): return motion()
def palette(): return coherence()
def broom(): return anti_blur()
def hole(): return depth()
def up(): return upscaling()
def numbers(): return _emoji('🔢')
def hybrid(): return _emoji('🎬')
def bulb(): return _emoji('💡')
def frame(): return _emoji('🖼️')
def control(): return _emoji('🎛️')
def net(): return _emoji('🥅')
def web(): return _emoji('🕸️')
def cadence(): return _emoji('⏱️')
def off(): return _emoji('❌')
def wan_video(): return _emoji('🎥')
def steps(): return _emoji('👣')
def sound(): return _emoji('🎵')
def music(): return _emoji('🎶')
def leaf(): return _emoji('🍃')
def wave(): return noise()


# Utility functions for consistent emoji usage
def tab_title(emoji_func, title):
    """Create a tab title with optional emoji
    
    Args:
        emoji_func: Function that returns emoji (or empty string)
        title: The tab title text
    
    Returns:
        Formatted title with emoji if enabled
    """
    emoji = emoji_func()
    if emoji:
        return f"{emoji} {title}"
    return title


def console(emoji_char, always_show=False):
    """Return emoji for console output with optional override
    
    Args:
        emoji_char: Direct emoji character (e.g., '🎬', '✅', '❌')
        always_show: If True, show emoji even when disabled in UI
    
    Returns:
        Emoji character or empty string based on settings
    """
    if always_show:
        return emoji_char
    return _emoji(emoji_char)


def log_prefix(emoji_func, always_show=False):
    """Create a log prefix with optional emoji (deprecated - use console() instead)
    
    Args:
        emoji_func: Function that returns emoji
        always_show: If True, show emoji even when disabled in UI
    
    Returns:
        Emoji for log prefix or empty string
    """
    if always_show:
        # Extract emoji character from function
        try:
            emoji_char = emoji_func()
            return emoji_char if emoji_char else ''
        except:
            return ''
    return emoji_func()
