#!/usr/bin/env python3
"""Test script to demonstrate all three logging themes.

Run this to see visual examples of slopcore, classic, and simple themes
with different log levels and emojis.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deforum.utils.system.logging import get_logger, set_logger_config, reset_logger


def demo_theme(theme_name: str):
    """Demonstrate a single theme with all log levels and emojis."""

    # Reset and configure logger for this theme
    reset_logger()
    set_logger_config(theme=theme_name, log_level='DEBUG', emojis_enabled=True)
    logger = get_logger()

    print(f"\n{'='*80}")
    print(f"THEME: {theme_name.upper()}")
    print(f"{'='*80}\n")

    # Header example
    logger.header(f"{theme_name.upper()} THEME DEMONSTRATION")

    # Log level examples
    logger.debug("This is a DEBUG message - only shows when log_level=DEBUG")
    logger.info("This is an INFO message - normal operation", emoji='run')
    logger.warning("This is a WARNING message - something to be aware of")
    logger.error("This is an ERROR message - something went wrong")
    logger.critical("This is a CRITICAL message - severe failure")

    # Separator
    logger.separator()

    # Emoji examples
    logger.info("Processing started", emoji='run')
    logger.info("Loading model", emoji='gear')
    logger.info("Generating frames", emoji='frame')
    logger.info("Creating video", emoji='movie_camera')
    logger.info("Wan interpolation", emoji='wan_video')
    logger.info("Audio sync", emoji='sound')
    logger.info("Keyframe distribution", emoji='distribution')
    logger.info("Refresh complete", emoji='refresh')

    # Progress bar example (if tqdm available)
    try:
        import time
        print("\nProgress bar example:")
        for i in logger.progress(range(20), desc=f"{theme_name} progress"):
            time.sleep(0.05)
    except ImportError:
        logger.info("(tqdm not available - progress bars will fallback to plain iterator)")

    logger.separator()


def demo_log_levels():
    """Demonstrate log level filtering."""

    print(f"\n{'='*80}")
    print("LOG LEVEL FILTERING DEMONSTRATION")
    print(f"{'='*80}\n")

    for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
        reset_logger()
        set_logger_config(theme='slopcore', log_level=level, emojis_enabled=True)
        logger = get_logger()

        print(f"\n--- Log Level: {level} ---")
        logger.debug("DEBUG message")
        logger.info("INFO message")
        logger.warning("WARNING message")
        logger.error("ERROR message")


def demo_emoji_toggle():
    """Demonstrate emoji on/off."""

    print(f"\n{'='*80}")
    print("EMOJI ON/OFF DEMONSTRATION")
    print(f"{'='*80}\n")

    for emojis_enabled in [True, False]:
        reset_logger()
        set_logger_config(theme='slopcore', log_level='INFO', emojis_enabled=emojis_enabled)
        logger = get_logger()

        print(f"\n--- Emojis: {'ENABLED' if emojis_enabled else 'DISABLED'} ---")
        logger.info("Starting render", emoji='run')
        logger.info("Loading model", emoji='gear')
        logger.info("Creating video", emoji='movie_camera')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test Deforum logging themes')
    parser.add_argument('--theme', choices=['slopcore', 'classic', 'simple', 'all'],
                       default='all', help='Theme to test (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Skip progress bar demo (faster)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("DEFORUM LOGGING SYSTEM - THEME DEMONSTRATION")
    print("="*80)

    if args.theme == 'all':
        # Demo all three themes
        demo_theme('slopcore')
        demo_theme('classic')
        demo_theme('simple')

        # Demo log level filtering
        demo_log_levels()

        # Demo emoji toggle
        demo_emoji_toggle()
    else:
        # Demo single theme
        demo_theme(args.theme)

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nTo use in WebUI:")
    print("  1. Go to Settings → Deforum → Console & UI Output Settings")
    print("  2. Select Log Level (DEBUG/INFO/WARNING/ERROR/CRITICAL)")
    print("  3. Select Console Theme (slopcore/classic/simple)")
    print("  4. Toggle 'Disable emojis' checkbox")
    print("  5. Apply settings and restart if needed")
    print()
