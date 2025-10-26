#!/usr/bin/env python3
"""Automated print statement migration to new logging system.

Converts print() calls to logger.debug/info/warning/error calls with:
- Proper log level detection
- Theme-aware emoji mapping
- Preserves formatting and f-strings
"""

import re
import sys
from pathlib import Path
from typing import Tuple, Optional


# Emoji to logger emoji name mapping
EMOJI_MAP = {
    '🎨': 'palette',
    '🎬': 'movie_camera',
    '🎥': 'wan_video',
    '📹': 'video_camera',
    '🎵': 'sound',
    '🎶': 'music',
    '🔑': 'key',
    '🖼️': 'frame',
    '🏎️': 'run',
    '⚙️': 'gear',
    '🔧': 'wrench',
    '🔄': 'refresh',
    '💡': 'bulb',
    '🎛️': 'control',
    '🥅': 'net',
    '🕸️': 'web',
    '✍️': 'prompts',
    '⏱️': 'cadence',
    '❌': 'off',
    '📊': 'distribution',
    '💪': 'strength',
    '📏': 'scale',
    '📄': 'document',
    '👣': 'steps',
    '🔢': 'numbers',
    '🎞️': 'frames',
    '🆙': 'up',
    '🌰': 'seed',
    '🥜': 'subseed',
    '🍃': 'leaf',
    '🚲': 'bicycle',
    '🕳️': 'hole',
    '🌊': 'wave',
    '🧹': 'broom',
    '🎭': 'masking',
    '⏱️': 'stopwatch',
    '🛠️': 'tools',
}


def detect_log_level(content: str) -> Tuple[str, Optional[str]]:
    """Detect appropriate log level from print content.

    Returns:
        (log_level, emoji_name) tuple
    """
    content_lower = content.lower()

    # DEBUG patterns
    if any(pattern in content for pattern in ['🔍 DEBUG:', 'DEBUG:']):
        # Extract emoji after DEBUG marker if present
        for emoji, name in EMOJI_MAP.items():
            if emoji in content and content.index(emoji) > content.index('DEBUG:'):
                return ('debug', name)
        return ('debug', None)

    # ERROR patterns
    if any(pattern in content_lower for pattern in [
        'error:', '❌ error', '** error', 'failed', 'exception:',
        'traceback', 'could not', 'unable to'
    ]):
        return ('error', None)

    # WARNING patterns
    if any(pattern in content for pattern in ['⚠️', 'WARNING:', 'Warning:']):
        return ('warning', None)

    # INFO with emoji
    for emoji, name in EMOJI_MAP.items():
        if emoji in content:
            return ('info', name)

    # Default to INFO
    return ('info', None)


def extract_emoji_from_print(line: str) -> Tuple[str, Optional[str]]:
    """Extract emoji and name from print statement.

    Returns:
        (cleaned_line, emoji_name) tuple
    """
    # Check for emojis at start of string
    for emoji, name in EMOJI_MAP.items():
        # Pattern: print(f"🎨 Message...")
        if f'"{emoji} ' in line or f"'{emoji} " in line:
            # Remove emoji from message
            line = line.replace(f'{emoji} ', '')
            return (line, name)

    return (line, None)


def migrate_print_statement(match: re.Match) -> str:
    """Convert a single print statement to logger call.

    Args:
        match: Regex match object containing print statement

    Returns:
        Converted logger call
    """
    indent = match.group(1) or ''
    print_content = match.group(2)

    # Detect log level and emoji
    level, emoji = detect_log_level(print_content)

    # Clean up content
    content = print_content

    # Remove DEBUG markers
    content = re.sub(r'🔍 DEBUG:\s*', '', content)
    content = re.sub(r'DEBUG:\s*', '', content)

    # Remove WARNING markers
    content = re.sub(r'⚠️\s+Warning:\s*', '', content)
    content = re.sub(r'WARNING:\s*', '', content)
    content = re.sub(r'Warning:\s*', '', content)

    # Remove ERROR markers
    content = re.sub(r'❌\s+Error:\s*', '', content)
    content = re.sub(r'\*\*\s+(.+?)\s+\*\*\s+Error:\s*', r'\1 - ', content)
    content = re.sub(r'Error:\s*', '', content)

    # Extract emoji from content
    content, extracted_emoji = extract_emoji_from_print(content)
    if extracted_emoji:
        emoji = extracted_emoji

    # Build logger call
    if emoji:
        result = f"{indent}logger.{level}({content}, emoji='{emoji}')"
    else:
        result = f"{indent}logger.{level}({content})"

    return result


def add_logger_import(content: str) -> str:
    """Add logger import to file if not present.

    Args:
        content: File content

    Returns:
        Content with logger import added
    """
    if 'from deforum.utils.system.logging import get_logger' in content:
        return content

    # Find the last import statement
    lines = content.split('\n')
    last_import_idx = -1

    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            last_import_idx = i

    if last_import_idx >= 0:
        # Insert after last import
        lines.insert(last_import_idx + 1, 'from deforum.utils.system.logging import get_logger')
        lines.insert(last_import_idx + 2, '')
        lines.insert(last_import_idx + 3, '# Initialize logger')
        lines.insert(last_import_idx + 4, 'logger = get_logger()')
        lines.insert(last_import_idx + 5, '')
        return '\n'.join(lines)

    return content


def migrate_file(filepath: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Migrate print statements in a single file.

    Args:
        filepath: Path to Python file
        dry_run: If True, don't write changes

    Returns:
        (total_prints, migrated_prints) tuple
    """
    with open(filepath, 'r') as f:
        original = f.read()

    content = original

    # Pattern to match print statements
    # Captures: (indent)(print content)
    pattern = re.compile(
        r'^( *)print\((.*?)\)$',
        re.MULTILINE
    )

    # Count total prints
    total_prints = len(pattern.findall(content))

    if total_prints == 0:
        return (0, 0)

    # Replace print statements
    content = pattern.sub(migrate_print_statement, content)

    # Add logger import if needed
    if 'logger.' in content and 'logger = get_logger()' not in original:
        content = add_logger_import(content)

    # Count successful migrations
    migrated = total_prints - len(re.findall(r'\bprint\(', content))

    if not dry_run and content != original:
        with open(filepath, 'w') as f:
            f.write(content)

    return (total_prints, migrated)


def migrate_directory(directory: Path, pattern: str = "*.py", dry_run: bool = False):
    """Migrate all Python files in directory.

    Args:
        directory: Directory to process
        pattern: Glob pattern for files
        dry_run: If True, don't write changes
    """
    total_files = 0
    total_prints = 0
    total_migrated = 0

    print(f"Scanning {directory} for {pattern} files...")

    for filepath in directory.rglob(pattern):
        # Skip external repos and test files
        if any(skip in str(filepath) for skip in [
            '/external_repos/', '/tests/', '/__pycache__/',
            'migrate_prints_to_logger.py'
        ]):
            continue

        prints, migrated = migrate_file(filepath, dry_run=dry_run)

        if prints > 0:
            total_files += 1
            total_prints += prints
            total_migrated += migrated

            status = "✓" if migrated == prints else "⚠"
            mode = "[DRY RUN] " if dry_run else ""
            print(f"{mode}{status} {filepath.relative_to(directory)}: {migrated}/{prints} prints migrated")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files processed: {total_files}")
    print(f"  Total prints found: {total_prints}")
    print(f"  Successfully migrated: {total_migrated}")
    print(f"  Remaining: {total_prints - total_migrated}")
    if dry_run:
        print(f"\n(DRY RUN - no files were modified)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Migrate print statements to logger')
    parser.add_argument('path', help='File or directory to process')
    parser.add_argument('--dry-run', action='store_true', help="Don't write changes")
    parser.add_argument('--pattern', default='*.py', help='File pattern (default: *.py)')

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)

    if path.is_file():
        prints, migrated = migrate_file(path, dry_run=args.dry_run)
        mode = "[DRY RUN] " if args.dry_run else ""
        print(f"{mode}Migrated {migrated}/{prints} print statements in {path}")
    else:
        migrate_directory(path, pattern=args.pattern, dry_run=args.dry_run)
