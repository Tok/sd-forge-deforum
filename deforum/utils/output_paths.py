"""Centralized output path management for Deforum.

This module provides a single source of truth for all output directory paths,
ensuring consistency across the codebase and easy adaptation to different WebUI forks.

Forge Neo uses 'output/' not 'outputs/' - this module handles that centrally.
"""

from pathlib import Path
from typing import Optional


class OutputPaths:
    """Centralized output directory path constants and utilities.

    All output paths should be accessed through this class to ensure consistency
    and easy maintenance.
    """

    # Base output directory (Forge Neo uses 'output', not 'outputs')
    BASE = "output"

    # Deforum-specific output directories
    DEFORUM = f"{BASE}/deforum"
    DEFORUM_TUNING = f"{BASE}/deforum-tuning"
    DEFORUM_TESTS = f"{BASE}/deforum-tests"

    # Standard Forge output directories (for reference)
    TXT2IMG = f"{BASE}/txt2img-images"
    IMG2IMG = f"{BASE}/img2img-images"

    @classmethod
    def get_deforum_output(cls, batch_name: Optional[str] = None) -> Path:
        """Get path to Deforum output directory.

        Args:
            batch_name: Optional batch subdirectory name (e.g., 'Deforum_20231025123456')

        Returns:
            Path object for the deforum output directory

        Examples:
            >>> OutputPaths.get_deforum_output()
            Path('output/deforum')
            >>> OutputPaths.get_deforum_output('Deforum_20231025123456')
            Path('output/deforum/Deforum_20231025123456')
        """
        base = Path(cls.DEFORUM)
        if batch_name:
            return base / batch_name
        return base

    @classmethod
    def get_tuning_output(cls, test_name: Optional[str] = None) -> Path:
        """Get path to tuning output directory.

        Args:
            test_name: Optional test subdirectory name

        Returns:
            Path object for the tuning output directory
        """
        base = Path(cls.DEFORUM_TUNING)
        if test_name:
            return base / test_name
        return base

    @classmethod
    def get_test_output(cls, test_name: Optional[str] = None) -> Path:
        """Get path to test output directory.

        Args:
            test_name: Optional test subdirectory name

        Returns:
            Path object for the test output directory
        """
        base = Path(cls.DEFORUM_TESTS)
        if test_name:
            return base / test_name
        return base

    @classmethod
    def find_settings_file(cls, timestring: str, outdir: Optional[str] = None) -> Optional[Path]:
        """Find settings file for a given timestring.

        Searches multiple possible locations for the settings file, trying:
        1. Standard Deforum output paths
        2. Custom outdir if provided

        Args:
            timestring: The timestring to search for (e.g., '20231025123456')
            outdir: Optional custom output directory

        Returns:
            Path to settings file if found, None otherwise
        """
        possible_paths = [
            # Standard Deforum output paths (Forge uses 'output' not 'outputs')
            Path(f"{cls.DEFORUM}/Deforum_{timestring}/{timestring}_settings.txt"),
            Path(f"{cls.DEFORUM}/{timestring}/{timestring}_settings.txt"),
        ]

        # Add custom outdir paths if provided
        if outdir:
            possible_paths.extend([
                Path(outdir) / f"{timestring}_settings.txt",
                Path(outdir) / timestring / f"{timestring}_settings.txt",
            ])

        # Find first existing path
        for path in possible_paths:
            if path.exists():
                return path

        return None

    @classmethod
    def ensure_directory(cls, path: str | Path) -> Path:
        """Ensure a directory exists, creating it if necessary.

        Args:
            path: Path to directory (string or Path object)

        Returns:
            Path object for the directory
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
