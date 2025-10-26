"""Pure functions for file hashing and checksum calculation.

This module contains functions for computing file hashes and checksums,
following functional programming principles with minimized side effects.
"""

import hashlib
from pathlib import Path
from typing import Callable, Union


def compute_file_checksum(
    file_path: Union[str, Path], hash_algorithm: str = "blake2b", chunk_size: int = 8192
) -> str:
    """Compute checksum/hash of a file.

    Args:
        file_path: Path to file
        hash_algorithm: Hash algorithm name (md5, sha1, sha256, blake2b, etc.)
        chunk_size: Size of chunks to read (bytes)

    Returns:
        Hexadecimal hash digest string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If hash algorithm is not supported

    Examples:
        >>> # Example with actual file (results vary by content)
        >>> compute_file_checksum('/path/to/file.txt', 'sha256')
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        >>> compute_file_checksum('/path/to/file.txt', 'md5')
        'd41d8cd98f00b204e9800998ecf8427e'
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Get hash object
    try:
        hash_obj = hashlib.new(hash_algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm '{hash_algorithm}': {str(e)}") from e

    # Read and hash file in chunks
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def compute_file_checksum_with_factory(
    file_path: Union[str, Path],
    hash_factory: Callable = hashlib.blake2b,
    chunk_num_blocks: int = 128,
) -> str:
    """Compute checksum using a hash factory function.

    This is the original implementation from general_utils.py that allows
    passing a hash factory function directly.

    Args:
        file_path: Path to file
        hash_factory: Hash factory function (e.g., hashlib.blake2b)
        chunk_num_blocks: Number of blocks per chunk

    Returns:
        Hexadecimal hash digest string

    Raises:
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> import hashlib
        >>> compute_file_checksum_with_factory('/path/to/file.txt', hashlib.md5)
        'd41d8cd98f00b204e9800998ecf8427e'
        >>> compute_file_checksum_with_factory('/path/to/file.txt', hashlib.sha256)
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    h = hash_factory()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_num_blocks * h.block_size):
            h.update(chunk)

    return h.hexdigest()


def compute_string_hash(data: Union[str, bytes], hash_algorithm: str = "sha256") -> str:
    """Compute hash of a string or bytes.

    Args:
        data: String or bytes to hash
        hash_algorithm: Hash algorithm name

    Returns:
        Hexadecimal hash digest string

    Raises:
        ValueError: If hash algorithm is not supported

    Examples:
        >>> compute_string_hash("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        >>> compute_string_hash("test", "md5")
        '098f6bcd4621d373cade4e832627b4f6'
        >>> compute_string_hash(b"binary data")
        '4b40ad541e86dd6d8c032c84b0f842bc8e0c19abfcd6f5c8e4bce8d6a8d1f5f5'
    """
    # Get hash object
    try:
        hash_obj = hashlib.new(hash_algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm '{hash_algorithm}': {str(e)}") from e

    # Convert string to bytes if needed
    if isinstance(data, str):
        data = data.encode("utf-8")

    hash_obj.update(data)
    return hash_obj.hexdigest()


def verify_file_checksum(
    file_path: Union[str, Path], expected_checksum: str, hash_algorithm: str = "blake2b"
) -> bool:
    """Verify file checksum matches expected value.

    Args:
        file_path: Path to file
        expected_checksum: Expected hexadecimal checksum
        hash_algorithm: Hash algorithm to use

    Returns:
        True if checksum matches, False otherwise

    Examples:
        >>> verify_file_checksum('/path/to/file.txt', 'abc123...', 'sha256')
        True
        >>> verify_file_checksum('/path/to/file.txt', 'wrong', 'sha256')
        False
    """
    try:
        actual_checksum = compute_file_checksum(file_path, hash_algorithm)
        return actual_checksum.lower() == expected_checksum.lower()
    except (FileNotFoundError, ValueError):
        return False


def get_supported_hash_algorithms() -> list[str]:
    """Get list of supported hash algorithm names.

    Returns:
        List of algorithm names

    Examples:
        >>> algorithms = get_supported_hash_algorithms()
        >>> 'sha256' in algorithms
        True
        >>> 'md5' in algorithms
        True
        >>> 'blake2b' in algorithms
        True
    """
    return sorted(hashlib.algorithms_available)
