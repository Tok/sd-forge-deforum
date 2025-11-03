"""HTTP client utilities with automatic retry logic.

Provides a configured HTTP session with exponential backoff for failed requests.
"""

from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Default retry configuration
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF_FACTOR = 0.3
DEFAULT_STATUS_FORCELIST = (429, 500, 502, 503, 504)


def get_http_client(
    retries: int = DEFAULT_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    status_forcelist: tuple[int, ...] = DEFAULT_STATUS_FORCELIST,
    session: Optional[requests.Session] = None,
) -> requests.Session:
    """Create HTTP client with automatic retry logic.

    Returns a requests.Session configured with exponential backoff retry strategy.
    This is low overhead so it's safe to call on every request.

    Args:
        retries: Number of retry attempts (default: 5)
        backoff_factor: Backoff multiplier for retries (default: 0.3)
        status_forcelist: HTTP status codes that trigger retry (default: 429, 500, 502, 503, 504)
        session: Existing session to configure, or None to create new one

    Returns:
        Configured requests.Session with retry adapter mounted

    Example:
        >>> client = get_http_client(retries=3)
        >>> response = client.get('https://example.com/api/data')
    """
    session = session or requests.Session()

    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    return session