"""Unit tests for HTTP client utilities."""

import pytest
from unittest.mock import MagicMock, patch
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from deforum.utils.system.http import (
    get_http_client,
    DEFAULT_RETRIES,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_STATUS_FORCELIST,
)


class TestConstants:
    """Test module constants."""

    def test_default_retries(self):
        """Test default retries constant value."""
        assert DEFAULT_RETRIES == 5

    def test_default_backoff_factor(self):
        """Test default backoff factor constant value."""
        assert DEFAULT_BACKOFF_FACTOR == 0.3

    def test_default_status_forcelist(self):
        """Test default status forcelist constant value."""
        assert DEFAULT_STATUS_FORCELIST == (429, 500, 502, 503, 504)


class TestGetHttpClient:
    """Test get_http_client function."""

    def test_returns_session(self):
        """get_http_client should return a requests.Session."""
        client = get_http_client()
        assert isinstance(client, requests.Session)

    def test_creates_new_session_by_default(self):
        """get_http_client should create new session when none provided."""
        client1 = get_http_client()
        client2 = get_http_client()
        assert client1 is not client2

    def test_reuses_provided_session(self):
        """get_http_client should reuse provided session."""
        existing_session = requests.Session()
        client = get_http_client(session=existing_session)
        assert client is existing_session

    def test_uses_default_parameters(self):
        """get_http_client should use default parameters when not specified."""
        with patch('deforum.utils.system.http.Retry') as mock_retry:
            with patch('deforum.utils.system.http.HTTPAdapter') as mock_adapter:
                get_http_client()

                # Check Retry was called with defaults
                mock_retry.assert_called_once_with(
                    total=DEFAULT_RETRIES,
                    read=DEFAULT_RETRIES,
                    connect=DEFAULT_RETRIES,
                    backoff_factor=DEFAULT_BACKOFF_FACTOR,
                    status_forcelist=list(DEFAULT_STATUS_FORCELIST),
                )

    def test_uses_custom_retries(self):
        """get_http_client should use custom retry count."""
        with patch('deforum.utils.system.http.Retry') as mock_retry:
            with patch('deforum.utils.system.http.HTTPAdapter'):
                get_http_client(retries=10)

                call_kwargs = mock_retry.call_args.kwargs
                assert call_kwargs['total'] == 10
                assert call_kwargs['read'] == 10
                assert call_kwargs['connect'] == 10

    def test_uses_custom_backoff_factor(self):
        """get_http_client should use custom backoff factor."""
        with patch('deforum.utils.system.http.Retry') as mock_retry:
            with patch('deforum.utils.system.http.HTTPAdapter'):
                get_http_client(backoff_factor=0.5)

                call_kwargs = mock_retry.call_args.kwargs
                assert call_kwargs['backoff_factor'] == 0.5

    def test_uses_custom_status_forcelist(self):
        """get_http_client should use custom status forcelist."""
        custom_statuses = (404, 503)
        with patch('deforum.utils.system.http.Retry') as mock_retry:
            with patch('deforum.utils.system.http.HTTPAdapter'):
                get_http_client(status_forcelist=custom_statuses)

                call_kwargs = mock_retry.call_args.kwargs
                assert call_kwargs['status_forcelist'] == list(custom_statuses)

    def test_mounts_http_adapter(self):
        """get_http_client should mount adapter for http://."""
        client = get_http_client()
        http_adapter = client.get_adapter('http://example.com')
        assert isinstance(http_adapter, HTTPAdapter)

    def test_mounts_https_adapter(self):
        """get_http_client should mount adapter for https://."""
        client = get_http_client()
        https_adapter = client.get_adapter('https://example.com')
        assert isinstance(https_adapter, HTTPAdapter)

    def test_http_and_https_use_same_adapter(self):
        """get_http_client should use same adapter for http and https."""
        client = get_http_client()
        http_adapter = client.get_adapter('http://example.com')
        https_adapter = client.get_adapter('https://example.com')
        assert http_adapter is https_adapter

    def test_adapter_has_retry_strategy(self):
        """get_http_client adapter should have retry strategy configured."""
        client = get_http_client(retries=3)
        adapter = client.get_adapter('https://example.com')

        # HTTPAdapter has max_retries attribute which is the Retry object
        assert isinstance(adapter.max_retries, Retry)
        assert adapter.max_retries.total == 3

    def test_all_custom_parameters(self):
        """get_http_client should accept all custom parameters together."""
        existing_session = requests.Session()
        custom_statuses = (418, 500)

        client = get_http_client(
            retries=7,
            backoff_factor=1.5,
            status_forcelist=custom_statuses,
            session=existing_session,
        )

        assert client is existing_session
        adapter = client.get_adapter('https://example.com')
        assert adapter.max_retries.total == 7
        assert adapter.max_retries.backoff_factor == 1.5

    def test_zero_retries(self):
        """get_http_client should handle zero retries."""
        client = get_http_client(retries=0)
        adapter = client.get_adapter('https://example.com')
        assert adapter.max_retries.total == 0

    def test_empty_status_forcelist(self):
        """get_http_client should handle empty status forcelist."""
        with patch('deforum.utils.system.http.Retry') as mock_retry:
            with patch('deforum.utils.system.http.HTTPAdapter'):
                get_http_client(status_forcelist=())

                call_kwargs = mock_retry.call_args.kwargs
                assert call_kwargs['status_forcelist'] == []

    def test_session_ready_for_requests(self):
        """get_http_client should return a session ready for HTTP requests."""
        client = get_http_client()

        # Session should have standard methods
        assert hasattr(client, 'get')
        assert hasattr(client, 'post')
        assert hasattr(client, 'put')
        assert hasattr(client, 'delete')
        assert callable(client.get)


class TestRetryBehavior:
    """Test retry behavior configuration."""

    def test_retry_on_configured_status_codes(self):
        """Retry strategy should include configured status codes."""
        client = get_http_client(status_forcelist=(500, 502))
        adapter = client.get_adapter('https://example.com')

        # The status_forcelist should be in the retry config
        retry = adapter.max_retries
        assert 500 in retry.status_forcelist
        assert 502 in retry.status_forcelist

    def test_backoff_factor_in_retry(self):
        """Retry strategy should include backoff factor."""
        client = get_http_client(backoff_factor=2.0)
        adapter = client.get_adapter('https://example.com')

        retry = adapter.max_retries
        assert retry.backoff_factor == 2.0

    def test_read_retries_match_total(self):
        """Read retries should match total retries."""
        client = get_http_client(retries=8)
        adapter = client.get_adapter('https://example.com')

        retry = adapter.max_retries
        assert retry.read == 8

    def test_connect_retries_match_total(self):
        """Connect retries should match total retries."""
        client = get_http_client(retries=8)
        adapter = client.get_adapter('https://example.com')

        retry = adapter.max_retries
        assert retry.connect == 8
