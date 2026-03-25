"""Tests for the LegiScan API client.

Uses mock API responses to test caching, rate limiting, error handling,
and response parsing without making real API calls.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.legiscan_client import LegiScanClient, LegiScanError, RateLimitError


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    """Provide a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def client(tmp_cache: Path) -> LegiScanClient:
    """Create a LegiScanClient with a temp cache and test API key."""
    return LegiScanClient(
        api_key="test_key_123",
        cache_dir=tmp_cache,
        rate_limit_delay=0.0,  # disable for tests
    )


class TestClientInit:
    def test_raises_without_api_key(self, tmp_cache: Path) -> None:
        with patch("src.data.legiscan_client.LEGISCAN_API_KEY", ""):
            with pytest.raises(LegiScanError, match="No API key"):
                LegiScanClient(cache_dir=tmp_cache)

    def test_creates_cache_dir(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "new_cache"
        client = LegiScanClient(api_key="test", cache_dir=cache_dir)
        assert cache_dir.exists()


class TestCaching:
    def test_cache_write_and_read(self, client: LegiScanClient) -> None:
        data = {"status": "OK", "sessions": [{"session_id": 1}]}
        client._write_cache("getSessionList", {"state": "OH"}, data)
        cached = client._read_cache("getSessionList", {"state": "OH"})
        assert cached == data

    def test_cache_miss_returns_none(self, client: LegiScanClient) -> None:
        result = client._read_cache("nonexistent", {})
        assert result is None

    def test_cache_key_deterministic(self, client: LegiScanClient) -> None:
        key1 = client._cache_key("getBill", {"id": "123"})
        key2 = client._cache_key("getBill", {"id": "123"})
        assert key1 == key2

    def test_cache_key_different_params(self, client: LegiScanClient) -> None:
        key1 = client._cache_key("getBill", {"id": "123"})
        key2 = client._cache_key("getBill", {"id": "456"})
        assert key1 != key2

    def test_invalidate_cache(self, client: LegiScanClient) -> None:
        client._write_cache("test_op", {"id": "1"}, {"data": True})
        assert client._read_cache("test_op", {"id": "1"}) is not None
        client.invalidate_cache("test_op", id="1")
        assert client._read_cache("test_op", {"id": "1"}) is None


class TestAPIRequests:
    @patch("requests.Session.get")
    def test_successful_request(self, mock_get: MagicMock, client: LegiScanClient) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "sessions": [{"session_id": 1, "session_name": "Test"}],
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = client._request("getSessionList", state="OH")
        assert result["status"] == "OK"

    @patch("requests.Session.get")
    def test_api_error_raises(self, mock_get: MagicMock, client: LegiScanClient) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ERROR",
            "alert": {"message": "Invalid key"},
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with pytest.raises(LegiScanError, match="Invalid key"):
            client._request("getSessionList", use_cache=False, state="OH")

    @patch("requests.Session.get")
    def test_cached_request_skips_api(self, mock_get: MagicMock, client: LegiScanClient) -> None:
        # Pre-populate cache
        cached_data = {"status": "OK", "sessions": []}
        client._write_cache("getSessionList", {"state": "OH"}, cached_data)

        result = client._request("getSessionList", state="OH")
        assert result == cached_data
        mock_get.assert_not_called()


class TestSessionList:
    @patch("requests.Session.get")
    def test_get_session_list(self, mock_get: MagicMock, client: LegiScanClient) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "sessions": [
                {"session_id": 1850, "session_name": "136th GA", "year_start": 2025},
                {"session_id": 1700, "session_name": "135th GA", "year_start": 2023},
            ],
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        sessions = client.get_session_list("OH")
        assert len(sessions) == 2
        assert sessions[0]["session_name"] == "136th GA"


class TestMasterList:
    @patch("requests.Session.get")
    def test_get_master_list_strips_session_key(
        self, mock_get: MagicMock, client: LegiScanClient
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "masterlist": {
                "session": {"session_id": 1850},
                "0": {"bill_id": 100, "number": "HB 1"},
                "1": {"bill_id": 101, "number": "SB 1"},
            },
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        master = client.get_master_list(1850)
        assert "session" not in master
        assert len(master) == 2


class TestGetBill:
    @patch("requests.Session.get")
    def test_get_bill(self, mock_get: MagicMock, client: LegiScanClient) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "bill": {
                "bill_id": 12345,
                "bill_number": "HB 123",
                "title": "Test Bill",
                "sponsors": [],
                "history": [],
            },
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        bill = client.get_bill(12345)
        assert bill["bill_id"] == 12345
        assert bill["bill_number"] == "HB 123"


class TestChangedBills:
    @patch("requests.Session.get")
    def test_detects_new_bills(self, mock_get: MagicMock, client: LegiScanClient) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "masterlist": {
                "0": {"bill_id": 100, "change_hash": "abc123"},
                "1": {"bill_id": 101, "change_hash": "def456"},
            },
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        changed = client.get_changed_bills(1850, known_hashes={100: "abc123"})
        assert 101 in changed
        assert 100 not in changed

    @patch("requests.Session.get")
    def test_detects_changed_hash(self, mock_get: MagicMock, client: LegiScanClient) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "masterlist": {
                "0": {"bill_id": 100, "change_hash": "new_hash"},
            },
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        changed = client.get_changed_bills(1850, known_hashes={100: "old_hash"})
        assert 100 in changed
