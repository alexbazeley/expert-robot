"""LegiScan API client with caching, rate limiting, and change-hash sync.

Wraps the LegiScan Pull API (https://api.legiscan.com/) to provide typed
access to legislative data. Responses are cached locally as JSON files to
minimize API calls against the 30,000/month public tier limit.

Usage:
    from src.data.legiscan_client import LegiScanClient
    client = LegiScanClient()
    sessions = client.get_session_list("OH")
    master = client.get_master_list(session_id=1234)
    bill = client.get_bill(bill_id=567890)
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from src.config import LEGISCAN_API_KEY, LEGISCAN_BASE_URL, LEGISCAN_CACHE_DIR

logger = logging.getLogger(__name__)


class LegiScanError(Exception):
    """Raised when the LegiScan API returns an error response."""


class RateLimitError(LegiScanError):
    """Raised when the API rate limit is exceeded."""


class LegiScanClient:
    """Client for the LegiScan Pull API with local JSON caching.

    Args:
        api_key: LegiScan API key. Defaults to LEGISCAN_API_KEY from env.
        cache_dir: Directory for cached JSON responses.
        rate_limit_delay: Minimum seconds between API calls.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        rate_limit_delay: float = 0.5,
    ) -> None:
        self.api_key = api_key or LEGISCAN_API_KEY
        if not self.api_key:
            raise LegiScanError(
                "No API key provided. Set LEGISCAN_API_KEY in .env or pass api_key parameter."
            )
        self.base_url = LEGISCAN_BASE_URL
        self.cache_dir = cache_dir or LEGISCAN_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "legislative-prediction/0.1"})

    def _rate_limit(self) -> None:
        """Enforce minimum delay between API requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

    def _cache_key(self, operation: str, params: dict[str, Any]) -> str:
        """Generate a deterministic cache filename from operation and params."""
        # Sort params for deterministic hashing
        param_str = json.dumps(params, sort_keys=True)
        hash_val = hashlib.sha256(f"{operation}:{param_str}".encode()).hexdigest()[:16]
        return f"{operation}_{hash_val}.json"

    def _get_cache_path(self, operation: str, params: dict[str, Any]) -> Path:
        """Return the full path for a cached response."""
        subdir = self.cache_dir / operation
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / self._cache_key(operation, params)

    def _read_cache(self, operation: str, params: dict[str, Any]) -> dict | None:
        """Read a cached response if it exists."""
        path = self._get_cache_path(operation, params)
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Cache read failed for %s: %s", path, e)
        return None

    def _write_cache(self, operation: str, params: dict[str, Any], data: dict) -> None:
        """Write a response to the cache."""
        path = self._get_cache_path(operation, params)
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except OSError as e:
            logger.warning("Cache write failed for %s: %s", path, e)

    def _request(
        self,
        operation: str,
        use_cache: bool = True,
        **params: Any,
    ) -> dict:
        """Make a request to the LegiScan API.

        Args:
            operation: The API operation (e.g., "getSessionList").
            use_cache: Whether to check/write local cache.
            **params: Additional query parameters for the API call.

        Returns:
            The parsed JSON response body.

        Raises:
            LegiScanError: If the API returns an error or the request fails.
            RateLimitError: If the API rate limit is hit.
        """
        # Check cache first
        if use_cache:
            cached = self._read_cache(operation, params)
            if cached is not None:
                logger.debug("Cache hit for %s %s", operation, params)
                return cached

        # Rate limit
        self._rate_limit()

        # Build request
        query_params = {
            "key": self.api_key,
            "op": operation,
            **params,
        }

        logger.info("API request: %s %s", operation, {k: v for k, v in params.items()})

        try:
            response = self._session.get(
                self.base_url,
                params=query_params,
                timeout=30,
            )
            self._last_request_time = time.monotonic()
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                raise RateLimitError("LegiScan API rate limit exceeded") from e
            raise LegiScanError(f"HTTP error {response.status_code}: {e}") from e
        except requests.exceptions.RequestException as e:
            raise LegiScanError(f"Request failed: {e}") from e

        data = response.json()

        if data.get("status") == "ERROR":
            raise LegiScanError(f"API error: {data.get('alert', {}).get('message', 'Unknown')}")

        # Cache successful responses
        if use_cache:
            self._write_cache(operation, params, data)

        return data

    # --- Session endpoints ---

    def get_session_list(self, state: str) -> list[dict]:
        """Get all available sessions for a state.

        Args:
            state: Two-letter state abbreviation (e.g., "OH").

        Returns:
            List of session dicts with keys: session_id, state_id, year_start,
            year_end, special, session_name, etc.
        """
        response = self._request("getSessionList", state=state)
        sessions = response.get("sessions", [])
        if isinstance(sessions, dict):
            sessions = list(sessions.values())
        logger.info("Found %d sessions for %s", len(sessions), state)
        return sessions

    # --- Bill list endpoints ---

    def get_master_list(self, session_id: int) -> dict[str, dict]:
        """Get summary info for all bills in a session.

        Args:
            session_id: LegiScan session ID.

        Returns:
            Dict mapping bill_id (as string) to bill summary dicts.
        """
        response = self._request("getMasterList", id=str(session_id))
        master = response.get("masterlist", {})
        # The first entry is session metadata, not a bill
        if "session" in master:
            del master["session"]
        logger.info("Master list for session %d: %d bills", session_id, len(master))
        return master

    def get_master_list_raw(self, session_id: int) -> dict[str, dict]:
        """Get raw master list with change_hash for efficient sync.

        Same as get_master_list but uses getMasterListRaw which includes
        change_hash for each bill, enabling delta sync.

        Args:
            session_id: LegiScan session ID.

        Returns:
            Dict mapping bill_id to bill summary dicts including change_hash.
        """
        response = self._request("getMasterListRaw", id=str(session_id), use_cache=False)
        master = response.get("masterlist", {})
        if "session" in master:
            del master["session"]
        return master

    # --- Bill detail ---

    def get_bill(self, bill_id: int) -> dict:
        """Get full detail for a specific bill.

        Includes sponsors, cosponsors, history, committee referrals,
        texts, votes, amendments, and subjects.

        Args:
            bill_id: LegiScan bill ID.

        Returns:
            Complete bill detail dict.
        """
        response = self._request("getBill", id=str(bill_id))
        return response.get("bill", {})

    # --- Roll call / votes ---

    def get_roll_call(self, roll_call_id: int) -> dict:
        """Get vote details for a specific roll call.

        Args:
            roll_call_id: LegiScan roll call ID.

        Returns:
            Roll call dict with individual votes.
        """
        response = self._request("getRollCall", id=str(roll_call_id))
        return response.get("roll_call", {})

    # --- Legislator ---

    def get_person(self, people_id: int) -> dict:
        """Get details for a specific legislator.

        Args:
            people_id: LegiScan people ID.

        Returns:
            Legislator detail dict.
        """
        response = self._request("getPerson", id=str(people_id))
        return response.get("person", {})

    # --- Bulk datasets ---

    def get_dataset_list(self, state: str | None = None) -> list[dict]:
        """Get list of available bulk dataset ZIP archives.

        Args:
            state: Optional state abbreviation to filter by.

        Returns:
            List of dataset info dicts with session_id, dataset_hash, etc.
        """
        params: dict[str, str] = {}
        if state:
            params["state"] = state
        response = self._request("getDatasetList", **params)
        datasets = response.get("datasetlist", [])
        if isinstance(datasets, dict):
            datasets = list(datasets.values())
        logger.info("Found %d datasets%s", len(datasets), f" for {state}" if state else "")
        return datasets

    def get_dataset(self, session_id: int, access_key: str) -> dict:
        """Download a bulk dataset ZIP archive (base64-encoded).

        Args:
            session_id: LegiScan session ID.
            access_key: Access key from getDatasetList response.

        Returns:
            Dict with 'zip' key containing base64-encoded ZIP data.
        """
        response = self._request(
            "getDataset",
            id=str(session_id),
            access_key=access_key,
            use_cache=True,
        )
        return response.get("dataset", {})

    # --- Search ---

    def search(
        self,
        state: str,
        query: str,
        year: int = 0,
        page: int = 1,
    ) -> dict:
        """Search for bills by keyword.

        Args:
            state: Two-letter state abbreviation.
            query: Search query string.
            year: Filter by year (0 = all years).
            page: Results page number.

        Returns:
            Search results dict with 'summary' and 'results' keys.
        """
        response = self._request(
            "search",
            state=state,
            query=query,
            year=str(year) if year else "0",
            page=str(page),
        )
        return response.get("searchresult", {})

    def search_raw(
        self,
        state: str,
        query: str,
        year: int = 0,
        page: int = 1,
    ) -> dict:
        """Search for bills (raw format with relevance scores).

        Args:
            state: Two-letter state abbreviation.
            query: Search query string.
            year: Filter by year (0 = all years).
            page: Results page number.

        Returns:
            Raw search results dict.
        """
        response = self._request(
            "searchRaw",
            state=state,
            query=query,
            year=str(year) if year else "0",
            page=str(page),
        )
        return response.get("searchresult", {})

    # --- Sync utilities ---

    def get_changed_bills(
        self,
        session_id: int,
        known_hashes: dict[int, str],
    ) -> list[int]:
        """Identify bills that have changed since last sync.

        Compares change_hash values from getMasterListRaw against known
        hashes to find bills requiring re-fetch.

        Args:
            session_id: LegiScan session ID.
            known_hashes: Dict mapping bill_id to last-known change_hash.

        Returns:
            List of bill_ids that need to be re-fetched.
        """
        current = self.get_master_list_raw(session_id)
        changed = []
        for key, bill_info in current.items():
            bill_id = bill_info.get("bill_id")
            if bill_id is None:
                continue
            current_hash = bill_info.get("change_hash", "")
            if str(bill_id) not in known_hashes and bill_id not in known_hashes:
                changed.append(bill_id)
            elif known_hashes.get(bill_id, known_hashes.get(str(bill_id))) != current_hash:
                changed.append(bill_id)
        logger.info(
            "Session %d: %d/%d bills changed since last sync",
            session_id,
            len(changed),
            len(current),
        )
        return changed

    def invalidate_cache(self, operation: str, **params: Any) -> None:
        """Remove a specific cached response.

        Args:
            operation: The API operation.
            **params: The same parameters used in the original request.
        """
        path = self._get_cache_path(operation, params)
        if path.exists():
            path.unlink()
            logger.debug("Cache invalidated: %s", path)
