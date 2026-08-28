from __future__ import annotations

import time
from threading import Lock
from typing import Protocol

import httpx


class Page(Protocol):
    # A read-only property, not a plain attribute: httpx.Response.text is a
    # property, and a writable protocol member would exclude it.
    @property
    def text(self) -> str: ...


class Fetcher(Protocol):
    """All that ingestion needs from a client: fetch a URL, hand back the text."""

    def get(self, url: str) -> Page: ...


USER_AGENT = "shgk-corpus/0.1 (private research corpus; respectful crawler)"


class HttpClient:
    def __init__(
        self,
        *,
        delay: float = 0.0,
        timeout: float = 45.0,
        retries: int = 3,
    ):
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.retries = retries
        self._last_request_at = 0.0
        self._request_lock = Lock()
        self._client = httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/json,*/*;q=0.8",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get(self, url: str) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            # httpx.Client can be shared by worker threads. Serialize only the
            # request start times so concurrency overlaps network latency while
            # preserving one global per-host request interval.
            with self._request_lock:
                elapsed = time.monotonic() - self._last_request_at
                if elapsed < self.delay:
                    time.sleep(self.delay - elapsed)
                self._last_request_at = time.monotonic()
            try:
                response = self._client.get(url)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as error:
                last_error = error
                if error.response.status_code not in {429, 500, 502, 503, 504}:
                    raise
                retry_after = error.response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(wait)
            except (httpx.TransportError, TimeoutError) as error:
                last_error = error
                time.sleep(2**attempt)
        assert last_error is not None
        raise last_error

