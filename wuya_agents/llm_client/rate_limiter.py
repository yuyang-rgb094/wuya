"""
Rate limiting for WuYa LLM Client.

Implements a sliding window rate limiter supporting both RPM
(requests per minute) and TPM (tokens per minute) limits.
Uses asyncio.Lock for concurrency safety without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from .config import RateLimitConfig

logger = logging.getLogger(__name__)


@dataclass
class RateLimitStats:
    """Statistics for rate limiter monitoring."""

    total_accepted: int = 0
    """Total requests that were accepted (not throttled)."""

    total_rejected: int = 0
    """Total requests that were throttled (had to wait)."""

    total_wait_time: float = 0.0
    """Cumulative wait time in seconds due to throttling."""

    @property
    def avg_wait_time(self) -> float:
        """Average wait time per throttled request."""
        if self.total_rejected == 0:
            return 0.0
        return self.total_wait_time / self.total_rejected


class RateLimiter:
    """
    Sliding window rate limiter for LLM API calls.

    Tracks request timestamps within a 60-second sliding window and
    enforces both RPM (requests per minute) and TPM (tokens per minute)
    limits. When a limit would be exceeded, the caller awaits until
    the oldest entry in the window expires.

    Thread-safe for use within a single event loop via asyncio.Lock.

    Example::

        limiter = RateLimiter(rpm=60, tpm=40000)

        # Before each API call
        await limiter.acquire(token_count=500)

        # After receiving response, record actual tokens
        limiter.record_tokens(480)

        # Check stats
        print(limiter.stats)
    """

    def __init__(
        self,
        rpm: int = 100,
        tpm: int = 100_000,
        window_seconds: float = 60.0,
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            rpm: Maximum requests per minute. 0 means unlimited.
            tpm: Maximum tokens per minute. 0 means unlimited.
            window_seconds: Sliding window duration in seconds. Default: 60.
        """
        self._rpm = max(0, rpm)
        self._tpm = max(0, tpm)
        self._window = window_seconds

        # Sliding window: deque of (timestamp, token_count) tuples
        self._request_window: Deque[tuple[float, int]] = deque()

        # Current window totals
        self._current_request_count: int = 0
        self._current_token_count: int = 0

        # Async lock for concurrency safety
        self._lock = asyncio.Lock()

        # Statistics
        self.stats = RateLimitStats()

    @classmethod
    def from_config(cls, config: RateLimitConfig) -> RateLimiter:
        """Create a RateLimiter from a RateLimitConfig."""
        return cls(rpm=config.rpm, tpm=config.tpm)

    def _prune_window(self, now: float) -> None:
        """
        Remove expired entries from the sliding window.

        Args:
            now: Current timestamp (time.time()).
        """
        cutoff = now - self._window
        while self._request_window and self._request_window[0][0] <= cutoff:
            _, tokens = self._request_window.popleft()
            self._current_request_count -= 1
            self._current_token_count -= tokens

        # Safety: ensure non-negative
        self._current_request_count = max(0, self._current_request_count)
        self._current_token_count = max(0, self._current_token_count)

    def _calculate_wait_time(self, now: float) -> float:
        """
        Calculate how long to wait before the next request can be made.

        Returns the maximum wait time needed to satisfy both RPM and TPM
        constraints. Returns 0 if no waiting is needed.

        Args:
            now: Current timestamp.

        Returns:
            Wait time in seconds (0 if no wait needed).
        """
        wait = 0.0

        # Check RPM limit
        if self._rpm > 0 and self._current_request_count >= self._rpm:
            # Wait until the oldest request expires
            if self._request_window:
                oldest_time = self._request_window[0][0]
                rpm_wait = (oldest_time + self._window) - now
                wait = max(wait, rpm_wait)

        # Check TPM limit
        if self._tpm > 0 and self._current_token_count >= self._tpm:
            # Calculate how many tokens need to expire
            excess = self._current_token_count - self._tpm + 1
            tokens_to_expire = 0
            expire_time = now
            for ts, tk in self._request_window:
                tokens_to_expire += tk
                expire_time = ts
                if tokens_to_expire >= excess:
                    break

            tpm_wait = (expire_time + self._window) - now
            wait = max(wait, tpm_wait)

        return max(0.0, wait)

    async def acquire(self, token_count: int = 0) -> None:
        """
        Acquire permission to make an API request.

        If the request would exceed rate limits, this method asynchronously
        waits until the window has enough capacity. The estimated token count
        is pre-registered in the window to prevent concurrent requests from
        overshooting the limit.

        Args:
            token_count: Estimated number of tokens this request will consume.
                Used for TPM enforcement. Actual tokens should be updated via
                record_tokens() after the response is received.
        """
        async with self._lock:
            now = time.time()
            self._prune_window(now)

            wait = self._calculate_wait_time(now)

            if wait > 0:
                self.stats.total_rejected += 1
                self.stats.total_wait_time += wait
                logger.debug(
                    "Rate limit reached. Waiting %.2fs "
                    "(RPM: %d/%d, TPM: %d/%d)",
                    wait,
                    self._current_request_count,
                    self._rpm,
                    self._current_token_count,
                    self._tpm,
                )
                await asyncio.sleep(wait)

                # Re-prune after waiting
                now = time.time()
                self._prune_window(now)

            # Register the request in the window
            self._request_window.append((now, token_count))
            self._current_request_count += 1
            self._current_token_count += token_count
            self.stats.total_accepted += 1

    def record_tokens(self, actual_tokens: int) -> None:
        """
        Update the token count for the most recent request.

        Call this after receiving the API response to correct the
        estimated token count with the actual value.

        Args:
            actual_tokens: Actual number of tokens consumed.
        """
        if self._request_window:
            _, estimated = self._request_window[-1]
            diff = actual_tokens - estimated
            self._current_token_count += diff
            self._request_window[-1] = (
                self._request_window[-1][0],
                actual_tokens,
            )

    @property
    def current_rpm(self) -> int:
        """Current requests in the sliding window."""
        return self._current_request_count

    @property
    def current_tpm(self) -> int:
        """Current tokens in the sliding window."""
        return self._current_token_count

    @property
    def rpm_limit(self) -> int:
        """Configured RPM limit."""
        return self._rpm

    @property
    def tpm_limit(self) -> int:
        """Configured TPM limit."""
        return self._tpm

    def reset(self) -> None:
        """Reset the rate limiter state and statistics."""
        self._request_window.clear()
        self._current_request_count = 0
        self._current_token_count = 0
        self.stats = RateLimitStats()

    def __repr__(self) -> str:
        return (
            f"RateLimiter(rpm={self._rpm}, tpm={self._tpm}, "
            f"current_rpm={self._current_request_count}, "
            f"current_tpm={self._current_token_count})"
        )
