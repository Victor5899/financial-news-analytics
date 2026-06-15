"""
Thread-safe token bucket rate limiter.

The token bucket algorithm allows short bursts (up to ``capacity`` tokens)
while enforcing a sustainable average throughput of ``rate`` tokens/second.

This is used by the Finnhub client to stay well within the free-tier
per-minute quota (60 requests / minute).

Usage
-----
    from src.utils.rate_limiter import TokenBucketRateLimiter

    limiter = TokenBucketRateLimiter(rate=0.5, capacity=5)

    for ticker in tickers:
        limiter.acquire()       # blocks if tokens exhausted
        response = session.get(url)
"""

from __future__ import annotations

import threading
import time


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter.

    Parameters
    ----------
    rate:
        Tokens refilled per second (sustained throughput).
    capacity:
        Maximum tokens stored at once (controls burst size).
    """

    def __init__(self, rate: float, capacity: float) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity          # start full so first calls are instant
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────────

    def acquire(self, tokens: float = 1.0) -> None:
        """Block the calling thread until ``tokens`` are available.

        For typical usage pass no arguments (acquires exactly 1 token).
        """
        if tokens > self._capacity:
            raise ValueError(
                f"Requested {tokens} tokens but bucket capacity is only {self._capacity}"
            )

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
            # Release the lock while sleeping to allow other threads to proceed.
            time.sleep(0.05)

    @property
    def available_tokens(self) -> float:
        """Current token count (snapshot — may change immediately after read)."""
        with self._lock:
            self._refill()
            return self._tokens

    # ── Private helpers ──────────────────────────────────────────────────────

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time. Must be called under lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        gained = elapsed * self._rate
        self._tokens = min(self._capacity, self._tokens + gained)
        self._last_refill = now
