"""In-process rate limiting, one limiter per app.

A sliding window of request timestamps per caller. Deliberately small and owned
here rather than delegated to a library: the limiter is enforced as a route
*dependency*, so it either runs or the route does not exist. An earlier
middleware-based attempt could silently classify a route as exempt and apply no
limit at all — a rate limiter that fails open is worse than none, because it
looks like protection. See DECISIONS #23.

Scope and limits, stated plainly:

- **Per process.** Counters live in memory, so multiple workers each keep their
  own. This matches the single-process deployment this project targets.
- **Per client IP.** Behind a reverse proxy every caller looks like the proxy
  unless ``X-Forwarded-For`` is handled, which this does not do.
- **Thread-safe.** Synchronous FastAPI endpoints run in a threadpool, so
  concurrent access is real, not hypothetical; a lock guards every mutation.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from fastapi import HTTPException, Request, status

_WINDOW_SECONDS: dict[str, float] = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
}

#: Ceiling on distinct keys held at once, so a flood of unique callers cannot
#: grow the table without bound. Expired keys are pruned before the cap bites.
MAX_TRACKED_KEYS = 10_000

_TOO_MANY = "Rate limit exceeded. Try again shortly."


@dataclass(frozen=True)
class RateLimitPolicy:
    """How many requests are allowed per window."""

    max_requests: int
    window_seconds: float


def parse_rate_limit(value: str) -> RateLimitPolicy:
    """Parse a ``"20/minute"`` style limit.

    Raises:
        ValueError: the value is not ``<positive int>/<second|minute|hour|day>``.
    """
    text = value.strip().lower()
    count_text, separator, unit_text = text.partition("/")
    unit = unit_text.strip().rstrip("s")
    if not separator or unit not in _WINDOW_SECONDS:
        raise ValueError(
            f"Invalid rate limit {value!r}; expected '<count>/<second|minute|hour|day>'"
        )
    try:
        count = int(count_text.strip())
    except ValueError:
        raise ValueError(f"Invalid rate limit count in {value!r}") from None
    if count < 1:
        raise ValueError(f"Rate limit count must be at least 1, got {count}")
    return RateLimitPolicy(max_requests=count, window_seconds=_WINDOW_SECONDS[unit])


class RateLimiter:
    """Sliding-window limiter for one application instance."""

    def __init__(
        self,
        policy: RateLimitPolicy,
        *,
        enabled: bool = True,
        clock: Callable[[], float] = monotonic,
        max_tracked_keys: int = MAX_TRACKED_KEYS,
    ) -> None:
        """Args:
        policy: allowance per window.
        enabled: when False every request is allowed.
        clock: monotonic time source; injectable so tests can advance it
            without sleeping.
        max_tracked_keys: cap on distinct callers held in memory.
        """
        self._policy = policy
        self._enabled = enabled
        self._clock = clock
        self._max_tracked_keys = max_tracked_keys
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> float | None:
        """Record a request from ``key``.

        Returns:
            ``None`` when the request is allowed, otherwise the seconds to wait
            before the oldest request in the window expires.
        """
        if not self._enabled:
            return None

        now = self._clock()
        cutoff = now - self._policy.window_seconds

        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self._max_tracked_keys:
                    self._prune(cutoff)
                hits = self._hits.setdefault(key, deque())

            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self._policy.max_requests:
                return hits[0] + self._policy.window_seconds - now

            hits.append(now)
            return None

    def _prune(self, cutoff: float) -> None:
        """Drop keys with no live requests. Caller must hold the lock."""
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]


def client_key(request: Request) -> str:
    """Identify the caller. Falls back to a constant when the peer is unknown."""
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    """Route dependency: reject the request when the caller is over its limit.

    Raises:
        HTTPException: 429 with a ``Retry-After`` header.
    """
    # No getattr fallback on purpose: an app without a limiter must fail loudly
    # rather than quietly serve unlimited traffic. Failing open is the exact
    # bug that made the previous middleware-based attempt useless.
    limiter: RateLimiter = request.app.state.rate_limiter

    retry_after = limiter.check(client_key(request))
    if retry_after is None:
        return

    raise HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        _TOO_MANY,
        headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
    )
