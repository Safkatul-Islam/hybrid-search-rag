"""API-key authentication for the query and ingest endpoints.

One shared key, supplied as an ``X-API-Key`` header and compared with
``hmac.compare_digest`` so a near-miss key costs the same time as a wrong one.
Rejections return a single generic 401: the caller is never told whether the
key was missing, malformed, or simply wrong.

``/health`` is deliberately left unauthenticated so a liveness probe does not
need a credential.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request, status

API_KEY_HEADER = "X-API-Key"

_UNAUTHORIZED = "Invalid or missing API key."


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """Reject the request unless it carries the configured API key.

    Raises:
        HTTPException: 401 when the header is absent or does not match.
    """
    expected: str = getattr(request.app.state, "api_key", "")
    if not expected:
        # create_app refuses to build an app without a key, so this is
        # unreachable in normal operation. Fail closed regardless: an
        # unconfigured key must never mean "allow everyone".
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _UNAUTHORIZED)

    if x_api_key is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _UNAUTHORIZED)

    # Compare as bytes: compare_digest rejects str inputs outside ASCII.
    if not hmac.compare_digest(x_api_key.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _UNAUTHORIZED)
