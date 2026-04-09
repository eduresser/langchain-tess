"""Typed exception hierarchy for the Tess AI API.

Maps HTTP status codes returned by the Tess API to specific exception
classes so callers can handle authentication failures, rate limits, and
validation errors differently from generic server errors.

Reference: https://docs.tess.im/en/errors
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class TessAPIError(Exception):
    """Base exception for all Tess AI API errors.

    Attributes:
        status_code: HTTP status code from the API response.
        body: Parsed JSON body of the error response (may be ``None``
            if the body could not be decoded).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class TessAuthenticationError(TessAPIError):
    """Raised on HTTP 403 — invalid or expired API key."""


class TessValidationError(TessAPIError):
    """Raised on HTTP 400 — invalid request payload."""


class TessPayloadTooLargeError(TessAPIError):
    """Raised on HTTP 413 — request body exceeds allowed size."""


class TessRateLimitError(TessAPIError):
    """Raised on HTTP 429 — rate limit exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying, as reported by
            the API.  Falls back to ``60`` when the field is absent.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 429,
        body: Optional[Dict[str, Any]] = None,
        retry_after: int = 60,
    ) -> None:
        super().__init__(message, status_code=status_code, body=body)
        self.retry_after = retry_after


class TessServerError(TessAPIError):
    """Raised on HTTP 5xx — internal server error."""


_STATUS_TO_EXCEPTION = {
    400: TessValidationError,
    403: TessAuthenticationError,
    413: TessPayloadTooLargeError,
    429: TessRateLimitError,
    500: TessServerError,
}


def raise_for_tess_status(status_code: int, body: Optional[Dict[str, Any]]) -> None:
    """Raise the appropriate :class:`TessAPIError` subclass for *status_code*.

    Does nothing when *status_code* < 400.  For 429 responses, extracts
    ``retry_after`` from *body* when available.
    """
    if status_code < 400:
        return

    body = body or {}
    error_msg = body.get("error", f"Tess API error (HTTP {status_code})")

    exc_cls = _STATUS_TO_EXCEPTION.get(status_code)
    if exc_cls is None:
        if status_code >= 500:
            exc_cls = TessServerError
        else:
            exc_cls = TessAPIError

    if exc_cls is TessRateLimitError:
        retry_after = int(body.get("retry_after", 60))
        raise TessRateLimitError(
            error_msg,
            status_code=status_code,
            body=body,
            retry_after=retry_after,
        )

    raise exc_cls(error_msg, status_code=status_code, body=body)
