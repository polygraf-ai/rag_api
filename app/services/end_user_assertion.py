"""End-user attribution through a shared agent: rag_api's half.

Secure LLM sends a short-lived assertion, signed by the governance dashboard for
the signed-in user, in ``X-Polygraf-End-User-Assertion``. rag_api never parses
or verifies it. It reads it once per embedding request, applies the gateway's
own header rules up front, and passes it on the gateway ``/v1/embeddings`` call
of that same request, so the embedding is credited to (and enforced for) that
person rather than the agent's owner.

The value is a bearer credential for that user on that agent for up to five
minutes. It must never be logged, echoed in a response, stored, or sent
anywhere but the gateway. ``EndUserAssertionRedactingFilter`` is a second line
of defence for logs, not a substitute for never logging it.
"""

from __future__ import annotations

import logging
import re

from fastapi import HTTPException, status
from starlette.requests import Request

END_USER_ASSERTION_HEADER = "X-Polygraf-End-User-Assertion"
# The gateway's limit on the raw header value (secure-ai-provider end_user.py).
MAX_ASSERTION_BYTES = 4096

_HEADER_KEY = END_USER_ASSERTION_HEADER.lower().encode("latin-1")
# A compact JWS is base64url plus dots; anything outside printable ASCII would
# make httpx fail to encode the outbound header.
_PRINTABLE_ASCII = re.compile(r"[\x20-\x7e]+")


def _rejected(message: str) -> HTTPException:
    """The 400 for an unusable header. The message never repeats the value."""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "code": "invalid_end_user_assertion",
            "type": "invalid_request_error",
            "message": message,
        },
    )


def read_end_user_assertion(request: Request) -> str | None:
    """The end-user assertion sent with *request*, or None when there is none.

    Mirrors the gateway's rules so a header the gateway would refuse is refused
    here, before any upload is saved or any embedding is paid for:

    * absent, or blank after stripping: None (the request goes on unattributed);
    * sent more than once: 400 (Starlette would otherwise hand the route only
      the first value and hide the duplicate);
    * more than 4096 bytes, measured before stripping: 400;
    * anything outside printable ASCII: 400.

    Read from the raw header list rather than declared as a FastAPI ``Header``,
    so request validation can never echo the value in a 422 body or log.
    """
    values = [value for key, value in request.headers.raw if key.lower() == _HEADER_KEY]
    if not values:
        return None
    if len(values) > 1:
        raise _rejected(f"Send at most one {END_USER_ASSERTION_HEADER} header")
    raw = values[0]
    if len(raw) > MAX_ASSERTION_BYTES:
        raise _rejected(
            f"{END_USER_ASSERTION_HEADER} exceeds {MAX_ASSERTION_BYTES} bytes"
        )
    assertion = raw.decode("latin-1").strip()
    if not assertion:
        return None
    if not _PRINTABLE_ASCII.fullmatch(assertion):
        raise _rejected(f"{END_USER_ASSERTION_HEADER} must be printable ASCII")
    return assertion


_REDACTED = "[redacted]"
# The header name followed by a value, in the shapes a log line can take:
# "Name: value", "name=value", "'name': 'value'", "(b'name', b'value')".
_HEADER_WITH_VALUE = re.compile(
    r"(x-polygraf-end-user-assertion['\"]?\s*[:=,]\s*b?['\"]?)[^\s'\",)}\]]+",
    re.IGNORECASE,
)
# A compact JWS whose protected header is JSON: base64url of '{"' is "eyJ".
# The dashboard's assertion is one ({"alg": "HS256", "typ": "pg-eua+jwt", ...}).
_COMPACT_JWS = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


def redact_end_user_assertions(text: str) -> str:
    text = _HEADER_WITH_VALUE.sub(lambda match: match.group(1) + _REDACTED, text)
    return _COMPACT_JWS.sub(_REDACTED, text)


class EndUserAssertionRedactingFilter(logging.Filter):
    """Replace end-user assertions in a log record with ``[redacted]``.

    Covers the formatted message (including %-style args), exception text and
    stack info. Records with nothing to redact are left exactly as they were.
    """

    _formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            # A malformed record; let the handler report it as it would anyway.
            return True
        redacted = redact_end_user_assertions(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()

        if record.exc_info and not record.exc_text:
            record.exc_text = self._formatter.formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact_end_user_assertions(record.exc_text)
        if record.stack_info:
            record.stack_info = redact_end_user_assertions(record.stack_info)
        return True
