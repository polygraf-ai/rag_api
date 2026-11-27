"""Unit tests for the end-user assertion header helper and its log redaction.

The assertion is a bearer credential for one user on one agent, so every test
here also checks that the value never appears in an error, a response or a log.
"""

import io
import json
import logging

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from app.services.end_user_assertion import (
    END_USER_ASSERTION_HEADER,
    MAX_ASSERTION_BYTES,
    EndUserAssertionRedactingFilter,
    read_end_user_assertion,
)

MARKER = "eua-marker-7f3c9a"
# Shaped like the dashboard's compact JWS: header {"alg":"HS256","typ":"pg-eua+jwt"}.
JWS_MARKER = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6InBnLWV1YStqd3QifQ"
    ".eyJzdWIiOiJ1c2VyLTEyMyJ9"
    ".c2lnbmF0dXJlLW1hcmtlcg"
)
_HEADER_KEY = b"x-polygraf-end-user-assertion"


def _request(*values: bytes, key: bytes = _HEADER_KEY) -> Request:
    headers = [(b"authorization", b"Bearer rag-jwt")]
    headers += [(key, value) for value in values]
    return Request({"type": "http", "method": "POST", "path": "/query", "headers": headers})


def test_header_name_matches_gateway():
    assert END_USER_ASSERTION_HEADER == "X-Polygraf-End-User-Assertion"
    assert MAX_ASSERTION_BYTES == 4096


def test_absent_header_is_none():
    assert read_end_user_assertion(_request()) is None


@pytest.mark.parametrize("raw", [b"", b"   ", b"\t \t"])
def test_blank_header_counts_as_absent(raw):
    assert read_end_user_assertion(_request(raw)) is None


def test_single_value_is_returned_stripped():
    assert read_end_user_assertion(_request(f"  {MARKER}  ".encode())) == MARKER


def test_header_name_is_case_insensitive():
    request = _request(MARKER.encode(), key=b"X-Polygraf-End-User-Assertion")
    assert read_end_user_assertion(request) == MARKER


def test_value_of_exactly_4096_bytes_passes():
    value = "a" * MAX_ASSERTION_BYTES
    assert read_end_user_assertion(_request(value.encode())) == value


def test_compact_jws_passes_unchanged():
    assert read_end_user_assertion(_request(JWS_MARKER.encode())) == JWS_MARKER


@pytest.mark.parametrize(
    "values",
    [
        pytest.param((MARKER.encode(), MARKER.encode()), id="duplicate"),
        pytest.param((b"", MARKER.encode()), id="duplicate-one-blank"),
        pytest.param((("a" * 4096 + MARKER[:1]).encode(),), id="4097-bytes"),
        # Measured before stripping, as the gateway does.
        pytest.param((b" " * 4097,), id="4097-bytes-of-whitespace"),
        pytest.param(((MARKER + "é").encode("utf-8"),), id="non-ascii"),
        pytest.param(((MARKER + "\x01").encode(),), id="control-character"),
        pytest.param(((MARKER + "\x7f").encode(),), id="delete-character"),
    ],
)
def test_unusable_header_is_a_400_that_never_repeats_the_value(values):
    with pytest.raises(HTTPException) as error:
        read_end_user_assertion(_request(*values))

    assert error.value.status_code == 400
    detail = error.value.detail
    assert detail["code"] == "invalid_end_user_assertion"
    assert detail["type"] == "invalid_request_error"
    assert isinstance(detail["message"], str) and detail["message"]
    assert MARKER not in json.dumps(detail)
    assert MARKER not in str(error.value)
    # Secure LLM renews on this substring; a 400 must never contain it.
    assert "end_user_assertion_invalid" not in json.dumps(detail)


# --- log redaction ---------------------------------------------------------


@pytest.fixture
def redacting_logger():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(EndUserAssertionRedactingFilter())
    log = logging.getLogger("tests.end_user_assertion.redaction")
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    yield log, stream
    log.removeHandler(handler)


@pytest.mark.parametrize(
    "message",
    [
        f"X-Polygraf-End-User-Assertion: {MARKER}",
        f"x-polygraf-end-user-assertion={MARKER}",
        f"headers={{'x-polygraf-end-user-assertion': '{MARKER}', 'host': 'rag'}}",
        f'headers={{"X-Polygraf-End-User-Assertion": "{MARKER}"}}',
        f"raw=[(b'x-polygraf-end-user-assertion', b'{MARKER}')]",
        f"bare token {JWS_MARKER} in text",
    ],
)
def test_filter_redacts_assertion_in_message(redacting_logger, message):
    log, stream = redacting_logger
    log.info(message)
    output = stream.getvalue()
    assert MARKER not in output
    assert JWS_MARKER not in output
    assert "[redacted]" in output


def test_filter_redacts_assertion_in_format_args(redacting_logger):
    log, stream = redacting_logger
    log.debug("request headers: %s", {END_USER_ASSERTION_HEADER: JWS_MARKER})
    assert JWS_MARKER not in stream.getvalue()
    assert "[redacted]" in stream.getvalue()


def test_filter_redacts_assertion_in_exception_text(redacting_logger):
    log, stream = redacting_logger
    try:
        raise RuntimeError(f"upstream echoed {JWS_MARKER}")
    except RuntimeError:
        log.error("failed", exc_info=True)
    assert "RuntimeError" in stream.getvalue()
    assert JWS_MARKER not in stream.getvalue()


def test_filter_leaves_unrelated_records_untouched(redacting_logger):
    log, stream = redacting_logger
    record = logging.LogRecord(
        "x", logging.INFO, __file__, 1, "file %s has %d chunks", ("abc", 3), None
    )
    assert EndUserAssertionRedactingFilter().filter(record) is True
    assert record.msg == "file %s has %d chunks"
    assert record.args == ("abc", 3)


def test_root_handler_has_the_redacting_filter():
    from app.config import handler

    assert any(isinstance(f, EndUserAssertionRedactingFilter) for f in handler.filters)


# --- 422 handler -----------------------------------------------------------


async def test_validation_handler_never_echoes_header_inputs(caplog):
    from main import validation_exception_handler

    exc = RequestValidationError(
        [
            {
                "type": "string_too_long",
                "loc": ("header", "x-polygraf-end-user-assertion"),
                "msg": "String should have at most 4096 characters",
                "input": MARKER,
                "ctx": {"max_length": 4096},
            },
            {
                "type": "missing",
                "loc": ("body", "query"),
                "msg": "Field required",
                "input": {"file_id": "f1"},
            },
        ]
    )
    with caplog.at_level(logging.DEBUG):
        response = await validation_exception_handler(_request(), exc)

    body = json.loads(response.body)
    assert response.status_code == 422
    assert MARKER not in response.body.decode()
    assert MARKER not in caplog.text
    header_error, body_error = body["detail"]
    assert header_error["loc"] == ["header", "x-polygraf-end-user-assertion"]
    assert "input" not in header_error
    # Body validation errors are reported as before.
    assert body_error["input"] == {"file_id": "f1"}
