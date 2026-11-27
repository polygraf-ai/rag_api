"""Route tests for end-user assertion forwarding.

The real SecureAIEmbeddings gateway call runs; only ``httpx.post`` is replaced,
by a stub that records each /v1/embeddings request and answers from a script.
"""

import json

import httpx
import pytest

from app.routes import document_routes
from tests.end_user_assertion_support import (  # noqa: F401 (fixtures)
    EMBEDDING_ROUTES,
    GATEWAY_KEY,
    HEADER,
    assert_secret_absent,
    auth,
    call,
    client,
    files_left,
    logs,
    store,
    upload_dir,
)

ASSERTION = "eua-route-marker-91c2e4"
GATEWAY_MESSAGE = "gateway wording that rag_api must not copy"


def _embeddings(count: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(count)]},
    )


def _refusal(status_code: int, code: str, error_type: str) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "error": {
                "message": GATEWAY_MESSAGE,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


class GatewayStub:
    """Stands in for httpx.post to the gateway's /v1/embeddings."""

    def __init__(self):
        self.calls = []
        self.script = []

    def __call__(self, url, *, headers, json, timeout, **kwargs):
        self.calls.append({"url": url, "headers": dict(headers), "kwargs": kwargs})
        if self.script:
            answer = self.script.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        return _embeddings(len(json["input"]))


@pytest.fixture
def gateway(monkeypatch):
    stub = GatewayStub()
    monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", stub)
    return stub


# --- R1 / R2: accepted and forwarded on the gateway call only ----------------


@pytest.mark.parametrize("route", EMBEDDING_ROUTES)
def test_assertion_reaches_the_gateway_once(route, auth, upload_dir, gateway, store, logs):
    response = call(
        route, auth, upload_dir, [(HEADER, ASSERTION), ("X-Polygraf-Source-App", "direct_api")]
    )

    assert response.status_code == 200, response.text
    assert len(gateway.calls) == 1
    sent = gateway.calls[0]
    assert sent["url"] == "http://secure-ai.test/v1/embeddings"
    assert sent["headers"] == {
        "Authorization": f"Bearer {GATEWAY_KEY}",
        "X-Polygraf-Source-App": "secure_llm_document_search",
        HEADER: ASSERTION,
    }
    assert not sent["kwargs"].get("follow_redirects", False)
    assert ASSERTION not in json.dumps(store.added)
    assert_secret_absent(response, logs, ASSERTION)


# --- R7: no assertion header forwarded --------------------------------------


@pytest.mark.parametrize("route", EMBEDDING_ROUTES)
@pytest.mark.parametrize("extra", [(), ((HEADER, "   "),)], ids=["absent", "blank"])
def test_without_assertion_gateway_request_omits_assertion_header(
    route, extra, auth, upload_dir, gateway, store
):
    response = call(route, auth, upload_dir, extra)

    assert response.status_code == 200, response.text
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["headers"] == {
        "Authorization": f"Bearer {GATEWAY_KEY}",
        "X-Polygraf-Source-App": "secure_llm_document_search",
    }


# --- R3: the gateway's header rules, before any work ------------------------

UNUSABLE_HEADERS = {
    "duplicate": [(HEADER, ASSERTION), (HEADER, ASSERTION)],
    "4097-bytes": [(HEADER, ASSERTION + "a" * (4097 - len(ASSERTION)))],
    "non-ascii": [(HEADER, (ASSERTION + "é").encode("utf-8"))],
}


@pytest.mark.parametrize("route", EMBEDDING_ROUTES)
@pytest.mark.parametrize("case", sorted(UNUSABLE_HEADERS))
def test_unusable_header_is_refused_before_any_work(
    route, case, auth, upload_dir, gateway, store, logs
):
    response = call(route, auth, upload_dir, UNUSABLE_HEADERS[case])

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_end_user_assertion"
    assert detail["type"] == "invalid_request_error"
    assert gateway.calls == []
    assert store.added == []
    assert files_left(upload_dir) == []
    assert_secret_absent(response, logs, ASSERTION)


def test_header_of_exactly_4096_bytes_is_forwarded(auth, upload_dir, gateway, store):
    value = ASSERTION + "a" * (4096 - len(ASSERTION))
    response = call("/query", auth, upload_dir, [(HEADER, value)])

    assert response.status_code == 200, response.text
    assert gateway.calls[0]["headers"][HEADER] == value


# --- R4: relay the gateway's refusal ----------------------------------------

REFUSALS = [
    pytest.param(
        _refusal(401, "end_user_assertion_invalid", "authentication_error"),
        401, "end_user_assertion_invalid", "authentication_error", id="assertion-invalid",
    ),
    pytest.param(
        _refusal(403, "end_user_assertion_required", "permission_error"),
        403, "end_user_assertion_required", "permission_error", id="assertion-required",
    ),
    pytest.param(
        _refusal(400, "invalid_end_user_assertion", "invalid_request_error"),
        400, "invalid_end_user_assertion", "invalid_request_error", id="assertion-400",
    ),
    pytest.param(
        _refusal(401, "gateway_key_revoked", "authentication_error"),
        401, "gateway_key_revoked", "authentication_error", id="key-revoked",
    ),
    pytest.param(
        _refusal(401, "invalid_api_key", "authentication_error"),
        401, "invalid_api_key", "authentication_error", id="invalid-api-key",
    ),
    pytest.param(
        _refusal(500, "server_error", "server_error"), 502, None, None, id="gateway-500"
    ),
    pytest.param(httpx.ReadTimeout("timed out"), 504, None, None, id="timeout"),
    pytest.param(httpx.ConnectError("refused"), 502, None, None, id="connect-error"),
]


@pytest.mark.parametrize("route", ["/query", "/embed"])
@pytest.mark.parametrize("answer, status_code, code, error_type", REFUSALS)
def test_gateway_refusal_is_relayed_with_its_code(
    route, answer, status_code, code, error_type, auth, upload_dir, gateway, store, logs
):
    gateway.script.append(answer)

    response = call(route, auth, upload_dir, [(HEADER, ASSERTION)])

    assert response.status_code == status_code, response.text
    detail = response.json()["detail"]
    assert detail["code"] == code
    assert detail["type"] == error_type
    assert isinstance(detail["message"], str) and detail["message"]
    assert GATEWAY_MESSAGE not in response.text
    # Secure LLM renews on this substring, so it must appear only for that code.
    assert ("end_user_assertion_invalid" in response.text) == (
        code == "end_user_assertion_invalid"
    )
    assert len(gateway.calls) == 1  # no retry inside rag_api
    assert files_left(upload_dir) == []
    assert_secret_absent(response, logs, ASSERTION)


@pytest.fixture
def three_batches(monkeypatch):
    monkeypatch.setattr(document_routes, "CHUNK_SIZE", 20)
    monkeypatch.setattr(document_routes, "CHUNK_OVERLAP", 0)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 1)
    # Three paragraphs of at most 20 characters: three chunks, three batches.
    return b"alpha alpha alpha\n\nbravo bravo bravo\n\ncharlie charlie"


def test_every_ingestion_batch_carries_the_assertion(
    auth, upload_dir, gateway, store, three_batches, logs
):
    response = call("/embed", auth, upload_dir, [(HEADER, ASSERTION)], three_batches)

    assert response.status_code == 200, response.text
    assert len(gateway.calls) == 3
    assert {call["headers"][HEADER] for call in gateway.calls} == {ASSERTION}
    assert len(store.added) == 3
    assert ASSERTION not in json.dumps(store.added)
    assert_secret_absent(response, logs, ASSERTION)


def test_mid_ingestion_401_rolls_back_and_is_relayed(
    auth, upload_dir, gateway, store, three_batches, logs
):
    gateway.script += [
        _embeddings(1),
        _refusal(401, "end_user_assertion_invalid", "authentication_error"),
    ]

    response = call("/embed", auth, upload_dir, [(HEADER, ASSERTION)], three_batches)

    assert response.status_code == 401, response.text
    assert response.json()["detail"]["code"] == "end_user_assertion_invalid"
    assert len(gateway.calls) == 2  # stopped at the refusal, no third batch, no retry
    assert len(store.added) == 1
    attempt_id = store.added[0]["_rag_ingestion_attempt_id"]
    assert store.deleted_by_metadata == [
        {"file_id": "testid1", "_rag_ingestion_attempt_id": attempt_id}
    ]
    assert files_left(upload_dir) == []
    assert_secret_absent(response, logs, ASSERTION)


def test_failed_rollback_still_relays_the_gateway_refusal(
    auth, upload_dir, gateway, store, three_batches, logs, monkeypatch
):
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    async def rollback_fails(self, metadata_filter, executor=None):
        raise RuntimeError("database unavailable during rollback")

    monkeypatch.setattr(AsyncPgVector, "delete_by_metadata", rollback_fails)
    gateway.script += [
        _embeddings(1),
        _refusal(401, "end_user_assertion_invalid", "authentication_error"),
    ]

    response = call("/embed", auth, upload_dir, [(HEADER, ASSERTION)], three_batches)

    # Secure LLM must still see the 401 it renews on, not the rollback error.
    assert response.status_code == 401, response.text
    assert response.json()["detail"]["code"] == "end_user_assertion_invalid"
    assert "database unavailable" not in response.text
    # The rows left behind are reported, not silently dropped.
    assert "Rollback of partial ingestion failed" in logs.text
    assert files_left(upload_dir) == []
    assert_secret_absent(response, logs, ASSERTION)


# --- R6: routes that do not embed ignore the header -------------------------

NON_EMBEDDING_REQUESTS = {
    "ids": ("GET", "/ids", {}),
    "documents": ("GET", "/documents", {"params": {"ids": ["testid1"]}}),
    "delete": ("DELETE", "/documents", {"json": ["testid1"]}),
    "context": ("GET", "/documents/testid1/context", {}),
    "health": ("GET", "/health", {}),
}


@pytest.mark.parametrize("name", sorted(NON_EMBEDDING_REQUESTS))
@pytest.mark.parametrize("case", ["single", *sorted(UNUSABLE_HEADERS)])
def test_non_embedding_routes_ignore_the_header(
    name, case, auth, gateway, store, logs
):
    method, path, kwargs = NON_EMBEDDING_REQUESTS[name]
    extra = [(HEADER, ASSERTION)] if case == "single" else UNUSABLE_HEADERS[case]

    baseline = client.request(method, path, headers=auth, **kwargs)
    with_header = client.request(method, path, headers=[*auth, *extra], **kwargs)

    assert baseline.status_code == 200, baseline.text
    assert with_header.status_code == baseline.status_code
    assert with_header.json() == baseline.json()
    assert gateway.calls == []
    assert_secret_absent(with_header, logs, ASSERTION)
