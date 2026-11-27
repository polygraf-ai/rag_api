from unittest.mock import Mock

import httpx
import pytest

from app.services import secure_ai_embeddings
from app.services.secure_ai_embeddings import (
    NORMALIZATION_VERSION,
    SecureAIEmbeddingError,
    SecureAIEmbeddings,
    parse_embedding_model,
)


# secure-llm#81
def test_secure_llm_81_upload_sends_default_source_app_with_gateway_key_and_model(monkeypatch):
    response = Mock(
        is_success=True,
        json=Mock(
            return_value={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }
        ),
    )
    post = Mock(return_value=response)
    monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", post)

    embedder = SecureAIEmbeddings("pgk_live_fake_test", "gemini/text-embedding-004")

    assert embedder.embed_documents(["first", "second"]) == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    post.assert_called_once()
    assert post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer pgk_live_fake_test",
        "X-Polygraf-Source-App": "secure_llm_document_search",
    }
    assert post.call_args.kwargs["json"] == {
        "model": "gemini/text-embedding-004",
        "input": ["first", "second"],
    }
    assert embedder.fingerprint(2) == {
        "provider": "gemini",
        "model": "text-embedding-004",
        "dimensions": 2,
        "normalization_version": NORMALIZATION_VERSION,
    }


@pytest.mark.parametrize("value", ["", "model-only", "/model", "provider/", "  /  "])
def test_embedding_model_must_be_fully_qualified(value):
    with pytest.raises(ValueError, match="provider/model"):
        parse_embedding_model(value)


def test_rejection_does_not_retry_or_change_authority(monkeypatch):
    response = Mock(is_success=False, status_code=403)
    post = Mock(return_value=response)
    monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", post)

    with pytest.raises(SecureAIEmbeddingError) as error:
        SecureAIEmbeddings("pgk_live_revoked_test", "cohere/embed-v4").embed_query(
            "query"
        )

    assert error.value.status_code == 403
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["model"] == "cohere/embed-v4"


def test_timeout_does_not_retry(monkeypatch):
    post = Mock(side_effect=httpx.ReadTimeout("timeout"))
    monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", post)

    with pytest.raises(SecureAIEmbeddingError) as error:
        SecureAIEmbeddings("pgk_live_fake_test", "gemini/embed").embed_query("query")

    assert error.value.status_code == 504
    post.assert_called_once()


def test_secret_is_not_in_error(monkeypatch):
    secret = "pgk_live_must_never_leak"
    post = Mock(side_effect=httpx.ConnectError(f"failed with {secret}"))
    monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", post)

    with pytest.raises(SecureAIEmbeddingError) as error:
        SecureAIEmbeddings(secret, "gemini/embed").embed_query("query")

    assert secret not in str(error.value)


# --- end-user assertion (R2, R4, R7) ----------------------------------------

ASSERTION = "eua-marker-3b1d55"


def _ok_response(count: int = 1) -> Mock:
    return Mock(
        is_success=True,
        json=Mock(
            return_value={
                "data": [{"index": i, "embedding": [0.1, 0.2]} for i in range(count)]
            }
        ),
    )


def _refusal(status_code: int, body=None, json_error: Exception | None = None) -> Mock:
    json_mock = Mock(side_effect=json_error) if json_error else Mock(return_value=body)
    return Mock(is_success=False, status_code=status_code, json=json_mock)


# secure-llm#81
def test_secure_llm_81_query_sends_default_source_app(monkeypatch):
    post = Mock(return_value=_ok_response())
    monkeypatch.setattr(secure_ai_embeddings.httpx, "post", post)

    SecureAIEmbeddings("pgk_live_fake_test", "gemini/embed").embed_query("query")

    post.assert_called_once()
    assert (
        post.call_args.kwargs["headers"].get("X-Polygraf-Source-App")
        == "secure_llm_document_search"
    )


# secure-llm#81
def test_secure_llm_81_uses_configured_source_app(monkeypatch):
    monkeypatch.setattr(
        secure_ai_embeddings, "SECURE_AI_SOURCE_APP", "custom_document_search", raising=False
    )
    post = Mock(return_value=_ok_response())
    monkeypatch.setattr(secure_ai_embeddings.httpx, "post", post)

    SecureAIEmbeddings("pgk_live_fake_test", "gemini/embed").embed_query("query")

    post.assert_called_once()
    assert (
        post.call_args.kwargs["headers"].get("X-Polygraf-Source-App")
        == "custom_document_search"
    )


# secure-llm#81
def test_secure_llm_81_source_app_is_sent_with_assertion(monkeypatch):
    post = Mock(return_value=_ok_response())
    monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", post)

    SecureAIEmbeddings(
        "pgk_live_fake_test", "gemini/embed", end_user_assertion=ASSERTION
    ).embed_query("query")

    post.assert_called_once()
    assert post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer pgk_live_fake_test",
        "X-Polygraf-End-User-Assertion": ASSERTION,
        "X-Polygraf-Source-App": "secure_llm_document_search",
    }


@pytest.mark.parametrize("assertion", [None, ""])
def test_without_assertion_omits_assertion_header(monkeypatch, assertion):
    post = Mock(return_value=_ok_response())
    monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", post)

    SecureAIEmbeddings(
        "pgk_live_fake_test", "gemini/embed", end_user_assertion=assertion
    ).embed_query("query")

    headers = post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer pgk_live_fake_test"
    assert "X-Polygraf-End-User-Assertion" not in headers


def test_every_batch_carries_the_same_assertion(monkeypatch):
    post = Mock(side_effect=lambda *a, **kw: _ok_response(len(kw["json"]["input"])))
    monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", post)
    embedder = SecureAIEmbeddings(
        "pgk_live_fake_test", "gemini/embed", end_user_assertion=ASSERTION
    )

    for batch in (["a", "b"], ["c"], ["d", "e"]):
        embedder.embed_documents(batch)

    assert post.call_count == 3
    for call in post.call_args_list:
        assert call.kwargs["headers"]["X-Polygraf-End-User-Assertion"] == ASSERTION


# secure-llm#81
def test_secure_llm_81_source_app_does_not_reach_redirect_target(monkeypatch):
    gateway_url = "https://gateway.test"
    redirect_url = "https://redirect.test/embeddings"
    monkeypatch.setattr(secure_ai_embeddings, "SECURE_AI_BASE_URL", gateway_url)

    def respond(request):
        if request.url.host == "gateway.test":
            return httpx.Response(307, headers={"Location": redirect_url})
        return httpx.Response(200, json=_ok_response().json())

    handler = Mock(side_effect=respond)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(secure_ai_embeddings.httpx, "post", client.post)

        with pytest.raises(SecureAIEmbeddingError, match="status 307") as error:
            SecureAIEmbeddings(
                "pgk_live_fake_test", "gemini/embed", end_user_assertion=ASSERTION
            ).embed_query("query")

    assert error.value.status_code == 502
    handler.assert_called_once()
    request = handler.call_args.args[0]
    assert request.url == f"{gateway_url}/v1/embeddings"
    assert request.headers["X-Polygraf-End-User-Assertion"] == ASSERTION
    assert request.headers.get("X-Polygraf-Source-App") == "secure_llm_document_search"


def _gateway_error(code, error_type, message="gateway text that must not leak"):
    return {"error": {"message": message, "type": error_type, "param": None, "code": code}}


@pytest.mark.parametrize(
    "status_code, body, expected_status, expected_code, expected_type",
    [
        (401, _gateway_error("end_user_assertion_invalid", "authentication_error"), 401, "end_user_assertion_invalid", "authentication_error"),
        (403, _gateway_error("end_user_assertion_required", "permission_error"), 403, "end_user_assertion_required", "permission_error"),
        (400, _gateway_error("invalid_end_user_assertion", "invalid_request_error"), 400, "invalid_end_user_assertion", "invalid_request_error"),
        (401, _gateway_error("gateway_key_revoked", "authentication_error"), 401, "gateway_key_revoked", "authentication_error"),
        (401, _gateway_error("invalid_api_key", "authentication_error"), 401, "invalid_api_key", "authentication_error"),
        (403, _gateway_error("embedding_not_enabled", "permission_error"), 403, "embedding_not_enabled", "permission_error"),
        (429, _gateway_error("budget_exceeded", "rate_limit_error"), 429, "budget_exceeded", "rate_limit_error"),
        # A status rag_api rewrites to 502 carries no gateway code.
        (500, _gateway_error("server_error", "server_error"), 502, None, None),
        (503, _gateway_error("upstream_unavailable", "server_error"), 502, None, None),
        # Bodies that do not carry a usable code.
        (401, {"detail": "not the openai shape"}, 401, None, None),
        (401, {"error": "a string, not an object"}, 401, None, None),
        (401, ["not", "an", "object"], 401, None, None),
        (401, _gateway_error("End User Invalid!", "authentication_error"), 401, None, "authentication_error"),
        (401, _gateway_error("end_user_assertion_invalid", "Bad Type"), 401, "end_user_assertion_invalid", None),
        (401, _gateway_error("a" * 65, "authentication_error"), 401, None, "authentication_error"),
        (401, _gateway_error(17, None), 401, None, None),
    ],
)
def test_refusal_keeps_status_and_a_safe_gateway_code(
    monkeypatch, status_code, body, expected_status, expected_code, expected_type
):
    post = Mock(return_value=_refusal(status_code, body))
    monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", post)

    with pytest.raises(SecureAIEmbeddingError) as error:
        SecureAIEmbeddings(
            "pgk_live_fake_test", "gemini/embed", end_user_assertion=ASSERTION
        ).embed_query("query")

    assert error.value.status_code == expected_status
    assert error.value.code == expected_code
    assert error.value.error_type == expected_type
    assert "gateway text that must not leak" not in str(error.value)
    assert ASSERTION not in str(error.value)
    post.assert_called_once()


def test_non_json_refusal_body_keeps_status_without_code(monkeypatch):
    post = Mock(return_value=_refusal(401, json_error=ValueError("not json")))
    monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", post)

    with pytest.raises(SecureAIEmbeddingError) as error:
        SecureAIEmbeddings("pgk_live_fake_test", "gemini/embed").embed_query("query")

    assert error.value.status_code == 401
    assert error.value.code is None
    assert error.value.error_type is None


def test_transport_failures_carry_no_code(monkeypatch):
    for side_effect, expected_status in (
        (httpx.ReadTimeout("timeout"), 504),
        (httpx.ConnectError("refused"), 502),
    ):
        post = Mock(side_effect=side_effect)
        monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", post)
        with pytest.raises(SecureAIEmbeddingError) as error:
            SecureAIEmbeddings("pgk_live_fake_test", "gemini/embed").embed_query("q")
        assert error.value.status_code == expected_status
        assert error.value.code is None
        post.assert_called_once()


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_assertion_refusal_is_not_retried(monkeypatch, status_code):
    post = Mock(
        return_value=_refusal(
            status_code, _gateway_error("end_user_assertion_invalid", "authentication_error")
        )
    )
    monkeypatch.setattr("app.services.secure_ai_embeddings.httpx.post", post)

    with pytest.raises(SecureAIEmbeddingError):
        SecureAIEmbeddings(
            "pgk_live_fake_test", "gemini/embed", end_user_assertion=ASSERTION
        ).embed_query("query")

    post.assert_called_once()


def test_assertion_is_not_in_errors_repr_or_logs(monkeypatch, caplog):
    embedder = SecureAIEmbeddings(
        "pgk_live_fake_test", "gemini/embed", end_user_assertion=ASSERTION
    )
    assert ASSERTION not in repr(embedder)

    with caplog.at_level("DEBUG"):
        for side_effect in (
            httpx.ConnectError(f"failed with {ASSERTION}"),
            httpx.ReadTimeout(f"timed out with {ASSERTION}"),
        ):
            monkeypatch.setattr(
                "app.services.secure_ai_embeddings.httpx.post",
                Mock(side_effect=side_effect),
            )
            with pytest.raises(SecureAIEmbeddingError) as error:
                embedder.embed_query("query")
            assert ASSERTION not in str(error.value)
            assert ASSERTION not in repr(error.value)

    assert ASSERTION not in caplog.text
