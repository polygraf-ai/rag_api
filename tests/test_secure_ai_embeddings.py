from unittest.mock import Mock

import httpx
import pytest

from app.services.secure_ai_embeddings import (
    NORMALIZATION_VERSION,
    SecureAIEmbeddingError,
    SecureAIEmbeddings,
    parse_embedding_model,
)


def test_exact_gateway_key_and_model_are_forwarded_once(monkeypatch):
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
        "Authorization": "Bearer pgk_live_fake_test"
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
