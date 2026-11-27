import re
from dataclasses import dataclass

import httpx
from langchain_core.embeddings import Embeddings

from app.config import SECURE_AI_BASE_URL, SECURE_AI_TIMEOUT_SECONDS
from app.services.end_user_assertion import END_USER_ASSERTION_HEADER

NORMALIZATION_VERSION = "none-v1"

# Gateway refusals whose status rag_api passes on; anything else becomes 502.
_RELAYED_STATUSES = (400, 401, 403, 404, 409, 429)
# Shape of a gateway error.code / error.type that is safe to pass on.
_SAFE_ERROR_TOKEN = re.compile(r"[a-z0-9_]{1,64}")


def _embeddings_url() -> str:
    if SECURE_AI_BASE_URL.endswith("/v1"):
        return f"{SECURE_AI_BASE_URL}/embeddings"
    return f"{SECURE_AI_BASE_URL}/v1/embeddings"


class SecureAIEmbeddingError(RuntimeError):
    """A failed gateway embedding call.

    The message is always rag_api's own fixed text. ``code`` and ``error_type``
    are the gateway's ``error.code`` / ``error.type`` when its refusal is relayed
    with its own status and they are plain identifiers; otherwise None.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        code: str | None = None,
        error_type: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.error_type = error_type


def _safe_token(value: object) -> str | None:
    if isinstance(value, str) and _SAFE_ERROR_TOKEN.fullmatch(value):
        return value
    return None


def _gateway_error_fields(response: httpx.Response) -> tuple[str | None, str | None]:
    """The gateway's ``error.code`` and ``error.type``, never its message."""
    try:
        body = response.json()
    except ValueError:
        return None, None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None, None
    return _safe_token(error.get("code")), _safe_token(error.get("type"))


@dataclass(frozen=True)
class EmbeddingModel:
    provider: str
    model: str

    @property
    def qualified_name(self) -> str:
        return f"{self.provider}/{self.model}"


def parse_embedding_model(value: str) -> EmbeddingModel:
    normalized = value.strip()
    provider, separator, model = normalized.partition("/")
    if not separator or not provider.strip() or not model.strip():
        raise ValueError("embedding_model must be a non-empty provider/model value")
    return EmbeddingModel(provider=provider.strip().lower(), model=model.strip())


class SecureAIEmbeddings(Embeddings):
    def __init__(
        self,
        gateway_key: str,
        embedding_model: str,
        end_user_assertion: str | None = None,
    ):
        if not gateway_key.strip():
            raise ValueError("X-Polygraf-Gateway-Key is required")
        self._gateway_key = gateway_key
        # Passed explicitly by the route that received it, never read from
        # request-global state: run_in_executor does not carry contextvars into
        # the worker thread. Sent only on the gateway /v1/embeddings call below.
        self._end_user_assertion = end_user_assertion or None
        self.model = parse_embedding_model(embedding_model)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        headers = {"Authorization": f"Bearer {self._gateway_key}"}
        if self._end_user_assertion:
            headers[END_USER_ASSERTION_HEADER] = self._end_user_assertion

        try:
            # Redirects stay off (httpx's default): on a cross-origin redirect
            # httpx drops Authorization but would forward the assertion.
            response = httpx.post(
                _embeddings_url(),
                headers=headers,
                json={"model": self.model.qualified_name, "input": texts},
                timeout=SECURE_AI_TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException as exc:
            raise SecureAIEmbeddingError(
                "Secure AI embedding request timed out", 504
            ) from exc
        except httpx.HTTPError as exc:
            raise SecureAIEmbeddingError("Secure AI embedding request failed") from exc

        if not response.is_success:
            relayed = response.status_code in _RELAYED_STATUSES
            code, error_type = (
                _gateway_error_fields(response) if relayed else (None, None)
            )
            raise SecureAIEmbeddingError(
                f"Secure AI rejected the embedding request with status {response.status_code}",
                response.status_code if relayed else 502,
                code=code,
                error_type=error_type,
            )

        try:
            data = response.json()["data"]
            ordered = sorted(data, key=lambda item: item["index"])
            embeddings = [item["embedding"] for item in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise SecureAIEmbeddingError(
                "Secure AI returned an invalid embedding response"
            ) from exc

        if len(embeddings) != len(texts) or not embeddings:
            raise SecureAIEmbeddingError(
                "Secure AI returned an incomplete embedding response"
            )
        dimensions = len(embeddings[0])
        if dimensions == 0 or any(len(vector) != dimensions for vector in embeddings):
            raise SecureAIEmbeddingError(
                "Secure AI returned inconsistent embedding dimensions"
            )
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def fingerprint(self, dimensions: int) -> dict[str, str | int]:
        return {
            "provider": self.model.provider,
            "model": self.model.model,
            "dimensions": dimensions,
            "normalization_version": NORMALIZATION_VERSION,
        }
