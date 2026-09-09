from dataclasses import dataclass

import httpx
from langchain_core.embeddings import Embeddings

from app.config import SECURE_AI_BASE_URL, SECURE_AI_TIMEOUT_SECONDS

NORMALIZATION_VERSION = "none-v1"


def _embeddings_url() -> str:
    if SECURE_AI_BASE_URL.endswith("/v1"):
        return f"{SECURE_AI_BASE_URL}/embeddings"
    return f"{SECURE_AI_BASE_URL}/v1/embeddings"


class SecureAIEmbeddingError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


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
    def __init__(self, gateway_key: str, embedding_model: str):
        if not gateway_key.strip():
            raise ValueError("X-Polygraf-Gateway-Key is required")
        self._gateway_key = gateway_key
        self.model = parse_embedding_model(embedding_model)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = httpx.post(
                _embeddings_url(),
                headers={"Authorization": f"Bearer {self._gateway_key}"},
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
            status_code = (
                response.status_code
                if response.status_code in (400, 401, 403, 404, 409, 429)
                else 502
            )
            raise SecureAIEmbeddingError(
                f"Secure AI rejected the embedding request with status {response.status_code}",
                status_code,
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
