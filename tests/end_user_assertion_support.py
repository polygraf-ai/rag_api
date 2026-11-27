"""Shared fixtures for the end-user assertion route and contract tests.

These tests keep the real SecureAIEmbeddings gateway call, so they cannot use
tests/test_main.py's autouse fixture (it replaces embed_query/embed_documents).
The vector store is stubbed the same way, and records what would have been
written or rolled back. Import the fixtures by name into a test module.
"""

import datetime
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.routes import document_routes
from app.services.vector_store.async_pg_vector import AsyncPgVector
from main import app

HEADER = "X-Polygraf-End-User-Assertion"
GATEWAY_KEY = "pgk_live_fake_test"
MODEL = "gemini/text-embedding-004"
JWT_SECRET = "testsecret"

EMBEDDING_ROUTES = ["/query", "/query_multiple", "/embed", "/embed-upload", "/local/embed"]
UPLOAD_ROUTES = ["/embed", "/embed-upload"]

client = TestClient(app)


@pytest.fixture
def store(monkeypatch):
    recorded = SimpleNamespace(added=[], deleted_by_metadata=[])

    async def get_documents_by_ids(self, ids, owners=None, executor=None):
        return [
            Document(
                page_content="Test content",
                metadata={
                    "file_id": file_id,
                    "user_id": "testuser",
                    "provider": "gemini",
                    "model": "text-embedding-004",
                    "dimensions": 3,
                    "normalization_version": "none-v1",
                },
            )
            for file_id in ids
        ]

    async def asimilarity_search_with_score_by_vector(
        self, embedding, k, filter=None, executor=None
    ):
        doc = Document(
            page_content="Queried content",
            metadata={"file_id": "testid1", "user_id": "testuser"},
        )
        return [(doc, 0.1)]

    async def aadd_documents_with_embeddings(
        self, docs, embeddings, ids=None, executor=None
    ):
        recorded.added.extend(dict(doc.metadata) for doc in docs)
        return ids

    async def delete_by_metadata(self, metadata_filter, executor=None):
        recorded.deleted_by_metadata.append(dict(metadata_filter))

    async def get_all_ids(self, owners=None, executor=None):
        return ["testid1"]

    async def get_filtered_ids(self, ids, owners=None, executor=None):
        return [file_id for file_id in ids if file_id == "testid1"]

    async def delete_scoped(self, ids=None, owners=None, executor=None):
        return None

    async def healthy():
        return True

    for name, fn in {
        "get_documents_by_ids": get_documents_by_ids,
        "asimilarity_search_with_score_by_vector": asimilarity_search_with_score_by_vector,
        "aadd_documents_with_embeddings": aadd_documents_with_embeddings,
        "delete_by_metadata": delete_by_metadata,
        "get_all_ids": get_all_ids,
        "get_filtered_ids": get_filtered_ids,
        "delete_scoped": delete_scoped,
    }.items():
        monkeypatch.setattr(AsyncPgVector, name, fn)
    monkeypatch.setattr(document_routes, "is_health_ok", healthy)

    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="test-worker"
        )
    return recorded


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(document_routes, "RAG_UPLOAD_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def auth(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    token = jwt.encode(
        {
            "id": "testuser",
            "exp": datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(hours=1),
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    return [("Authorization", f"Bearer {token}"), ("X-Polygraf-Gateway-Key", GATEWAY_KEY)]


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


def call(route, auth, upload_dir, extra_headers=(), content=b"hello world"):
    """Call one of the five embedding routes the way Secure LLM would."""
    headers = [*auth, *extra_headers]
    if route == "/query":
        return client.post(
            route,
            json={"query": "q", "file_id": "testid1", "embedding_model": MODEL},
            headers=headers,
        )
    if route == "/query_multiple":
        return client.post(
            route,
            json={"query": "q", "file_ids": ["testid1"], "embedding_model": MODEL},
            headers=headers,
        )
    if route in UPLOAD_ROUTES:
        field = "file" if route == "/embed" else "uploaded_file"
        return client.post(
            route,
            files={field: ("doc.txt", content, "text/plain")},
            data={"file_id": "testid1", "embedding_model": MODEL},
            headers=headers,
        )
    if route == "/local/embed":
        (upload_dir / "local.txt").write_bytes(content)
        return client.post(
            route,
            json={
                "filepath": "local.txt",
                "filename": "local.txt",
                "file_content_type": "text/plain",
                "file_id": "testid1",
                "embedding_model": MODEL,
            },
            headers=headers,
        )
    raise AssertionError(route)


def assert_secret_absent(response, logs, *secrets):
    for secret in secrets:
        assert secret not in response.text
        assert secret not in json.dumps(dict(response.headers))
        assert secret not in logs.text


def files_left(upload_dir):
    """Upload temp files left behind (the /local/embed input is not one)."""
    return [p for p in upload_dir.rglob("*") if p.is_file() and p.name != "local.txt"]
