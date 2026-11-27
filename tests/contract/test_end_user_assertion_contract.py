"""Contract test: rag_api in front of a stub gateway, over real HTTP.

A real HTTP server on 127.0.0.1 stands in for the gateway's /v1/embeddings and
answers the way the gateway does (secure-ai-provider ``end_user.py``,
``dashboard_resolver.py``):

* assertion ``assertion-good``: 200 with embeddings;
* any other assertion: 401 ``authentication_error`` / ``end_user_assertion_invalid``;
* no assertion: 403 ``permission_error`` / ``end_user_assertion_required`` in
  "required" mode, 200 in "optional" mode.

rag_api runs unmodified apart from the vector-store stubs and the gateway URL,
with the real httpx client. This proves rag_api's half of the chain: it passes
the header through unchanged and turns the gateway's 401 into a 401 whose body
contains ``end_user_assertion_invalid``, which is exactly what Secure LLM's
renew-and-resend matches (secure-llm ``assertion.ts`` / ``assertion.spec.ts``
"against a RAG API that relays the gateway refusal").
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.end_user_assertion_support import (  # noqa: F401 (fixtures)
    GATEWAY_KEY,
    HEADER,
    assert_secret_absent,
    auth,
    call,
    files_left,
    logs,
    store,
    upload_dir,
)

GOOD = "assertion-good"
STALE = "assertion-stale"


class GatewayState:
    def __init__(self):
        self.mode = "optional"
        self.requests = []


def _error(message, error_type, code):
    return {"error": {"message": message, "type": error_type, "param": None, "code": code}}


def _handler_for(state: GatewayState):
    class GatewayHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            assertions = self.headers.get_all(HEADER) or []
            state.requests.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "assertions": assertions,
                }
            )
            if not assertions:
                if state.mode == "required":
                    return self._send(
                        403,
                        _error(
                            "This agent requires a signed-in user",
                            "permission_error",
                            "end_user_assertion_required",
                        ),
                    )
                return self._send(200, self._embeddings(body))
            if assertions == [GOOD]:
                return self._send(200, self._embeddings(body))
            return self._send(
                401,
                _error(
                    "End-user identity could not be verified",
                    "authentication_error",
                    "end_user_assertion_invalid",
                ),
            )

        @staticmethod
        def _embeddings(body):
            return {
                "data": [
                    {"index": i, "embedding": [0.1, 0.2, 0.3]}
                    for i in range(len(body["input"]))
                ]
            }

        def _send(self, status_code, payload):
            data = json.dumps(payload).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass  # keep the stub out of the captured logs

    return GatewayHandler


@pytest.fixture
def gateway(monkeypatch):
    state = GatewayState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # The module imported the constant, so patching app.config is not enough.
    monkeypatch.setattr(
        "app.services.secure_ai_embeddings.SECURE_AI_BASE_URL",
        f"http://127.0.0.1:{server.server_address[1]}",
    )
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.lower(), raising=False)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def test_good_assertion_passes_through_unchanged(auth, upload_dir, gateway, store, logs):
    response = call("/query", auth, upload_dir, [(HEADER, GOOD)])

    assert response.status_code == 200, response.text
    assert gateway.requests == [
        {
            "path": "/v1/embeddings",
            "authorization": f"Bearer {GATEWAY_KEY}",
            "assertions": [GOOD],
        }
    ]
    assert_secret_absent(response, logs, GOOD)


def test_stale_assertion_on_query_is_a_401_secure_llm_renews_on(
    auth, upload_dir, gateway, store, logs
):
    response = call("/query", auth, upload_dir, [(HEADER, STALE)])

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "end_user_assertion_invalid"
    # Secure LLM's check: status 401 and the body contains this text.
    assert "end_user_assertion_invalid" in response.text
    assert "End-user identity could not be verified" not in response.text
    assert len(gateway.requests) == 1
    assert_secret_absent(response, logs, STALE)


def test_stale_assertion_on_embed_stores_nothing(auth, upload_dir, gateway, store, logs):
    response = call("/embed", auth, upload_dir, [(HEADER, STALE)])

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "end_user_assertion_invalid"
    assert store.added == []
    assert files_left(upload_dir) == []
    assert_secret_absent(response, logs, STALE)


@pytest.mark.parametrize("route", ["/query", "/embed"])
def test_required_agent_without_assertion_is_a_403(
    route, auth, upload_dir, gateway, store, logs
):
    gateway.mode = "required"

    response = call(route, auth, upload_dir)

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "end_user_assertion_required"
    assert response.json()["detail"]["type"] == "permission_error"


def test_optional_agent_without_assertion_sends_no_header(
    auth, upload_dir, gateway, store
):
    response = call("/query", auth, upload_dir)

    assert response.status_code == 200, response.text
    assert [r["assertions"] for r in gateway.requests] == [[]]


def test_duplicate_header_never_reaches_the_gateway(
    auth, upload_dir, gateway, store, logs
):
    response = call("/query", auth, upload_dir, [(HEADER, GOOD), (HEADER, STALE)])

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_end_user_assertion"
    assert gateway.requests == []
    assert_secret_absent(response, logs, GOOD, STALE)
