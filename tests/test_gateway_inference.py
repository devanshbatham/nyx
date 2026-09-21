import asyncio
import hashlib
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from nyx import inference
from nyx.contract import candidates
from nyx.hosted_api import create_app


KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeAsyncClient:
    async def get(self, path, timeout):
        assert path == "/health" and timeout == 2
        return httpx.Response(200, request=httpx.Request("GET", "http://backend/health"))

    async def aclose(self):
        pass


class FakeModel:
    async_client = FakeAsyncClient()
    abort_client = None

    class Client:
        def close(self):
            pass

    class Pool:
        def shutdown(self, **kwargs):
            pass

    client = Client()
    pool = Pool()

    async def aclose(self):
        pass


class FakeEngine:
    backend = "llamacpp"
    model = FakeModel()

    def __init__(self, max_length):
        self.max_length = max_length

    def prepare(self, items):
        prompts = [[index] for index, _ in enumerate(items)]
        return prompts, [1] * len(items), [len(candidates(question)[0]) for _, question in items]

    async def score_async(self, prepared):
        _, lengths, sizes = prepared
        rows = [[float(index == size - 1) for index in range(size)] for size in sizes]
        return rows, lengths, {"cached_tokens": [0] * len(rows), "extra_input_tokens": 2, "extra_output_tokens": 1}

    @staticmethod
    def calibrate_probs(probabilities, primitive):
        return probabilities


@pytest.mark.asyncio
async def test_authenticated_end_to_end_decision(monkeypatch):
    monkeypatch.setattr(inference, "Engine", FakeEngine)
    monkeypatch.setenv("NYX_HOSTED_WORKERS", "2")
    app = create_app(hashlib.sha256(KEY.encode()).hexdigest())
    payload = {
        "model": "devanshbatham/nyx",
        "state": {"message": "example"},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "route it",
                "criteria": {"first": None, "second": {"nested": True}},
            },
            "severity": {
                "type": "score",
                "criteria": ["low", {"label": "high"}],
            },
            "accepted": {"type": "noul", "instructions": "is accepted?"},
        },
    }
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/health")).status_code == 200
            response = await client.post("/v1/systemone", headers=AUTH, json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "nyx"
    assert body["answers"]["route"]["choice"] == "second"
    assert body["answers"]["severity"]["score"] == 1
    assert body["answers"]["severity"]["legend"] == {"0": "low", "1": {"label": "high"}}
    assert body["answers"]["accepted"] == {"type": "noul", "noul": 1.0}
    assert body["usage"] == {"input_tokens": 5, "output_tokens": 4}
    assert response.headers["x-nyx-request-id"]
    assert response.headers["x-typesafe-request-id"] == response.headers["x-nyx-request-id"]
    assert response.headers["cache-control"] == "no-store"


def test_official_python_sdk_is_drop_in(monkeypatch):
    import typesafe_sdk

    monkeypatch.setattr(inference, "Engine", FakeEngine)
    monkeypatch.setenv("NYX_HOSTED_WORKERS", "2")
    app = create_app(hashlib.sha256(KEY.encode()).hexdigest())
    socket_handle = socket.socket()
    socket_handle.bind(("127.0.0.1", 0))
    socket_handle.listen(128)
    port = socket_handle.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [socket_handle]}, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(url + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        else:
            raise AssertionError("Local test server did not become ready")

        with typesafe_sdk.TypeSafeClient(api_key=KEY, base_url=url, model="jev-latest", timeout=10) as client:
            result = client.system_one(
                state={"message": "example"},
                questions={
                    "route": typesafe_sdk.Choice(
                        instructions="Route it",
                        criteria={"first": None, "second": "Second route"},
                    ),
                    "severity": typesafe_sdk.Score(
                        instructions="How severe?",
                        criteria=["low", "high"],
                    ),
                    "accepted": typesafe_sdk.Noul(instructions="Is accepted?"),
                },
            )
            models = client.models.list()

        assert result.model == "nyx"
        assert result.choices["route"].choice == "second"
        assert result.scores["severity"].score == 1
        assert result.nouls["accepted"].noul == 1
        assert result.request_id
        assert models.request_id
        assert {model.name for model in models.models} >= {"nyx", "jev-latest", "jev-1.13.0"}
    finally:
        server.should_exit = True
        thread.join(timeout=5)
