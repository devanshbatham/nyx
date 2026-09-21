import asyncio
import hashlib

import httpx
import pytest

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


class FakeEngine:
    backend = "sglang"
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
    assert response.headers["cache-control"] == "no-store"
