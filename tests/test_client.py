import httpx
import pytest

from nyx import AsyncNyxClient, NyxClient


def response(request: httpx.Request) -> httpx.Response:
    assert request.headers["authorization"] == "Bearer secret"
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"models": [{"name": "nyx"}]})
    assert request.url.path == "/v1/systemone"
    assert request.read()
    return httpx.Response(200, json={"model": "nyx", "answers": {}, "usage": {}})


def test_sync_client():
    with NyxClient(api_key="secret", base_url="https://nyx.invalid", transport=httpx.MockTransport(response)) as client:
        assert client.models()["models"][0]["name"] == "nyx"
        assert client.systemone(state="x", questions={})["model"] == "nyx"


@pytest.mark.asyncio
async def test_async_client():
    async with AsyncNyxClient(
        api_key="secret", base_url="https://nyx.invalid", transport=httpx.MockTransport(response)
    ) as client:
        assert (await client.models())["models"][0]["name"] == "nyx"


def test_client_requires_key(monkeypatch):
    monkeypatch.delenv("NYX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NYX_API_KEY"):
        NyxClient()
