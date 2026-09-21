import hashlib

import httpx
import pytest

from nyx.hosted_api import create_app


@pytest.mark.asyncio
async def test_gateway_fails_closed_without_valid_bearer():
    key = "correct-secret"
    app = create_app(hashlib.sha256(key.encode()).hexdigest())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.get("/v1/models")
        assert denied.status_code == 401
        assert denied.headers["cache-control"] == "no-store"
        allowed = await client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
        assert allowed.status_code == 200
        assert allowed.json()["models"][0]["name"] == "nyx"


def test_gateway_rejects_invalid_key_hash():
    with pytest.raises(ValueError, match="SHA256"):
        # Middleware validation is deferred until the ASGI stack is built.
        app = create_app("invalid")
        app.build_middleware_stack()
