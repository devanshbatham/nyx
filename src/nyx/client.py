"""Typed synchronous and asynchronous clients for the nyx API."""

from __future__ import annotations

import os
from typing import Any, Literal, NotRequired, TypedDict

import httpx


class ChoiceQuestion(TypedDict):
    type: Literal["choice"]
    instructions: NotRequired[Any]
    criteria: dict[str, Any]


class ScoreQuestion(TypedDict):
    type: Literal["score"]
    instructions: NotRequired[Any]
    criteria: list[Any]


class NoulQuestion(TypedDict):
    type: Literal["noul"]
    instructions: NotRequired[Any]
    criteria: NotRequired[dict[str, Any] | None]


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


def _settings(api_key: str | None, base_url: str | None) -> tuple[str, str]:
    key = api_key or os.getenv("NYX_API_KEY")
    if not key:
        raise ValueError("Set NYX_API_KEY or pass api_key")
    url = (base_url or os.getenv("NYX_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
    return key, url


class Client:
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, transport: httpx.BaseTransport | None = None):
        key, url = _settings(api_key, base_url)
        self._client = httpx.Client(
            base_url=url, headers={"Authorization": f"Bearer {key}", "User-Agent": "nyx-python/1.0"},
            timeout=timeout, transport=transport, follow_redirects=False,
        )

    def systemone(self, *, state: Any, questions: dict[str, Question], model: str = "nyx") -> dict[str, Any]:
        response = self._client.post("/v1/systemone", json={"model": model, "state": state, "questions": questions})
        response.raise_for_status()
        return response.json()

    def models(self) -> dict[str, Any]:
        response = self._client.get("/v1/models")
        response.raise_for_status()
        return response.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AsyncClient:
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, transport: httpx.AsyncBaseTransport | None = None):
        key, url = _settings(api_key, base_url)
        self._client = httpx.AsyncClient(
            base_url=url, headers={"Authorization": f"Bearer {key}", "User-Agent": "nyx-python/1.0"},
            timeout=timeout, transport=transport, follow_redirects=False,
        )

    async def systemone(self, *, state: Any, questions: dict[str, Question], model: str = "nyx") -> dict[str, Any]:
        response = await self._client.post("/v1/systemone", json={"model": model, "state": state, "questions": questions})
        response.raise_for_status()
        return response.json()

    async def models(self) -> dict[str, Any]:
        response = await self._client.get("/v1/models")
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()
