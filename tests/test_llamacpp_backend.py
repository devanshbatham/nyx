import asyncio
import math

import httpx
import pytest

from nyx.llamacpp_backend import LlamaCppBackend


class FakeAsyncClient:
    def __init__(self, body):
        self.body = body
        self.payload = None

    async def post(self, path, json):
        assert path == "/completion"
        self.payload = json
        return httpx.Response(
            200,
            json=self.body,
            request=httpx.Request("POST", "http://backend/completion"),
        )


def backend(body):
    value = LlamaCppBackend.__new__(LlamaCppBackend)
    value.labels = ("A", "B", "C")
    value.label_ids = (10, 11, 12)
    value.slots = asyncio.Semaphore(1)
    value.async_client = FakeAsyncClient(body)
    return value


@pytest.mark.asyncio
async def test_constrained_candidate_probabilities_are_returned_as_logits():
    model = backend({
        "tokens_evaluated": 3,
        "tokens_cached": 2,
        "completion_probabilities": [{"top_probs": [
            {"id": 11, "prob": 0.7},
            {"id": 10, "prob": 0.2},
            {"id": 12, "prob": 0.1},
        ]}],
    })
    logits, meta = await model._generate([1, 2, 3], [10, 11, 12])
    assert logits == pytest.approx([math.log(0.2), math.log(0.7), math.log(0.1)])
    assert meta == {"cached_tokens": 2, "prompt_tokens": 3}
    payload = model.async_client.payload
    assert payload["prompt"] == [1, 2, 3]
    assert payload["grammar"] == 'root ::= "A" | "B" | "C"'
    assert payload["n_predict"] == 1
    assert payload["n_probs"] == payload["min_keep"] == 3
    assert payload["post_sampling_probs"] is True
    assert payload["samplers"] == ["temperature"]
    assert payload["cache_prompt"] is False


@pytest.mark.asyncio
async def test_missing_or_noncanonical_candidates_fail_closed():
    model = backend({
        "completion_probabilities": [{
            "top_probs": [{"id": 10, "prob": 1.0}],
        }],
    })
    with pytest.raises(ValueError, match="every requested candidate"):
        await model._generate([1], [10, 11])
    with pytest.raises(ValueError, match="canonical label prefix"):
        await model._generate([1], [11, 10])
