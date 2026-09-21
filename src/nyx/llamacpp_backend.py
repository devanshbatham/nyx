"""llama.cpp bridge for the nyx CPU, Metal, CUDA, and HIP backend."""

from __future__ import annotations

from array import array
import asyncio
import hashlib
import math
import os
import struct
import time

import httpx

from .bounded_inference import bounded_map
from .inference_cache import InferenceCache
from .shared_tokens import SharedTokens, common_prefix_length


class LlamaCppBackend:
    def __init__(self, labels: list[str], label_ids: list[int]):
        capacity = int(os.getenv("NYX_LLAMACPP_CONCURRENCY", "1"))
        if capacity != 1:
            raise ValueError("NYX_LLAMACPP_CONCURRENCY must be 1 for the qualified profile")
        base_url = os.getenv("NYX_LLAMACPP_URL", "http://127.0.0.1:30000")
        self.client = httpx.Client(base_url=base_url, timeout=120)
        self.client.get("/health").raise_for_status()
        props = self.client.get("/props")
        props.raise_for_status()
        self.model_info = props.json()
        self.async_client = httpx.AsyncClient(
            base_url=base_url,
            timeout=120,
            limits=httpx.Limits(max_connections=capacity, max_keepalive_connections=capacity),
        )
        self.labels = tuple(labels)
        self.label_ids = tuple(label_ids)
        self.slots = asyncio.Semaphore(capacity)
        self.request_fanout = capacity
        self.inference_cache = InferenceCache()

    @staticmethod
    def cache_key(prompt, ids):
        if isinstance(prompt, SharedTokens):
            digest = prompt.prefix.digest(len(prompt))
            digest.update(array("I", prompt.suffix).tobytes())
            digest.update(array("I", ids).tobytes())
            return digest.digest()
        return hashlib.sha256(
            struct.pack("!I", len(prompt))
            + array("I", prompt).tobytes()
            + array("I", ids).tobytes()
        ).digest()

    def grammar(self, size: int) -> str:
        choices = " | ".join(f'"{label}"' for label in self.labels[:size])
        return f"root ::= {choices}"

    async def _generate(self, prompt, ids):
        size = len(ids)
        if tuple(ids) != self.label_ids[:size]:
            raise ValueError("llama.cpp candidate IDs must be the canonical label prefix")
        payload = {
            "prompt": list(prompt),
            "n_predict": 1,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "grammar": self.grammar(size),
            "n_probs": size,
            "min_keep": size,
            "post_sampling_probs": True,
            "return_tokens": True,
            "cache_prompt": False,
            "samplers": ["temperature"],
        }
        async with self.slots:
            response = await self.async_client.post("/completion", json=payload)
        response.raise_for_status()
        result = response.json()
        rows = result.get("completion_probabilities") or []
        if len(rows) != 1:
            raise ValueError("Expected exactly one llama.cpp readout token")
        candidates = rows[0].get("top_probs") or []
        probabilities = {item["id"]: float(item["prob"]) for item in candidates}
        if set(ids) - probabilities.keys():
            raise ValueError("llama.cpp did not return every requested candidate")
        logits = [math.log(max(probabilities[token], 1e-300)) for token in ids]
        meta = {
            "cached_tokens": result.get("tokens_cached"),
            "prompt_tokens": result.get("tokens_evaluated", len(prompt)),
        }
        return logits, meta

    async def async_generate(self, prompt, ids):
        digest = self.cache_key(prompt, ids)
        result, status = await self.inference_cache.get(
            digest, lambda: self._generate(prompt, ids)
        )
        values, meta = result
        return values, {**meta, "question_cache": status}

    async def async_score(self, prompts, label_ids):
        started = time.perf_counter()
        outputs = await bounded_map(
            self.async_generate,
            list(zip(prompts, label_ids)),
            self.request_fanout,
        )
        diagnostics = {
            "common_prefix_tokens": common_prefix_length(prompts),
            "warm_prefill_ms": 0.0,
            "branch_ms": (time.perf_counter() - started) * 1000,
            "extra_input_tokens": 0,
            "extra_output_tokens": 0,
            "cached_tokens": [item[1].get("cached_tokens") for item in outputs],
        }
        return [item[0] for item in outputs], diagnostics

    async def aclose(self):
        await self.async_client.aclose()
        self.client.close()
