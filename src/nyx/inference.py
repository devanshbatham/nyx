"""Bounded request workers for the Nyx SGLang gateway."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import time

from .contract import MODEL_ID, Request, answer
from .engine import Engine
from .inference_cache import inference_for_waiters


@dataclass
class Job:
    req: Request
    prepared: tuple
    future: asyncio.Future


async def worker(app):
    while True:
        job = await app.state.queue.get()
        if job.future.cancelled():
            continue
        started = time.perf_counter()
        try:
            probabilities, lengths, diagnostics = await inference_for_waiters(
                app.state.engine.score_async(job.prepared), [job.future]
            )
            calibrate = app.state.engine.calibrate_probs
            result = {
                "model": MODEL_ID,
                "answers": {
                    question_id: answer(question, calibrate(row, question.type))
                    for (question_id, question), row in zip(job.req.questions.items(), probabilities)
                },
                "usage": {
                    "input_tokens": getattr(job.prepared, "context_tokens", sum(lengths) + diagnostics.get("extra_input_tokens", 0)),
                    "output_tokens": len(job.req.questions) + diagnostics.get("extra_output_tokens", 0),
                },
            }
            headers = {
                "Server-Timing": (
                    f"inference;dur={(time.perf_counter() - started) * 1000:.3f}, "
                    f"warm;dur={diagnostics.get('warm_prefill_ms', 0):.3f}, "
                    f"branches;dur={diagnostics.get('branch_ms', 0):.3f}"
                ),
                "x-local-batch-questions": str(len(job.req.questions)),
            }
            cached = diagnostics.get("cached_tokens", [])
            if cached and all(value is not None for value in cached):
                headers["x-local-cached-tokens"] = str(sum(cached))
            alignment = getattr(job.prepared, "alignment", None)
            if alignment is not None:
                headers["x-local-prefix-alignment"] = json.dumps(alignment, separators=(",", ":"))
            if not job.future.done():
                job.future.set_result((result, headers))
        except Exception as error:
            if not job.future.done():
                job.future.set_exception(error)
