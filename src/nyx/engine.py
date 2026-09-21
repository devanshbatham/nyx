"""Nyx prompt compiler and SGLang selected-logit inference engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
from transformers import AutoTokenizer

from .contract import LABELS, candidates, messages


class Engine:
    def __init__(self, max_length: int = 32768):
        model_path = os.getenv("NYX_MODEL_PATH")
        if not model_path:
            raise ValueError("NYX_MODEL_PATH must point to a downloaded devanshbatham/nyx repository")
        self.path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.ids = []
        for label in LABELS:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Candidate label is not one token: {label}")
            self.ids.append(ids[0])

        from .compiler import PromptCompiler
        from .sglang_backend import SGLangBackend

        self.compiler = PromptCompiler(self.tokenizer, max_length)
        self.max_length = max_length
        self.backend = "sglang"
        self.temperature = 1.0
        self.temperatures = {}
        calibration = os.getenv("NYX_CALIBRATION")
        if calibration:
            config = json.loads(Path(calibration).read_text())
            self.temperature = config.get("temperature", 1.0)
            self.temperatures = config.get("temperatures", {})
            if not all(np.isfinite(value) and value > 0 for value in [self.temperature, *self.temperatures.values()]):
                raise ValueError("Invalid calibration temperature")

        self.model = SGLangBackend()
        expected = os.getenv("NYX_BACKEND_MODEL_PATH", "/model")
        actual = self.model.model_info.get("model_path")
        if actual != expected:
            raise ValueError(f"Backend model path mismatch: expected {expected!r}, received {actual!r}")

    def encode(self, state, question):
        if os.getenv("NYX_PREFIX_TOKENIZATION", "1") == "1":
            return self.compiler.encode(state, question)
        ids = self.tokenizer.apply_chat_template(
            messages(state, question), tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        if len(ids) > self.max_length - 1:
            raise ValueError("Prompt exceeds context limit; no truncation performed")
        return ids

    def prepare(self, items):
        if os.getenv("NYX_CONTEXT_ACCOUNTING") == "shared":
            return self.prepare_shared(items)
        align_enabled = os.getenv("NYX_PREFIX_ALIGNMENT", "0") == "1"
        started = time.perf_counter() if align_enabled else None
        prompts = []
        total_tokens = 0
        for state, question in items:
            prompt = self.encode(state, question)
            total_tokens += len(prompt)
            if total_tokens > 640000:
                raise ValueError("Total prompt budget exceeds 640000 tokens")
            prompts.append(prompt)
        alignment = None
        if align_enabled:
            prompts, alignment = self.compiler.align_shared(items, prompts)
        lengths = [len(prompt) for prompt in prompts]
        sizes = [len(candidates(question)[0]) for _, question in items]
        if alignment is not None:
            from .prefix_alignment import PreparedPrompts
            alignment["prepare_ms"] = (time.perf_counter() - started) * 1000
            return PreparedPrompts(prompts, lengths, sizes, alignment)
        return prompts, lengths, sizes

    def prepare_shared(self, items):
        from .shared_tokens import ContextBudgetExceeded, PreparedContext
        prompts = []
        sizes = []
        prefix = None
        state = None
        total = 0
        for current, question in items:
            if prefix is None:
                state = current
                prefix = self.compiler.shared_prefix(state)
                total = len(prefix)
            elif current is not state and current != state:
                raise ValueError("Shared context preparation requires one state")
            prompt = self.compiler.encode_shared(prefix, question)
            total += len(prompt.suffix)
            if len(prompt) > 32768 or total > 65536:
                raise ContextBudgetExceeded("Shared context budget exceeded")
            prompts.append(prompt)
            sizes.append(len(candidates(question)[0]))
        return PreparedContext(prompts, [len(prompt) for prompt in prompts], sizes, total)

    async def score_async(self, prepared):
        prompts, lengths, sizes = prepared
        logits, diagnostics = await self.model.async_score(prompts, [self.ids[:size] for size in sizes])
        probabilities = []
        for row in logits:
            values = np.array(row, dtype=float) / self.temperature
            values = np.exp(values - values.max())
            probabilities.append((values / values.sum()).tolist())
        return probabilities, lengths, diagnostics

    def calibrate_probs(self, probabilities, primitive):
        key = "choice_large" if primitive == "choice" and len(probabilities) > 32 and "choice_large" in self.temperatures else primitive
        temperature = self.temperatures.get(key, 1.0)
        if temperature == 1.0:
            return probabilities
        values = np.log(np.clip(probabilities, 1e-300, 1)) / temperature
        values -= values.max()
        values = np.exp(values)
        return (values / values.sum()).tolist()
