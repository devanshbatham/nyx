"""Exact, immutable token sequences that share a long state in host memory."""
from collections.abc import Sequence
from itertools import chain
from array import array
import hashlib
import struct


class ContextBudgetExceeded(ValueError):
    pass


class PreparedContext(tuple):
    def __new__(cls, prompts, lengths, sizes, context_tokens):
        value = super().__new__(cls, (prompts, lengths, sizes))
        value.context_tokens = context_tokens
        return value


class SharedPrefix:
    __slots__ = ('tokens', 'packed', 'hashes')

    def __init__(self, tokens):
        self.tokens = tuple(tokens)
        self.packed = array('I', self.tokens).tobytes()
        self.hashes = {}

    def __len__(self):
        return len(self.tokens)

    def digest(self, prompt_length):
        digest = self.hashes.get(prompt_length)
        if digest is None:
            digest = hashlib.sha256(struct.pack('!I', prompt_length))
            digest.update(self.packed)
            if len(self.hashes) < 64:
                self.hashes[prompt_length] = digest
        return digest.copy()


class SharedTokens(Sequence):
    __slots__ = ('prefix', 'suffix')

    def __init__(self, prefix, suffix):
        self.prefix = prefix if isinstance(prefix, SharedPrefix) else SharedPrefix(prefix)
        self.suffix = tuple(suffix)

    def __len__(self):
        return len(self.prefix) + len(self.suffix)

    def __iter__(self):
        return chain(self.prefix.tokens, self.suffix)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self.prefix.tokens + self.suffix)[index]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.prefix.tokens[index] if index < len(self.prefix) else self.suffix[index-len(self.prefix)]


def common_prefix_length(prompts):
    if len(prompts) < 2:
        return 0
    first = prompts[0]
    if isinstance(first, SharedTokens) and all(
            isinstance(p, SharedTokens) and (p.prefix is first.prefix or p.prefix.tokens == first.prefix.tokens)
            for p in prompts):
        low = min(p.suffix for p in prompts)
        high = max(p.suffix for p in prompts)
        extra = next((i for i, (a, b) in enumerate(zip(low, high)) if a != b), min(len(low), len(high)))
        return len(first.prefix) + extra
    common = 0
    for values in zip(*prompts):
        if len(set(values)) != 1:
            break
        common += 1
    return common
