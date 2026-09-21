"""Thread-safe tokenization reuse, bounded by entries and charged Python memory."""
from collections import OrderedDict, Counter
from threading import Lock
import sys


class TokenizationCache:
    def __init__(self, encode, max_entries=256, max_bytes=16*1024*1024, max_tokens=32768):
        self.encode = encode
        self.max_entries, self.max_bytes, self.max_tokens = max_entries, max_bytes, max_tokens
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = Lock()
        self.counts = Counter()

    def __call__(self, text):
        with self.lock:
            entry = self.entries.get(text)
            if entry is not None:
                self.entries.move_to_end(text)
                self.counts['hits'] += 1
                return entry[0]
            self.counts['misses'] += 1
        # Never hold the cache lock while Rust tokenization runs.
        tokens = tuple(self.encode(text))
        size = sys.getsizeof(text) + sys.getsizeof(tokens) + len(tokens)*sys.getsizeof(0) + 256
        if len(tokens) > self.max_tokens or size > self.max_bytes or not self.max_entries:
            return tokens
        with self.lock:
            existing = self.entries.get(text)
            if existing is not None:
                return existing[0]
            while self.entries and (len(self.entries) >= self.max_entries or self.bytes+size > self.max_bytes):
                _, (_, previous) = self.entries.popitem(last=False)
                self.bytes -= previous
                self.counts['evictions'] += 1
            self.entries[text] = (tokens, size)
            self.bytes += size
        return tokens

    def snapshot(self):
        with self.lock:
            return {'entries': len(self.entries), 'charged_bytes': self.bytes,
                    'max_bytes': self.max_bytes, 'max_entries': self.max_entries,
                    'counts': dict(self.counts)}
