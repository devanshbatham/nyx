"""Bounded exact-input cache with shared work and subscriber-aware cancellation."""
import asyncio
from collections import OrderedDict, Counter
import json
import time


class InferenceCache:
    def __init__(self, max_entries=8192, max_bytes=64*1024*1024, ttl=300):
        self.max_entries, self.max_bytes, self.ttl = max_entries, max_bytes, ttl
        self.entries = OrderedDict()
        self.inflight = {}
        self.bytes = 0
        self.counts = Counter()

    def contains(self, key):
        entry = self.entries.get(key)
        return entry is not None and entry[0] > time.monotonic()

    def _expire(self, now):
        while self.entries:
            oldest = next(iter(self.entries))
            if self.entries[oldest][0] > now:
                break
            _, _, size = self.entries.pop(oldest)
            self.bytes -= size

    def get_completed(self, keys):
        """Synchronous snapshot, or None if any key needs asynchronous work.

        Owned by one event loop. Partial lookups neither count hits nor subscribe
        to in-flight work; the existing get() path handles misses and cancellation.
        Keys may be lazy so a cold batch stops hashing at its first miss.
        """
        now = time.monotonic()
        self._expire(now)
        values = []
        for key in keys:
            entry = self.entries.get(key)
            if entry is None or entry[0] <= now:
                return None
            values.append(entry[1])
        if values:
            self.counts['hits'] += len(values)
        return values

    async def get(self, key, factory):
        now = time.monotonic()
        self._expire(now)
        entry = self.entries.get(key)
        if entry is not None:
            self.counts['hits'] += 1
            return entry[1], 'hit'
        shared = self.inflight.get(key)
        status = 'coalesced' if shared else 'miss'
        self.counts[status] += 1
        if shared is None:
            async def compute():
                value = await factory()
                size = len(key) + len(json.dumps(value).encode())
                if size <= self.max_bytes and self.max_entries:
                    while self.entries and (len(self.entries) >= self.max_entries or self.bytes+size > self.max_bytes):
                        _, (_, _, old_size) = self.entries.popitem(last=False)
                        self.bytes -= old_size
                        self.counts['evictions'] += 1
                    self.entries[key] = (time.monotonic()+self.ttl, value, size)
                    self.bytes += size
                return value
            shared = {'task': asyncio.create_task(compute()), 'waiters': 0}
            self.inflight[key] = shared
        shared['waiters'] += 1
        try:
            return await asyncio.shield(shared['task']), status
        finally:
            shared['waiters'] -= 1
            if shared['waiters'] == 0:
                if self.inflight.get(key) is shared:
                    del self.inflight[key]
                if not shared['task'].done():
                    self.counts['abandoned'] += 1
                    shared['task'].cancel()
                await asyncio.gather(shared['task'], return_exceptions=True)

    def snapshot(self):
        return {'counts': dict(self.counts), 'entries': len(self.entries), 'bytes': self.bytes,
                'inflight_keys': len(self.inflight), 'ttl_seconds': self.ttl,
                'max_entries': self.max_entries, 'max_bytes': self.max_bytes}


class AbandonedInference(Exception):
    pass


async def inference_for_waiters(awaitable, futures):
    task = asyncio.create_task(awaitable)
    def check(_):
        if all(future.cancelled() for future in futures):
            task.cancel()
    for future in futures:
        future.add_done_callback(check)
    check(None)
    try:
        return await task
    except asyncio.CancelledError:
        if not asyncio.current_task().cancelling() and all(future.cancelled() for future in futures):
            raise AbandonedInference() from None
        raise
    finally:
        for future in futures:
            future.remove_done_callback(check)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
