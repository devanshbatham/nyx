import asyncio
import hashlib
import pytest
from nyx.inference_cache import InferenceCache, inference_for_waiters, AbandonedInference
from nyx.hosted_api import AccessGuard

KEY = 'test-key'
HASH = hashlib.sha256(KEY.encode()).hexdigest()


def test_exact_cache_coalesces_and_reuses_completed_results():
    async def run():
        cache = InferenceCache()
        calls = 0
        async def compute():
            nonlocal calls
            calls += 1
            await asyncio.sleep(.01)
            return [1, 2]
        results = await asyncio.gather(*(cache.get(b'exact', compute) for _ in range(100)))
        assert calls == 1 and all(v == [1,2] for v, _ in results)
        assert sum(s == 'coalesced' for _, s in results) == 99
        assert await cache.get(b'exact', compute) == ([1,2], 'hit')
        assert calls == 1
        await cache.get(b'different', compute)
        assert calls == 2 and not cache.inflight
    asyncio.run(run())


def test_errors_not_cached_and_byte_entry_expiry_limits():
    async def run():
        cache = InferenceCache(max_entries=1, max_bytes=20, ttl=.01)
        async def fail():
            raise ValueError('failure')
        with pytest.raises(ValueError):
            await cache.get(b'failed', fail)
        assert not cache.entries and not cache.inflight
        async def small():
            return 42
        await cache.get(b'a', small)
        await cache.get(b'b', small)
        assert len(cache.entries) == 1 and b'a' not in cache.entries
        async def big():
            return 'x'*100
        await cache.get(b'big', big)
        assert b'big' not in cache.entries and cache.bytes <= 20
        await asyncio.sleep(.02)
        _, status = await cache.get(b'b', small)
        assert status == 'miss' and cache.bytes <= 20
    asyncio.run(run())


def test_cancelling_one_subscriber_keeps_shared_inference_for_other():
    async def run():
        cache = InferenceCache()
        started = asyncio.Event()
        release = asyncio.Event()
        stopped = asyncio.Event()
        async def compute():
            started.set()
            try:
                await release.wait()
                return 7
            finally:
                stopped.set()
        first = asyncio.create_task(cache.get(b'x', compute))
        second = asyncio.create_task(cache.get(b'x', compute))
        await started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not stopped.is_set()
        release.set()
        assert await second == (7, 'coalesced')
        assert stopped.is_set() and not cache.inflight
    asyncio.run(run())


def test_last_subscriber_cancels_pending_inference():
    async def run():
        cache = InferenceCache()
        started, stopped = asyncio.Event(), asyncio.Event()
        async def compute():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        task = asyncio.create_task(cache.get(b'x', compute))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set() and not cache.inflight and not cache.entries
    asyncio.run(run())


def test_worker_cancels_model_when_response_future_abandoned():
    async def run():
        future = asyncio.get_running_loop().create_future()
        started, stopped = asyncio.Event(), asyncio.Event()
        async def model():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        task = asyncio.create_task(inference_for_waiters(model(), [future]))
        await started.wait()
        future.cancel()
        with pytest.raises(AbandonedInference):
            await task
        assert stopped.is_set()
    asyncio.run(run())


def test_client_disconnect_cancels_application_and_releases_admission():
    async def run():
        entered, disconnected, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        received = False
        async def receive():
            nonlocal received
            if not received:
                received = True
                return {'type': 'http.request', 'body': b'{}', 'more_body': False}
            await disconnected.wait()
            return {'type': 'http.disconnect'}
        async def app(scope, receive, send):
            await receive()
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        async def send(message):
            raise AssertionError('Disconnected response should not be sent')
        guard = AccessGuard(app, HASH, max_inflight=1, max_waiting=10, requests_per_minute=0)
        scope = {'type':'http','method':'POST','path':'/v1/systemone','headers':[(b'authorization',('Bearer '+KEY).encode())]}
        task = asyncio.create_task(guard(scope, receive, send))
        await entered.wait()
        disconnected.set()
        await task
        assert cancelled.is_set() and guard.inflight == 0 and not guard.slots.locked()
        assert guard.metrics.statuses['499'] == 1
    asyncio.run(run())
