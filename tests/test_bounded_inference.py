import asyncio
import pytest
from nyx.bounded_inference import bounded_map


def test_large_batch_does_not_queue_all_questions_ahead_of_small_request():
    async def run():
        slots = asyncio.Semaphore(4)
        release = asyncio.Event()
        entered = asyncio.Event()
        order = []
        async def generate(index):
            async with slots:
                order.append(index)
                if len(order) == 4:
                    entered.set()
                await release.wait()
                await asyncio.sleep(0)
                return index * 2
        large = asyncio.create_task(bounded_map(generate, [(i,) for i in range(1000)], 4))
        await entered.wait()
        small = asyncio.create_task(bounded_map(generate, [(-1,)], 4))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release.set()
        assert await small == [-2]
        assert order.index(-1) <= 8
        assert await large == [i * 2 for i in range(1000)]
    asyncio.run(run())


@pytest.mark.parametrize('fail', [False, True])
def test_cancel_or_failure_cleans_up_all_inflight_work(fail):
    async def run():
        active = 0
        entered = asyncio.Event()
        release = asyncio.Event()
        async def generate(i):
            nonlocal active
            active += 1
            if active == 4:
                entered.set()
            try:
                await release.wait()
                if i == 0 and fail:
                    raise ValueError('backend failure')
                await asyncio.sleep(10)
            finally:
                active -= 1
        task = asyncio.create_task(bounded_map(generate, [(i,) for i in range(1000)], 4))
        await entered.wait()
        if fail:
            release.set()
            with pytest.raises(ValueError, match='backend failure'):
                await task
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert active == 0
    asyncio.run(run())


def test_empty_batch_and_invalid_capacity():
    async def run():
        async def generate(i):
            return i
        assert await bounded_map(generate, [], 4) == []
        with pytest.raises(ValueError):
            await bounded_map(generate, [(1,)], 0)
    asyncio.run(run())
