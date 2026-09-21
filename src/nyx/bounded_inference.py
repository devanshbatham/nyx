"""Bound question fan-out while preserving order and full single-batch capacity."""
import asyncio


async def bounded_map(function, items, capacity):
    if capacity < 1:
        raise ValueError('Inference fan-out must be positive')
    outputs = [None] * len(items)
    pending = iter(enumerate(items))

    async def worker():
        for index, arguments in pending:
            outputs[index] = await function(*arguments)

    tasks = [asyncio.create_task(worker()) for _ in range(min(capacity, len(items)))]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return outputs
