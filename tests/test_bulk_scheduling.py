import asyncio
import types
from nyx.sglang_backend import SGLangBackend
from nyx.inference_cache import InferenceCache


def test_bulk_queue_does_not_block_small_or_fully_cached_requests():
    async def run():
        backend=SGLangBackend.__new__(SGLangBackend)
        backend.bulk_slots=asyncio.Semaphore(1)
        backend.inference_cache=InferenceCache()
        entered=asyncio.Event();release=asyncio.Event()
        async def score(self,prompts,labels,fully_cached):
            if len(prompts)>1 and not fully_cached:
                entered.set();await release.wait()
            return fully_cached,{}
        backend._async_score=types.MethodType(score,backend)
        prompts=[[i]*5000 for i in range(4)];labels=[[10,11]]*4
        first=asyncio.create_task(backend.async_score(prompts,labels))
        await entered.wait()
        waiting=asyncio.create_task(backend.async_score(prompts,labels))
        assert await backend.async_score([[1]],[[10,11]])==(False,{})
        async def result():return ([.2,.8],{})
        for p,ids in zip(prompts,labels):await backend.inference_cache.get(backend.cache_key(p,ids),result)
        values,diag=await backend.async_score(prompts,labels)
        assert values==[[.2,.8]]*4 and diag['extra_input_tokens']==0
        waiting.cancel();await asyncio.gather(waiting,return_exceptions=True)
        release.set();await first
        assert not backend.bulk_slots.locked()
    asyncio.run(run())


def test_fully_cached_batch_never_warms_gpu():
    async def run():
        backend=SGLangBackend.__new__(SGLangBackend)
        backend.bulk_slots=asyncio.Semaphore(1);backend.inference_cache=InferenceCache()
        backend.warm_mode='auto';backend.recent_prefixes={};backend.warm_cache_seconds=0
        backend.request_fanout=4
        async def forbidden(*args):raise AssertionError('Cached request must not call GPU')
        backend._async_generate_uncached=forbidden
        prompts=[[1]*5000+[i] for i in range(4)];labels=[[10,11]]*4
        async def result():return ([.2,.8],{'cached_tokens':5000})
        for p,ids in zip(prompts,labels):await backend.inference_cache.get(backend.cache_key(p,ids),result)
        values,diag=await backend.async_score(prompts,labels)
        assert values==[[.2,.8]]*4 and diag['extra_input_tokens']==0
    asyncio.run(run())
