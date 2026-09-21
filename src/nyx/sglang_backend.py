"""Independent SGLang bridge; selected-token readout and measured prefix warming."""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import asyncio,os,time
import httpx
from .bounded_inference import bounded_map
from .inference_cache import InferenceCache
from .backend_cancellation import generate_with_abort
from .shared_tokens import common_prefix_length, SharedTokens
from array import array
import hashlib, struct

class SGLangBackend:
    def __init__(self):
        capacity=int(os.getenv('NYX_SGLANG_CONCURRENCY','64'))
        if not 1<=capacity<=1024:raise ValueError('Backend concurrency must be 1..1024')
        base_url=os.getenv('NYX_SGLANG_URL','http://127.0.0.1:30000')
        self.client=httpx.Client(base_url=base_url,timeout=120,limits=httpx.Limits(max_connections=capacity,max_keepalive_connections=capacity))
        self.client.get('/health').raise_for_status()
        model_info=self.client.get('/model_info');model_info.raise_for_status()
        self.model_info=model_info.json()
        self.pool=ThreadPoolExecutor(max_workers=32)
        if os.getenv('NYX_ASYNC_HTTP_TRANSPORT', 'httpx') == 'aiohttp':
            from .async_http import AiohttpClient
            self.async_client = AiohttpClient(base_url, capacity)
        else:
            self.async_client=httpx.AsyncClient(base_url=base_url,timeout=120,limits=httpx.Limits(max_connections=capacity,max_keepalive_connections=capacity))
        self.slots=asyncio.Semaphore(capacity)
        bulk_capacity=int(os.getenv('NYX_SGLANG_BULK_CONCURRENCY','8'))
        if not 1<=bulk_capacity<=64:raise ValueError('Bulk concurrency must be 1..64')
        self.bulk_slots=asyncio.Semaphore(bulk_capacity)
        self.request_fanout=capacity
        self.inference_cache=InferenceCache()
        from .async_http import AiohttpClient
        self.abort_client=AiohttpClient(base_url,8,timeout=2)
        self.warm_mode=os.getenv('NYX_WARM_PREFIX','auto')
        self.last_diagnostics={}
        self.warm_cache_seconds=float(os.getenv('NYX_WARM_PREFIX_CACHE_SECONDS','0'))
        self.recent_prefixes=OrderedDict()

    def generate(self,prompt,ids=None):
        if not isinstance(prompt,list):prompt=list(prompt)
        data={'input_ids':prompt,'sampling_params':{'temperature':1.,'max_new_tokens':1,'top_k':-1,'top_p':1.},'return_logprob':ids is not None}
        if ids is not None:data.update(token_ids_logprob=ids,logprob_start_len=-1,top_logprobs_num=0,return_text_in_logprobs=False)
        r=self.client.post('/generate',json=data);r.raise_for_status();result=r.json();meta=result['meta_info']
        if meta['completion_tokens']!=1:raise ValueError('Expected exactly one readout token')
        if ids is None:return None,meta
        values={x[1]:x[0] for x in meta['output_token_ids_logprobs'][0]}
        return [values[i] for i in ids],meta

    def score(self,prompts,label_ids):
        common=0
        if len(prompts)>1:
            for tup in zip(*prompts):
                if len(set(tup))!=1:break
                common+=1
        warm=self.warm_mode=='always' and common>0 or self.warm_mode=='auto' and len(prompts)>=4 and common>=256
        t=time.perf_counter();extra_tokens=0
        if warm:
            _,meta=self.generate(prompts[0][:common]);extra_tokens=meta['prompt_tokens']
        prefill_ms=(time.perf_counter()-t)*1000;t=time.perf_counter()
        outputs=list(self.pool.map(lambda args:self.generate(*args),zip(prompts,label_ids)))
        self.last_diagnostics={'common_prefix_tokens':common,'warm_prefill_ms':prefill_ms,'branch_ms':(time.perf_counter()-t)*1000,'extra_input_tokens':extra_tokens,'extra_output_tokens':int(warm),'cached_tokens':[x[1].get('cached_tokens') for x in outputs]}
        return [x[0] for x in outputs]

    async def async_generate(self,prompt,ids=None):
        if ids is None:
            return await self._async_generate_uncached(prompt,ids)
        digest=self.cache_key(prompt,ids)
        result,status=await self.inference_cache.get(digest,lambda:self._async_generate_uncached(prompt,ids))
        values,meta=result
        return values,{**meta,'question_cache':status}

    async def _async_generate_uncached(self,prompt,ids=None):
        data={'input_ids':prompt,'sampling_params':{'temperature':1.,'max_new_tokens':1,'top_k':-1,'top_p':1.},'return_logprob':ids is not None}
        if ids is not None:data.update(token_ids_logprob=ids,logprob_start_len=-1,top_logprobs_num=0,return_text_in_logprobs=False)
        async with self.slots:
            # Allocate the complete input only when a GPU dispatch slot is held.
            if not isinstance(prompt,list):data['input_ids']=list(prompt)
            r=await generate_with_abort(self.async_client,self.abort_client,data,self.inference_cache.counts)
        r.raise_for_status();result=r.json();meta=result['meta_info']
        if meta['completion_tokens']!=1:raise ValueError('Expected exactly one readout token')
        if ids is None:return None,meta
        values={x[1]:x[0] for x in meta['output_token_ids_logprobs'][0]}
        return [values[i] for i in ids],meta

    @staticmethod
    def cache_key(prompt,ids):
        if isinstance(prompt,SharedTokens):
            digest=prompt.prefix.digest(len(prompt))
            digest.update(array('I',prompt.suffix).tobytes())
            digest.update(array('I',ids).tobytes())
            return digest.digest()
        return hashlib.sha256(struct.pack('!I',len(prompt))+array('I',prompt).tobytes()+array('I',ids).tobytes()).digest()

    async def async_score(self,prompts,label_ids):
        # A completed batch needs no question tasks, repeated key hashing,
        # GPU warm-up, or bulk admission. Misses keep subscriber-aware handling.
        def completed():
            tick=time.perf_counter()
            outputs=self.inference_cache.get_completed(self.cache_key(p,ids)
                        for p,ids in zip(prompts,label_ids))
            if outputs is None:return None
            diag={'warm_skipped_recent_prefix':False,'common_prefix_tokens':common_prefix_length(prompts),
                  'warm_prefill_ms':0.,'branch_ms':(time.perf_counter()-tick)*1000,
                  'extra_input_tokens':0,'extra_output_tokens':0,
                  'cached_tokens':[x[1].get('cached_tokens') for x in outputs]}
            return [x[0] for x in outputs],diag
        result=completed()
        if result is not None:return result
        bulk=len(prompts)>=4 and sum(map(len,prompts))>=16384
        if bulk:
            tick=time.perf_counter()
            async with self.bulk_slots:
                waited=(time.perf_counter()-tick)*1000
                cached_result=completed()
                result,diag=cached_result if cached_result is not None else await self._async_score(prompts,label_ids,False)
            return result,{**diag,'bulk_wait_ms':waited}
        return await self._async_score(prompts,label_ids,False)

    async def _async_score(self,prompts,label_ids,fully_cached=False):
        common=common_prefix_length(prompts)
        warm=self.warm_mode=='always' and common>0 or self.warm_mode=='auto' and len(prompts)>=4 and common>=256
        warm=warm and not fully_cached
        key=tuple(prompts[0][:common]) if warm else None
        already_warm=key in self.recent_prefixes and time.monotonic()-self.recent_prefixes[key]<self.warm_cache_seconds
        warm=warm and not already_warm
        t=time.perf_counter();extra_tokens=0
        if warm:
            _,meta=await self.async_generate(prompts[0][:common]);extra_tokens=meta['prompt_tokens']
            if self.warm_cache_seconds>0:
                self.recent_prefixes[key]=time.monotonic();self.recent_prefixes.move_to_end(key)
                while len(self.recent_prefixes)>64:self.recent_prefixes.popitem(last=False)
        prefill_ms=(time.perf_counter()-t)*1000;t=time.perf_counter()
        outputs=await bounded_map(self.async_generate, list(zip(prompts,label_ids)), self.request_fanout)
        diag={'warm_skipped_recent_prefix':already_warm,'common_prefix_tokens':common,'warm_prefill_ms':prefill_ms,'branch_ms':(time.perf_counter()-t)*1000,'extra_input_tokens':extra_tokens,'extra_output_tokens':int(warm),'cached_tokens':[x[1].get('cached_tokens') for x in outputs]}
        return [x[0] for x in outputs],diag
