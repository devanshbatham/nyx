from concurrent.futures import ThreadPoolExecutor
from nyx.tokenization_cache import TokenizationCache


def test_reuse_limits_and_oversize_inputs_are_never_retained():
    calls=[]
    def encode(text):
        calls.append(text)
        return list(range(len(text)))
    cache=TokenizationCache(encode,max_entries=2,max_bytes=4000,max_tokens=20)
    assert cache('hello')==cache('hello')==(0,1,2,3,4)
    assert calls==['hello']
    cache('second');cache('third')
    assert len(cache.entries)==2 and cache.bytes<=4000
    cache('x'*30);cache('x'*30)
    assert 'x'*30 not in cache.entries
    assert calls.count('x'*30)==2


def test_concurrent_callers_preserve_values_and_memory_accounting():
    cache=TokenizationCache(lambda text:[ord(x) for x in text],max_entries=8,max_bytes=5000)
    words=[f'word-{i%16}' for i in range(1000)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        outputs=list(pool.map(cache,words))
    assert outputs==[tuple(map(ord,w)) for w in words]
    assert len(cache.entries)<=8 and cache.bytes<=5000
    assert cache.bytes==sum(size for _,size in cache.entries.values())
