import pytest

from nyx.sglang_backend import SGLangBackend
from nyx.shared_tokens import SharedPrefix, SharedTokens, common_prefix_length


def test_shared_representation_preserves_tokens_and_cache_keys():
    prefix = SharedPrefix(range(10_000))
    prompts = [SharedTokens(prefix, [40, 41, index, 42]) for index in range(4)]
    assert common_prefix_length(prompts) == 10_002
    assert common_prefix_length([list(prompt) for prompt in prompts]) == 10_002
    for prompt in prompts:
        full = list(prompt)
        for part in [slice(None, 5), slice(-8, None), slice(9998, 10004), slice(None, None, -1)]:
            assert prompt[part] == full[part]
        assert prompt[-1] == full[-1]
        with pytest.raises(IndexError):
            _ = prompt[-len(prompt) - 1]
        assert SGLangBackend.cache_key(prompt, [15, 16]) == SGLangBackend.cache_key(full, [15, 16])
    assert len(prefix.hashes) == 1
