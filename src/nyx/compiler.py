"""Reuse exact tokenized state prefixes without caching decisions or dropping input."""
from .tokenization_cache import TokenizationCache
from .contract import messages,canonical

class PromptCompiler:
    def __init__(self,tokenizer,max_length):
        self.tokenizer=tokenizer;self.max_length=max_length
        stub=messages('',{'type':'choice','instructions':'placeholder','criteria':{'placeholder':None}})
        rendered=tokenizer.apply_chat_template(stub,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        body=stub[-1]['content'];at=rendered.index(body)
        self.before=rendered[:at];self.after=rendered[at+len(body):]
        # Cache immutable token tuples only; bounded process-local state cache.
        encode=lambda text:tokenizer.encode(text,add_special_tokens=False)
        self.prefix_tokens=TokenizationCache(encode,max_entries=64,max_bytes=32*1024*1024)
        self.suffix_tokens=TokenizationCache(encode,max_entries=4096,max_bytes=16*1024*1024)

    def encode(self,state,q):
        msgs=messages(state,q)
        content=msgs[-1]['content'];state_prefix='State (data to evaluate):\n'+canonical(state)+'\n\n'
        prefix=self.before+state_prefix;suffix=content[len(state_prefix):]+self.after
        # Qwen BPE separates the double-newline from the following 'Question:'.
        # Exact equivalence is independently checked against full-template tokenization.
        ids=list(self.prefix_tokens(prefix))+list(self.suffix_tokens(suffix))
        if len(ids)>self.max_length-1:raise ValueError(f'Prompt has {len(ids)} tokens; limit is {self.max_length-1}; no truncation performed')
        return ids

    def shared_prefix(self,state):
        from .shared_tokens import SharedPrefix
        return SharedPrefix(self.prefix_tokens(self.before+'State (data to evaluate):\n'+canonical(state)+'\n\n'))

    def encode_shared(self,prefix,q):
        from .shared_tokens import SharedTokens
        empty_prefix='State (data to evaluate):\n\n\n'
        content=messages('',q)[-1]['content']
        assert content.startswith(empty_prefix)
        suffix=content[len(empty_prefix):]+self.after
        return SharedTokens(prefix,self.suffix_tokens(suffix))

    def align_shared(self,items,prompts):
        from .prefix_alignment import SharedPrefixAligner
        if not hasattr(self,'shared_aligner'):self.shared_aligner=SharedPrefixAligner(self)
        return self.shared_aligner.align(items,prompts)
