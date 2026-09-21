"""Experimental bounded separator-only shared-prefix alignment, default off.

No warmup change. Whitespace changes model inputs/positions and needs a paired
quality ablation. Failure to prove safe tokenization returns original prompts.
"""
from functools import lru_cache
import time

from .contract import canonical,messages

GRID=64
MAX_ADDED_TOKENS=63
MAX_PADDING_BYTES=512
MAX_SEARCH=68
MAX_TAIL_CHARACTERS=512
MAX_VERIFICATION_CHARACTERS=131072


class PreparedPrompts(tuple):
    def __new__(cls,prompts,lengths,sizes,alignment):
        value=super().__new__(cls,(prompts,lengths,sizes));value.alignment=alignment;return value


def common_length(prompts):
    # Lexicographic extremes determine the LCP of every list between them.
    # List comparisons run in C, avoiding a Python set of64 IDs at each position.
    low,high=min(prompts),max(prompts)
    return next((i for i,(a,b) in enumerate(zip(low,high)) if a!=b),min(len(low),len(high)))


class SharedPrefixAligner:
    def __init__(self,compiler):
        self.compiler=compiler;self.tokenizer=compiler.tokenizer
        self.boundary=lru_cache(maxsize=64)(self._boundary)
        self.padding=lru_cache(maxsize=64)(self._padding)
        self.verify_tail=lru_cache(maxsize=256)(self._verify_tail)

    def encode(self,text):return tuple(self.tokenizer.encode(text,add_special_tokens=False))

    def _boundary(self,prefix):
        backend=getattr(self.tokenizer,'backend_tokenizer',None)
        if backend is None or repr(backend.normalizer) not in {'None','NFC()'}:return None
        pieces=backend.pre_tokenizer.pre_tokenize_str(prefix)
        if len(pieces)<2:return None
        at=pieces[-2][1][0];tail=prefix[at:]
        if len(tail)>MAX_TAIL_CHARACTERS or not tail.endswith('\n\n'):return None
        base=self.encode(prefix[:at]);original=tuple(self.compiler.prefix_tokens(prefix))
        if base+self.encode(tail)!=original:return None
        return base,tail,original

    def _padding(self,prefix,needed):
        boundary=self.boundary(prefix)
        if boundary is None:return None,'unverified_prefix_boundary',0
        base,tail,original=boundary
        # Usually one near-direct trial; bounded fallback never re-encodes state.
        patterns=[' \n'*2] if needed==1 else []
        indices=list(dict.fromkeys([max(1,needed-1),max(1,needed),max(1,needed-2),min(64,needed+1),*range(1,65)]))
        patterns += ['\n \t'*n for n in indices]
        patterns += [' \n','\t\n']
        for trial,pad in enumerate(patterns[:MAX_SEARCH],1):
            if len(pad.encode())>MAX_PADDING_BYTES:continue
            padded_tail=tail[:-2]+pad+'\n\n';ids=base+self.encode(padded_tail)
            if len(ids)-len(original)==needed:
                return (ids,padded_tail,pad),'candidate',trial
        return None,'bounded_search_no_alignment',min(len(patterns),MAX_SEARCH)

    def _verify_tail(self,tail,suffix):
        # Native tokenization of the bounded changed region plus actual suffix;
        # every question is checked, never64 copies of the complete long state.
        suffix_ids=self.encode(suffix)
        return suffix_ids if self.encode(tail+suffix)==self.encode(tail)+suffix_ids else None

    def align(self,items,prompts):
        started=time.perf_counter();diag={'enabled':True,'applied':False,'grid':GRID,'warmup_policy_changed':False}
        def finish(reason,new=None):
            diag['reason']=reason;diag['alignment_ms']=(time.perf_counter()-started)*1000
            return prompts if new is None else new,diag
        if not 4<=len(prompts)<=64:return finish('requires4_to64_questions')
        shared=items[0][0];state=canonical(shared)
        if any(value is not shared and canonical(value)!=state for value,_ in items):return finish('different_states')
        common=common_length(prompts);diag['common_before']=common
        if common<256:return finish('common_prefix_below256')
        if any(len(p)==common for p in prompts):return finish('no_suffix_after_common_prefix')
        needed=(-common)%GRID
        if needed==0:diag['common_after']=common;return finish('already_aligned')
        prefix=self.compiler.before+'State (data to evaluate):\n'+state+'\n\n'
        original=tuple(self.compiler.prefix_tokens(prefix))
        if any(tuple(p[:len(original)])!=original for p in prompts):return finish('unexpected_compiler_prefix')
        result,reason,attempts=self.padding(prefix,needed);diag['cached_plan_search_candidates']=attempts
        if result is None:return finish(reason)
        new_prefix,tail,padding=result
        if not 0<len(new_prefix)-len(original)<=MAX_ADDED_TOKENS:return finish('added_token_bound')
        candidate=[list(new_prefix)+p[len(original):] for p in prompts]
        if any(len(p)>self.compiler.max_length-1 for p in candidate) or sum(map(len,candidate))>640000:return finish('would_exceed_original_context_limits')
        suffixes=[]
        for _,q in items:
            body=messages('',q)[-1]['content'];marker='State (data to evaluate):\n\n\n'
            if not body.startswith(marker):return finish('unexpected_question_template')
            suffixes.append(body[len(marker):]+self.compiler.after)
        if sum(len(tail)+len(suffix) for suffix in suffixes)>MAX_VERIFICATION_CHARACTERS:return finish('bounded_suffix_verification_limit')
        for suffix,prompt in zip(suffixes,prompts):
            checked=self.verify_tail(tail,suffix)
            if checked is None:return finish('native_boundary_tokenization_mismatch')
            if checked!=tuple(prompt[len(original):]):return finish('original_suffix_tokenization_mismatch')
        actual=common_length(candidate)
        if actual%GRID or actual!=common+needed:return finish('actual_common_prefix_not_aligned')
        diag.update(applied=True,common_after=actual,added_tokens_per_question=needed,added_tokens_total=needed*len(prompts),
                    padding_utf8_bytes=len(padding.encode()),separator_padding=repr(padding),native_boundary_checks=len(suffixes))
        return finish('aligned',candidate)
