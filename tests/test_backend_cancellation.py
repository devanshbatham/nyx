import asyncio
from collections import Counter
import httpx
import pytest
from nyx.backend_cancellation import generate_with_abort, drain_aborts


@pytest.mark.parametrize('abort_fails', [False, True])
def test_cancel_targets_only_its_unique_request(abort_fails):
    async def run():
        started=asyncio.Event(); requests=[]; aborted=[]; counts=Counter()
        class Client:
            async def post(self,path,json):
                requests.append(json)
                if len(requests)==2:
                    started.set()
                await asyncio.Event().wait()
        class Control:
            async def post(self,path,json):
                assert path=='/abort_request'
                aborted.append(json)
                if abort_fails:
                    raise httpx.ConnectError('Unavailable')
                return httpx.Response(200,request=httpx.Request('POST','http://backend'+path))
        tasks=[asyncio.create_task(generate_with_abort(Client(),Control(),{'input_ids':[1,2]},counts)) for _ in range(2)]
        await started.wait()
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        assert not tasks[1].done()
        assert requests[0]['rid']!=requests[1]['rid']
        assert len(requests[0]['rid'])==42
        assert aborted==[{'rid':requests[0]['rid'],'abort_all':False}]
        assert counts['backend_abort_attempts']==1
        assert counts['backend_abort_unconfirmed' if abort_fails else 'backend_abort_acknowledged']==1
        tasks[1].cancel()
        await asyncio.gather(tasks[1],return_exceptions=True)
    asyncio.run(run())


def test_success_never_sends_abort_and_preserves_payload():
    async def run():
        original={'input_ids':[1,2]}
        class Client:
            async def post(self,path,json):
                assert json['input_ids']==[1,2] and json['rid']
                return 'result'
        class Control:
            async def post(self,*args,**kwargs):
                raise AssertionError('Must not abort completed work')
        counts=Counter()
        assert await generate_with_abort(Client(),Control(),original,counts)=='result'
        assert original=={'input_ids':[1,2]} and not counts
    asyncio.run(run())


def test_second_cancellation_does_not_cancel_abort_delivery():
    async def run():
        started, abort_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        counts = Counter()
        class Client:
            async def post(self, path, json):
                started.set()
                await asyncio.Event().wait()
        class Control:
            async def post(self, path, json):
                abort_started.set()
                await release.wait()
                return httpx.Response(200,request=httpx.Request('POST','http://backend'+path))
        task=asyncio.create_task(generate_with_abort(Client(),Control(),{},counts))
        await started.wait();task.cancel();await abort_started.wait();task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set();await drain_aborts()
        assert counts['backend_abort_acknowledged']==1 and counts['backend_abort_unconfirmed']==0
    asyncio.run(run())


@pytest.mark.parametrize('mode', ['disconnect', '503', 'always_disconnect', '400'])
def test_bounded_transient_retry_uses_new_id_and_never_retries_bad_input(mode):
    async def run():
        calls=[];aborted=[];counts=Counter()
        class Client:
            async def post(self,path,json):
                calls.append(dict(json))
                if mode=='always_disconnect' or mode=='disconnect' and len(calls)==1:
                    raise httpx.ReadError('Connection reset')
                status=400 if mode=='400' else 503 if mode=='503' and len(calls)==1 else 200
                return httpx.Response(status,request=httpx.Request('POST','http://backend/generate'))
        class Control:
            async def post(self,path,json):
                aborted.append(json['rid'])
                return httpx.Response(200,request=httpx.Request('POST','http://backend/abort_request'))
        if mode=='always_disconnect':
            with pytest.raises(httpx.ReadError):await generate_with_abort(Client(),Control(),{},counts)
        else:
            result=await generate_with_abort(Client(),Control(),{},counts)
            assert result.status_code==(400 if mode=='400' else 200)
        assert len(calls)==(1 if mode=='400' else 2)
        assert counts['backend_transient_retries']==(0 if mode=='400' else 1)
        if mode!='400':
            assert calls[0]['rid']!=calls[1]['rid']
            assert aborted[0]==calls[0]['rid']
    asyncio.run(run())
