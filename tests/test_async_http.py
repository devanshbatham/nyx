import asyncio
from aiohttp import web
import pytest
from nyx.async_http import AiohttpClient


def test_pool_isolation_timeouts_and_recovery():
    async def run():
        active = peak = 0
        async def answer(request):
            nonlocal active, peak
            active += 1; peak = max(peak, active)
            try:
                data = await request.json()
                await asyncio.sleep(data.get('delay', .002))
                return web.json_response(data, status=data.get('status', 200))
            finally:
                active -= 1
        app = web.Application(); app.router.add_post('/generate', answer)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = AiohttpClient(f'http://127.0.0.1:{port}', 8)
        try:
            responses = await asyncio.gather(*(client.post('/generate', json={'id':i}) for i in range(100)))
            assert [r.json()['id'] for r in responses] == list(range(100))
            assert peak <= 8
            with pytest.raises(TimeoutError):
                await client.post('/generate', json={'delay':.1}, timeout=.01)
            response = await client.post('/generate', json={'id':101})
            assert response.json()['id'] == 101
            response = await client.post('/generate', json={'status':503})
            assert response.status_code == 503
        finally:
            await client.aclose(); await runner.cleanup()
        assert client.session.closed
    asyncio.run(run())
