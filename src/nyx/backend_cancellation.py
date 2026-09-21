"""Cancel only our own backend request, without waiting for disconnect polling."""
import asyncio
import uuid
import random
import aiohttp
import httpx

_pending_aborts = set()


async def drain_aborts():
    await asyncio.gather(*tuple(_pending_aborts), return_exceptions=True)


async def generate_with_abort(client, control_client, data, counters):
    # Unique nonempty IDs are essential: SGLang abort IDs can match a prefix.
    request_id = 'nyx-local-' + uuid.uuid4().hex
    payload = {**data, 'rid': request_id}
    async def send_abort():
        abort_id = request_id
        counters['backend_abort_attempts'] += 1
        async def abort():
            try:
                async with asyncio.timeout(2):
                    response = await control_client.post('/abort_request',
                        json={'rid': abort_id, 'abort_all': False})
                    response.raise_for_status()
                    counters['backend_abort_acknowledged'] += 1
            except (Exception, asyncio.CancelledError):
                counters['backend_abort_unconfirmed'] += 1
        task = asyncio.create_task(abort())
        _pending_aborts.add(task)
        task.add_done_callback(_pending_aborts.discard)
        # A second cancellation propagates to the caller while the retained
        # control task continues independently, bounded by its own deadline.
        await asyncio.shield(task)
    for attempt in range(2):
        # Each retry has its own ID so cancellation cannot affect a later attempt.
        if attempt:
            request_id = 'nyx-local-' + uuid.uuid4().hex
            payload['rid'] = request_id
        try:
            response = await client.post('/generate', json=payload)
            if getattr(response, 'status_code', None) in {502, 503, 504} and not attempt:
                counters['backend_transient_retries'] += 1
                await send_abort()
                await asyncio.sleep(random.uniform(.01, .03))
                continue
            return response
        except asyncio.CancelledError:
            await send_abort()
            raise
        except (aiohttp.ClientConnectionError, httpx.NetworkError, httpx.RemoteProtocolError):
            await send_abort()
            if attempt:
                counters['backend_transport_failures'] += 1
                raise
            counters['backend_transient_retries'] += 1
            await asyncio.sleep(random.uniform(.01, .03))
