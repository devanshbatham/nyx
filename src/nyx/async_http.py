"""Optional bounded aiohttp transport for the local inference connection pool."""
import aiohttp
import httpx


class AiohttpClient:
    def __init__(self, base_url, capacity, timeout=120):
        self.base_url = base_url
        self.capacity = capacity
        self.timeout = timeout
        self.session = None

    async def request(self, method, path, **kwargs):
        # Engine construction happens in a thread; create resources on first use
        # on the serving event loop, with no await between checking and assignment.
        if self.session is None:
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=self.capacity, limit_per_host=self.capacity),
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        if 'timeout' in kwargs:
            kwargs['timeout'] = aiohttp.ClientTimeout(total=kwargs['timeout'])
        url = self.base_url + path
        async with self.session.request(method, url, **kwargs) as response:
            content = await response.read()
            return httpx.Response(response.status, content=content,
                                  request=httpx.Request(method, url))

    async def post(self, path, **kwargs):
        return await self.request('POST', path, **kwargs)

    async def get(self, path, **kwargs):
        return await self.request('GET', path, **kwargs)

    async def aclose(self):
        if self.session is not None:
            await self.session.close()
