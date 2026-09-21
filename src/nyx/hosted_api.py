"""Authenticated production gateway for the Nyx decision API."""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Response, Request as HttpRequest
from pydantic import Field, field_validator
from starlette.responses import JSONResponse

from . import inference as server
from .contract import MODEL_ID, Request, canonical, structured, max_questions, MAX_CHOICES
from .validation_logging import install_validation_logging
from . import wire
from .hosted_metrics import HostedMetrics
from .backend_errors import log_backend_failure
from .shared_tokens import ContextBudgetExceeded

ALIASES = {MODEL_ID, 'devanshbatham/nyx'}
BODY_LIMIT = 2 * 1024 * 1024


class HostedQuestion(wire.Question):
    @property
    def type(self):
        return self.root.type

    @property
    def instructions(self):
        return self.root.instructions

    @property
    def criteria(self):
        value = self.root.criteria
        if self.type == 'noul' and value is not None:
            return value.model_dump(exclude_none=True)
        return value

    def inference_question(self):
        c = self.criteria
        if self.type == 'choice':
            c = {key: None if value is None else canonical(value) for key, value in c.items()}
        elif self.type == 'score':
            c = [canonical(value) for value in c]
        elif c is not None:
            c = {key: canonical(value) for key, value in c.items() if value is not None}
        return dict(type=self.type, instructions='' if self.instructions is None else self.instructions, criteria=c)


class HostedRequest(wire.SystemOneRequest):
    questions: dict[str, HostedQuestion] = Field(min_length=1)
    _state = field_validator('state')(structured)

    @field_validator('questions')
    @classmethod
    def question_count(cls, value):
        limit = max_questions()
        if limit is not None and len(value) > limit:
            raise ValueError(f'Provide 1..{limit} questions')
        return value


class AccessGuard:
    """Authenticate before buffering; bound both body memory and active work."""
    def __init__(self, app, key_hash, max_inflight=8, requests_per_minute=120,
                 max_waiting=0, admission_timeout=120, metrics=None):
        if len(key_hash) != 64 or any(c not in '0123456789abcdef' for c in key_hash):
            raise ValueError('A SHA256 API key hash is required; refusing unauthenticated startup')
        self.app, self.key_hash = app, key_hash
        self.max_inflight, self.inflight = max_inflight, 0
        self.rate = requests_per_minute
        self.tokens, self.updated = float(self.rate), time.monotonic()
        self.max_waiting, self.waiting = max_waiting, 0
        self.admission_timeout = admission_timeout
        self.slots = asyncio.Semaphore(max_inflight)
        self.metrics = metrics or HostedMetrics()
        self.metrics.guard = self

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        request_id = str(uuid.uuid4())
        scope.setdefault('state', {})['request_id'] = request_id
        detail = scope['state']['request_metrics'] = {}
        started = time.perf_counter()
        status = 499

        async def send_with_id(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
                message['headers'] = list(message.get('headers', [])) + [
                    (b'x-nyx-request-id', request_id.encode()),
                    (b'cache-control', b'no-store'),
                    (b'x-content-type-options', b'nosniff'),
                    (b'x-local-admission-ms', f"{detail.get('admission_ms', 0):.3f}".encode())]
            await send(message)

        async def reject(code, message, headers=None, reason='rejected'):
            self.metrics.rejections[reason] += 1
            await JSONResponse({'detail': message}, status_code=code, headers=headers)(scope, receive, send_with_id)
            self.metrics.finish(scope, code, (time.perf_counter()-started)*1000)

        if scope['path'] == '/health' and scope['method'] == 'GET':
            return await self.app(scope, receive, send_with_id)
        auth = [v for k, v in scope['headers'] if k.lower() == b'authorization']
        value = auth[0].split(b' ', 1) if len(auth) == 1 else []
        if (len(value) != 2 or value[0].lower() != b'bearer' or not value[1]
                or not hmac.compare_digest(hashlib.sha256(value[1]).hexdigest(), self.key_hash)):
            return await reject(401, 'Missing or invalid API key', {'WWW-Authenticate': 'Bearer'})
        if scope['path'] == '/internal/metrics' and scope['method'] == 'GET':
            return await self.app(scope, receive, send_with_id)
        now = time.monotonic()
        self.tokens = min(float(self.rate), self.tokens + (now-self.updated)*self.rate/60)
        self.updated = now
        if self.rate and self.tokens < 1:
            return await reject(429, 'Rate limit exceeded', {'Retry-After': '1'}, reason='rate')
        if self.rate:
            self.tokens -= 1
        acquired = False
        if self.max_waiting:
            if self.slots.locked() and self.waiting >= self.max_waiting:
                return await reject(503, 'Server waiting queue is full', {'Retry-After': '1'}, reason='waiting_queue_full')
            self.waiting += 1
            self.metrics.max_waiting = max(self.metrics.max_waiting, self.waiting)
            try:
                async with asyncio.timeout(self.admission_timeout):
                    await self.slots.acquire()
                acquired = True
            except TimeoutError:
                return await reject(504, 'Admission queue deadline exceeded', reason='admission_deadline')
            finally:
                self.waiting -= 1
        elif self.inflight >= self.max_inflight:
            return await reject(429, 'Concurrency limit exceeded', {'Retry-After': '1'}, reason='concurrency')
        detail['admission_ms'] = (time.perf_counter()-started)*1000
        self.inflight += 1
        self.metrics.max_active = max(self.metrics.max_active, self.inflight)
        try:
            body = bytearray()
            try:
                async with asyncio.timeout(30):
                    while True:
                        message = await receive()
                        if message['type'] == 'http.disconnect':
                            return
                        body.extend(message.get('body', b''))
                        if len(body) > BODY_LIMIT:
                            return await reject(413, 'Body exceeds 2 MiB')
                        if not message.get('more_body', False):
                            break
            except TimeoutError:
                return await reject(408, 'Request body deadline exceeded')
            delivered = False
            disconnected = asyncio.Event()

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                await disconnected.wait()
                return {'type': 'http.disconnect'}

            async def watch_disconnect():
                while True:
                    message = await receive()
                    if message['type'] == 'http.disconnect':
                        disconnected.set()
                        return
                    await asyncio.sleep(0)

            application = asyncio.create_task(self.app(scope, replay, send_with_id))
            watcher = asyncio.create_task(watch_disconnect())
            try:
                done, _ = await asyncio.wait([application, watcher], return_when=asyncio.FIRST_COMPLETED)
                if application in done:
                    await application
                else:
                    self.metrics.rejections['client_disconnected'] += 1
                    application.cancel()
            finally:
                for task in [application, watcher]:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(application, watcher, return_exceptions=True)
        finally:
            self.inflight -= 1
            if acquired:
                self.slots.release()
            # Body rejections already recorded themselves.
            if status not in {408, 413}:
                self.metrics.finish(scope, status, (time.perf_counter()-started)*1000)


@asynccontextmanager
async def lifespan(app):
    limit = int(os.getenv('NYX_HOSTED_MAX_PROMPT_TOKENS', '262143'))
    if not 1 <= limit <= 262143:
        raise ValueError('Hosted prompt limit must be 1..262143')
    app.state.engine = await asyncio.to_thread(server.Engine, max_length=limit+1)
    queue_size = int(os.getenv('NYX_HOSTED_QUEUE_SIZE', '128'))
    if not 1 <= queue_size <= 4096:
        raise ValueError('Hosted queue size must be 1..4096')
    app.state.queue = asyncio.Queue(maxsize=queue_size)
    remote = True
    workers = int(os.getenv('NYX_HOSTED_WORKERS', '64'))
    if not 1 <= workers <= 256:
        raise ValueError('Hosted workers must be 1..256')
    tasks = [asyncio.create_task(server.worker(app)) for _ in range(workers)]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if remote:
            model = app.state.engine.model
            await model.async_client.aclose()
            if getattr(model, 'abort_client', None) is not None:
                from .backend_cancellation import drain_aborts
                await drain_aborts()
                await model.abort_client.aclose()
            model.client.close()
            model.pool.shutdown(wait=False, cancel_futures=True)


def create_app(key_hash=None):
    app = FastAPI(title='Nyx Decision API', version='1.0.0', lifespan=lifespan)
    app.state.metrics = HostedMetrics()
    install_validation_logging(app)
    max_questions()  # Fail startup if the configured bound is invalid.
    max_inflight = int(os.getenv('NYX_HOSTED_MAX_INFLIGHT', '8'))
    rate = int(os.getenv('NYX_HOSTED_REQUESTS_PER_MINUTE', '120'))
    max_waiting = int(os.getenv('NYX_HOSTED_MAX_WAITING', '0'))
    if not 1 <= max_inflight <= 4096 or not 0 <= rate <= 60000 or not 0 <= max_waiting <= 16384:
        raise ValueError('Invalid hosting admission limits')
    app.add_middleware(AccessGuard, key_hash=key_hash or os.environ.get('NYX_HOSTED_API_KEY_SHA256', ''),
                       max_inflight=max_inflight, requests_per_minute=rate, max_waiting=max_waiting,
                       metrics=app.state.metrics)

    @app.get('/health')
    async def health():
        engine = app.state.engine
        if getattr(engine, 'backend', None) == 'sglang':
            try:
                check = await engine.model.async_client.get('/health', timeout=2)
                check.raise_for_status()
            except Exception:
                raise HTTPException(503, 'Model backend unavailable')
        return {'status': 'ready', 'model': MODEL_ID}

    @app.get('/v1/models')
    def models():
        return {'models': [{'name': name, 'description': 'Nyx 35B-A3B INT4+FP8; Choice, Score, and Noul decisions.', 'release_date': '2026-09-21'}
                           for name in [MODEL_ID, *sorted(ALIASES - {MODEL_ID})]]}


    @app.get('/v1/limits')
    def limits():
        shared=os.getenv('NYX_CONTEXT_ACCOUNTING')=='shared'
        return dict(max_questions=max_questions(), max_choices=MAX_CHOICES, max_score_levels=10,
                    max_prompt_tokens_per_question=32768 if shared else app.state.engine.max_length-1,
                    max_total_prompt_tokens=65536 if shared else 640000, max_body_bytes=BODY_LIMIT,
                    context_accounting='shared_state_once_local_tokenizer' if shared else 'sum_compiled_prompts',
                    max_queued_requests=app.state.queue.maxsize,
                    max_concurrent_requests=max_inflight, max_waiting_requests=max_waiting,
                    requests_per_minute=rate or None,
                    inference_deadline_seconds=120,
                    confidence='Nyx empirical confidence statistic')

    @app.get('/internal/metrics')
    async def metrics():
        cache = getattr(getattr(app.state.engine, 'model', None), 'inference_cache', None)
        compiler = getattr(app.state.engine, 'compiler', None)
        token_caches = {name: getattr(compiler, name).snapshot() for name in ['prefix_tokens','suffix_tokens']
                        if hasattr(getattr(compiler,name,None),'snapshot')}
        return {**app.state.metrics.snapshot(), 'model_queue_depth': app.state.queue.qsize(),
                'tokenization_cache': token_caches,
                'question_cache': cache.snapshot() if cache is not None else None}

    @app.post('/v1/systemone')
    async def decide(req: HostedRequest, response: Response, request: HttpRequest):
        started = time.perf_counter()
        if req.model not in ALIASES:
            raise HTTPException(400, {'error_type':'api_usage_error', 'message':f'Unknown model: {req.model}'})
        for qid, question in req.questions.items():
            if question.type == 'choice':
                if not question.criteria:
                    raise HTTPException(400, f'Choice question must have at least one choice: {qid}')
                if len(question.criteria) > MAX_CHOICES:
                    raise HTTPException(400, f'Too many choices. Must have at most {MAX_CHOICES} choices.')
            elif question.type == 'score' and len(question.criteria) > 10:
                raise HTTPException(400, 'Too many score levels. Must have at most 10 levels.')
            elif question.type == 'noul' and not question.instructions and not any((question.criteria or {}).values()):
                raise HTTPException(400, f'Noul question must have criteria or instructions: {qid}')
        normalized = Request(model=MODEL_ID, state=req.state,
                             questions={key: q.inference_question() for key, q in req.questions.items()})
        try:
            prepared = await asyncio.to_thread(app.state.engine.prepare, [(normalized.state, q) for q in normalized.questions.values()])
        except ContextBudgetExceeded as error:
            raise HTTPException(400, {'error_type':'max_tokens_exceeded'}) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        prepared_at = time.perf_counter()
        request.scope['state']['request_metrics'].update(questions=len(prepared[0]), prompt_tokens=sum(prepared[1]))
        future = asyncio.get_running_loop().create_future()
        try:
            app.state.queue.put_nowait(server.Job(normalized, prepared, future))
        except asyncio.QueueFull:
            raise HTTPException(529, 'Model queue full', headers={'Retry-After': '1'})
        try:
            async with asyncio.timeout(120):
                result, headers = await future
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except TimeoutError:
            raise HTTPException(504, 'Inference deadline exceeded')
        except Exception as error:
            log_backend_failure(error, request.scope['state']['request_id'])
            raise HTTPException(502, 'Model backend failed')
        for key, question in req.questions.items():
            if question.type == 'score':
                result['answers'][key]['legend'] = {str(i): level for i, level in enumerate(question.criteria)}
        response.headers.update(headers)
        response.headers['Server-Timing'] = headers.get('Server-Timing', '') + f', prepare_api;dur={(prepared_at-started)*1000:.3f}, api_total;dur={(time.perf_counter()-started)*1000:.3f}'
        # The decision builder emits JSON-native values. Avoid FastAPI's second
        # recursive conversion of large probability/legend trees before encoding.
        return JSONResponse(content=result, headers=dict(response.headers))

    return app
