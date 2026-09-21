"""Bounded aggregate telemetry; never retain input text, question keys or credentials."""
from collections import Counter, deque
import time


class HostedMetrics:
    def __init__(self):
        self.started = time.monotonic()
        self.statuses = Counter()
        self.rejections = Counter()
        self.recent = deque(maxlen=8192)
        self.questions = self.prompt_tokens = 0
        self.max_active = self.max_waiting = 0
        self.guard = None

    def finish(self, scope, status, elapsed_ms):
        if scope['path'] != '/v1/systemone':
            return
        detail = scope.get('state', {}).get('request_metrics', {})
        self.statuses[str(status)] += 1
        self.questions += detail.get('questions', 0)
        self.prompt_tokens += detail.get('prompt_tokens', 0)
        self.recent.append((time.monotonic(), status, elapsed_ms, detail.get('admission_ms', 0)))

    def snapshot(self):
        now = time.monotonic()
        last = [r for r in self.recent if now-r[0] <= 60]
        latencies = sorted(r[2] for r in last)
        return {'uptime_seconds': now-self.started, 'completed_by_status': dict(self.statuses),
                'rejections_by_reason': dict(self.rejections), 'prepared_questions': self.questions,
                'prepared_prompt_tokens': self.prompt_tokens,
                'active_requests': self.guard.inflight if self.guard else 0,
                'waiting_requests': self.guard.waiting if self.guard else 0,
                'peak_active_requests': self.max_active, 'peak_waiting_requests': self.max_waiting,
                'last_60s_retained_completions': len(last), 'recent_capacity': self.recent.maxlen,
                'recent_latency_ms': {f'p{p}': latencies[min(len(latencies)-1, int((len(latencies)-1)*p/100))]
                                      for p in [50, 95, 99]} if latencies else {}}
