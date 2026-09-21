"""Failure diagnostics that never serialize prompts, credentials or response bodies."""
import json
import logging
import traceback
from pathlib import Path


def log_backend_failure(error, request_id):
    chain = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        item = {'type': type(error).__name__, 'frames': [
            {'file': Path(frame.filename).name, 'line': frame.lineno, 'function': frame.name}
            for frame in traceback.extract_tb(error.__traceback__)]}
        status = getattr(getattr(error, 'response', None), 'status_code', None)
        if isinstance(status, int):
            item['upstream_status'] = status
        errno = getattr(error, 'errno', None)
        if isinstance(errno, int):
            item['errno'] = errno
        chain.append(item)
        error = error.__cause__ or error.__context__
    logging.getLogger('nyx.backend').error(json.dumps(
        {'event': 'backend_failure', 'request_id': request_id, 'chain': chain}))
