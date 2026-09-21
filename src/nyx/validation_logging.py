"""Error-only validation diagnostics without request values, headers or user keys."""
import json
import logging

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler

logger = logging.getLogger('nyx.validation')
FIELDS = {'body', 'model', 'state', 'questions', 'type', 'instructions', 'criteria'}
REASONS = {
    'Choice requires 1..128 options': 'choice_options_invalid',
    'Score requires 2..10 ordered levels': 'score_levels_invalid',
    'Noul criteria accepts only true and false': 'noul_criteria_invalid',
    'Expected string, object, or array': 'structured_value_required',
    'Provide 1..64 questions': 'question_count_invalid',
    'Provide 1..1024 questions': 'question_count_invalid',
    'Unknown model; see /v1/models': 'unknown_model',
    'Total prompt budget exceeds 640000 tokens': 'aggregate_prompt_limit',
    'Prompt exceeds context limit; no truncation performed': 'prompt_context_limit',
    'Invalid model probabilities': 'invalid_model_output',
}


def reason(message):
    if isinstance(message, dict) and str(message.get('message', '')).startswith('Unknown model: '):
        return 'unknown_model'
    value = str(message).removeprefix('Value error, ')
    if value.startswith('Prompt has ') and '; no truncation performed' in value:
        return 'prompt_context_limit'
    return REASONS.get(value, 'validation_error')


def safe_location(location):
    return [part if part in FIELDS and not (i >= 2 and location[i-1] in {'questions', 'criteria'})
            else '<item>' for i, part in enumerate(location)]


def install_validation_logging(app):
    @app.exception_handler(RequestValidationError)
    async def schema_error(request, error):
        entries = [{'location': safe_location(item['loc']), 'type': item['type'],
                    'reason': reason(item['msg'])} for item in error.errors()[:16]]
        logger.warning('request_validation_failed request_id=%s errors=%s',
                       getattr(request.state, 'request_id', 'unavailable'), json.dumps(entries))
        return await request_validation_exception_handler(request, error)

    @app.exception_handler(HTTPException)
    async def application_error(request, error):
        if error.status_code in {400, 422}:
            logger.warning('request_rejected request_id=%s status=%s reason=%s',
                           getattr(request.state, 'request_id', 'unavailable'), error.status_code, reason(error.detail))
        return await http_exception_handler(request, error)
