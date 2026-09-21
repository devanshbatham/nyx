import json
import logging
from nyx.backend_errors import log_backend_failure


def test_failure_log_has_frames_but_no_sensitive_exception_text(caplog):
    with caplog.at_level(logging.ERROR, logger='nyx.backend'):
        try:
            raise ConnectionRefusedError(111, 'secret-key and private prompt')
        except Exception as error:
            log_backend_failure(error, 'request-123')
    assert 'secret-key' not in caplog.text and 'private prompt' not in caplog.text
    record=json.loads(caplog.records[-1].message)
    assert record['chain'][0]['type']=='ConnectionRefusedError'
    assert record['chain'][0]['errno']==111 and record['chain'][0]['frames']
