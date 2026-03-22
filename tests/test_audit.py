"""
Tests for audit.py — log format, field presence, rotation namer.
"""
from __future__ import annotations

import json
import logging
import sys
from io import StringIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixture: capture log output in memory, no disk I/O
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def log_capture():
    """Replace the audit logger's handlers with an in-memory stream."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger('ai_proxy.audit')
    original = logger.handlers[:]
    logger.handlers = [handler]
    yield stream
    logger.handlers = original


def _entries(stream: StringIO) -> list[dict]:
    stream.seek(0)
    return [json.loads(l) for l in stream.getvalue().splitlines() if l.strip()]


def _last(stream: StringIO) -> dict:
    entries = _entries(stream)
    assert entries, "No log entries written"
    return entries[-1]


# ---------------------------------------------------------------------------
# Log format
# ---------------------------------------------------------------------------

class TestLogFormat:

    def test_entry_is_valid_json(self, log_capture):
        import audit
        audit.log('test.event')
        assert isinstance(_last(log_capture), dict)

    def test_entry_has_ts_field(self, log_capture):
        import audit
        audit.log('test.event')
        entry = _last(log_capture)
        assert 'ts' in entry
        assert 'T' in entry['ts']   # ISO 8601

    def test_entry_has_event_field(self, log_capture):
        import audit
        audit.log('my.event.name')
        assert _last(log_capture)['event'] == 'my.event.name'

    def test_extra_kwargs_included(self, log_capture):
        import audit
        audit.log('api.request', method='POST', path='/v1/chat/completions',
                  ip='127.0.0.1', token_id='abc', label='my-app')
        entry = _last(log_capture)
        assert entry['method'] == 'POST'
        assert entry['path'] == '/v1/chat/completions'
        assert entry['ip'] == '127.0.0.1'
        assert entry['token_id'] == 'abc'
        assert entry['label'] == 'my-app'

    def test_auth_rejected_fields(self, log_capture):
        import audit
        audit.log('auth.rejected', method='GET', path='/v1/models',
                  ip='10.0.0.1', reason='missing_token')
        entry = _last(log_capture)
        assert entry['event'] == 'auth.rejected'
        assert entry['reason'] == 'missing_token'

    def test_multiple_events_written_in_order(self, log_capture):
        import audit
        audit.log('event.one')
        audit.log('event.two')
        audit.log('event.three')
        events = [e['event'] for e in _entries(log_capture)]
        assert events == ['event.one', 'event.two', 'event.three']


# ---------------------------------------------------------------------------
# Rotation namer
# ---------------------------------------------------------------------------

class TestRotationNamer:

    def test_dot_suffix_becomes_dash(self):
        import audit
        assert audit._namer('/app/data/audit_logs/log.1') == '/app/data/audit_logs/log-1'

    def test_higher_number(self):
        import audit
        assert audit._namer('/app/data/audit_logs/log.10') == '/app/data/audit_logs/log-10'

    def test_non_numeric_suffix_unchanged(self):
        import audit
        result = audit._namer('/app/data/audit_logs/log')
        assert result == '/app/data/audit_logs/log'
