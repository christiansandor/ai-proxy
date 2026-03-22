"""
Tests for the admin API plugin and the proxy's client-token auth middleware.

  Admin API  /api/tokens  — CRUD, admin token gating, audit logging
  Proxy auth /v1/*        — client tokens required, 401 on failure
  Path guard              — anything outside /v1, /v1beta, /api → 404
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

ADMIN_TOKEN = 'super-secret-admin'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    import auth
    auth.TOKENS_PATH = tmp_path / 'tokens.json'
    with auth._lock:
        auth._cache = None
    yield


def _make_handler(
    method='GET',
    path='/api/tokens',
    body=b'',
    bearer=ADMIN_TOKEN,
    config_override=None,
    client_ip='127.0.0.1',
):
    """Minimal mock handler compatible with handle_admin_request and handle_request."""
    handler = MagicMock()
    handler.command = method
    handler.path = path
    handler.client_address = (client_ip, 9000)
    handler._body = body
    handler.config = config_override or {
        'adminToken': ADMIN_TOKEN,
        'services': [],
        'modelAliases': {},
    }
    handler.headers = {'Authorization': f'Bearer {bearer}'} if bearer else {}

    sent = {'status': None, 'headers': {}, 'body': b''}

    handler.send_response.side_effect  = lambda code: sent.update(status=code)
    handler.send_header.side_effect    = lambda k, v: sent['headers'].update({k.lower(): v})
    handler.end_headers.side_effect    = lambda: None
    handler.wfile.write.side_effect    = lambda d: sent.update(body=sent['body'] + d)
    handler.send_error.side_effect     = lambda code, msg='': sent.update(
        status=code, body=msg.encode() if isinstance(msg, str) else msg
    )

    handler._sent = sent
    return handler


def _json(handler) -> dict:
    return json.loads(handler._sent['body'])


# ============================================================
# Admin API — authentication
# ============================================================

class TestAdminAuth:

    def test_missing_authorization_header_returns_401(self):
        from plugins.admin_api import handle_admin_request
        h = _make_handler(bearer=None)
        handle_admin_request(h)
        assert h._sent['status'] == 401

    def test_wrong_token_returns_403(self):
        from plugins.admin_api import handle_admin_request
        h = _make_handler(bearer='wrong-token')
        handle_admin_request(h)
        assert h._sent['status'] == 403

    def test_correct_token_is_allowed(self):
        from plugins.admin_api import handle_admin_request
        h = _make_handler(method='GET', path='/api/tokens')
        handle_admin_request(h)
        assert h._sent['status'] == 200

    def test_unconfigured_admin_token_returns_500(self):
        from plugins.admin_api import handle_admin_request
        h = _make_handler(config_override={'adminToken': '', 'services': [], 'modelAliases': {}})
        handle_admin_request(h)
        assert h._sent['status'] == 500

    def test_rejected_auth_is_audited(self):
        from plugins.admin_api import handle_admin_request
        with patch('audit.log') as mock_log:
            h = _make_handler(bearer='bad')
            handle_admin_request(h)
            events = [c[0][0] for c in mock_log.call_args_list]
            assert 'admin.auth.rejected' in events


# ============================================================
# Admin API — GET /api/tokens
# ============================================================

class TestAdminListTokens:

    def test_empty_list_on_fresh_store(self):
        from plugins.admin_api import handle_admin_request
        h = _make_handler(method='GET', path='/api/tokens')
        handle_admin_request(h)
        assert h._sent['status'] == 200
        assert _json(h) == {'tokens': []}

    def test_lists_all_created_tokens(self):
        import auth
        from plugins.admin_api import handle_admin_request
        auth.create_token('app-a')
        auth.create_token('app-b')
        h = _make_handler(method='GET', path='/api/tokens')
        handle_admin_request(h)
        labels = {t['label'] for t in _json(h)['tokens']}
        assert labels == {'app-a', 'app-b'}

    def test_list_is_audited(self):
        from plugins.admin_api import handle_admin_request
        with patch('audit.log') as mock_log:
            h = _make_handler(method='GET', path='/api/tokens')
            handle_admin_request(h)
            assert 'admin.tokens.list' in [c[0][0] for c in mock_log.call_args_list]


# ============================================================
# Admin API — POST /api/tokens
# ============================================================

class TestAdminCreateToken:

    def test_creates_token_with_label(self):
        from plugins.admin_api import handle_admin_request
        h = _make_handler(method='POST', path='/api/tokens', body=b'{"label": "my-app"}')
        handle_admin_request(h)
        assert h._sent['status'] == 201
        body = _json(h)
        assert body['label'] == 'my-app'
        assert body['revoked'] is False

    def test_returned_id_is_uuid(self):
        import re
        from plugins.admin_api import handle_admin_request
        h = _make_handler(method='POST', path='/api/tokens', body=b'{"label": "uuid-check"}')
        handle_admin_request(h)
        assert re.match(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            _json(h)['id'],
        )

    def test_missing_label_returns_400(self):
        from plugins.admin_api import handle_admin_request
        h = _make_handler(method='POST', path='/api/tokens', body=b'{}')
        handle_admin_request(h)
        assert h._sent['status'] == 400

    def test_invalid_json_returns_400(self):
        from plugins.admin_api import handle_admin_request
        h = _make_handler(method='POST', path='/api/tokens', body=b'not json')
        handle_admin_request(h)
        assert h._sent['status'] == 400

    def test_creation_is_audited(self):
        from plugins.admin_api import handle_admin_request
        with patch('audit.log') as mock_log:
            h = _make_handler(method='POST', path='/api/tokens', body=b'{"label": "audit-test"}')
            handle_admin_request(h)
            assert 'admin.token.created' in [c[0][0] for c in mock_log.call_args_list]

    def test_new_token_is_immediately_valid(self):
        import auth
        from plugins.admin_api import handle_admin_request
        h = _make_handler(method='POST', path='/api/tokens', body=b'{"label": "live-check"}')
        handle_admin_request(h)
        assert auth.validate_token(_json(h)['id']) is not None


# ============================================================
# Admin API — DELETE /api/tokens/<id>
# ============================================================

class TestAdminRevokeToken:

    def test_revoke_existing_returns_200(self):
        import auth
        from plugins.admin_api import handle_admin_request
        record = auth.create_token('doomed')
        h = _make_handler(method='DELETE', path=f'/api/tokens/{record["id"]}')
        handle_admin_request(h)
        assert h._sent['status'] == 200

    def test_revoked_token_no_longer_valid(self):
        import auth
        from plugins.admin_api import handle_admin_request
        record = auth.create_token('kill-me')
        h = _make_handler(method='DELETE', path=f'/api/tokens/{record["id"]}')
        handle_admin_request(h)
        assert auth.validate_token(record['id']) is None

    def test_revoke_nonexistent_returns_404(self):
        from plugins.admin_api import handle_admin_request
        h = _make_handler(method='DELETE',
                          path='/api/tokens/00000000-0000-0000-0000-000000000000')
        handle_admin_request(h)
        assert h._sent['status'] == 404

    def test_revocation_is_audited(self):
        import auth
        from plugins.admin_api import handle_admin_request
        record = auth.create_token('audit-revoke')
        with patch('audit.log') as mock_log:
            h = _make_handler(method='DELETE', path=f'/api/tokens/{record["id"]}')
            handle_admin_request(h)
            assert 'admin.token.revoked' in [c[0][0] for c in mock_log.call_args_list]


# ============================================================
# Admin API — unknown routes / methods
# ============================================================

class TestAdminRouting:

    def test_unknown_path_returns_404(self):
        from plugins.admin_api import handle_admin_request
        h = _make_handler(method='GET', path='/api/unknown')
        handle_admin_request(h)
        assert h._sent['status'] == 404

    def test_put_on_tokens_returns_405(self):
        from plugins.admin_api import handle_admin_request
        h = _make_handler(method='PUT', path='/api/tokens')
        handle_admin_request(h)
        assert h._sent['status'] == 405


# ============================================================
# Proxy — path guard
# ============================================================

class TestPathGuard:

    def _run(self, path, method='GET'):
        from proxy import ProxyHandler
        h = MagicMock()
        h.command = method
        h.path = path
        h.client_address = ('127.0.0.1', 9000)
        h.headers = {'Content-Length': '0'}
        h.rfile.read = MagicMock(return_value=b'')
        h.config = {'services': [], 'modelAliases': {}, 'adminToken': ADMIN_TOKEN}
        h.send_error = MagicMock()
        ProxyHandler.handle_request(h)
        return h

    def test_root_returns_404(self):
        h = self._run('/')
        h.send_error.assert_called_once_with(404, 'Not found')

    def test_arbitrary_path_returns_404(self):
        h = self._run('/metrics')
        h.send_error.assert_called_once_with(404, 'Not found')

    def test_health_endpoint_returns_404(self):
        h = self._run('/health')
        h.send_error.assert_called_once_with(404, 'Not found')

    def test_v1_path_passes_guard(self):
        """Path guard passes /v1; auth check fires next (401, not path 404)."""
        h = self._run('/v1/models')
        if h.send_error.called:
            assert h.send_error.call_args[0] != (404, 'Not found')

    def test_v1beta_path_passes_guard(self):
        h = self._run('/v1beta/models')
        if h.send_error.called:
            assert h.send_error.call_args[0] != (404, 'Not found')

    def test_api_path_dispatches_to_admin(self):
        with patch('plugins.admin_api.handle_admin_request') as mock_admin:
            h = self._run('/api/tokens')
            mock_admin.assert_called_once()


# ============================================================
# Proxy — client token auth middleware
# ============================================================

class TestProxyClientAuth:

    def _run_v1(self, bearer=None, extra_services=None):
        h = MagicMock()
        h.command = 'GET'
        h.path = '/v1/models'
        h.client_address = ('127.0.0.1', 1234)
        h.config = {'services': extra_services or [], 'modelAliases': {},
                    'adminToken': ADMIN_TOKEN}
        h.headers = {'Content-Length': '0'}
        if bearer:
            h.headers['Authorization'] = f'Bearer {bearer}'
        h.rfile.read = MagicMock(return_value=b'')
        h.send_error = MagicMock()
        h.find_services_for_route = MagicMock(return_value=[])
        return h

    def test_no_auth_header_returns_401(self):
        from proxy import ProxyHandler
        h = self._run_v1()
        ProxyHandler.handle_request(h)
        h.send_error.assert_called_once_with(401, 'Unauthorized')

    def test_invalid_token_returns_401(self):
        from proxy import ProxyHandler
        h = self._run_v1(bearer='not-a-real-token')
        ProxyHandler.handle_request(h)
        h.send_error.assert_called_once_with(401, 'Unauthorized')

    def test_valid_token_passes_auth(self):
        import auth
        from proxy import ProxyHandler
        record = auth.create_token('proxy-test')
        h = self._run_v1(bearer=record['id'])
        ProxyHandler.handle_request(h)
        codes = [c[0][0] for c in h.send_error.call_args_list]
        assert 401 not in codes

    def test_revoked_token_returns_401(self):
        import auth
        from proxy import ProxyHandler
        record = auth.create_token('revoke-proxy')
        auth.revoke_token(record['id'])
        h = self._run_v1(bearer=record['id'])
        ProxyHandler.handle_request(h)
        h.send_error.assert_called_once_with(401, 'Unauthorized')

    def test_successful_auth_emits_api_request_audit(self):
        import auth
        from proxy import ProxyHandler
        record = auth.create_token('audit-proxy')
        with patch('audit.log') as mock_log:
            h = self._run_v1(bearer=record['id'])
            ProxyHandler.handle_request(h)
            assert 'api.request' in [c[0][0] for c in mock_log.call_args_list]

    def test_failed_auth_emits_auth_rejected_audit(self):
        from proxy import ProxyHandler
        with patch('audit.log') as mock_log:
            h = self._run_v1(bearer='garbage')
            ProxyHandler.handle_request(h)
            assert 'auth.rejected' in [c[0][0] for c in mock_log.call_args_list]

    def test_missing_token_reason_is_missing_token(self):
        from proxy import ProxyHandler
        with patch('audit.log') as mock_log:
            h = self._run_v1()   # no bearer at all
            ProxyHandler.handle_request(h)
            rejected = [c for c in mock_log.call_args_list if c[0][0] == 'auth.rejected']
            assert rejected and rejected[0][1]['reason'] == 'missing_token'

    def test_bad_token_reason_is_invalid_token(self):
        from proxy import ProxyHandler
        with patch('audit.log') as mock_log:
            h = self._run_v1(bearer='bad-value')
            ProxyHandler.handle_request(h)
            rejected = [c for c in mock_log.call_args_list if c[0][0] == 'auth.rejected']
            assert rejected and rejected[0][1]['reason'] == 'invalid_token'

    def test_client_bearer_stripped_before_forwarding(self):
        """The client's UUID token must never reach the backend service.

        forward_request builds its own headers dict, explicitly dropping
        'authorization'.  We call it directly and intercept conn.request
        to see exactly what headers would be sent to the backend.
        """
        import auth
        from proxy import ProxyHandler

        record = auth.create_token('strip-test')
        svc = {
            'name': 'test-svc', 'baseUrl': 'http://localhost:9999',
            'health': '/v1/models', 'routes': ['/v1/models'],
            'maxConcurrent': 1, 'queueTimeoutSeconds': 5,
        }

        h = MagicMock()
        h.command = 'GET'
        h.path = '/v1/models'
        h.headers = {
            'Authorization': f'Bearer {record["id"]}',
            'Content-Type': 'application/json',
        }
        h._body = None

        captured = {}

        mock_conn = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.getheaders.return_value = []
        mock_response.getheader.return_value = 'application/json'
        mock_response.__iter__ = MagicMock(return_value=iter([]))
        mock_conn.getresponse.return_value = mock_response
        mock_conn.request.side_effect = lambda method, path, body, headers: captured.update(headers)

        with patch('http.client.HTTPConnection', return_value=mock_conn):
            # Stub out the write side so it doesn't blow up
            h.send_response = MagicMock()
            h.send_header = MagicMock()
            h.end_headers = MagicMock()
            h.wfile = MagicMock()
            h.connection = MagicMock()
            ProxyHandler.forward_request(h, svc)

        auth_sent = captured.get('Authorization', '')
        assert record['id'] not in auth_sent


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
