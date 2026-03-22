"""
Admin API  —  /api/tokens
==========================
Called directly from ProxyHandler.handle_request (not via the plugin loop)
so it can use its own auth independent of client tokens.

Requires ``adminToken`` to be set in config.yaml.

  GET    /api/tokens            list all tokens (including revoked)
  POST   /api/tokens            create a token   body: {"label": "..."}
  DELETE /api/tokens/<uuid>     revoke a token
"""
from __future__ import annotations

import json
import re

_UUID_PAT = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
_UUID_RE  = re.compile(f'^{_UUID_PAT}$')
_ROUTE_RE = re.compile(rf'^/api/tokens(?:/({_UUID_PAT}))?(?:\?.*)?$')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send_json(handler, status: int, payload: object) -> None:
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json')
    handler.send_header('Content-Length', str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _check_admin_auth(handler) -> bool:
    admin_token = (handler.config or {}).get('adminToken', '')
    if not admin_token:
        _send_json(handler, 500, {'error': 'adminToken not configured in config.yaml'})
        return False
    auth_header = handler.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        _send_json(handler, 401, {'error': 'Missing Authorization header'})
        return False
    if auth_header[len('Bearer '):].strip() != admin_token:
        _send_json(handler, 403, {'error': 'Invalid admin token'})
        return False
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def handle_admin_request(handler) -> None:
    """Called directly from ProxyHandler.handle_request for /api/* paths."""
    import auth
    import audit

    ip = handler.client_address[0]

    if not _check_admin_auth(handler):
        audit.log('admin.auth.rejected', method=handler.command,
                  path=handler.path, ip=ip)
        return

    method = handler.command
    m = _ROUTE_RE.match(handler.path)
    if not m:
        _send_json(handler, 404, {'error': 'Unknown admin endpoint'})
        return

    token_id = m.group(1)   # None for /api/tokens, a UUID for /api/tokens/<id>

    # ── GET /api/tokens ──────────────────────────────────────────────────────
    if method == 'GET' and token_id is None:
        tokens = auth.list_tokens()
        _send_json(handler, 200, {'tokens': tokens})
        audit.log('admin.tokens.list', count=len(tokens), ip=ip)
        return

    # ── POST /api/tokens ─────────────────────────────────────────────────────
    if method == 'POST' and token_id is None:
        body = handler._body or b'{}'
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            _send_json(handler, 400, {'error': 'Invalid JSON body'})
            return
        label = str(payload.get('label', '')).strip()
        if not label:
            _send_json(handler, 400, {'error': '"label" is required'})
            return
        record = auth.create_token(label)
        _send_json(handler, 201, record)
        audit.log('admin.token.created', label=label,
                  token_id=record['id'], ip=ip)
        return

    # ── DELETE /api/tokens/<id> ───────────────────────────────────────────────
    if method == 'DELETE' and token_id is not None:
        ok = auth.revoke_token(token_id)
        if ok:
            _send_json(handler, 200, {'ok': True, 'id': token_id})
            audit.log('admin.token.revoked', token_id=token_id, ip=ip)
        else:
            _send_json(handler, 404, {'error': 'Token not found'})
        return

    _send_json(handler, 405, {'error': f'Method {method} not supported here'})
