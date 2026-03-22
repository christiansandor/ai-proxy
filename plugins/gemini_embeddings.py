"""
Plugin: Gemini Embedding Bridge
================================
Intercepts the two Gemini embedding endpoints that AFFiNE (and similar tools)
expect, and converts them to/from the OpenAI-compatible /embeddings format.

Activated for any service that has `geminiEmbeddingPath` set in config.yaml:

    services:
      - name: Mac
        baseUrl: http://100.64.0.1:1234
        token: not-needed
        health: /v1/models
        geminiEmbeddingPath: /v1/embeddings   # ← enables this plugin
        routes:
          - /v1/models
          - /v1/chat/completions
          - /v1/embeddings

Handled routes
--------------
  GET  /v1beta/models                          → static model list
  POST /v1beta/models/<model>:embedContent     → Gemini→OpenAI→Gemini bridge
"""

import http.client
import json
import re
from urllib.parse import urlparse

from plugins import ProxyPlugin

GEMINI_EMBED_RE = re.compile(r'^/v1beta/models/([^/:?]+):embedContent')


def _gemini_request_to_openai(model: str, gemini_body: dict) -> dict:
    parts = gemini_body.get('content', {}).get('parts', [])
    texts = [p['text'] for p in parts if 'text' in p]
    print(f"→ [gemini] Extracted {len(texts)} text part(s)", flush=True)
    return {
        'model': f'gemini/{model}',
        'input': texts[0] if len(texts) == 1 else texts,
    }


def _openai_response_to_gemini(openai_body: dict) -> dict:
    embeddings = [item['embedding'] for item in openai_body.get('data', [])]
    if len(embeddings) == 1:
        return {'embedding': {'values': embeddings[0]}}
    return {'embedding': [{'values': v} for v in embeddings]}


def _extract_bearer_token(headers) -> str | None:
    auth = headers.get('Authorization') or headers.get('authorization')
    if auth:
        m = re.match(r'^(?:Bearer|Api-Key|ApiKey|Key)\s+(.+)$', auth, re.IGNORECASE)
        return m.group(1) if m else auth
    for key_header in ('x-goog-api-key', 'x-api-key', 'apikey'):
        val = headers.get(key_header)
        if val:
            return val
    return None


def _post_json(service, path: str, payload: dict, headers: dict):
    """POST JSON to a service. Returns (parsed_response_dict, status) or (None, None)."""
    parsed_base = urlparse(service['baseUrl'])
    host = parsed_base.hostname
    port = parsed_base.port or (443 if parsed_base.scheme == 'https' else 80)
    body = json.dumps(payload).encode()
    headers = {**headers, 'Content-Length': str(len(body))}

    conn = None
    try:
        conn = (
            http.client.HTTPSConnection(host, port, timeout=600)
            if parsed_base.scheme == 'https'
            else http.client.HTTPConnection(host, port, timeout=600)
        )
        conn.request('POST', path, body=body, headers=headers)
        response = conn.getresponse()
        return json.loads(response.read()), response.status
    except Exception as e:
        print(f"✗ [gemini] _post_json failed for {service['name']}: {e}", flush=True)
        return None, None
    finally:
        if conn:
            conn.close()


def _send_json(handler, status: int, payload: dict, extra_headers: dict = None):
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json')
    handler.send_header('Content-Length', str(len(body)))
    for k, v in (extra_headers or {}).items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(body)


class GeminiEmbeddingPlugin(ProxyPlugin):

    def match(self, method: str, path: str) -> bool:
        if method == 'GET' and path.rstrip('/') == '/v1beta/models':
            return True
        if method == 'POST' and GEMINI_EMBED_RE.match(path):
            return True
        return False

    def handle(self, handler) -> None:
        if handler.command == 'GET':
            self._handle_models(handler)
        else:
            m = GEMINI_EMBED_RE.match(handler.path)
            self._handle_embed(handler, m.group(1))

    # ── /v1beta/models ───────────────────────────────────────────────────────

    def _handle_models(self, handler):
        capable = [s for s in handler.config['services'] if s.get('geminiEmbeddingPath')]
        base_names = {'gemini-embedding-001'} if capable else set()
        aliases = handler.config.get('modelAliases', {})
        alias_names = {
            k[len('gemini/'):] for k in aliases
            if k.startswith('gemini/') and capable
        }
        all_names = base_names | alias_names
        models = [{'name': n, 'type': 'embedding'} for n in sorted(all_names)]
        _send_json(handler, 200, {'models': models})
        print(f"✓ [gemini] Served /v1beta/models ({len(models)} model(s))", flush=True)

    # ── /v1beta/models/:model:embedContent ───────────────────────────────────

    def _handle_embed(self, handler, model: str):
        capable = [s for s in handler.config['services'] if s.get('geminiEmbeddingPath')]
        if not capable:
            handler.send_error(404, "No service configured with geminiEmbeddingPath")
            return

        content_length = int(handler.headers.get('Content-Length', 0))
        raw_body = handler.rfile.read(content_length) if content_length > 0 else b'{}'

        try:
            gemini_body = json.loads(raw_body)
        except json.JSONDecodeError as e:
            handler.send_error(400, f"Invalid JSON body: {e}")
            return

        openai_payload = _gemini_request_to_openai(model, gemini_body)
        incoming_token = _extract_bearer_token(handler.headers)

        aliases = handler.config.get('modelAliases', {})
        original_model = openai_payload['model']
        if original_model in aliases:
            openai_payload['model'] = aliases[original_model]
            print(f"→ [alias] {original_model!r} → {openai_payload['model']!r}", flush=True)

        from proxy import _get_semaphore, _invalidate_health_cache

        # Separate healthy services into free and locked, mirroring the
        # four-pass routing logic in handle_request so that Gemini embedding
        # requests respect the per-service semaphore and never trigger a
        # mid-generation model eviction on LM Studio.
        healthy = [s for s in capable if handler.is_service_healthy(s)]
        if not healthy:
            handler.send_error(503, "No healthy service with geminiEmbeddingPath available")
            return

        def _forward(service):
            """Attempt the embedding call. Returns True on success."""
            service_token = service.get('token')
            token = (
                incoming_token
                or (service_token if service_token != 'not-needed' else None)
            )
            forward_headers = {'Content-Type': 'application/json'}
            if token:
                forward_headers['Authorization'] = f'Bearer {token}'

            target_path = service['geminiEmbeddingPath']
            print(f"→ [gemini] Forwarding to {service['name']} at {target_path}", flush=True)

            openai_body, status = _post_json(service, target_path, openai_payload, forward_headers)

            if openai_body is None:
                print(f"✗ [gemini] {service['name']} failed, trying next...", flush=True)
                _invalidate_health_cache(service['name'])
                return False

            if status != 200:
                _send_json(handler, status, openai_body, {'X-Ai-Proxy-Server': service['name']})
                return True   # response sent (even if an error), stop iterating

            _send_json(
                handler, 200,
                _openai_response_to_gemini(openai_body),
                {'X-Ai-Proxy-Server': service['name']},
            )
            print(f"✓ [gemini] Done embedContent for model={model}", flush=True)
            return True

        # Pass 1 — try free services first (non-blocking acquire)
        remaining = list(healthy)
        for service in list(remaining):
            sem = _get_semaphore(service)
            if sem.acquire(blocking=False):
                try:
                    if _forward(service):
                        return
                finally:
                    sem.release()
                remaining.remove(service)   # forward failed, drop from Pass 2

        if not remaining:
            handler.send_error(503, "No healthy service with geminiEmbeddingPath available")
            return

        # Pass 2 — queue behind each locked service in turn
        for service in remaining:
            timeout = service.get('queueTimeoutSeconds', 300)
            print(
                f"→ [gemini] All services busy, queuing behind {service['name']} "
                f"(timeout={timeout}s)",
                flush=True,
            )
            sem = _get_semaphore(service)
            if not sem.acquire(timeout=timeout):
                # Pass 4 — timed out
                print(f"✗ [gemini] Queue timeout ({timeout}s) waiting for {service['name']}", flush=True)
                handler.send_error(429, "Service busy, please retry later")
                return
            try:
                if _forward(service):
                    return
            finally:
                sem.release()

        handler.send_error(503, "No healthy service with geminiEmbeddingPath available")


PLUGIN = GeminiEmbeddingPlugin()
