import http.client
import json
import os
import sys
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib
import pkgutil
import socket
import time
import urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import yaml

import auth
import audit

_CONFIG_PATH = 'config.yaml'

# Load configuration from config.yaml
with open(_CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# ---------------------------------------------------------------------------
# Hot-reload watcher
# ---------------------------------------------------------------------------

def _reload_config() -> None:
    """
    Re-read config.yaml and update the shared ``config`` dict *in-place* so
    that every existing reference (ProxyHandler.config, plugins, etc.) picks
    up the new values without a restart.

    Side-effects on successful reload:
      • health cache cleared  — stale service state is no longer valid
      • model aggregator cache cleared — model list / aliases may have changed
    """
    try:
        with open(_CONFIG_PATH, 'r') as f:
            new_config = yaml.safe_load(f)
        config.clear()
        config.update(new_config)
        print("✓ [config] config.yaml reloaded", flush=True)
    except Exception as e:
        print(f"✗ [config] Reload failed: {e}", flush=True)
        return  # keep the old config intact

    # Clear health cache — services or URLs may have changed
    with _health_cache_lock:
        _health_cache.clear()
    print("  [config] Health cache cleared", flush=True)

    # Invalidate the model aggregator's cache so the next /v1/models request
    # fetches fresh data reflecting any service / alias changes.
    try:
        import plugins.model_aggregator as ma
        ma.reset_cache()
    except Exception as e:
        print(f"  [config] Could not reset model cache: {e}", flush=True)


def _watch_config(interval: float = 2.0) -> None:
    """
    Background daemon thread: polls config.yaml every *interval* seconds and
    calls _reload_config() when the file's mtime changes.
    """
    try:
        last_mtime = os.path.getmtime(_CONFIG_PATH)
    except OSError:
        last_mtime = 0.0

    while True:
        time.sleep(interval)
        try:
            mtime = os.path.getmtime(_CONFIG_PATH)
            if mtime != last_mtime:
                print(f"→ [config] Change detected, reloading config.yaml…", flush=True)
                _reload_config()
                last_mtime = mtime
        except Exception as e:
            print(f"✗ [config] Watch error: {e}", flush=True)

# ---------------------------------------------------------------------------
# Health cache
# ---------------------------------------------------------------------------

HEALTH_CACHE_TTL_HEALTHY   = 30   # seconds — re-check healthy services
HEALTH_CACHE_TTL_UNHEALTHY =  5   # seconds — retry unhealthy services sooner

# service_name -> (is_healthy: bool, model_ids: set|None, timestamp: float)
_health_cache: dict[str, tuple[bool, set | None, float]] = {}
_health_cache_lock = threading.Lock()
_health_check_in_progress: set[str] = set()   # names currently being checked


def _invalidate_health_cache(name: str) -> None:
    """Thread-safe removal of a service from the health cache."""
    with _health_cache_lock:
        _health_cache.pop(name, None)


# ---------------------------------------------------------------------------
# Per-service semaphores
# ---------------------------------------------------------------------------
# Each service gets a Semaphore(maxConcurrent) that gates how many requests
# may be forwarded simultaneously.  For LM Studio this is 1 (serial queue)
# so that model eviction mid-generation can never happen.
# Semaphores are created lazily and never destroyed.

_service_semaphores: dict[str, threading.Semaphore] = {}
_service_semaphores_lock = threading.Lock()


def _get_semaphore(service: dict) -> threading.Semaphore:
    name = service['name']
    with _service_semaphores_lock:
        if name not in _service_semaphores:
            max_concurrent = service.get('maxConcurrent', 1)
            _service_semaphores[name] = threading.Semaphore(max_concurrent)
        return _service_semaphores[name]


# ---------------------------------------------------------------------------
# Model alias helpers
# ---------------------------------------------------------------------------

def apply_model_alias(body: bytes, aliases: dict) -> bytes:
    """
    If the JSON body contains a "model" key whose value matches an alias,
    rewrite it to the target model name and re-encode the body.
    Returns the original bytes unchanged on any error or miss.
    """
    if not body or not aliases:
        return body
    try:
        data = json.loads(body)
        original = data.get('model')
        if original and original in aliases:
            data['model'] = aliases[original]
            print(f"→ [alias] {original!r} → {data['model']!r}", flush=True)
            return json.dumps(data).encode()
    except Exception:
        pass
    return body

# ---------------------------------------------------------------------------
# Plugin loader
# ---------------------------------------------------------------------------

_plugins = []

def _load_plugins():
    import plugins as pkg
    for _, name, _ in pkgutil.iter_modules(pkg.__path__):
        # admin_api is not a plugin — it's called directly from handle_request
        if name == 'admin_api':
            continue
        module = importlib.import_module(f'plugins.{name}')
        plugin = getattr(module, 'PLUGIN', None)
        if plugin is not None:
            _plugins.append(plugin)
            print(f"  ↳ plugin loaded: plugins.{name}", flush=True)

# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ProxyHandler(BaseHTTPRequestHandler):
    """Proxy HTTP request handler."""
    # Expose shared state to plugins via the handler instance
    config = config

    def do_GET(self):    self.handle_request()
    def do_POST(self):   self.handle_request()
    def do_PUT(self):    self.handle_request()
    def do_DELETE(self): self.handle_request()

    def handle_request(self):
        self.close_connection = True
        self._client_label = None   # set after auth; used in subsequent logs
        print(f"→ {self.command} {self.path}", flush=True)  # unauthenticated at this point; see _who log below

        # ── Read body once up-front ──────────────────────────────────────────
        content_length = int(self.headers.get('Content-Length', 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else None
        self._body = raw_body   # set early; admin handler reads this directly

        # ── Path restriction: only /v1*, /v1beta/, /api/ ─────────────────────
        p = self.path
        if not (
            p.startswith('/v1/') or p == '/v1'
            or p.startswith('/v1beta/')
            or p.startswith('/api/')
        ):
            self.send_error(404, "Not found")
            return

        # ── Admin API — own auth, no client token required ───────────────────
        if p.startswith('/api/'):
            from plugins.admin_api import handle_admin_request
            handle_admin_request(self)
            return

        # ── Client token auth (/v1* and /v1beta/) ────────────────────────────
        auth_header = self.headers.get('Authorization', '')
        token_value = (
            auth_header[len('Bearer '):].strip()
            if auth_header.startswith('Bearer ')
            else None
        )
        token_record = auth.validate_token(token_value) if token_value else None

        if not token_record:
            audit.log(
                'auth.rejected',
                method=self.command,
                path=self.path,
                ip=self.client_address[0],
                reason='invalid_token' if token_value else 'missing_token',
            )
            self.send_error(401, "Unauthorized")
            return

        self._auth_token = token_record
        self._client_label = token_record['label']
        print(f"→ {self._who} {self.command} {self.path}", flush=True)
        audit.log(
            'api.request',
            method=self.command,
            path=self.path,
            ip=self.client_address[0],
            token_id=token_record['id'],
            label=token_record['label'],
        )

        # ── Apply model alias (after auth, before routing) ───────────────────
        body = apply_model_alias(raw_body, config.get('modelAliases', {}))
        self._body = body

        model_id = None
        if body:
            try:
                model_id = json.loads(body).get('model')
            except Exception:
                pass

        # ── Try plugins first ────────────────────────────────────────────────
        for plugin in _plugins:
            if plugin.match(self.command, self.path):
                plugin.handle(self)
                return

        # ── Normal proxy routing ─────────────────────────────────────────────
        services = self.find_services_for_route(self.path)
        if not services:
            print(f"✗ {self._who} No service for {self.path}", flush=True)
            self.send_error(404, "No service found for this route")
            return

        # Compute healthy candidates once
        healthy = [s for s in services if self.is_service_healthy(s, model_id=model_id)]

        # ── Pass 3: nothing healthy at all ──
        if not healthy:
            print(f"✗ {self._who} No healthy service for {self.path}", flush=True)
            self.send_error(503, "No healthy service available for this route")
            return

        # ── Pass 1: healthy + free ──
        remaining = list(healthy)
        for service in list(remaining):
            sem = _get_semaphore(service)
            if sem.acquire(blocking=False):
                print(f"✓ {self._who} Routing to {service['name']} (free)", flush=True)
                try:
                    if self.forward_request(service):
                        print(f"✓ {self._who} Done {self.command} {self.path}", flush=True)
                        return
                finally:
                    sem.release()
                print(f"✗ {self._who} {service['name']} failed forwarding, invalidating", flush=True)
                _invalidate_health_cache(service['name'])
                remaining.remove(service)

        # ── Pass 3 (second check): all free services failed, nothing left ──
        if not remaining:
            print(f"✗ {self._who} All services exhausted for {self.path}", flush=True)
            self.send_error(503, "No healthy service available for this route")
            return

        # ── Pass 2: healthy + locked — queue behind each in turn ──
        for service in remaining:
            timeout = service.get('queueTimeoutSeconds', 300)
            print(
                f"→ {self._who} All services busy, queuing behind {service['name']} "
                f"(timeout={timeout}s)",
                flush=True,
            )

            sem = _get_semaphore(service)
            acquired = sem.acquire(timeout=timeout)

            # ── Pass 4: timed out waiting for a slot ──
            if not acquired:
                print(
                    f"✗ {self._who} Queue timeout ({timeout}s) waiting for {service['name']}",
                    flush=True,
                )
                self.send_error(429, "Service busy, please retry later")
                return

            print(f"✓ {self._who} Routing to {service['name']} (queued)", flush=True)
            try:
                if self.forward_request(service):
                    print(f"✓ {self._who} Done {self.command} {self.path}", flush=True)
                    return
            finally:
                sem.release()

            print(f"✗ {self._who} {service['name']} failed after queuing, trying next...", flush=True)
            _invalidate_health_cache(service['name'])

        print(f"✗ {self._who} All queued services failed for {self.path}", flush=True)
        self.send_error(503, "No healthy service available for this route")

    @property
    def _who(self) -> str:
        """Short client label for log lines, e.g. '[my-app]'."""
        label = getattr(self, '_client_label', None)
        return f'[{label}]' if label else '[unauthenticated]'

    def find_services_for_route(self, path):
        return [
            s for s in config['services']
            if any(route in path for route in s.get('routes', []))
        ]

    def is_service_healthy(self, service, model_id: str | None = None) -> bool:
        """
        Returns True when the service passes its health check AND (if model_id
        is given) that model is present in the service's advertised model list.

        Fail-closed on model list: if the health endpoint did not return a
        parseable model list AND a model_id was requested, we refuse to route
        to this service rather than assuming it can handle any model.  This
        prevents routing to services whose /v1/models endpoint returned an
        unexpected shape (e.g. auth error body, non-standard schema).
        """
        name = service['name']
        now = time.time()

        with _health_cache_lock:
            cached = _health_cache.get(name)
            if cached is not None:
                is_healthy, model_ids, checked_at = cached
                ttl = HEALTH_CACHE_TTL_HEALTHY if is_healthy else HEALTH_CACHE_TTL_UNHEALTHY
                if now - checked_at < ttl:
                    if not is_healthy:
                        return False
                    if model_id:
                        if model_ids is None:
                            print(
                                f"  [routing] {name} returned no model list, "
                                f"skipping for model {model_id!r}",
                                flush=True,
                            )
                            return False
                        if model_id not in model_ids:
                            print(
                                f"  [routing] {name} healthy but lacks model {model_id!r}",
                                flush=True,
                            )
                            return False
                    return True

            if name in _health_check_in_progress:
                if cached is not None:
                    return cached[0]
                return False

            _health_check_in_progress.add(name)

        is_healthy, model_ids = self._check_health(service)

        with _health_cache_lock:
            _health_cache[name] = (is_healthy, model_ids, now)
            _health_check_in_progress.discard(name)

        if not is_healthy:
            return False
        if model_id:
            if model_ids is None:
                print(
                    f"  [routing] {name} returned no model list, "
                    f"skipping for model {model_id!r}",
                    flush=True,
                )
                return False
            if model_id not in model_ids:
                print(
                    f"  [routing] {name} healthy but lacks model {model_id!r}",
                    flush=True,
                )
                return False
        return True

    def _check_health(self, service) -> tuple[bool, set | None]:
        """Fetches the service health URL. Retries once after 0.5s."""
        for attempt in range(2):
            try:
                health_url = f"{service['baseUrl']}{service['health']}"
                print(f"→ Health check: {health_url}", flush=True)
                req = urllib.request.Request(health_url)
                token = service.get('token')
                if token and token != 'not-needed':
                    req.add_header('Authorization', f"Bearer {token}")
                response = urllib.request.urlopen(req, timeout=12)
                code = response.getcode()
                print(f"→ Health check result: {code}", flush=True)

                if code != 200:
                    return False, None

                model_ids: set | None = None
                try:
                    body = json.loads(response.read())
                    data = body.get('data')
                    if isinstance(data, list):
                        model_ids = {m['id'] for m in data if 'id' in m}
                        print(
                            f"  [health] {service['name']} advertises "
                            f"{len(model_ids)} model(s)",
                            flush=True,
                        )
                except Exception:
                    pass

                return True, model_ids

            except Exception as e:
                if attempt < 1:
                    time.sleep(0.5)
                else:
                    print(
                        f"✗ Health check failed for {service['name']} "
                        f"after 2 attempts: {e}",
                        flush=True,
                    )

        return False, None

    def forward_request(self, service) -> bool:
        """
        Forward the request to the given service and stream the response back.

        Returns True if the request was handled (response sent to client),
        False if the backend was unreachable or returned a 5xx error — in
        which case the caller should try the next available service.

        4xx responses ARE forwarded and return True: they indicate a problem
        with the request itself (bad input, auth failure, etc.) rather than a
        service failure, so retrying on another backend would not help.
        """
        parsed_base = urlparse(service['baseUrl'])
        host = parsed_base.hostname
        port = parsed_base.port or (443 if parsed_base.scheme == 'https' else 80)

        body = self._body

        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ('host', 'content-length', 'transfer-encoding',
                                  'authorization')
        }
        token = service.get('token')
        if token and token != 'not-needed':
            headers['Authorization'] = f"Bearer {token}"

        conn = None
        try:
            conn = (
                http.client.HTTPSConnection(host, port, timeout=600)
                if parsed_base.scheme == 'https'
                else http.client.HTTPConnection(host, port, timeout=600)
            )
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()

            # Treat 5xx as a service failure so the router can fall through to
            # the next healthy backend.  4xx means the request itself is at
            # fault and retrying elsewhere won't help, so forward those normally.
            if response.status >= 500:
                print(
                    f"✗ {service['name']} returned {response.status}, "
                    "treating as service failure",
                    flush=True,
                )
                _invalidate_health_cache(service['name'])
                return False

            self.send_response(response.status)
            for k, v in response.getheaders():
                if k.lower() not in ('transfer-encoding',):
                    self.send_header(k, v)
            self.send_header('Connection', 'close')
            self.send_header('X-Ai-Proxy-Server', service['name'])
            self.end_headers()

            content_type = response.getheader('content-type', '')
            is_streaming = (
                'text/event-stream' in content_type
                or 'application/json' in content_type
            )

            try:
                if is_streaming:
                    for line in response:
                        self.wfile.write(line)
                        self.wfile.flush()
                        if b'[DONE]' in line:
                            break
                else:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except BrokenPipeError:
                pass
            except Exception as e:
                print(f"Error during streaming: {e}", flush=True)
            finally:
                try:
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_WR)
                except Exception:
                    pass

            return True

        except Exception as e:
            print(f"✗ Could not reach {service['name']}: {e}", flush=True)
            return False
        finally:
            if conn:
                conn.close()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_server():
    print("Loading plugins...")
    _load_plugins()

    # Start hot-reload watcher — picks up config.yaml changes without restart
    watcher = threading.Thread(target=_watch_config, daemon=True, name="config-watcher")
    watcher.start()
    print(f"→ [config] Watching {_CONFIG_PATH} for changes (poll every 2s)", flush=True)

    print(f"→ [auth]  Client token auth enabled  (tokens: {auth.TOKENS_PATH})", flush=True)
    print(f"→ [audit] Logging to {audit._LOG_DIR}", flush=True)

    server_address = ('', 8080)
    httpd = ThreadingHTTPServer(server_address, ProxyHandler)  # type: ignore[type-arg]
    httpd.daemon_threads = True
    print("Proxy server starting on http://localhost:8080")
    print("Configured services:")
    for service in config['services']:
        print(f"  - {service['name']} ({service['baseUrl']})")
    httpd.serve_forever()


if __name__ == '__main__':
    run_server()
