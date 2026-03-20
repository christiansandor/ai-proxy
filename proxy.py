import http.client
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib
import pkgutil
import socket
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import yaml

# Load configuration from config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# service_name -> (is_healthy: bool, model_ids: set|None, timestamp: float)
_health_cache: dict[str, tuple[bool, set | None, float]] = {}

HEALTH_CACHE_TTL = 30  # seconds

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
        module = importlib.import_module(f'plugins.{name}')
        plugin = getattr(module, 'PLUGIN', None)
        if plugin is not None:
            _plugins.append(plugin)
            print(f"  ↳ plugin loaded: plugins.{name}", flush=True)

# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ProxyHandler(BaseHTTPRequestHandler):
    # Expose shared state to plugins via the handler instance
    config = config

    def do_GET(self):    self.handle_request()
    def do_POST(self):   self.handle_request()
    def do_PUT(self):    self.handle_request()
    def do_DELETE(self): self.handle_request()

    def handle_request(self):
        self.close_connection = True
        print(f"→ {self.command} {self.path}", flush=True)

        # ── Read body once up-front ──
        # Must happen before plugins or forwarding so the stream isn't consumed
        # twice, and so we can inspect the model field for routing.
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Apply alias rewrite before routing so model_id reflects the real name
        body = apply_model_alias(body, config.get('modelAliases', {}))

        # Extract model ID for model-aware service selection
        model_id = None
        if body:
            try:
                model_id = json.loads(body).get('model')
            except Exception:
                pass

        # Stash for use by forward_request (avoids re-reading the socket)
        self._body = body

        # ── Try plugins first ──
        for plugin in _plugins:
            if plugin.match(self.command, self.path):
                plugin.handle(self)
                return

        # ── Normal proxy routing ──
        services = self.find_services_for_route(self.path)
        if not services:
            print(f"✗ No service for {self.path}", flush=True)
            self.send_error(404, "No service found for this route")
            return

        for service in services:
            if self.is_service_healthy(service, model_id=model_id):
                print(f"✓ Routing to {service['name']}", flush=True)
                if self.forward_request(service):
                    print(f"✓ Done {self.command} {self.path}", flush=True)
                    return
                print(f"✗ {service['name']} failed, trying next...", flush=True)
                _health_cache.pop(service['name'], None)

        print(f"✗ No healthy service for {self.path}", flush=True)
        self.send_error(503, "No healthy service available for this route")

    def find_services_for_route(self, path):
        return [
            s for s in config['services']
            if any(route in path for route in s.get('routes', []))
        ]

    def is_service_healthy(self, service, model_id: str | None = None) -> bool:
        """
        Returns True when:
          1. The service passes its health check, AND
          2. If model_id is given, that model is present in the service's model list.

        Results are cached for HEALTH_CACHE_TTL seconds.  The model list comes
        for free from the /v1/models health endpoint, so model filtering has no
        extra network cost.
        """
        name = service['name']
        now = time.time()

        cached = _health_cache.get(name)
        if cached is not None:
            is_healthy, model_ids, checked_at = cached
            if now - checked_at < HEALTH_CACHE_TTL:
                if not is_healthy:
                    return False
                if model_id and model_ids is not None and model_id not in model_ids:
                    print(
                        f"  [routing] {name} is healthy but lacks model {model_id!r}",
                        flush=True,
                    )
                    return False
                return True

        is_healthy, model_ids = self._check_health(service)
        _health_cache[name] = (is_healthy, model_ids, now)

        if not is_healthy:
            return False
        if model_id and model_ids is not None and model_id not in model_ids:
            print(
                f"  [routing] {name} is healthy but lacks model {model_id!r}",
                flush=True,
            )
            return False
        return True

    def _check_health(self, service) -> tuple[bool, set | None]:
        """
        Fetches the service health URL and returns (is_healthy, model_ids).

        When the health endpoint is /v1/models (the common case) the response
        body is parsed to extract model IDs so they can be used for routing
        without a second request.  If parsing fails or the endpoint is not a
        model listing, model_ids is returned as None (meaning "unknown").
        """
        try:
            health_url = f"{service['baseUrl']}{service['health']}"
            print(f"→ Health check: {health_url}", flush=True)
            req = urllib.request.Request(health_url)
            if 'token' in service:
                req.add_header('Authorization', f"Bearer {service['token']}")
            response = urllib.request.urlopen(req, timeout=5)
            code = response.getcode()
            print(f"→ Health check result: {code}", flush=True)

            if code != 200:
                return False, None

            # Try to extract model IDs from the response body.
            # Works whenever health == /v1/models; harmless otherwise.
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
                pass  # body isn't a model listing — that's fine

            return True, model_ids

        except Exception as e:
            print(f"✗ Health check failed for {service['name']}: {e}", flush=True)
            return False, None

    def forward_request(self, service):
        parsed_base = urlparse(service['baseUrl'])
        host = parsed_base.hostname
        port = parsed_base.port or (443 if parsed_base.scheme == 'https' else 80)

        # Body was read and alias-rewritten in handle_request
        body = self._body

        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ('host', 'content-length', 'transfer-encoding')
        }
        if 'token' in service and service['token'] != 'not-needed':
            headers['Authorization'] = f"Bearer {service['token']}"

        conn = None
        try:
            conn = (
                http.client.HTTPSConnection(host, port, timeout=600)
                if parsed_base.scheme == 'https'
                else http.client.HTTPConnection(host, port, timeout=600)
            )
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()

            self.send_response(response.status)
            for k, v in response.getheaders():
                if k.lower() not in ('transfer-encoding',):
                    self.send_header(k, v)
            self.send_header('Connection', 'close')
            self.send_header('X-Ai-Proxy-Server', service['name'])
            self.end_headers()

            content_type = response.getheader('content-type', '')
            is_streaming = 'text/event-stream' in content_type or 'application/json' in content_type

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

    server_address = ('', 8080)
    httpd = HTTPServer(server_address, lambda request, client_address, server: ProxyHandler(request, client_address, server))
    print("Proxy server starting on http://localhost:8080")
    print("Configured services:")
    for service in config['services']:
        print(f"  - {service['name']} ({service['baseUrl']})")
    httpd.serve_forever()


if __name__ == '__main__':
    run_server()
