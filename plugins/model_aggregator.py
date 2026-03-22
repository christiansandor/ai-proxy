# plugins/model_aggregator.py
"""
Plugin: Model Aggregator
========================
Intercepts GET /v1/models and serves a cached union of models that are
available across *all* healthy services that expose the /v1/models route.

Why union?  Because different services may host different models; the client
should see everything that can be routed to, not just the common subset.
The proxy will route each request to the first healthy service that actually
carries the requested model.

Cache
-----
  - Populated lazily on the first request, then refreshed in the background
    every MODEL_CACHE_TTL seconds (default: 300 / 5 minutes).
  - Stale-while-revalidate: an expired cache is served immediately while a
    background thread fetches fresh data, so callers never block.
"""

import http.client
import json
import threading
import time
from urllib.parse import urlparse

from plugins import ProxyPlugin

MODEL_CACHE_TTL = 300  # seconds (5 minutes)

# ---------------------------------------------------------------------------
# Module-level cache — shared across all handler threads
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cached_models: list | None = None   # list of OpenAI model objects
_cache_timestamp: float = 0.0        # epoch of last successful fetch
_refresh_in_progress = False         # guard against concurrent refreshes


def reset_cache() -> None:
    """
    Atomically clear all cache state.  Called by the config hot-reload watcher
    in proxy.py whenever config.yaml changes.

    Resetting _refresh_in_progress is the critical part: without it, if a
    background refresh was running at the moment of reload, the flag stays True
    and no new refresh can be triggered — models would be served forever from
    the (now-invalidated) stale cache.
    """
    global _cached_models, _cache_timestamp, _refresh_in_progress
    with _cache_lock:
        _cached_models = None
        _cache_timestamp = 0.0
        _refresh_in_progress = False
    print("  [models] Cache reset (config reload)", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_models_from_service(service: dict) -> list | None:
    parsed = urlparse(service['baseUrl'])
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    conn = None
    try:
        conn = (
            http.client.HTTPSConnection(host, port, timeout=10)
            if parsed.scheme == 'https'
            else http.client.HTTPConnection(host, port, timeout=10)
        )
        headers = {}
        token = service.get('token')
        if token and token != 'not-needed':
            headers['Authorization'] = f'Bearer {token}'

        conn.request('GET', '/v1/models', headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            print(f"  [models] {service['name']} returned HTTP {resp.status}", flush=True)
            return None

        body = json.loads(resp.read())
        models = body.get('data', [])
        print(f"  [models] {service['name']} has {len(models)} model(s)", flush=True)
        return models
    except Exception as e:
        print(f"  [models] Failed to fetch from {service['name']}: {e}", flush=True)
        return None
    finally:
        if conn:
            conn.close()


def _compute_union(all_service_models: list[list]) -> list:
    """
    All unique models across every service; first occurrence of each ID wins.
    This lets the client discover models that are only on a subset of backends.
    """
    seen: set[str] = set()
    result: list = []
    for models in all_service_models:
        for m in models:
            mid = m.get('id', '')
            if mid and mid not in seen:
                seen.add(mid)
                result.append(m)
    return result


def _apply_aliases(models: list, aliases: dict) -> list:
    """
    For each alias → target in aliases, if the target model is present in the
    list and the alias id is not already there, inject a cloned entry with
    id = alias.  The clone inherits all other fields from the target object so
    the client sees a fully-formed model entry.
    """
    if not aliases:
        return models
    existing_ids = {m.get('id') for m in models}
    extra = []
    for alias, target in aliases.items():
        if target in existing_ids and alias not in existing_ids:
            original = next(m for m in models if m.get('id') == target)
            extra.append({**original, 'id': alias})
            print(f"  [models] alias {alias!r} → {target!r}", flush=True)
    if extra:
        print(f"  [models] injected {len(extra)} alias(es)", flush=True)
    return models + extra


def _eligible_services(config: dict) -> list:
    return [
        s for s in config.get('services', [])
        if any('/v1/models' in r for r in s.get('routes', []))
    ]


def _refresh_cache(config: dict, handler=None) -> list:
    global _cached_models, _cache_timestamp, _refresh_in_progress

    services = _eligible_services(config)
    print(f"→ [models] Refreshing from {len(services)} service(s)...", flush=True)

    all_service_models = []
    for service in services:
        healthy = handler.is_service_healthy(service) if handler else True
        if not healthy:
            print(f"  [models] Skipping unhealthy: {service['name']}", flush=True)
            continue
        models = _get_models_from_service(service)
        if models is not None:
            all_service_models.append(models)

    if not all_service_models:
        print("✗ [models] No services responded; keeping previous cache.", flush=True)
        with _cache_lock:
            _refresh_in_progress = False
        return _cached_models or []

    union = _compute_union(all_service_models)
    print(
        f"✓ [models] {len(union)} unique model(s) across "
        f"{len(all_service_models)} service(s)",
        flush=True,
    )
    with _cache_lock:
        _cached_models = union
        _cache_timestamp = time.time()
        _refresh_in_progress = False
    return union


def _send_json(handler, payload: dict):
    body = json.dumps(payload).encode()
    handler.send_response(200)
    handler.send_header('Content-Type', 'application/json')
    handler.send_header('Content-Length', str(len(body)))
    handler.send_header('X-Ai-Proxy-Cache-Age', str(int(time.time() - _cache_timestamp)))
    handler.end_headers()
    handler.wfile.write(body)


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class ModelAggregatorPlugin(ProxyPlugin):

    def match(self, method: str, path: str) -> bool:
        return method == 'GET' and path.rstrip('/') in ('/v1/models', '/models')

    def handle(self, handler) -> None:
        global _refresh_in_progress

        now = time.time()
        with _cache_lock:
            has_cache = _cached_models is not None
            is_stale  = (now - _cache_timestamp) >= MODEL_CACHE_TTL
            should_refresh = is_stale and not _refresh_in_progress
            if should_refresh:
                _refresh_in_progress = True

        if not has_cache:
            # Cold start — block until we have something to serve
            print("→ [models] Cold start, fetching synchronously...", flush=True)
            models = _refresh_cache(handler.config, handler)
        else:
            models = _cached_models  # serve stale immediately
            if should_refresh:
                threading.Thread(
                    target=_refresh_cache,
                    args=(handler.config, handler),
                    daemon=True,
                ).start()
                print(
                    f"→ [models] Cache stale ({int(now - _cache_timestamp)}s), "
                    "serving stale + refreshing in background",
                    flush=True,
                )

        aliases = handler.config.get('modelAliases', {})
        models = _apply_aliases(models, aliases)
        _send_json(handler, {'object': 'list', 'data': models})
        print(f"✓ [models] Served {len(models)} model(s)", flush=True)


PLUGIN = ModelAggregatorPlugin()
