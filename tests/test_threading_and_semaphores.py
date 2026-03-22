"""
Tests for proxy.py — threading and semaphore correctness.

Covers:
  - ThreadingHTTPServer is used (not single-threaded HTTPServer)
  - daemon_threads is set
  - Per-service semaphore respects maxConcurrent
  - Pass 1: free service → routed immediately, no blocking
  - Pass 2: locked service → request queues and proceeds once slot is free
  - Pass 3: no healthy services → 503
  - Pass 4: locked service, wait times out → 429
  - Concurrent requests don't race past the semaphore
  - _invalidate_health_cache is thread-safe under concurrent access
"""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(name="svc", max_concurrent=1, queue_timeout=2):
    return {
        'name': name,
        'baseUrl': 'http://localhost:9999',
        'health': '/v1/models',
        'routes': ['/v1/chat/completions'],
        'maxConcurrent': max_concurrent,
        'queueTimeoutSeconds': queue_timeout,
    }


def _make_handler(services=None, body=None):
    """
    Build a ProxyHandler-like object with the minimal interface needed to
    exercise routing logic, without requiring a real socket.
    """
    from proxy import ProxyHandler

    handler = MagicMock(spec=ProxyHandler)
    handler.command = 'POST'
    handler.path = '/v1/chat/completions'
    handler.headers = {}
    handler._body = body or b'{"model": "test-model"}'
    handler.config = {'services': services or [], 'modelAliases': {}}

    # Wire real semaphore logic through to the actual module functions
    handler.find_services_for_route = ProxyHandler.find_services_for_route.__get__(handler)

    return handler


# ---------------------------------------------------------------------------
# Server-level tests
# ---------------------------------------------------------------------------

class TestServerSetup:

    def test_uses_threading_http_server(self):
        import proxy
        # run_server() must instantiate ThreadingHTTPServer, not HTTPServer
        created = []
        original = __builtins__  # keep reference

        with patch('proxy.ThreadingHTTPServer') as mock_cls:
            mock_instance = MagicMock()
            mock_instance.serve_forever.side_effect = KeyboardInterrupt
            mock_cls.return_value = mock_instance
            try:
                proxy.run_server()
            except (KeyboardInterrupt, SystemExit):
                pass
            mock_cls.assert_called_once()
            # First arg to constructor is the address tuple
            args, kwargs = mock_cls.call_args
            assert args[0] == ('', 8080)

    def test_daemon_threads_enabled(self):
        with patch('proxy.ThreadingHTTPServer') as mock_cls:
            mock_instance = MagicMock()
            mock_instance.serve_forever.side_effect = KeyboardInterrupt
            mock_cls.return_value = mock_instance
            try:
                import proxy
                proxy.run_server()
            except (KeyboardInterrupt, SystemExit):
                pass
            assert mock_instance.daemon_threads is True


# ---------------------------------------------------------------------------
# Semaphore creation tests
# ---------------------------------------------------------------------------

class TestSemaphoreCreation:

    def setup_method(self):
        # Clear semaphore registry between tests
        from proxy import _service_semaphores
        _service_semaphores.clear()

    def test_default_max_concurrent_is_one(self):
        from proxy import _get_semaphore
        svc = _make_service(name='default-svc')
        del svc['maxConcurrent']   # remove explicit value
        sem = _get_semaphore(svc)
        # A Semaphore(1) allows exactly one non-blocking acquire
        assert sem.acquire(blocking=False) is True
        assert sem.acquire(blocking=False) is False
        sem.release()

    def test_max_concurrent_respected(self):
        from proxy import _get_semaphore
        svc = _make_service(name='multi-svc', max_concurrent=3)
        sem = _get_semaphore(svc)
        acquired = [sem.acquire(blocking=False) for _ in range(4)]
        # First 3 should succeed, 4th must fail
        assert acquired[:3] == [True, True, True]
        assert acquired[3] is False
        for _ in range(3):
            sem.release()

    def test_same_service_returns_same_semaphore(self):
        from proxy import _get_semaphore
        svc = _make_service(name='stable-svc')
        assert _get_semaphore(svc) is _get_semaphore(svc)

    def test_different_services_get_independent_semaphores(self):
        from proxy import _get_semaphore
        svc_a = _make_service(name='svc-a')
        svc_b = _make_service(name='svc-b')
        assert _get_semaphore(svc_a) is not _get_semaphore(svc_b)

    def test_semaphore_creation_is_thread_safe(self):
        """Many threads requesting the semaphore for the same service
        must all get back the *same* object."""
        from proxy import _get_semaphore
        svc = _make_service(name='thread-safe-svc')
        results = []
        barrier = threading.Barrier(20)

        def grab():
            barrier.wait()   # all threads hit _get_semaphore simultaneously
            results.append(_get_semaphore(svc))

        threads = [threading.Thread(target=grab) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(set(id(s) for s in results)) == 1


# ---------------------------------------------------------------------------
# Pass 1 — free service routed immediately
# ---------------------------------------------------------------------------

class TestPass1FreeThenRoute:

    def setup_method(self):
        from proxy import _service_semaphores, _health_cache
        _service_semaphores.clear()
        _health_cache.clear()

    def test_free_service_is_acquired_without_blocking(self):
        from proxy import _get_semaphore
        svc = _make_service(name='free-svc')
        sem = _get_semaphore(svc)

        start = time.monotonic()
        acquired = sem.acquire(blocking=False)
        elapsed = time.monotonic() - start

        assert acquired is True
        assert elapsed < 0.05   # must not have waited
        sem.release()

    def test_pass1_does_not_block_when_slot_available(self):
        """
        Simulate handle_request Pass 1: if a service's semaphore is free,
        forward_request must be called without any sleep or timeout wait.

        NOTE: patch.object patches the *class*, but handle_request calls
        self.method() where self is the MagicMock instance — which has its own
        auto-generated attributes that shadow the class patch.  We must set
        the methods on the handler instance directly.
        """
        from proxy import ProxyHandler

        svc = _make_service(name='instant-svc', queue_timeout=1)

        handler = MagicMock(spec=ProxyHandler)
        handler.command = 'POST'
        handler.path = '/v1/chat/completions'
        handler.headers = {}
        handler._body = b'{"model": "test-model"}'
        handler.config = {'services': [svc], 'modelAliases': {}}
        # Set on the instance — these are what handle_request actually calls
        handler.find_services_for_route = MagicMock(return_value=[svc])
        handler.is_service_healthy = MagicMock(return_value=True)
        handler.forward_request = MagicMock(return_value=True)
        handler.send_error = MagicMock()

        start = time.monotonic()
        ProxyHandler.handle_request(handler)
        elapsed = time.monotonic() - start

        handler.forward_request.assert_called_once_with(svc)
        handler.send_error.assert_not_called()
        assert elapsed < 0.5


# ---------------------------------------------------------------------------
# Pass 2 — queuing behind a locked service
# ---------------------------------------------------------------------------

class TestPass2Queuing:

    def setup_method(self):
        from proxy import _service_semaphores, _health_cache
        _service_semaphores.clear()
        _health_cache.clear()

    def test_queued_request_proceeds_after_slot_released(self):
        """
        Hold the semaphore in a background thread for 0.3s; the queued request
        must succeed *after* the hold expires, not before.
        """
        from proxy import _get_semaphore
        svc = _make_service(name='queue-svc', queue_timeout=2)
        sem = _get_semaphore(svc)

        # Occupy the only slot
        sem.acquire()
        release_at = time.monotonic() + 0.3
        threading.Timer(0.3, sem.release).start()

        start = time.monotonic()
        acquired = sem.acquire(timeout=2)
        elapsed = time.monotonic() - start

        assert acquired is True
        assert elapsed >= 0.25       # had to wait for the slot
        assert elapsed < 1.0         # but didn't wait unnecessarily long
        sem.release()

    def test_second_request_waits_for_first_to_finish(self):
        """
        Two threads compete for a maxConcurrent=1 semaphore.
        Their critical sections must not overlap.
        """
        from proxy import _get_semaphore
        svc = _make_service(name='serial-svc', max_concurrent=1)
        sem = _get_semaphore(svc)

        timeline = []
        errors = []

        def worker(name, hold_seconds=0.2):
            if not sem.acquire(timeout=3):
                errors.append(f"{name} timed out")
                return
            try:
                timeline.append(('enter', name, time.monotonic()))
                time.sleep(hold_seconds)
                timeline.append(('exit', name, time.monotonic()))
            finally:
                sem.release()

        t1 = threading.Thread(target=worker, args=('A',))
        t2 = threading.Thread(target=worker, args=('B',))
        t1.start()
        time.sleep(0.02)   # give t1 a head start so it acquires first
        t2.start()
        t1.join(); t2.join()

        assert not errors

        # Extract enter/exit times
        enters = {e[1]: e[2] for e in timeline if e[0] == 'enter'}
        exits  = {e[1]: e[2] for e in timeline if e[0] == 'exit'}

        # B must enter after A exits — no overlap
        assert enters['B'] >= exits['A'] - 0.01


# ---------------------------------------------------------------------------
# Pass 3 — no healthy services → 503
# ---------------------------------------------------------------------------

class TestPass3NoHealthyServices:

    def setup_method(self):
        from proxy import _service_semaphores, _health_cache
        _service_semaphores.clear()
        _health_cache.clear()

    def test_503_when_no_healthy_services(self):
        from proxy import ProxyHandler
        svc = _make_service(name='dead-svc')

        handler = MagicMock(spec=ProxyHandler)
        handler.command = 'POST'
        handler.path = '/v1/chat/completions'
        handler.headers = {}
        handler._body = b'{"model": "test-model"}'
        handler.config = {'services': [svc], 'modelAliases': {}}
        handler.find_services_for_route = MagicMock(return_value=[svc])
        handler.is_service_healthy = MagicMock(return_value=False)
        handler.forward_request = MagicMock(return_value=False)
        handler.send_error = MagicMock()

        ProxyHandler.handle_request(handler)

        handler.forward_request.assert_not_called()
        handler.send_error.assert_called_once_with(503, "No healthy service available for this route")

    def test_503_when_no_services_match_route(self):
        from proxy import ProxyHandler

        handler = MagicMock(spec=ProxyHandler)
        handler.command = 'POST'
        handler.path = '/v1/chat/completions'
        handler.headers = {}
        handler._body = b'{}'
        handler.config = {'services': [], 'modelAliases': {}}
        handler.find_services_for_route = MagicMock(return_value=[])
        handler.send_error = MagicMock()

        ProxyHandler.handle_request(handler)

        handler.send_error.assert_called_once_with(404, "No service found for this route")


# ---------------------------------------------------------------------------
# Pass 4 — timeout waiting for locked service → 429
# ---------------------------------------------------------------------------

class TestPass4Timeout:

    def setup_method(self):
        from proxy import _service_semaphores, _health_cache
        _service_semaphores.clear()
        _health_cache.clear()

    def test_429_when_queue_timeout_expires(self):
        """
        Occupy the semaphore and never release it; the proxy must return 429
        once queueTimeoutSeconds elapses.
        """
        from proxy import ProxyHandler, _get_semaphore

        svc = _make_service(name='busy-svc', queue_timeout=1)
        sem = _get_semaphore(svc)
        sem.acquire()   # lock it and never release — simulates ongoing inference

        handler = MagicMock(spec=ProxyHandler)
        handler.command = 'POST'
        handler.path = '/v1/chat/completions'
        handler.headers = {}
        handler._body = b'{"model": "test-model"}'
        handler.config = {'services': [svc], 'modelAliases': {}}
        handler.find_services_for_route = MagicMock(return_value=[svc])
        handler.is_service_healthy = MagicMock(return_value=True)
        handler.send_error = MagicMock()

        start = time.monotonic()
        ProxyHandler.handle_request(handler)
        elapsed = time.monotonic() - start

        sem.release()   # cleanup

        handler.send_error.assert_called_once_with(429, "Service busy, please retry later")
        assert elapsed >= 1.0        # had to wait the full timeout
        assert elapsed < 3.0         # but not indefinitely

    def test_429_not_503_when_service_is_healthy_but_busy(self):
        """
        A busy (locked) service that times out must produce 429, not 503.
        503 means broken; 429 means busy — semantically different.
        """
        from proxy import ProxyHandler, _get_semaphore

        svc = _make_service(name='semantics-svc', queue_timeout=1)
        sem = _get_semaphore(svc)
        sem.acquire()

        error_codes = []

        handler = MagicMock(spec=ProxyHandler)
        handler.command = 'POST'
        handler.path = '/v1/chat/completions'
        handler.headers = {}
        handler._body = b'{"model": "test-model"}'
        handler.config = {'services': [svc], 'modelAliases': {}}
        handler.find_services_for_route = MagicMock(return_value=[svc])
        handler.is_service_healthy = MagicMock(return_value=True)
        handler.send_error = MagicMock(side_effect=lambda code, msg: error_codes.append(code))

        ProxyHandler.handle_request(handler)
        sem.release()

        assert 429 in error_codes
        assert 503 not in error_codes


# ---------------------------------------------------------------------------
# Concurrency stress test — semaphore must hold under load
# ---------------------------------------------------------------------------

class TestConcurrencyStress:

    def setup_method(self):
        from proxy import _service_semaphores, _health_cache
        _service_semaphores.clear()
        _health_cache.clear()

    def test_semaphore_max_concurrent_never_exceeded_under_load(self):
        """
        20 threads all try to acquire a maxConcurrent=2 semaphore.
        At no point should more than 2 hold it simultaneously.
        """
        from proxy import _get_semaphore

        svc = _make_service(name='stress-svc', max_concurrent=2)
        sem = _get_semaphore(svc)

        max_seen = [0]
        current = [0]
        lock = threading.Lock()
        errors = []

        def worker():
            if not sem.acquire(timeout=5):
                errors.append('timeout')
                return
            try:
                with lock:
                    current[0] += 1
                    if current[0] > max_seen[0]:
                        max_seen[0] = current[0]
                    if current[0] > 2:
                        errors.append(f'concurrency exceeded: {current[0]}')
                time.sleep(0.05)
                with lock:
                    current[0] -= 1
            finally:
                sem.release()

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        assert max_seen[0] <= 2
        assert max_seen[0] >= 1   # sanity: at least some concurrency happened

    def test_serial_service_processes_requests_sequentially(self):
        """
        With maxConcurrent=1, requests must be strictly serial — we record
        start/end timestamps and verify no two intervals overlap.
        """
        from proxy import _get_semaphore

        svc = _make_service(name='serial-stress-svc', max_concurrent=1)
        sem = _get_semaphore(svc)
        intervals = []
        lock = threading.Lock()

        def worker():
            if not sem.acquire(timeout=10):
                return
            try:
                start = time.monotonic()
                time.sleep(0.05)
                end = time.monotonic()
                with lock:
                    intervals.append((start, end))
            finally:
                sem.release()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        # Sort by start time and verify no overlap between adjacent intervals
        intervals.sort()
        for i in range(len(intervals) - 1):
            _, end_i = intervals[i]
            start_next, _ = intervals[i + 1]
            assert start_next >= end_i - 0.005, (
                f"Overlap detected: interval {i} ended at {end_i:.4f}, "
                f"interval {i+1} started at {start_next:.4f}"
            )


# ---------------------------------------------------------------------------
# Health cache thread safety
# ---------------------------------------------------------------------------

class TestHealthCacheThreadSafety:

    def setup_method(self):
        from proxy import _health_cache, _health_check_in_progress
        _health_cache.clear()
        _health_check_in_progress.clear()

    def test_invalidate_health_cache_is_thread_safe(self):
        """
        Many threads simultaneously invalidating different and overlapping keys
        must not raise or corrupt the dict.
        """
        from proxy import _health_cache, _invalidate_health_cache
        import time as time_mod

        # Pre-populate with 50 entries
        for i in range(50):
            _health_cache[f'svc-{i}'] = (True, set(), time_mod.time())

        errors = []

        def invalidator(name):
            try:
                _invalidate_health_cache(name)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=invalidator, args=(f'svc-{i % 50}',))
            for i in range(200)
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors

    def test_concurrent_health_checks_dont_duplicate(self):
        """
        When many threads trigger is_service_healthy simultaneously on a cold
        cache, the in-progress guard must ensure _check_health is only called
        once, not N times.

        NOTE: patch.object inside each thread creates a race — multiple threads
        simultaneously patching/unpatching the same class attribute corrupts
        the patch state.  Patch once at the test level instead.
        """
        from proxy import ProxyHandler

        svc = _make_service(name='cold-svc')
        check_count = [0]
        count_lock = threading.Lock()

        def fake_check(self_inner, service):
            with count_lock:
                check_count[0] += 1
            time.sleep(0.1)   # simulate network latency
            return True, {'test-model'}

        results = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(10)

        def caller():
            handler = MagicMock(spec=ProxyHandler)
            # Set on the instance — is_service_healthy calls self._check_health(),
            # where self is this MagicMock.  patch.object patches the class but the
            # instance mock has its own attribute that shadows it, causing an
            # unpackable MagicMock return that raises ValueError on the checking
            # thread and silently drops its result.
            handler._check_health = lambda service: fake_check(handler, service)
            barrier.wait()   # all threads call is_service_healthy simultaneously
            result = ProxyHandler.is_service_healthy(handler, svc)
            with results_lock:
                results.append(result)

        threads = [threading.Thread(target=caller) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(results) == 10
        # Guard worked: only one actual network call despite 10 concurrent callers
        assert check_count[0] == 1


# ---------------------------------------------------------------------------
# Pass 2 fallthrough — locked service fails, next service attempted
# ---------------------------------------------------------------------------

class TestPass2Fallthrough:

    def setup_method(self):
        from proxy import _service_semaphores, _health_cache
        _service_semaphores.clear()
        _health_cache.clear()

    def test_falls_through_to_next_locked_service_when_first_fails_forwarding(self):
        """
        Two locked services; the first frees up but forward_request fails.
        Pass 2 must continue to the second service rather than 503ing.
        """
        from proxy import ProxyHandler, _get_semaphore

        svc_a = _make_service(name='locked-a', queue_timeout=3)
        svc_b = _make_service(name='locked-b', queue_timeout=3)

        sem_a = _get_semaphore(svc_a)
        sem_b = _get_semaphore(svc_b)
        sem_a.acquire()   # both locked to start
        sem_b.acquire()

        # Release both after a short delay so the test doesn't take long
        threading.Timer(0.2, sem_a.release).start()
        threading.Timer(0.4, sem_b.release).start()

        forward_calls = []

        def fake_forward(service):
            forward_calls.append(service['name'])
            # First service fails, second succeeds
            return service['name'] == 'locked-b'

        handler = MagicMock(spec=ProxyHandler)
        handler.command = 'POST'
        handler.path = '/v1/chat/completions'
        handler.headers = {}
        handler._body = b'{"model": "x"}'
        handler.config = {'services': [svc_a, svc_b], 'modelAliases': {}}
        handler.find_services_for_route = MagicMock(return_value=[svc_a, svc_b])
        handler.is_service_healthy = MagicMock(return_value=True)
        handler.forward_request = MagicMock(side_effect=fake_forward)
        handler.send_error = MagicMock()

        ProxyHandler.handle_request(handler)

        # Both services should have been attempted
        assert forward_calls == ['locked-a', 'locked-b']
        # Success via locked-b — no error
        handler.send_error.assert_not_called()

    def test_503_after_all_locked_services_fail_forwarding(self):
        """
        Both locked services free up but both fail to forward — must 503,
        not 429 (the service isn't busy, it's broken).
        """
        from proxy import ProxyHandler, _get_semaphore

        svc_a = _make_service(name='broken-a', queue_timeout=3)
        svc_b = _make_service(name='broken-b', queue_timeout=3)

        sem_a = _get_semaphore(svc_a)
        sem_b = _get_semaphore(svc_b)
        sem_a.acquire()
        sem_b.acquire()
        threading.Timer(0.2, sem_a.release).start()
        threading.Timer(0.4, sem_b.release).start()

        handler = MagicMock(spec=ProxyHandler)
        handler.command = 'POST'
        handler.path = '/v1/chat/completions'
        handler.headers = {}
        handler._body = b'{"model": "x"}'
        handler.config = {'services': [svc_a, svc_b], 'modelAliases': {}}
        handler.find_services_for_route = MagicMock(return_value=[svc_a, svc_b])
        handler.is_service_healthy = MagicMock(return_value=True)
        handler.forward_request = MagicMock(return_value=False)   # both fail
        handler.send_error = MagicMock()

        ProxyHandler.handle_request(handler)

        # Should 503 (broken), not 429 (busy)
        codes = [call.args[0] for call in handler.send_error.call_args_list]
        assert 503 in codes
        assert 429 not in codes


# ---------------------------------------------------------------------------
# Gemini plugin — semaphore respected
# ---------------------------------------------------------------------------

class TestGeminiSemaphore:

    def setup_method(self):
        from proxy import _service_semaphores, _health_cache
        _service_semaphores.clear()
        _health_cache.clear()

    def _make_gemini_service(self, name='gemini-svc', queue_timeout=2):
        return {
            'name': name,
            'baseUrl': 'http://localhost:9999',
            'health': '/v1/models',
            'routes': ['/v1/embeddings'],
            'geminiEmbeddingPath': '/v1/embeddings',
            'maxConcurrent': 1,
            'queueTimeoutSeconds': queue_timeout,
        }

    def test_gemini_acquires_semaphore_before_forwarding(self):
        """
        _handle_embed must hold the semaphore during the backend call so
        that a concurrent LLM generation cannot be evicted.
        """
        from proxy import _get_semaphore
        from plugins.gemini_embeddings import GeminiEmbeddingPlugin

        svc = self._make_gemini_service()
        sem = _get_semaphore(svc)

        semaphore_held_during_call = [False]

        def fake_post_json(service, path, payload, headers):
            # Check whether the semaphore is held (value == 0 means acquired)
            semaphore_held_during_call[0] = not sem.acquire(blocking=False)
            if not semaphore_held_during_call[0]:
                sem.release()   # wasn't held — release the probe acquire
            return {'data': [{'embedding': [0.1, 0.2]}]}, 200

        handler = MagicMock()
        handler.command = 'POST'
        handler.path = '/v1beta/models/gemini-embedding-001:embedContent'
        handler.headers = {'Content-Length': '50'}
        handler.rfile.read = MagicMock(return_value=b'{"content": {"parts": [{"text": "hello"}]}}')
        handler.config = {
            'services': [svc],
            'modelAliases': {},
        }
        handler.is_service_healthy = MagicMock(return_value=True)

        plugin = GeminiEmbeddingPlugin()

        with patch('plugins.gemini_embeddings._post_json', side_effect=fake_post_json):
            plugin._handle_embed(handler, 'gemini-embedding-001')

        assert semaphore_held_during_call[0], \
            "Semaphore was NOT held during _post_json — model eviction race still possible"

    def test_gemini_429_when_service_locked_and_timeout_expires(self):
        """
        If the service semaphore is held and the queue timeout expires,
        the Gemini plugin must return 429, not proceed with the call.
        """
        from proxy import _get_semaphore
        from plugins.gemini_embeddings import GeminiEmbeddingPlugin

        svc = self._make_gemini_service(queue_timeout=1)
        sem = _get_semaphore(svc)
        sem.acquire()   # hold it forever — simulates ongoing LLM generation

        handler = MagicMock()
        handler.command = 'POST'
        handler.headers = {'Content-Length': '50'}
        handler.rfile.read = MagicMock(return_value=b'{"content": {"parts": [{"text": "hello"}]}}')
        handler.config = {'services': [svc], 'modelAliases': {}}
        handler.is_service_healthy = MagicMock(return_value=True)
        handler.send_error = MagicMock()

        plugin = GeminiEmbeddingPlugin()

        start = time.monotonic()
        plugin._handle_embed(handler, 'gemini-embedding-001')
        elapsed = time.monotonic() - start

        sem.release()

        handler.send_error.assert_called_once_with(429, "Service busy, please retry later")
        assert elapsed >= 1.0
        assert elapsed < 3.0

    def test_gemini_free_service_not_blocked_by_locked_service(self):
        """
        Two services: first is locked, second is free.
        Pass 1 should route to the free second service without waiting.
        """
        from proxy import _get_semaphore
        from plugins.gemini_embeddings import GeminiEmbeddingPlugin

        svc_locked = self._make_gemini_service(name='locked-gemini', queue_timeout=30)
        svc_free   = self._make_gemini_service(name='free-gemini',   queue_timeout=30)

        sem_locked = _get_semaphore(svc_locked)
        sem_locked.acquire()   # hold it

        forwarded_to = []

        def fake_post_json(service, path, payload, headers):
            forwarded_to.append(service['name'])
            return {'data': [{'embedding': [0.1]}]}, 200

        handler = MagicMock()
        handler.command = 'POST'
        handler.headers = {'Content-Length': '50'}
        handler.rfile.read = MagicMock(return_value=b'{"content": {"parts": [{"text": "hi"}]}}')
        handler.config = {'services': [svc_locked, svc_free], 'modelAliases': {}}
        handler.is_service_healthy = MagicMock(return_value=True)

        plugin = GeminiEmbeddingPlugin()

        start = time.monotonic()
        with patch('plugins.gemini_embeddings._post_json', side_effect=fake_post_json):
            plugin._handle_embed(handler, 'gemini-embedding-001')
        elapsed = time.monotonic() - start

        sem_locked.release()

        assert forwarded_to == ['free-gemini'], \
            f"Expected routing to free-gemini, got {forwarded_to}"
        assert elapsed < 0.5, "Should have routed immediately to free service"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
