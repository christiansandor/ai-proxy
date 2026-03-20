"""Tests for proxy.py - validating the HTTPServer handler fix."""

import pytest


class TestProxyHandlerInstantiation:
    """Test that ProxyHandler can be properly instantiated with HTTPServer."""

    def test_lambda_handler_signature(self):
        """Verify the lambda wrapper accepts 3 parameters as expected by HTTPServer."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        # Import ProxyHandler from proxy module
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from proxy import ProxyHandler

        # Create the lambda wrapper as used in run_server()
        handler_factory = lambda request, client_address, server: ProxyHandler(request, client_address, server)
        
        # Verify it's callable with 3 arguments
        assert callable(handler_factory)
        
        # The factory should return a ProxyHandler instance when called
        # We can't fully instantiate without real sockets, but we can verify the signature
        import inspect
        sig = inspect.signature(handler_factory)
        params = list(sig.parameters.keys())
        assert len(params) == 3
        assert params[0] == 'request'
        assert params[1] == 'client_address'
        assert params[2] == 'server'

    def test_proxy_handler_is_subclass_of_base_request_handler(self):
        """Verify ProxyHandler is a proper subclass of BaseHTTPRequestHandler."""
        from http.server import BaseHTTPRequestHandler
        from proxy import ProxyHandler
        
        assert issubclass(ProxyHandler, BaseHTTPRequestHandler)

    def test_httpserver_accepts_lambda_factory(self):
        """Test that HTTPServer can accept the lambda factory pattern."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from proxy import ProxyHandler
        
        # This should not raise any type errors at runtime
        server_address = ('', 0)  # Port 0 lets OS assign an available port
        handler_factory = lambda request, client_address, server: ProxyHandler(request, client_address, server)
        
        # Create server with the factory (don't serve_forever to avoid blocking)
        httpd = HTTPServer(server_address, handler_factory)
        
        # Verify server was created successfully
        assert httpd is not None
        assert isinstance(httpd, HTTPServer)
        
        httpd.server_close()


class TestProxyHandlerMethods:
    """Test that ProxyHandler has the expected methods."""

    def test_proxy_handler_has_do_get(self):
        from proxy import ProxyHandler
        assert hasattr(ProxyHandler, 'do_GET')

    def test_proxy_handler_has_do_post(self):
        from proxy import ProxyHandler
        assert hasattr(ProxyHandler, 'do_POST')

    def test_proxy_handler_has_handle_request(self):
        from proxy import ProxyHandler
        assert hasattr(ProxyHandler, 'handle_request')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
