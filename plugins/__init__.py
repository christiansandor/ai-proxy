"""
Plugin interface for the AI proxy.

Each plugin is a .py file in this directory that subclasses ProxyPlugin and
sets PLUGIN = MyPlugin() at the module level.  The proxy will:
  1. call plugin.match(method, path) on every request
  2. if it returns True, call plugin.handle(handler) and stop routing

Plugins are loaded in alphabetical filename order; first match wins.
"""


class ProxyPlugin:
    def match(self, method: str, path: str) -> bool:
        """Return True if this plugin should handle the request."""
        raise NotImplementedError

    def handle(self, handler) -> None:
        """
        Handle the request.  `handler` is the live ProxyHandler instance, so
        you have access to:
          handler.command, handler.path, handler.headers
          handler.rfile        – read the request body from here
          handler.send_response(), handler.send_header(), handler.end_headers()
          handler.wfile        – write the response body here
          handler.send_error() – send an HTTP error
          handler.config       – the parsed config.yaml dict
          handler.is_service_healthy(service) – health-check helper
        """
        raise NotImplementedError
