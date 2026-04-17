"""Serve JupyterLite `_output` with a correct `.wasm` MIME type (Pyodide)."""
from __future__ import annotations

import http.server
import os
import socketserver

ROOT = os.environ.get("JUPYTERLITE_HTTP_ROOT", "/srv")
PORT = int(os.environ.get("PORT", "8000"))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = dict(
        http.server.SimpleHTTPRequestHandler.extensions_map,
        **{".wasm": "application/wasm"},
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)


if __name__ == "__main__":
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.serve_forever()
