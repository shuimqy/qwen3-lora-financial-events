"""Local static server with a fixed /api proxy for separate frontend development."""
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "static"
ALLOWED = {("POST", "/api/v1/chat"), ("GET", "/api/health/ready")}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, backend_url, **kwargs):
        self.backend_url = backend_url
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self):
        if urlsplit(self.path).path.startswith("/api/"):
            self.proxy("GET")
        else:
            super().do_GET()

    def do_POST(self):
        self.proxy("POST")

    def json_error(self, status, message):
        body = json.dumps({"error": {"message": message}}, ensure_ascii=False).encode("utf-8")
        self.respond(status, body, "application/json; charset=utf-8")

    def respond(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def proxy(self, method):
        path = urlsplit(self.path).path
        if (method, path) not in ALLOWED:
            self.json_error(404, "接口不存在。")
            return
        data = None
        if method == "POST":
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.json_error(400, "请求长度不正确。")
                return
            if not 0 < size <= 262144:
                self.close_connection = True
                self.json_error(413, "输入内容过长。")
                return
            data = self.rfile.read(size)
        request = Request(self.backend_url + path.removeprefix("/api"), data=data, method=method,
                          headers={"Content-Type": "application/json; charset=utf-8"})
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(request, timeout=180 if method == "POST" else 5) as response:
                self.respond(response.status, response.read(), "application/json; charset=utf-8")
        except HTTPError as error:
            self.respond(error.code, error.read(), "application/json; charset=utf-8")
        except (URLError, TimeoutError, OSError):
            self.json_error(503, "暂时无法连接服务，请确认后端已启动。")


def create_server(host="127.0.0.1", port=3000, backend_url="http://127.0.0.1:8000"):
    parsed = urlsplit(backend_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("FRONTEND_BACKEND_URL must be an HTTP(S) base URL")
    return ThreadingHTTPServer((host, port), partial(Handler, backend_url=backend_url.rstrip("/")))


def main():
    host = os.environ.get("FRONTEND_HOST", "127.0.0.1")
    port = int(os.environ.get("FRONTEND_PORT", "3000"))
    backend_url = os.environ.get("FRONTEND_BACKEND_URL", "http://127.0.0.1:8000")
    server = create_server(host, port, backend_url)
    print(f"Frontend: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
