import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from financial_frontend.server import create_server


@pytest.fixture
def service():
    received = []

    class Backend(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ready":true}')

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, data))
            self.send_response(422 if data.get("fail") else 200)
            self.end_headers()
            self.wfile.write(json.dumps({"echo": data}).encode())

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    frontend = create_server(port=0, backend_url=f"http://127.0.0.1:{upstream.server_port}")
    threads = [Thread(target=server.serve_forever, daemon=True) for server in (upstream, frontend)]
    for thread in threads:
        thread.start()
    try:
        yield f"http://127.0.0.1:{frontend.server_port}", received
    finally:
        for server in (upstream, frontend):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)


def test_page_assets_and_same_origin_proxy(service):
    url, received = service
    with urlopen(url) as response:
        assert response.status == 200 and '金融助手' in response.read().decode()
    for asset in ("/style.css", "/app.js"):
        with urlopen(url + asset) as response:
            assert response.status == 200
    data = {"messages": [{"role": "user", "content": "你好"}], "context": {"awaiting_document": True}}
    request = Request(url + "/api/v1/chat", data=json.dumps(data).encode(), method="POST")
    with urlopen(request) as response:
        assert json.load(response)["echo"] == data
    assert received == [("/v1/chat", data)]
    with urlopen(url + "/api/health/ready") as response:
        assert json.load(response)["ready"]


def test_upstream_status_is_preserved_and_arbitrary_api_paths_are_blocked(service):
    url, received = service
    request = Request(url + "/api/v1/chat", data=b'{"fail":true}', method="POST")
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 422
    with pytest.raises(HTTPError) as error:
        urlopen(url + "/api/v1/model")
    assert error.value.code == 404
    assert len(received) == 1


def test_frontend_returns_friendly_error_when_backend_is_offline():
    temporary = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = temporary.server_port
    temporary.server_close()
    frontend = create_server(port=0, backend_url=f"http://127.0.0.1:{port}")
    thread = Thread(target=frontend.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(f"http://127.0.0.1:{frontend.server_port}/api/health/ready")
        assert error.value.code == 503
        assert "后端" in json.load(error.value)["error"]["message"]
    finally:
        frontend.shutdown()
        frontend.server_close()
        thread.join(timeout=3)
