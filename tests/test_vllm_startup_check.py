"""tensorpilot.vllm_startup_check against a stand-in OpenAI-compatible server: which answers pass, which fail, and that the key
never reaches the command line."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tensorpilot import vllm_startup_check as check


def server(handler_for):
    class Handler(BaseHTTPRequestHandler):
        seen = []

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            Handler.seen.append((self.headers.get("Authorization"), body))
            status, payload = handler_for(body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, Handler


def good(body):
    return 200, {"choices": [{"text": " red, green and blue", "finish_reason": "length"}], "usage": {"completion_tokens": 16}}


@pytest.fixture
def stand_in():
    started = []

    def make(handler_for):
        httpd, handler = server(handler_for)
        started.append(httpd)
        return f"http://127.0.0.1:{httpd.server_port}", handler

    yield make
    for httpd in started:
        httpd.shutdown()


def test_every_sampling_path_is_exercised_and_the_key_is_sent_as_a_bearer_token(stand_in):
    url, handler = stand_in(good)
    report = check.run(url, "m", "sekrit", timeout=5)
    assert report["ok"] and [c["case"] for c in report["cases"]] == ["greedy", "top_k_top_p", "top_k", "top_p"]
    assert {auth for auth, _ in handler.seen} == {"Bearer sekrit"}
    bodies = [body for _, body in handler.seen]
    assert any(b.get("top_k") and b.get("top_p") for b in bodies), "the FlashInfer top-k/top-p path must be requested"
    assert bodies[0]["temperature"] == 0


@pytest.mark.parametrize("answer, problem", [
    ({"choices": [{"text": "  ", "finish_reason": "length"}], "usage": {"completion_tokens": 3}}, "empty text"),
    ({"choices": [{"text": "x", "finish_reason": None}], "usage": {"completion_tokens": 3}}, "finish_reason"),
    ({"choices": [{"text": "x", "finish_reason": "length"}], "usage": {"completion_tokens": 0}}, "0 tokens"),
    ({"unexpected": 1}, "malformed"),
])
def test_an_unusable_answer_fails_the_check(stand_in, answer, problem):
    url, _ = stand_in(lambda body: (200, answer))
    report = check.run(url, "m", "k", timeout=5)
    assert not report["ok"] and problem in report["cases"][0]["problem"]


def test_a_server_error_on_the_sampling_path_fails_only_that_case_and_names_it(stand_in):
    """What the second vLLM failure would have looked like to a client: the sampler path errors while greedy works."""
    url, _ = stand_in(lambda body: (500, {"error": "boom"}) if body.get("top_k") else good(body))
    report = check.run(url, "m", "k", timeout=5)
    by_case = {c["case"]: c for c in report["cases"]}
    assert not report["ok"] and by_case["greedy"]["ok"] and not by_case["top_k_top_p"]["ok"]
    assert "HTTPError" in by_case["top_k_top_p"]["problem"]


def test_an_unreachable_server_fails_without_a_traceback():
    report = check.run("http://127.0.0.1:9", "m", "k", timeout=2)
    assert not report["ok"] and all("URLError" in c["problem"] or "Error" in c["problem"] for c in report["cases"])


def test_the_cli_reads_the_key_from_the_environment_and_writes_the_report(stand_in, tmp_path, monkeypatch):
    url, handler = stand_in(good)
    monkeypatch.setenv("TENSORPILOT_API_KEY", "from-env")
    out = tmp_path / "out.json"
    monkeypatch.setattr("sys.argv", ["x", "--url", url, "--model", "m", "--output", str(out)])
    assert check.main() == 0
    assert json.loads(out.read_text())["ok"] and handler.seen[0][0] == "Bearer from-env"
    assert "--api-key" not in check.__doc__, "the key must not be a command-line argument"


def test_only_http_urls_are_accepted():
    report = check.run("file:///etc/passwd", "m", "k", timeout=1)
    assert not report["ok"] and all("http or https" in c["problem"] for c in report["cases"])
