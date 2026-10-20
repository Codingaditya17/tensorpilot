"""SIGTERM from the container runtime or Runpod must stop a live worker without hanging."""
import json
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_model import make_tiny_model_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = """
import sys
from aiohttp import web
from tensorpilot.cuda_backend import CUDABackend
from tensorpilot.server import create_app
app = create_app(lambda: CUDABackend(sys.argv[1], device="cpu", allow_cpu=True, dtype="float32"),
                 queue_size=4, max_tokens=512, warmup=True,
                 metadata={"model": "tiny", "backend": "cuda"}, instance_id="sigterm-test")
web.run_app(app, host="127.0.0.1", port=int(sys.argv[2]), handler_cancellation=True, access_log=None,
            print=None)
"""


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_ready(url, process, timeout_s=120):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"worker exited early with {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/readyz", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(.2)
    raise AssertionError("worker never became ready")


def test_sigterm_during_decode_drains_and_exits_cleanly(tmp_path):
    model_dir = make_tiny_model_dir(tmp_path / "model")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen([sys.executable, "-c", LAUNCHER, str(model_dir), str(port)], cwd=REPO,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        wait_ready(url, process)
        request = urllib.request.Request(url + "/v1/generate", method="POST",
                                         data=json.dumps({"prompt": "hello world the", "max_tokens": 512}).encode(),
                                         headers={"Content-Type": "application/json"})
        response = urllib.request.urlopen(request, timeout=60)
        first = json.loads(response.readline())
        assert first["type"] == "token"
        process.send_signal(signal.SIGTERM)
        rest = [json.loads(line) for line in response if line.strip()]
        response.close()
        terminal = [event for event in rest if event["type"] in ("done", "error")]
        assert terminal, "stream ended without a terminal event"
        assert terminal[-1]["type"] == "error"
        assert terminal[-1]["code"] in ("shutting_down", "cancelled")
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
