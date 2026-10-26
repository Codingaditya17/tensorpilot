"""The gateway as session.sh starts it: real control-plane and data-plane processes configured only through
environment variables, in front of an authenticated worker. Exercises the CLI wiring that in-process app
construction cannot: a credential missing from a process environment."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import aiohttp
from aiohttp.test_utils import TestServer

from tensorpilot.server import create_app as create_worker
from test_serving import FakeBackend

WORKER_KEY, CONTROL_KEY, DATA_KEY = "w" * 24, "c" * 24, "d" * 24


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def spawn(module, args, **env):
    return subprocess.Popen([sys.executable, "-m", f"tensorpilot.{module}", *args],
                            env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def ready(session, url, process):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        assert process.poll() is None, f"{url} exited early"
        try:
            async with session.get(url + "/readyz") as response:
                if response.status == 200:
                    return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(.2)
    raise AssertionError(f"{url} never became ready")


async def run_gateway(data_env):
    processes = []
    async with TestServer(create_worker(FakeBackend, api_key=WORKER_KEY, warmup=False), port=free_port()) as worker:
        control_port, data_port = free_port(), free_port()
        try:
            async with aiohttp.ClientSession() as session:
                processes.append(spawn("control_plane", ["--port", str(control_port),
                                                          "--worker", f"worker-a=http://127.0.0.1:{worker.port}"],
                                       TENSORPILOT_CONTROL_API_KEY=CONTROL_KEY, TENSORPILOT_WORKER_API_KEY=WORKER_KEY))
                await ready(session, f"http://127.0.0.1:{control_port}", processes[-1])
                processes.append(spawn("data_plane", ["--port", str(data_port), "--control-url",
                                                       f"http://127.0.0.1:{control_port}"],
                                       TENSORPILOT_CONTROL_API_KEY=CONTROL_KEY, TENSORPILOT_DATA_API_KEY=DATA_KEY,
                                       **data_env))
                await ready(session, f"http://127.0.0.1:{data_port}", processes[-1])
                async with session.post(f"http://127.0.0.1:{data_port}/v1/generate",
                                        json={"prompt": "hi", "max_tokens": 2},
                                        headers={"Authorization": "Bearer " + DATA_KEY}) as response:
                    return response.status, await response.read()
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                process.wait(timeout=10)


def test_gateway_processes_stream_when_the_data_plane_gets_the_worker_key():
    status, body = asyncio.run(run_gateway({"TENSORPILOT_WORKER_API_KEY": WORKER_KEY}))
    events = [json.loads(line) for line in body.decode().splitlines()]
    assert status == 200 and events[-1]["type"] == "done"


def test_gateway_processes_report_a_gateway_fault_when_the_worker_key_is_missing():
    """What session.sh did on the first GPU Pod: no TENSORPILOT_WORKER_API_KEY for the data plane."""
    status, body = asyncio.run(run_gateway({}))
    assert status == 502 and json.loads(body)["error"]["code"] == "worker_unauthorized"
