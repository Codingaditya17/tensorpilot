"""Distributed behavior with the CUDA backend class on CPU, over real loopback HTTP.

These tests exercise the control-plane and data-plane contracts with a real backend
instance behind real aiohttp servers. They do not measure GPU performance and do
not require NVIDIA hardware.
"""
import asyncio
import socket

import aiohttp
import pytest
from aiohttp import web

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_model import make_tiny_model_dir  # noqa: E402
from tensorpilot.control_plane import create_app as create_control_plane  # noqa: E402
from tensorpilot.cuda_backend import CUDABackend  # noqa: E402
from tensorpilot.data_plane import create_app as create_data_plane  # noqa: E402
from tensorpilot.protocol import StreamValidator, decode_event  # noqa: E402
from tensorpilot.server import ENGINE, create_app as create_worker_app  # noqa: E402

MODEL_NAME = "tiny-llama-cpu-fixture"
PROMPT = "hello world the"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return make_tiny_model_dir(tmp_path_factory.mktemp("cluster-model"))


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FaultyBackend(CUDABackend):
    """A real CUDABackend whose decode raises after a configurable number of tokens."""

    fail_after = None

    def stream(self, prompt, max_tokens):
        for count, token in enumerate(super().stream(prompt, max_tokens)):
            if self.fail_after is not None and count == self.fail_after:
                raise RuntimeError("injected backend fault")
            yield token


class Worker:
    """A worker process equivalent: the production worker app around a CUDA-interface backend on CPU."""

    def __init__(self, model_dir, worker_id, instance_id, *, port=0, backend_class=CUDABackend):
        self.model_dir, self.worker_id, self.instance_id = model_dir, worker_id, instance_id
        self.port, self.backend_class = port, backend_class
        self.requests = 0
        self.runner = None
        self.url = None

    def make_backend(self):
        return self.backend_class(str(self.model_dir), device="cpu", allow_cpu=True, dtype="float32")

    async def start(self):
        app = create_worker_app(self.make_backend, queue_size=2, max_tokens=64, warmup=True,
                                metadata={"model": MODEL_NAME, "backend": "cuda"},
                                instance_id=self.instance_id)

        @web.middleware
        async def count_generations(request, handler):
            if request.path == "/v1/generate":
                self.requests += 1
            return await handler(request)

        app.middlewares.append(count_generations)
        self.runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        return self

    @property
    def engine(self):
        return self.runner.app[ENGINE]

    async def stop(self):
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None


async def start_listener(app):
    runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


class Cluster:
    """Control plane and data plane over loopback HTTP, with the given workers registered."""

    def __init__(self, workers, *, probe_interval_s=.1, stale_after_s=5):
        self.workers = workers
        self.probe_interval_s = probe_interval_s
        self.stale_after_s = stale_after_s
        self.runners = []
        self.control_url = None
        self.data_url = None

    async def __aenter__(self):
        for worker in self.workers:
            await worker.start()
        control = create_control_plane(
            workers=[(worker.worker_id, worker.url) for worker in self.workers],
            probe_interval_s=self.probe_interval_s, stale_after_s=self.stale_after_s,
            heartbeat_total_timeout_s=1)
        control_runner, self.control_url = await start_listener(control)
        self.runners.append(control_runner)
        data_runner, self.data_url = await start_listener(create_data_plane(control_url=self.control_url))
        self.runners.append(data_runner)
        await wait_until(self.all_ready)
        return self

    async def __aexit__(self, *exc_info):
        for runner in reversed(self.runners):
            await runner.cleanup()
        for worker in self.workers:
            await worker.stop()

    async def all_ready(self):
        return (await self.control_json("/health"))["worker_counts"]["available"] == len(self.workers)

    async def in_flight_zero(self):
        return (await self.control_json("/health"))["worker_counts"]["in_flight_routes"] == 0

    async def control_json(self, path, method="GET", **kwargs):
        status, body = await self.control_status(path, method, **kwargs)
        return body

    async def control_status(self, path, method="GET", **kwargs):
        async with aiohttp.ClientSession() as session:
            async with session.request(method, self.control_url + path, **kwargs) as response:
                body = await response.json() if response.content_type == "application/json" else None
                return response.status, body

    async def generate(self, body):
        """Stream one request through the data plane; return (events, status, error_body)."""
        async with aiohttp.ClientSession() as session:
            async with session.post(self.data_url + "/v1/generate", json=body) as response:
                if response.status != 200:
                    return [], response.status, await response.json()
                events = []
                async for line in response.content:
                    events.append(decode_event(line))
                return events, 200, None

    async def worker_health(self, worker):
        async with aiohttp.ClientSession() as session:
            async with session.get(worker.url + "/health") as response:
                return await response.json()


async def wait_until(predicate, timeout=20):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(.05)
    raise AssertionError("condition not reached before timeout")


def validated(events, max_tokens):
    """Apply the shared client contract to a stream and return its terminal event."""
    validator = StreamValidator(events[0]["request_id"], max_tokens)
    for event in events:
        validator.feed(event)
    validator.eof()
    return events[-1]


def run(coroutine):
    asyncio.run(coroutine)


def test_cuda_worker_is_placed_streams_validly_and_releases_capacity(model_dir):
    async def scenario():
        worker = Worker(model_dir, "worker-a", "worker-a-gen1")
        async with Cluster([worker]) as cluster:
            events, status, _ = await cluster.generate({"prompt": PROMPT, "max_tokens": 5})
            assert status == 200
            terminal = validated(events, 5)
            assert terminal["type"] == "done"
            assert terminal["placement"]["worker_id"] == "worker-a"
            assert terminal["generated_tokens"] == sum(event.get("output", False) for event in events)
            assert terminal["estimated_kv_bytes"] > 0
            health = await cluster.worker_health(worker)
            assert health["backend"] == "cuda"
            assert health["gpu_telemetry"] is None
            assert health["execution_mode"] == "serial"
            assert health["instance_id"] == "worker-a-gen1"
            assert health["artifact_sha256"]
            assert await cluster.in_flight_zero()
    run(scenario())


def test_model_and_artifact_compatibility_is_enforced_before_dispatch(model_dir):
    async def scenario():
        worker = Worker(model_dir, "worker-a", "worker-a-gen1")
        async with Cluster([worker]) as cluster:
            health = await cluster.worker_health(worker)
            status, body = await cluster.control_status("/v1/routes", "POST", json={
                "prompt": PROMPT, "max_tokens": 2, "model": "some-other-model"})
            assert status == 503
            assert body["error"]["code"] == "no_worker_capacity"
            status, _ = await cluster.control_status("/v1/routes", "POST", json={
                "prompt": PROMPT, "max_tokens": 2, "artifact_sha256": "0" * 64})
            assert status == 503
            status, route = await cluster.control_status("/v1/routes", "POST", json={
                "prompt": PROMPT, "max_tokens": 2, "model": MODEL_NAME,
                "artifact_sha256": health["artifact_sha256"]})
            assert status == 201
            await cluster.control_status(f"/v1/routes/{route['route_id']}/complete", "POST",
                                         json={"outcome": "completed"})
            assert worker.requests == 0
    run(scenario())


def test_client_disconnect_cancels_decode_releases_capacity_and_worker_recovers(model_dir):
    async def scenario():
        worker = Worker(model_dir, "worker-a", "worker-a-gen1")
        async with Cluster([worker]) as cluster:
            async with aiohttp.ClientSession() as session:
                async with session.post(cluster.data_url + "/v1/generate",
                                        json={"prompt": PROMPT, "max_tokens": 64}) as response:
                    assert response.status == 200
                    await response.content.readline()
                    response.close()

            async def worker_idle():
                health = await cluster.worker_health(worker)
                return health["active_sequences"] == 0 and health["queue_depth"] == 0
            await wait_until(cluster.in_flight_zero)
            await wait_until(worker_idle)
            events, status, _ = await cluster.generate({"prompt": PROMPT, "max_tokens": 3})
            assert status == 200
            assert validated(events, 3)["type"] == "done"
    run(scenario())


def test_connection_failure_after_placement_is_terminal_and_releases_route(model_dir):
    async def scenario():
        worker = Worker(model_dir, "worker-a", "worker-a-gen1")
        # A long probe interval keeps the stopped worker "ready" during the proxy attempt,
        # so the data plane, not a heartbeat, must handle the transport failure.
        async with Cluster([worker], probe_interval_s=60, stale_after_s=600) as cluster:
            await worker.stop()
            _, status, body = await cluster.generate({"prompt": PROMPT, "max_tokens": 2})
            assert status == 503
            assert body["error"]["code"] == "worker_transport_error"
            assert await cluster.in_flight_zero()
    run(scenario())


def test_mid_stream_worker_fault_is_terminal_and_never_replayed(model_dir):
    async def scenario():
        worker = Worker(model_dir, "worker-a", "worker-a-gen1", backend_class=FaultyBackend)
        async with Cluster([worker]) as cluster:
            worker.engine.backend.fail_after = 2
            events, status, _ = await cluster.generate({"prompt": PROMPT, "max_tokens": 6})
            assert status == 200
            assert [event["type"] for event in events[:2]] == ["token", "token"]
            terminal = events[-1]
            assert terminal["type"] == "error"
            assert terminal["code"] == "inference_error"
            assert terminal["placement"]["worker_id"] == "worker-a"
            # The request reached the worker exactly once: no retry and no replayed tokens.
            assert worker.requests == 1
            await wait_until(cluster.in_flight_zero)
            worker.engine.backend.fail_after = None
            recovered, status, _ = await cluster.generate({"prompt": PROMPT, "max_tokens": 2})
            assert validated(recovered, 2)["type"] == "done"
    run(scenario())


def test_drain_blocks_new_placements_and_removal_follows_drain(model_dir):
    async def scenario():
        worker = Worker(model_dir, "worker-a", "worker-a-gen1")
        async with Cluster([worker]) as cluster:
            status, _ = await cluster.control_status("/v1/workers/worker-a/drain", "POST",
                                                     json={"draining": True})
            assert status == 200
            _, status, body = await cluster.generate({"prompt": PROMPT, "max_tokens": 2})
            assert status == 503
            assert body["error"]["code"] == "no_worker_capacity"
            status, _ = await cluster.control_status("/v1/workers/worker-a/drain", "POST",
                                                     json={"draining": False})
            assert status == 200
            status, _ = await cluster.control_status("/v1/workers/worker-a", "DELETE")
            assert status == 204
            _, status, _ = await cluster.generate({"prompt": PROMPT, "max_tokens": 2})
            assert status == 503
    run(scenario())


def test_restart_with_new_instance_identity_is_observed_and_serves_again(model_dir):
    async def scenario():
        port = free_port()
        first = Worker(model_dir, "worker-a", "worker-a-gen1", port=port)
        async with Cluster([first]) as cluster:
            await first.stop()

            async def first_instance_gone():
                return (await cluster.control_json("/health"))["worker_counts"]["available"] == 0
            await wait_until(first_instance_gone)

            second = await Worker(model_dir, "worker-a", "worker-a-gen2", port=port).start()
            try:
                async def new_instance_ready():
                    workers = (await cluster.control_json("/v1/workers"))["workers"]
                    return any(item["instance_id"] == "worker-a-gen2" and item["ready"] for item in workers)
                await wait_until(new_instance_ready)
                events, status, _ = await cluster.generate({"prompt": PROMPT, "max_tokens": 2})
                assert status == 200
                assert validated(events, 2)["placement"]["worker_id"] == "worker-a"
            finally:
                await second.stop()
    run(scenario())


def test_integrated_proxy_client_disconnect_does_not_fail_the_worker(model_dir):
    async def scenario():
        worker = Worker(model_dir, "worker-a", "worker-a-gen1")
        async with Cluster([worker]) as cluster:
            async with aiohttp.ClientSession() as session:
                async with session.post(cluster.control_url + "/v1/generate",
                                        json={"prompt": PROMPT, "max_tokens": 64}) as response:
                    assert response.status == 200
                    await response.content.readline()
                    response.close()
            await wait_until(cluster.in_flight_zero)
            summary = await cluster.control_json("/health")
            assert summary["worker_counts"]["available"] == 1
            events, status, _ = await cluster.generate({"prompt": PROMPT, "max_tokens": 2})
            assert status == 200
            assert validated(events, 2)["type"] == "done"
    run(scenario())


def test_worker_registers_through_the_api_and_then_serves(model_dir):
    async def scenario():
        worker = Worker(model_dir, "worker-a", "worker-a-gen1")
        await worker.start()
        control = create_control_plane(workers=(), probe_interval_s=60, stale_after_s=600,
                                       heartbeat_total_timeout_s=1)
        control_runner, control_url = await start_listener(control)
        data_runner, data_url = await start_listener(create_data_plane(control_url=control_url))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(control_url + "/v1/workers",
                                        json={"worker_id": "worker-a", "url": worker.url}) as response:
                    assert response.status == 201
                    registered = await response.json()
                assert registered["ready"] is True
                assert registered["model"] == MODEL_NAME
                assert registered["instance_id"] == "worker-a-gen1"
                async with session.post(data_url + "/v1/generate",
                                        json={"prompt": PROMPT, "max_tokens": 2}) as response:
                    assert response.status == 200
                    events = [decode_event(line) async for line in response.content]
            assert validated(events, 2)["placement"]["worker_id"] == "worker-a"
        finally:
            await data_runner.cleanup()
            await control_runner.cleanup()
            await worker.stop()
    run(scenario())
