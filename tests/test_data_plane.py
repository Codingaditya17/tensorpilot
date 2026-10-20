import asyncio
import json
import uuid

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tensorpilot.control_plane import create_app as create_control
from tensorpilot.data_plane import create_app as create_data


def worker_app(*, broken=False):
    app = web.Application()

    async def health(request):
        return web.json_response({"ready": True, "active_sequences": 0, "queue_depth": 0,
                                  "execution_concurrency": 4, "queue_capacity": 4,
                                  "admission": {"estimated_kv_reserved_bytes": 0,
                                                "estimated_kv_budget_bytes": 512 * 1024**2},
                                  "model": "test", "artifact_sha256": "a" * 64,
                                  "service_version": "test", "instance_id": "worker-1",
                                  "execution_mode": "synthetic"})

    async def generate(request):
        request_id = uuid.uuid4().hex
        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson",
                                               "X-Request-ID": request_id})
        await response.prepare(request)
        await response.write((json.dumps({"type": "token", "text": "x", "token_id": 1,
                                          "output": True, "request_id": request_id}) + "\n").encode())
        if broken:
            raise ConnectionResetError("injected")
        await response.write((json.dumps({"type": "done", "generated_tokens": 1,
                                          "requested": 1, "request_id": request_id}) + "\n").encode())
        return response

    app.router.add_get("/health", health)
    app.router.add_post("/v1/generate", generate)
    return app


def test_data_plane_uses_route_lease_and_releases_capacity():
    async def scenario():
        async with TestServer(worker_app()) as worker:
            control_app = create_control(workers=(("worker-1", str(worker.make_url("/"))),),
                                         probe_interval_s=10)
            async with TestServer(control_app) as control:
                data_app = create_data(control_url=str(control.make_url("/")))
                async with TestClient(TestServer(data_app)) as data:
                    async with data.post("/v1/generate", json={"prompt": "hi", "max_tokens": 1}) as response:
                        assert response.status == 200
                        events = [json.loads(line) async for line in response.content]
                        assert [event["type"] for event in events] == ["token", "done"]
                        assert events[-1]["placement"]["worker_id"] == "worker-1"
                    async with aiohttp.ClientSession() as control_client:
                        async with control_client.get(str(control.make_url("/health"))) as response:
                            health = await response.json()
                            assert health["worker_counts"]["in_flight_routes"] == 0
    asyncio.run(scenario())


def test_data_plane_never_replays_partial_worker_stream():
    calls = 0

    async def scenario():
        nonlocal calls
        app = worker_app(broken=True)
        original = next(route.handler for route in app.router.routes()
                        if route.resource.canonical == "/v1/generate")

        async def counted(request):
            nonlocal calls
            calls += 1
            return await original(request)

        broken = web.Application()
        broken.router.add_get("/health", next(route.handler for route in app.router.routes()
                                               if route.resource.canonical == "/health"))
        broken.router.add_post("/v1/generate", counted)
        async with TestServer(broken) as worker:
            async with TestServer(create_control(workers=(("worker-1", str(worker.make_url("/"))),),
                                                  probe_interval_s=10)) as control:
                async with TestClient(TestServer(create_data(control_url=str(control.make_url("/"))))) as data:
                    async with data.post("/v1/generate", json={"prompt": "hi", "max_tokens": 2}) as response:
                        events = [json.loads(line) async for line in response.content]
                        assert [event["type"] for event in events] == ["token", "error"]
                        assert events[-1]["code"] == "worker_transport_error"
        assert calls == 1
    asyncio.run(scenario())
