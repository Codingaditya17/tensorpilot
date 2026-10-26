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


# --- Authenticated worker. The fake worker above has no auth, which is why the data plane never sending
# a worker credential went unnoticed until a real GPU Pod: its gateway smoke got 5/5 HTTP 401.

WORKER_KEY = "worker-secret-key-0001"
CONTROL_KEY = "control-secret-key-0002"
DATA_KEY = "data-secret-key-0003"


def authenticated_worker():
    from tensorpilot.server import create_app as create_worker
    from test_serving import FakeBackend
    return create_worker(FakeBackend, api_key=WORKER_KEY, warmup=False)


async def gateway(worker, *, data_options, control_options=None, data_key=None):
    """A real control plane and data plane in front of ``worker``; returns an open data-plane client."""
    control_app = create_control(workers=(("worker-1", str(worker.make_url("/"))),), probe_interval_s=10,
                                 api_key=CONTROL_KEY, worker_api_key=WORKER_KEY, **(control_options or {}))
    control = TestServer(control_app)
    await control.start_server()
    data_app = create_data(control_url=str(control.make_url("/")), control_api_key=CONTROL_KEY,
                           api_key=data_key, **data_options)
    client = TestClient(TestServer(data_app))
    await client.start_server()
    return control, client


async def in_flight_routes(control):
    async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + CONTROL_KEY}) as session:
        async with session.get(str(control.make_url("/health"))) as response:
            return (await response.json())["worker_counts"]["in_flight_routes"]


def test_gateway_streams_from_an_authenticated_worker_when_given_the_worker_key():
    async def scenario():
        async with TestServer(authenticated_worker()) as worker:
            control, data = await gateway(worker, data_options={"worker_api_key": WORKER_KEY}, data_key=DATA_KEY)
            try:
                headers = {"Authorization": "Bearer " + DATA_KEY}
                async with data.post("/v1/generate", json={"prompt": "hi", "max_tokens": 2}, headers=headers) as response:
                    assert response.status == 200
                    events = [json.loads(line) async for line in response.content]
                assert events[-1]["type"] == "done" and events[-1]["placement"]["worker_id"] == "worker-1"
                assert await in_flight_routes(control) == 0
            finally:
                await data.close()
                await control.close()
    asyncio.run(scenario())


def test_missing_worker_credential_is_a_gateway_fault_not_a_client_401_and_frees_the_route():
    """The exact Pod failure: the worker returned 401 and the gateway relayed it to the caller."""
    async def scenario():
        async with TestServer(authenticated_worker()) as worker:
            control, data = await gateway(worker, data_options={}, data_key=DATA_KEY)
            try:
                headers = {"Authorization": "Bearer " + DATA_KEY}
                async with data.post("/v1/generate", json={"prompt": "hi", "max_tokens": 2}, headers=headers) as response:
                    body = await response.text()
                    assert response.status == 502
                    assert response.headers["X-TensorPilot-Worker"] == "worker-1"
                error = json.loads(body)["error"]
                assert error["code"] == "worker_unauthorized"
                assert "TENSORPILOT_WORKER_API_KEY" in error["message"]
                for secret in (WORKER_KEY, CONTROL_KEY, DATA_KEY):
                    assert secret not in body
                assert await in_flight_routes(control) == 0
            finally:
                await data.close()
                await control.close()
    asyncio.run(scenario())


def test_the_callers_own_token_is_still_enforced_by_the_data_plane():
    async def scenario():
        async with TestServer(authenticated_worker()) as worker:
            control, data = await gateway(worker, data_options={"worker_api_key": WORKER_KEY}, data_key=DATA_KEY)
            try:
                async with data.post("/v1/generate", json={"prompt": "hi", "max_tokens": 1}) as response:
                    assert response.status == 401
                    assert (await response.json())["error"]["message"] == "A valid data-plane token is required"
            finally:
                await data.close()
                await control.close()
    asyncio.run(scenario())


def test_worker_key_is_never_sent_to_a_host_outside_the_allow_list():
    seen = []

    def recording_worker():
        app = worker_app()

        @web.middleware
        async def record(request, handler):
            if request.path == "/v1/generate":
                seen.append(request.headers.get("Authorization"))
            return await handler(request)

        app.middlewares.append(record)
        return app

    async def scenario():
        async with TestServer(recording_worker()) as worker:
            # The worker is registered as 127.0.0.1; this data plane only trusts "localhost".
            control, data = await gateway(worker, data_options={"worker_api_key": WORKER_KEY,
                                                                "allowed_worker_hosts": ("localhost",)},
                                          control_options={})
            try:
                async with data.post("/v1/generate", json={"prompt": "hi", "max_tokens": 1},
                                     headers={}) as response:
                    assert response.status == 502
                    assert (await response.json())["error"]["code"] == "worker_host_not_allowed"
                assert seen == []
                assert await in_flight_routes(control) == 0
            finally:
                await data.close()
                await control.close()
    asyncio.run(scenario())
