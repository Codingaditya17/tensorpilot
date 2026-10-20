import asyncio
import json
import uuid

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tensorpilot.control_plane import create_app
from tensorpilot.protocol import StreamValidator


def worker_app(worker_id, artifact="artifact"):
    app = web.Application()

    async def health(request):
        return web.json_response({"ready": True, "active_sequences": 0, "queue_depth": 0,
                                  "execution_concurrency": 2, "queue_capacity": 2,
                                  "admission": {"estimated_kv_reserved_bytes": 0,
                                                "estimated_kv_budget_bytes": 512 * 1024**2},
                                  "model": "test-model", "artifact_sha256": artifact,
                                  "service_version": "0.7.0", "instance_id": worker_id,
                                  "execution_mode": "adaptive"})

    async def generate(request):
        body = await request.json()
        request_id = uuid.uuid4().hex
        await asyncio.sleep(.02)
        events = [{"type": "token", "text": worker_id, "token_id": 1, "output": True,
                   "request_id": request_id},
                  {"type": "done", "generated_tokens": 1, "queue_wait_s": 0,
                   "server_elapsed_s": .01, "ttft_s": .005, "request_id": request_id,
                   "requested": body["max_tokens"]}]
        return web.Response(text="".join(json.dumps(event) + "\n" for event in events),
                            content_type="application/x-ndjson", headers={"X-Request-ID": request_id})

    app.router.add_get("/health", health)
    app.router.add_post("/v1/generate", generate)
    return app


def test_proxy_balances_concurrent_streams_and_explains_placement():
    async def scenario():
        async with TestServer(worker_app("worker-a")) as first, TestServer(worker_app("worker-b")) as second:
            workers = (("worker-a", str(first.make_url("/"))), ("worker-b", str(second.make_url("/"))))
            async with TestClient(TestServer(create_app(workers=workers, probe_interval_s=.01))) as control:
                async def generate():
                    async with control.post("/v1/generate", json={"prompt": "hi", "max_tokens": 1}) as response:
                        assert response.status == 200
                        request_id = response.headers["X-Request-ID"]
                        events = [json.loads(line) async for line in response.content]
                        validator = StreamValidator(request_id, 1)
                        for event in events:
                            validator.feed(event)
                        validator.eof()
                        return response.headers["X-TensorPilot-Worker"], events[-1]
                results = await asyncio.gather(generate(), generate())
                assert {worker for worker, _ in results} == {"worker-a", "worker-b"}
                assert all(done["placement"]["reason"] == "lowest-load-then-memory"
                           for _, done in results)
                async with control.get("/health") as response:
                    body = await response.json()
                    assert body["ready"] and body["worker_counts"]["available"] == 2
                    assert body["placements"] == 2
                    assert "workers" not in body
    asyncio.run(scenario())


def test_worker_api_pages_and_rejects_cursor_after_membership_change():
    async def scenario():
        async with TestServer(worker_app("shared")) as worker:
            url = str(worker.make_url("/"))
            fleet = tuple((f"worker-{index:02d}", url) for index in range(5))
            async with TestClient(TestServer(create_app(workers=fleet))) as control:
                async with control.get("/v1/workers?limit=2") as response:
                    assert response.status == 200
                    first = await response.json()
                assert [item["worker_id"] for item in first["workers"]] == ["worker-00", "worker-01"]
                cursor = first["next_cursor"]
                async with control.get("/v1/workers", params={"limit": 2, "cursor": cursor}) as response:
                    second = await response.json()
                assert [item["worker_id"] for item in second["workers"]] == ["worker-02", "worker-03"]
                async with control.delete("/v1/workers/worker-04") as response:
                    assert response.status == 204
                async with control.get("/v1/workers", params={"limit": 2, "cursor": cursor}) as response:
                    assert response.status == 409
                    assert (await response.json())["error"]["code"] == "stale_cursor"
    asyncio.run(scenario())


def test_heartbeat_concurrency_is_bounded_for_slow_workers():
    active = 0
    maximum = 0

    async def scenario():
        nonlocal active, maximum
        app = worker_app("shared")
        health_route = next(route for route in app.router.routes()
                            if route.resource.canonical == "/health" and route.method == "GET")
        original = health_route.handler

        async def slow_health(request):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(.02)
                return await original(request)
            finally:
                active -= 1

        replacement = web.Application()
        replacement.router.add_get("/health", slow_health)
        replacement.router.add_post("/v1/generate", next(
            route.handler for route in app.router.routes()
            if route.resource.canonical == "/v1/generate" and route.method == "POST"))
        async with TestServer(replacement) as worker:
            url = str(worker.make_url("/"))
            fleet = tuple((f"worker-{index}", url) for index in range(8))
            control_app = create_app(workers=fleet, heartbeat_concurrency=2,
                                     probe_interval_s=10)
            async with TestClient(TestServer(control_app)) as control:
                async with control.get("/metrics") as response:
                    metrics = await response.text()
                assert "tensorpilot_cluster_heartbeat_duration_seconds_count 8.0" in metrics
        assert maximum == 2
    asyncio.run(scenario())


def test_drain_and_no_capacity_are_explicit():
    async def scenario():
        async with TestServer(worker_app("worker-a")) as worker:
            workers = (("worker-a", str(worker.make_url("/"))),)
            async with TestClient(TestServer(create_app(workers=workers))) as control:
                async with control.post("/v1/workers/worker-a/drain", json={"draining": True}) as response:
                    assert response.status == 200
                    assert (await response.json())["draining"]
                async with control.post("/v1/generate", json={"prompt": "hi"}) as response:
                    assert response.status == 503
                    assert (await response.json())["error"]["code"] == "no_worker_capacity"
                    assert response.headers["Retry-After"] == "1"
                async with control.post("/v1/workers/worker-a/drain", json={"draining": False}) as response:
                    assert response.status == 200
                async with control.post("/v1/generate", json={"prompt": "hi", "model": "wrong"}) as response:
                    assert response.status == 503
    asyncio.run(scenario())


def test_registration_auth_and_ssrf_boundaries():
    async def scenario():
        async with TestClient(TestServer(create_app(api_key="secret"))) as control:
            async with control.get("/v1/workers") as response:
                assert response.status == 401
            headers = {"Authorization": "Bearer secret"}
            async with control.post("/v1/workers", headers=headers,
                                    json={"worker_id": "remote", "url": "http://example.com:80"}) as response:
                assert response.status == 400
                assert (await response.json())["error"]["code"] == "invalid_worker"
            async with control.get("/livez") as response:
                assert response.status == 200
    asyncio.run(scenario())


def test_midstream_worker_failure_is_terminal_and_never_replayed():
    calls = [0, 0]

    async def broken(request):
        calls[0] += 1
        request_id = uuid.uuid4().hex
        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson",
                                               "X-Request-ID": request_id})
        await response.prepare(request)
        await response.write((json.dumps({"type": "token", "text": "x", "token_id": 1,
                                          "output": True, "request_id": request_id}) + "\n").encode())
        raise ConnectionResetError("injected worker failure")

    async def healthy(request):
        calls[1] += 1
        return web.Response(status=500)

    async def scenario():
        worker = worker_app("worker-a")
        worker.router.add_route("POST", "/broken", broken)
        # Replace the registered generate route by using a dedicated app.
        app = web.Application()
        app.router.add_get("/health", next(route.handler for route in worker.router.routes()
                                           if route.resource.canonical == "/health" and route.method == "GET"))
        app.router.add_post("/v1/generate", broken)
        app.router.add_post("/never", healthy)
        async with TestServer(app) as server:
            workers = (("worker-a", str(server.make_url("/"))),)
            async with TestClient(TestServer(create_app(workers=workers))) as control:
                async with control.post("/v1/generate", json={"prompt": "hi", "max_tokens": 2}) as response:
                    events = [json.loads(line) async for line in response.content]
                    assert [event["type"] for event in events] == ["token", "error"]
                    assert events[-1]["code"] == "worker_transport_error"
                    assert events[-1]["placement"]["worker_id"] == "worker-a"
        assert calls == [1, 0]
    asyncio.run(scenario())
